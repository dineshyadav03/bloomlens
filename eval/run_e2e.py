"""Capped, resumable end-to-end pilot (eval/PROTOCOL.md section 9): real identify() calls --
BioCLIP 2 + Qdrant + the live Gemini agent, using the frozen retrieval threshold -- on the fixed
`pilot` sample (eval/pilot_sample.py) only. **A pilot, not a benchmark**: wide CIs, no headline
claim; eval/e2e_pilot_results.md says so explicitly.

    uv run python eval/run_e2e.py --max-calls 20

Needs GEMINI_API_KEY. Each identify() call is several real Gemini requests (docs/telemetry_probe.md)
and spends free-tier quota, so progress is cached to a resumable JSONL file keyed by (the pilot
entry's own identity -- source id, plus variant if corrupted; see eval/open_world_data.py -- the
Gemini model, and the git revision): re-running only attempts entries that are still missing or
whose key has changed (a model or code update invalidates their cache entry, not the whole file).
Keyed by entry identity rather than image content on purpose: this dataset already contains two
byte-identical Oxford photos filed under different names (found in M15a's leakage test), so a
content hash would silently collapse two distinct pilot entries into one cache slot. `--max-calls`
bounds how many NEW identify() calls one invocation makes; a quota error also stops the run early
with the cache intact, so the next invocation resumes.

Per-call outcome fields come straight from IdentifyResult and from the shared telemetry row
identify() itself just wrote (src/telemetry.py) -- nothing here re-implements timing or token
counting. Nothing about a scan's own image, prompt or model text is stored, per the same rule
telemetry follows.
"""

import argparse
import hashlib
import json
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from eval import open_world_data  # noqa: E402
from eval.build_datasets import MANIFEST_DIR  # noqa: E402
from eval.frozen_config import git_revision  # noqa: E402
from eval.oxford102 import RAW_ROOT  # noqa: E402
from eval.pilot_sample import select_pilot_sample  # noqa: E402
from src import db  # noqa: E402
from src.identify import IdentifyError, identify  # noqa: E402
from src.versions import GEMINI_MODEL  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "e2e_pilot.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent / "e2e_pilot_results.md"
DEFAULT_MAX_CALLS = 20


def load_manifest(name: str) -> dict:
    return json.loads((MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8"))


def cache_entry_key(entry_key: str, model: str, code_revision: str) -> str:
    """One resumable-cache key: a code or model change invalidates just this entry, not the file.

    Keyed by the pilot *entry's own identity* (`eval.open_world_data.cache_key`: source id, plus
    variant for a corrupted one) -- not by the image's byte content. A content hash was tried
    first and had to be dropped: this dataset already contains two byte-identical Oxford photos
    saved under different names (found the hard way in M15a's leakage test), so two distinct
    pilot entries can share one image's bytes; keying on content silently collapsed them into a
    single cache slot and one of the two never got its own identify() call."""
    return hashlib.sha256(f"{entry_key}|{model}|{code_revision}".encode()).hexdigest()


def load_cache(path: Path | None = None) -> dict[str, dict]:
    """cache key -> its recorded result. Malformed trailing lines (a run killed mid-write) are
    skipped, not fatal."""
    path = CACHE_PATH if path is None else path
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[record["cache_key"]] = record
    return records


def append_result(record: dict, path: Path | None = None) -> None:
    path = CACHE_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def latest_telemetry_row(after_id: int) -> dict | None:
    """The scan_metrics row identify() itself just wrote (its id is > `after_id`), or None."""
    with closing(db.connect()) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(scan_metrics)")]
        row = conn.execute(
            "SELECT * FROM scan_metrics WHERE id > ? ORDER BY id DESC LIMIT 1", (after_id,)
        ).fetchone()
    return dict(zip(columns, row, strict=True)) if row else None


def latest_scan_metrics_id() -> int:
    with closing(db.connect()) as conn:
        row = conn.execute("SELECT MAX(id) FROM scan_metrics").fetchone()
    return row[0] or 0


def run_one(entry: dict, *, model: str, code_revision: str) -> dict:
    """Run identify() on one pilot entry and return its cache record. Never raises: a pipeline
    failure is recorded as an outcome, like any other result."""
    cache_key = cache_entry_key(open_world_data.cache_key(entry), model, code_revision)
    record = {
        "cache_key": cache_key,
        "source_id": entry["source_id"],
        "family": entry["family"],
        "label": entry["label"],
        "gemini_model": model,
        "code_revision": code_revision,
    }

    before_id = latest_scan_metrics_id()
    try:
        with Image.open(RAW_ROOT / entry["path"]) as opened:
            result = identify(opened.convert("RGB"))
        record.update(
            status="ok",
            failure_category=None,
            final_species=result.species,
            retrieval_top1_species=result.top_candidates[0]["common_name"] if result.top_candidates else None,
            agrees_with_retrieval=(
                result.species == result.top_candidates[0]["common_name"] if result.top_candidates else None
            ),
            confidence_tier=result.confidence_tier,
            abstained=result.abstained,
            abstain_source=result.abstain_source,
            schema_valid=True,
        )
    except IdentifyError as exc:
        record.update(
            status="error",
            failure_category=exc.category,
            final_species=None,
            retrieval_top1_species=None,
            agrees_with_retrieval=None,
            confidence_tier=None,
            abstained=None,
            abstain_source=None,
            schema_valid=False,
        )

    telemetry_row = latest_telemetry_row(before_id)
    for field in ("total_ms", "attempts", "input_tokens", "output_tokens", "est_cost_usd", "model_reported"):
        record[field] = telemetry_row.get(field) if telemetry_row else None
    return record


def summarize(records: list[dict]) -> dict:
    import numpy as np

    ok = [r for r in records if r["status"] == "ok"]
    by_family = {}
    for family in ("id", "near_ood", "far_ood"):
        subset = [r for r in records if r["family"] == family]
        ok_subset = [r for r in subset if r["status"] == "ok"]
        by_family[family] = {
            "n": len(subset),
            "n_ok": len(ok_subset),
            "abstained_rate": (
                sum(1 for r in ok_subset if r["abstained"]) / len(ok_subset) if ok_subset else None
            ),
            "agreement_with_retrieval": (
                sum(1 for r in ok_subset if r["agrees_with_retrieval"]) / len(ok_subset) if ok_subset else None
            ),
        }

    latencies = [r["total_ms"] for r in ok if r["total_ms"] is not None]
    attempts = [r["attempts"] for r in records if r["attempts"] is not None]
    tokens_in = [r["input_tokens"] for r in ok if r["input_tokens"] is not None]
    tokens_out = [r["output_tokens"] for r in ok if r["output_tokens"] is not None]
    costs = [r["est_cost_usd"] for r in ok if r["est_cost_usd"] is not None]
    failures: dict[str, int] = {}
    for r in records:
        if r["status"] == "error":
            failures[r["failure_category"]] = failures.get(r["failure_category"], 0) + 1

    return {
        "n_total": len(records),
        "n_ok": len(ok),
        "schema_valid_rate": len(ok) / len(records) if records else None,
        "failures_by_category": failures,
        "by_family": by_family,
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
            "n": len(latencies),
        },
        "mean_attempts": float(np.mean(attempts)) if attempts else None,
        "tokens": {
            "mean_input": float(np.mean(tokens_in)) if tokens_in else None,
            "mean_output": float(np.mean(tokens_out)) if tokens_out else None,
            "n_with_counts": len(tokens_in),
        },
        "estimated_cost_usd": {"total": sum(costs) if costs else None, "n_priced": len(costs)},
    }


def _cost_line(cost: dict) -> str:
    if cost["total"] is None:
        return "Estimated cost: not available"
    return (
        f"Estimated cost: ${cost['total']:.4f} over {cost['n_priced']} priced calls "
        "-- estimated list-price equivalent; actual billed cost unknown"
    )


def render(summary: dict, sample_size: int) -> str:
    complete = summary["n_total"] >= sample_size
    lines = [
        "# End-to-end pilot -- NOT a benchmark (eval/PROTOCOL.md section 9)",
        "",
        f"{summary['n_total']} of {sample_size} pilot images attempted so far"
        + ("" if complete else " (**partial**: capped by --max-calls; re-run to continue)")
        + f". {summary['n_ok']} succeeded"
        + (f" ({summary['schema_valid_rate']:.1%} schema-valid)." if summary["schema_valid_rate"] is not None else "."),
        "",
        "**This is a small, capped sample. Confidence intervals would be wide and are not computed; "
        "nothing here is a headline claim.**",
        "",
        "| family | n | n ok | abstained rate | agrees with retrieval top-1 |",
        "|---|---:|---:|---|---|",
    ]
    for family, stats in summary["by_family"].items():
        abstained = f"{stats['abstained_rate']:.1%}" if stats["abstained_rate"] is not None else "n/a"
        agree = stats["agreement_with_retrieval"]
        agreement = f"{agree:.1%}" if agree is not None else "n/a"
        lines.append(f"| {family} | {stats['n']} | {stats['n_ok']} | {abstained} | {agreement} |")

    latency = summary["latency_ms"]
    lines += [
        "",
        f"Latency (successful calls, n={latency['n']}): "
        + (f"p50 {latency['p50']:.0f} ms, p95 {latency['p95']:.0f} ms" if latency["n"] else "not enough data"),
        f"Mean attempts per call: {summary['mean_attempts']:.2f}" if summary["mean_attempts"] is not None else "",
        f"Tokens (n={summary['tokens']['n_with_counts']} reported): "
        + (
            f"mean {summary['tokens']['mean_input']:.0f} in / {summary['tokens']['mean_output']:.0f} out"
            if summary["tokens"]["n_with_counts"]
            else "none reported"
        ),
        _cost_line(summary["estimated_cost_usd"]),
        "",
        "Failures by category: "
        + (", ".join(f"{k}={v}" for k, v in summary["failures_by_category"].items()) or "none"),
    ]
    return "\n".join(line for line in lines if line != "") + "\n"


def sample_cache_key(entry: dict, model: str, code_revision: str) -> str:
    return cache_entry_key(open_world_data.cache_key(entry), model, code_revision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    args = parser.parse_args()

    sample = select_pilot_sample(load_manifest("id"), load_manifest("near_ood"), load_manifest("far_ood"))
    code_revision = git_revision()
    cache = load_cache()
    keys = {entry["source_id"]: sample_cache_key(entry, GEMINI_MODEL, code_revision) for entry in sample}

    calls_made = 0
    for entry in sample:
        key = keys[entry["source_id"]]
        if key in cache:
            continue
        remaining = sum(1 for e in sample if keys[e["source_id"]] not in cache)
        if calls_made >= args.max_calls:
            print(f"--max-calls ({args.max_calls}) reached; {remaining} pilot images remain.")
            break
        print(f"[{calls_made + 1}/{args.max_calls}] {entry['family']}:{entry['source_id']}...", end=" ", flush=True)
        try:
            record = run_one(entry, model=GEMINI_MODEL, code_revision=code_revision)
        except Exception as exc:  # noqa: BLE001 -- a quota/network failure must not lose prior progress
            print(f"stopped: {type(exc).__name__}")
            break
        append_result(record)
        cache[key] = record
        calls_made += 1
        print(record["status"])

    records = [cache[keys[entry["source_id"]]] for entry in sample if keys[entry["source_id"]] in cache]
    summary = summarize(records)
    RESULTS_PATH.write_text(render(summary, len(sample)), encoding="utf-8", newline="\n")
    print(f"\n{len(records)}/{len(sample)} pilot images done. Wrote {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
