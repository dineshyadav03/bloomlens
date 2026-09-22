"""Evaluate `test` against the frozen config -- exactly once (eval/PROTOCOL.md section 5).

    uv run python eval/run_open_world.py

Reads eval/frozen/v1.json (written by eval/select_threshold.py; refuses if it does not exist or
was edited since), embeds every `test`-split image of every family (cached -- see
eval/open_world_data.py), computes the per-family metrics of section 6, writes
eval/open_world_results.md, and only then writes eval/frozen/v1.lock. A second run for the same
config hash refuses outright: there is no "does it replicate?" step, and a poor result is
published as it is. Changing anything the config depends on requires a new protocol version.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import frozen_config, metrics, open_world_data, scores  # noqa: E402
from eval.build_datasets import MANIFEST_DIR, manifest_hash  # noqa: E402
from eval.select_threshold import abstention_rate  # noqa: E402
from src.versions import BIOCLIP_REVISION  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent / "open_world_results.md"


def load_test_entries(name: str) -> tuple[list[dict], dict]:
    document = json.loads((MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [e for e in document["entries"] if e["split"] == "test"], document


def embed(entries: list[dict], *, label: str) -> dict[str, list[dict]]:
    return open_world_data.similarities_for(
        entries, progress=lambda done, total: print(f"  {label} {done}/{total}", end="\r")
    )


def scores_of(similarities: dict, entries: list[dict], candidate: str) -> list[float]:
    return [
        scores.score_by_name(candidate, open_world_data.scores_only(similarities[open_world_data.cache_key(e)]))
        for e in entries
    ]


def group_by(entries: list[dict], key: str, values: list[float]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for entry, value in zip(entries, values, strict=True):
        groups[entry[key]].append(value)
    return groups


def family_metrics(id_test_scores: list[float], ood_entries: list[dict], ood_scores: list[float], tau: float) -> dict:
    groups = group_by(ood_entries, "group", ood_scores)
    return {
        "n_images": len(ood_scores),
        "n_categories": len(groups),
        "auroc": metrics.bootstrap_ci_joint(id_test_scores, groups, metrics.auroc),
        "aupr_out": metrics.bootstrap_ci_joint(id_test_scores, groups, metrics.aupr_out),
        "fpr_at_95tpr": metrics.bootstrap_ci_joint(id_test_scores, groups, metrics.fpr_at_tpr),
        "abstention_rate": metrics.bootstrap_ci_grouped(groups, lambda pooled, tau=tau: abstention_rate(pooled, tau)),
    }


def id_metrics(id_test: list[dict], id_similarities: dict, id_scores: list[float], tau: float) -> dict:
    predicted = [open_world_data.top1_species(id_similarities[open_world_data.cache_key(e)]) for e in id_test]
    correct = [entry["label"] == species for entry, species in zip(id_test, predicted, strict=True)]
    paired = list(zip(id_scores, correct, strict=True))

    def selective(pairs, field):
        s, c = zip(*pairs, strict=True)
        return metrics.selective_accuracy(list(s), list(c), threshold=tau)[field]

    return {
        "n_images": len(id_test),
        "top1_accuracy_unfiltered": sum(correct) / len(correct),
        "false_abstain_rate": metrics.bootstrap_ci(id_scores, lambda s: abstention_rate(s, tau)),
        "coverage_at_tau": metrics.bootstrap_ci(paired, lambda pairs: selective(pairs, "coverage")),
        "selective_accuracy_at_tau": metrics.bootstrap_ci(paired, lambda pairs: selective(pairs, "selective_accuracy")),
        "risk_coverage_auc": metrics.bootstrap_ci(
            paired, lambda pairs: metrics.risk_coverage_auc(*(list(x) for x in zip(*pairs, strict=True)))
        ),
    }


def corrupted_metrics(entries: list[dict], similarities: dict, candidate: str, tau: float) -> dict:
    by_variant: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for entry in entries:
        sims = similarities[open_world_data.cache_key(entry)]
        score = scores.score_by_name(candidate, open_world_data.scores_only(sims))
        correct = open_world_data.top1_species(sims) == entry["label"]
        by_variant[entry["variant"]].append((correct, score < tau))
    return {
        variant: {
            "n_images": len(pairs),
            "top1_accuracy": metrics.bootstrap_ci([p[0] for p in pairs], lambda c: sum(c) / len(c)),
            "abstention_rate": metrics.bootstrap_ci([p[1] for p in pairs], lambda a: sum(a) / len(a)),
        }
        for variant, pairs in sorted(by_variant.items())
    }


def _fmt_ci(block: dict, as_percent: bool = False) -> str:
    scale = 100 if as_percent else 1
    unit = "%" if as_percent else ""
    return f"{block['point'] * scale:.2f}{unit} [{block['low'] * scale:.2f}, {block['high'] * scale:.2f}]"


def render(config, family_results: dict, id_results: dict, corrupted_results: dict, config_hash: str) -> str:
    lines = [
        "# Open-world evaluation -- `test`, evaluated once (eval/PROTOCOL.md)",
        "",
        f"Frozen rule: `{config.effective_candidate}` at tau = {config.effective_tau:.6f} "
        f"({'adopted' if config.adopted else 'baseline kept -- see eval/tuning_report.md'}). "
        f"Config `{config_hash[:12]}...`, BioCLIP `{config.bioclip_revision}`, code `{config.code_revision[:12]}`.",
        "",
        "**Not evaluated: quality (no labels), price (simulated), or the full agent (see the M15c pilot).**",
        "",
        "## In-distribution (test)",
        f"n = {id_results['n_images']}. Unfiltered top-1 accuracy {id_results['top1_accuracy_unfiltered']:.1%}.",
        "",
        "| metric | value [95% CI] |",
        "|---|---|",
        f"| false-abstain rate at tau | {_fmt_ci(id_results['false_abstain_rate'], True)} |",
        f"| coverage at tau | {_fmt_ci(id_results['coverage_at_tau'], True)} |",
        f"| selective accuracy at tau | {_fmt_ci(id_results['selective_accuracy_at_tau'], True)} |",
        f"| risk-coverage AUC (lower is better) | {_fmt_ci(id_results['risk_coverage_auc'])} |",
        "",
    ]
    for family, result in family_results.items():
        lines += [
            f"## {family} (test)",
            f"n = {result['n_images']} images, {result['n_categories']} categories.",
            "",
            "| metric | value [95% CI] |",
            "|---|---|",
            f"| AUROC (ID positive) | {_fmt_ci(result['auroc'])} |",
            f"| AUPR-Out (OOD positive) | {_fmt_ci(result['aupr_out'])} |",
            f"| FPR @ 95% TPR | {_fmt_ci(result['fpr_at_95tpr'], True)} |",
            f"| abstention rate (correct rejection) | {_fmt_ci(result['abstention_rate'], True)} |",
            "",
        ]
    lines += [
        "## Corrupted-in-set (test) -- robustness, not OOD",
        "",
        "| corruption | n | top-1 accuracy | abstention rate |",
        "|---|---:|---|---|",
    ]
    for variant, result in corrupted_results.items():
        lines.append(
            f"| {variant} | {result['n_images']} | {_fmt_ci(result['top1_accuracy'], True)} "
            f"| {_fmt_ci(result['abstention_rate'], True)} |"
        )
    lines += ["", "No per-species claims are made: see eval/PROTOCOL.md for why."]
    return "\n".join(lines) + "\n"


def main() -> None:
    try:
        config, config_hash = frozen_config.read_frozen()
    except FileNotFoundError:
        sys.exit(f"{frozen_config.CONFIG_PATH} does not exist. Run eval/select_threshold.py first.")

    if frozen_config.already_run(config_hash):
        sys.exit(
            f"{frozen_config.LOCK_PATH} already records a completed run for this exact config "
            f"({config_hash[:12]}...). Per eval/PROTOCOL.md section 5, `test` is evaluated once: "
            "there is no re-run for the same frozen config. Start a new protocol version to change anything."
        )
    if config.bioclip_revision != BIOCLIP_REVISION:
        sys.exit(
            f"the frozen config was built against BioCLIP revision {config.bioclip_revision}, "
            f"but this checkout is pinned to {BIOCLIP_REVISION}. Re-run eval/select_threshold.py first."
        )

    id_test, id_doc = load_test_entries("id")
    near_ood_test, near_ood_doc = load_test_entries("near_ood")
    far_ood_test, _far_ood_doc = load_test_entries("far_ood")
    corrupted_test, _corrupted_doc = load_test_entries("corrupted")
    for name, document, split_field in (("id", id_doc, "id"), ("near_ood", near_ood_doc, "near_ood")):
        expected = config.manifest_hashes.get(split_field)
        if expected and expected != manifest_hash(document):
            sys.exit(f"eval/manifest/{name}.json has changed since the config was frozen. Re-run selection first.")

    print(
        f"embedding {len(id_test)} ID-test, {len(near_ood_test)} near-OOD-test, "
        f"{len(far_ood_test)} far-OOD-test, {len(corrupted_test)} corrupted-test images..."
    )
    id_similarities = embed(id_test, label="id-test")
    near_ood_similarities = embed(near_ood_test, label="near-ood-test")
    far_ood_similarities = embed(far_ood_test, label="far-ood-test")
    corrupted_similarities = embed(corrupted_test, label="corrupted-test")
    print()

    candidate = config.effective_candidate
    tau = config.effective_tau
    id_scores = scores_of(id_similarities, id_test, candidate)
    near_ood_scores = scores_of(near_ood_similarities, near_ood_test, candidate)
    far_ood_scores = scores_of(far_ood_similarities, far_ood_test, candidate)

    family_results = {
        "near_ood": family_metrics(id_scores, near_ood_test, near_ood_scores, tau),
        "far_ood": family_metrics(id_scores, far_ood_test, far_ood_scores, tau),
    }
    id_results = id_metrics(id_test, id_similarities, id_scores, tau)
    corrupted_results = corrupted_metrics(corrupted_test, corrupted_similarities, candidate, tau)

    RESULTS_PATH.write_text(
        render(config, family_results, id_results, corrupted_results, config_hash), encoding="utf-8", newline="\n"
    )
    frozen_config.write_lock(config_hash, results_path=RESULTS_PATH.name)
    print(f"wrote {RESULTS_PATH} and locked config {config_hash[:12]}... -- this config will not run again.")


if __name__ == "__main__":
    main()
