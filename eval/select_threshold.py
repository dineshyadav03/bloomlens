"""Tune the abstention rule on `dev` only, then freeze it (eval/PROTOCOL.md section 4).

    uv run python eval/select_threshold.py

Reads the committed `dev` entries of eval/manifest/id.json and near_ood.json, embeds every image
(cached -- see eval/open_world_data.py), computes every candidate score, picks one by dev AUROC,
sets tau, decides whether it beats the baseline enough to be adopted, and writes
eval/frozen/v1.json (with its own content hash) plus eval/tuning_report.md. It never touches `test`
or `pilot`. Running it again overwrites the frozen file -- once eval/run_open_world.py has locked
that hash, this script refuses too, so tuning cannot resume after the test set has been evaluated.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import frozen_config, metrics, open_world_data, scores  # noqa: E402
from eval.build_datasets import MANIFEST_DIR, manifest_hash  # noqa: E402
from eval.frozen_config import (  # noqa: E402
    BASELINE_CANDIDATE,
    BASELINE_TAU,
    CANDIDATE_PREFERENCE,
    MAX_ID_ABSTENTION,
    MIN_NEAR_OOD_GAIN,
    TIE_BAND,
    FrozenConfig,
)
from src.versions import BIOCLIP_REVISION  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "tuning_report.md"


def load_dev_entries(name: str) -> tuple[list[dict], dict]:
    document = json.loads((MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return [e for e in document["entries"] if e["split"] == "dev"], document


def candidate_scores(similarities_by_key: dict, entries: list[dict]) -> dict[str, list[float]]:
    """candidate name -> that candidate's score for every entry, in entry order."""
    per_entry = [
        scores.all_candidates(open_world_data.scores_only(similarities_by_key[open_world_data.cache_key(e)]))
        for e in entries
    ]
    return {name: [row[name] for row in per_entry] for name in per_entry[0]}


def pick_candidate(id_dev: dict[str, list[float]], ood_dev: dict[str, list[float]]) -> tuple[str, dict[str, float]]:
    """The candidate with the best dev AUROC (ID positive), ties within TIE_BAND broken by
    CANDIDATE_PREFERENCE order. Returns (chosen name, every candidate's AUROC)."""
    auroc_by_candidate = {name: metrics.auroc(id_dev[name], ood_dev[name]) for name in id_dev}
    best = max(auroc_by_candidate.values())
    tied = {name for name, value in auroc_by_candidate.items() if best - value <= TIE_BAND}
    chosen = next(name for name in CANDIDATE_PREFERENCE if name in tied)
    return chosen, auroc_by_candidate


def compute_tau(id_dev_scores: list[float]) -> float:
    """The largest value that still accepts at least 95% of `id_dev_scores` (the 5th percentile,
    'lower' interpolation so it is an actual achievable score, never an interpolated one)."""
    return float(np.percentile(id_dev_scores, 5, method="lower"))


def abstention_rate(values: list[float], tau: float) -> float:
    return float(np.mean(np.asarray(values) < tau))


def decide_adoption(near_ood_gain_pp: float, id_abstention: float) -> bool:
    """eval/PROTOCOL.md section 4 step 4, exactly: adopt only if the new rule abstains on at
    least MIN_NEAR_OOD_GAIN more of near-OOD-dev than the baseline, while abstaining on at most
    MAX_ID_ABSTENTION of ID-dev."""
    return near_ood_gain_pp >= MIN_NEAR_OOD_GAIN and id_abstention <= MAX_ID_ABSTENTION


def build_config(
    id_scores_by_candidate: dict[str, list[float]], ood_scores_by_candidate: dict[str, list[float]], *, manifests: dict
) -> FrozenConfig:
    chosen, per_candidate_auroc = pick_candidate(id_scores_by_candidate, ood_scores_by_candidate)
    tau = compute_tau(id_scores_by_candidate[chosen])
    new_ood_abstention = abstention_rate(ood_scores_by_candidate[chosen], tau)
    baseline_ood_abstention = abstention_rate(ood_scores_by_candidate[BASELINE_CANDIDATE], BASELINE_TAU)
    id_abstention = abstention_rate(id_scores_by_candidate[chosen], tau)
    gain_pp = new_ood_abstention - baseline_ood_abstention
    return FrozenConfig(
        candidate=chosen,
        tau=tau,
        adopted=decide_adoption(gain_pp, id_abstention),
        dev_near_ood_gain_pp=gain_pp,
        dev_id_abstention=id_abstention,
        candidate_auroc=per_candidate_auroc,
        manifest_hashes=manifests,
        bioclip_revision=BIOCLIP_REVISION,
        code_revision=frozen_config.git_revision(),
        created_at=frozen_config.now_iso(),
    )


def report(config: FrozenConfig) -> str:
    lines = [
        "# Threshold selection (dev only) -- eval/PROTOCOL.md section 4",
        "",
        f"Frozen: `{config.candidate}`, tau = {config.tau:.6f}.",
        f"**Adopted: {config.adopted}** "
        f"(near-OOD-dev abstention gain {config.dev_near_ood_gain_pp:+.1%} vs the "
        f"{MIN_NEAR_OOD_GAIN:.0%} bar; ID-dev abstention {config.dev_id_abstention:.1%} vs the "
        f"{MAX_ID_ABSTENTION:.0%} ceiling).",
        "" if config.adopted else "The baseline tier-`low` gate stays in force (see `effective_candidate`).",
        "",
        "| candidate | dev AUROC (ID vs near-OOD) |",
        "|---|---:|",
    ]
    for name, value in sorted(config.candidate_auroc.items(), key=lambda kv: -kv[1]):
        marker = " <- frozen" if name == config.candidate else ""
        lines.append(f"| {name} | {value:.4f}{marker} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    if frozen_config.CONFIG_PATH.exists():
        existing, existing_hash = frozen_config.read_frozen()
        if frozen_config.already_run(existing_hash):
            sys.exit(
                f"{frozen_config.LOCK_PATH} shows the test set was already evaluated for this exact config "
                f"({existing_hash[:12]}...). Tuning after that would defeat 'test once' -- see eval/PROTOCOL.md "
                "section 5. Start a new protocol version (v2) instead."
            )

    id_dev, id_doc = load_dev_entries("id")
    near_ood_dev, near_ood_doc = load_dev_entries("near_ood")
    print(f"embedding {len(id_dev)} ID-dev + {len(near_ood_dev)} near-OOD-dev images (cached across runs)...")

    def show(label):
        return lambda done, total: print(f"  {label} {done}/{total}", end="\r")

    id_similarities = open_world_data.similarities_for(id_dev, progress=show("id-dev"))
    near_ood_similarities = open_world_data.similarities_for(near_ood_dev, progress=show("near-ood-dev"))
    print()

    id_scores_by_candidate = candidate_scores(id_similarities, id_dev)
    ood_scores_by_candidate = candidate_scores(near_ood_similarities, near_ood_dev)
    config = build_config(
        id_scores_by_candidate,
        ood_scores_by_candidate,
        manifests={"id": manifest_hash(id_doc), "near_ood": manifest_hash(near_ood_doc)},
    )

    digest = frozen_config.write_frozen(config)
    REPORT_PATH.write_text(report(config), encoding="utf-8", newline="\n")
    print(report(config))
    print(f"wrote {frozen_config.CONFIG_PATH} ({digest[:12]}...)")


if __name__ == "__main__":
    main()
