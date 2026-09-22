"""Pick the fixed pilot sample for the M15c end-to-end run (eval/PROTOCOL.md section 9):
about 90 in-distribution images (5 per covered species) and about 30 out-of-distribution
images (15 near-OOD + 15 far-OOD), all from the `pilot` split -- never `dev` or `test`, which
are reserved for threshold tuning and the once-only open-world evaluation.

Pure function of the three manifest documents; deterministic via eval.splits' SHA-256 buckets,
so the sample is the same on every machine and every run.
"""

from collections import defaultdict

from eval import splits

ID_PER_SPECIES = 5
NEAR_OOD_COUNT = 15
FAR_OOD_COUNT = 15


def _pilot_entries(document: dict) -> list[dict]:
    return [e for e in document["entries"] if e["split"] == "pilot"]


def _tag(entries: list[dict], family: str) -> list[dict]:
    return [{**e, "family": family} for e in entries]


def select_id_sample(id_doc: dict, per_species: int = ID_PER_SPECIES) -> list[dict]:
    """`per_species` pilot images per covered species, the ones with the smallest bucket."""
    by_species: dict[str, list[dict]] = defaultdict(list)
    for entry in _pilot_entries(id_doc):
        by_species[entry["label"]].append(entry)
    chosen = []
    for species in sorted(by_species):
        ordered = sorted(by_species[species], key=lambda e: (splits.bucket(e["source_id"]), e["source_id"]))
        chosen.extend(ordered[:per_species])
    return _tag(chosen, "id")


def select_ood_sample(document: dict, family: str, count: int) -> list[dict]:
    """`count` pilot images from `document`, spread across as many categories as possible: one
    per category (by bucket order) until every category has one, then a second pass, and so on."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for entry in _pilot_entries(document):
        by_category[entry["group"]].append(entry)
    for category in by_category:
        by_category[category].sort(key=lambda e: (splits.bucket(e["source_id"]), e["source_id"]))

    chosen, round_index = [], 0
    categories = sorted(by_category)
    while len(chosen) < count and any(round_index < len(v) for v in by_category.values()):
        for category in categories:
            if round_index < len(by_category[category]):
                chosen.append(by_category[category][round_index])
                if len(chosen) == count:
                    break
        round_index += 1
    return _tag(chosen, family)


def select_pilot_sample(
    id_doc: dict,
    near_ood_doc: dict,
    far_ood_doc: dict,
    *,
    id_per_species: int = ID_PER_SPECIES,
    near_ood_count: int = NEAR_OOD_COUNT,
    far_ood_count: int = FAR_OOD_COUNT,
) -> list[dict]:
    """The full pilot sample: ID entries first, then near-OOD, then far-OOD, each internally
    sorted for a stable, reviewable order. Every entry gains a `family` key."""
    return [
        *select_id_sample(id_doc, id_per_species),
        *select_ood_sample(near_ood_doc, "near_ood", near_ood_count),
        *select_ood_sample(far_ood_doc, "far_ood", far_ood_count),
    ]
