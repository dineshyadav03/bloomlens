"""Ordinal inter-rater agreement statistics (docs/quality/PROTOCOL.md section 7).

Thin wrappers over vetted, independently-maintained libraries only -- no hand-rolled coefficient
math:
- `krippendorff.alpha(level_of_measurement="ordinal")` is the primary multi-rater statistic;
- `sklearn.metrics.cohen_kappa_score(weights="quadratic")` is the primary pairwise statistic;
- `statsmodels.stats.inter_rater.fleiss_kappa` is an optional, NOT ordinal-aware, secondary
  statistic (PROTOCOL.md section 7 explains why it's never a substitute for the alpha above);
- `scipy.stats.bootstrap` (paired resampling -- resampling ITEMS, keeping every rater's label for
  the same item together, never resampling label pairs independently) draws the bootstrap
  distribution behind every CI here. The interval is a plain percentile interval over the draws
  where the statistic is defined: a small or unanimous sample can produce a resample with no
  variation in the grades, where alpha and kappa are undefined (0/0) rather than zero, and those
  draws are counted (`n_undefined_resamples`) and left out, not silently turned into numbers.
  Percentile rather than scipy's default "BCa"/"basic", which can produce bounds outside the
  range a coefficient can take.

Every function here is checked in tests/unit/test_eval_quality_agreement.py against a published
worked example with an independently-known correct answer, not only synthetic sanity checks.
**Building this module is not itself validation** -- see docs/quality/PROTOCOL.md section 8.

The bottom half turns the labeler's stored labels (`eval/quality_labels_db.latest_labels()`) into
the protocol's report (`agreement_report` / `format_report`) and lists the items section 5 sends
to a third rater:

    uv run --extra eval-quality python eval/quality_agreement.py [--adjudicator RATER_ID ...]
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import warnings
from pathlib import Path

import krippendorff
import numpy as np
from scipy import stats as scipy_stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters
from statsmodels.stats.inter_rater import fleiss_kappa as _sm_fleiss_kappa

# BloomLens's own rating scale (docs/quality/STANDARD.md; src/tools.py:55-60): ordinal and
# worse-is-higher, so the distance between two grades reflects how severe a disagreement is.
GRADES: tuple[str, ...] = ("A", "B", "C")

BOOTSTRAP_RESAMPLES = 9_999
BOOTSTRAP_SEED = 20260922


def _grade_to_ordinal(labels: tuple[str, ...]) -> dict[str, int]:
    return {grade: index for index, grade in enumerate(labels)}


def _encode(ratings, labels: tuple[str, ...]) -> list[float]:
    """Map grade letters to floats for krippendorff/scipy. `None` -- an item a rater didn't grade,
    whether truly missing or marked "cannot grade" (docs/quality/PROTOCOL.md section 4) -- becomes
    NaN, krippendorff's own convention for a missing entry (never imputed, never its own category)."""
    mapping = _grade_to_ordinal(labels)
    return [float(mapping[g]) if g is not None else float("nan") for g in ratings]


def _alpha_of(matrix: np.ndarray) -> float:
    """`krippendorff.alpha` (ordinal) of a raters-by-items matrix, or NaN where alpha is undefined:
    when every grade given is the same (nothing to agree or disagree about, so 0/0) or when no
    item has been graded by two or more raters. The library raises on the first of these; an
    honest "undefined" is more useful to a caller than an exception."""
    observed = ~np.isnan(matrix)
    if np.unique(matrix[observed]).size < 2 or not (observed.sum(axis=0) >= 2).any():
        return float("nan")
    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"))


def krippendorff_alpha(rater_ratings: list[list[str | None]], *, labels: tuple[str, ...] = GRADES) -> float:
    """Ordinal Krippendorff's alpha across two or more raters. `rater_ratings[r]` is one rater's
    grades, one entry per item, in the SAME item order for every rater; `None` marks an item that
    rater didn't grade. Handles any number of raters and unequal per-rater coverage natively.
    NaN when undefined (see `_alpha_of`)."""
    return _alpha_of(np.array([_encode(r, labels) for r in rater_ratings], dtype=float))


def weighted_cohen_kappa(rater_a: list[str], rater_b: list[str], *, labels: tuple[str, ...] = GRADES) -> float:
    """Quadratic-weighted Cohen's kappa between exactly two raters on the same items -- drop any
    item either rater marked "cannot grade" before calling; this wrapper takes no missing marker.
    NaN when undefined (sklearn's own convention: e.g. both raters gave one identical grade)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return float(cohen_kappa_score(list(rater_a), list(rater_b), labels=list(labels), weights="quadratic"))


def fleiss_kappa(rater_ratings: list[list[str]], *, labels: tuple[str, ...] = GRADES) -> float:
    """Unweighted Fleiss' kappa across two or more raters (statsmodels). NOT ordinal-aware -- an
    A/C disagreement counts exactly the same as an A/B one -- so PROTOCOL.md section 7 reports this
    only as an optional secondary statistic, never in place of `krippendorff_alpha`. Every item
    must be graded by every rater (no missing entries; use `krippendorff_alpha` when coverage is
    incomplete). NaN when undefined (every rater gave one identical grade throughout)."""
    mapping = _grade_to_ordinal(labels)
    coded = np.array([[mapping[g] for g in rater] for rater in rater_ratings]).T  # (items, raters)
    table, _ = aggregate_raters(coded, n_cat=len(labels))
    with np.errstate(invalid="ignore", divide="ignore"):  # unanimous on one grade: 0/0, i.e. NaN
        return float(_sm_fleiss_kappa(table))


def _bootstrap_interval(columns: tuple, statistic, *, confidence: float, resamples: int, seed: int) -> dict:
    """The percentile interval of `statistic` over `scipy.stats.bootstrap`'s paired resamples of
    `columns` (one array per rater, resampled together by item). Draws where the statistic is
    undefined (NaN) are counted and excluded from the percentiles, never treated as numbers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy_stats.DegenerateDataWarning)
        result = scipy_stats.bootstrap(
            columns,
            statistic,
            paired=True,
            vectorized=False,
            confidence_level=confidence,
            n_resamples=resamples,
            method="percentile",
            random_state=np.random.default_rng(seed),
        )
    draws = np.asarray(result.bootstrap_distribution, dtype=float)
    defined = draws[~np.isnan(draws)]
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(defined, [tail, 100 - tail]) if defined.size else (float("nan"), float("nan"))
    return {"low": float(low), "high": float(high), "n_undefined_resamples": int(draws.size - defined.size)}


def _undefined_interval(n_items: int, resamples: int) -> dict:
    nan = float("nan")
    return {
        "point": nan,
        "low": nan,
        "high": nan,
        "n_items": n_items,
        "resamples": resamples,
        "n_undefined_resamples": 0,
    }


def bootstrap_alpha_ci(
    rater_ratings: list[list[str | None]],
    *,
    labels: tuple[str, ...] = GRADES,
    confidence: float = 0.95,
    resamples: int | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """A bootstrap CI for `krippendorff_alpha`. `scipy.stats.bootstrap`'s `paired=True` resamples
    ITEMS, keeping every rater's label for the same resampled item together -- matching
    PROTOCOL.md's stated resampling unit -- rather than resampling each rater's column
    independently, which would let raters drift apart across a resample and misstate the CI."""
    resamples = BOOTSTRAP_RESAMPLES if resamples is None else resamples
    encoded = [np.array(_encode(r, labels)) for r in rater_ratings]
    n = len(encoded[0]) if encoded else 0
    point = krippendorff_alpha(rater_ratings, labels=labels) if n and len(encoded) >= 2 else float("nan")
    if math.isnan(point):
        # No point estimate (too little data, or no variation in the grades): no interval either.
        return _undefined_interval(n, resamples)

    def statistic(*columns):
        return _alpha_of(np.array(columns))

    interval = _bootstrap_interval(tuple(encoded), statistic, confidence=confidence, resamples=resamples, seed=seed)
    return {"point": point, "n_items": n, "resamples": resamples, **interval}


def bootstrap_kappa_ci(
    rater_a: list[str],
    rater_b: list[str],
    *,
    labels: tuple[str, ...] = GRADES,
    confidence: float = 0.95,
    resamples: int | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """A bootstrap CI for `weighted_cohen_kappa`, resampling ITEMS paired across both raters
    (see `bootstrap_alpha_ci`)."""
    resamples = BOOTSTRAP_RESAMPLES if resamples is None else resamples
    rater_a, rater_b = list(rater_a), list(rater_b)
    n = len(rater_a)
    point = weighted_cohen_kappa(rater_a, rater_b, labels=labels) if n else float("nan")
    if math.isnan(point):
        return _undefined_interval(n, resamples)

    def statistic(a, b):
        return weighted_cohen_kappa(list(a), list(b), labels=labels)

    columns = (np.array(rater_a, dtype=object), np.array(rater_b, dtype=object))
    interval = _bootstrap_interval(columns, statistic, confidence=confidence, resamples=resamples, seed=seed)
    return {"point": point, "n_items": n, "resamples": resamples, **interval}


def cannot_grade_rate(rater_ratings: list[list[str | None]]) -> dict:
    """Per PROTOCOL.md section 4: the share of items where AT LEAST ONE rater could not grade,
    and the share where ALL raters agreed it was ungradable -- reported alongside, never folded
    into, the primary agreement statistics above (which exclude such items entirely)."""
    if not rater_ratings or not rater_ratings[0]:
        return {"any_cannot_grade": float("nan"), "all_cannot_grade": float("nan"), "n_items": 0}
    n_items = len(rater_ratings[0])
    any_count = sum(1 for i in range(n_items) if any(rater[i] is None for rater in rater_ratings))
    all_count = sum(1 for i in range(n_items) if all(rater[i] is None for rater in rater_ratings))
    return {
        "any_cannot_grade": any_count / n_items,
        "all_cannot_grade": all_count / n_items,
        "n_items": n_items,
    }


# --- the protocol's report, from stored labels -------------------------------------------------

CANNOT_GRADE = "CANNOT_GRADE"  # the label tools/label_quality.py stores for "cannot grade"

LabelsByItem = dict[str, dict[str, str]]  # {item_id: {rater_id: "A"|"B"|"C"|"CANNOT_GRADE"}}


def items_needing_adjudication(labels: LabelsByItem) -> list[str]:
    """PROTOCOL.md section 5: items where two raters' grades differ by MORE than one letter (A vs
    C) go to a third rater. A one-letter gap is normal disagreement and is not adjudicated, and
    a "cannot grade" is never a disagreement about the grade."""
    needing = []
    for item_id, by_rater in sorted(labels.items()):
        ordinals = [GRADES.index(g) for g in by_rater.values() if g in GRADES]
        if ordinals and max(ordinals) - min(ordinals) > 1:
            needing.append(item_id)
    return needing


def _matrix(labels: LabelsByItem, raters: list[str], items: list[str], *, keep) -> list[list[str | None]]:
    """One row per rater, one column per item: the rater's grade where `keep(grade)` holds, else
    None (unrated, or a grade `keep` rejects)."""
    matrix = []
    for rater in raters:
        row = []
        for item in items:
            grade = labels[item].get(rater)
            row.append(grade if grade is not None and keep(grade) else None)
        matrix.append(row)
    return matrix


def _has_pairable_item(matrix: list[list[str | None]]) -> bool:
    """Krippendorff's alpha needs at least one item graded by two or more raters."""
    return any(sum(value is not None for value in column) >= 2 for column in zip(*matrix, strict=True))


def without_raters(labels: LabelsByItem, raters) -> LabelsByItem:
    """`labels` with the given raters' entries removed, and any item left with no labels dropped."""
    raters = set(raters)
    kept = {item: {r: g for r, g in by_rater.items() if r not in raters} for item, by_rater in labels.items()}
    return {item: by_rater for item, by_rater in kept.items() if by_rater}


def agreement_report(labels: LabelsByItem, *, resamples: int | None = None, adjudicators=()) -> dict:
    """Every number PROTOCOL.md sections 4 and 7 call for, pooled over all items -- never per
    species. `labels` is `eval/quality_labels_db.latest_labels()`'s shape.

    `adjudicators` are the third raters of section 5. Their labels are recorded and reported as
    outcomes, but they are kept OUT of every agreement statistic: adjudication resolves an
    item, it does not change what the original raters actually disagreed on."""
    resamples = BOOTSTRAP_RESAMPLES if resamples is None else resamples
    adjudications = {
        item: {r: g for r, g in by_rater.items() if r in set(adjudicators)} for item, by_rater in labels.items()
    }
    labels = without_raters(labels, adjudicators)
    raters = sorted({rater for by_rater in labels.values() for rater in by_rater})
    items = sorted(labels)
    report: dict = {"n_items": len(items), "raters": raters, "n_raters": len(raters)}

    # Section 4's primary set: an item ANY rater marked "cannot grade" is excluded outright.
    graded_items = [i for i in items if not any(g == CANNOT_GRADE for g in labels[i].values())]
    report["n_items_in_primary"] = len(graded_items)

    primary = _matrix(labels, raters, graded_items, keep=lambda g: g in GRADES)
    can_alpha = len(raters) >= 2 and bool(primary) and _has_pairable_item(primary)
    report["alpha"] = bootstrap_alpha_ci(primary, resamples=resamples) if can_alpha else None

    pairwise = []
    for a, b in itertools.combinations(raters, 2):
        common = [i for i in graded_items if labels[i].get(a) in GRADES and labels[i].get(b) in GRADES]
        if len(common) >= 2:
            ci = bootstrap_kappa_ci([labels[i][a] for i in common], [labels[i][b] for i in common], resamples=resamples)
            pairwise.append({"raters": (a, b), **ci})
    report["pairwise_kappa"] = pairwise
    points = [p["point"] for p in pairwise if not math.isnan(p["point"])]  # a degenerate pair has no kappa
    report["mean_pairwise_kappa"] = sum(points) / len(points) if points else None

    complete = [i for i in graded_items if all(labels[i].get(r) in GRADES for r in raters)]
    report["fleiss_kappa"] = (
        {"value": fleiss_kappa([[labels[i][r] for i in complete] for r in raters]), "n_items": len(complete)}
        if len(raters) >= 2 and complete
        else None
    )

    # Section 4's rates and sensitivity analysis. The rate is over the items EVERY rater has
    # labeled, so an item nobody has reached yet is not miscounted as "cannot grade".
    fully_rated = [i for i in items if all(r in labels[i] for r in raters)]
    report["cannot_grade"] = cannot_grade_rate(_matrix(labels, raters, fully_rated, keep=lambda g: g != CANNOT_GRADE))
    with_cannot_grade = _matrix(labels, raters, items, keep=lambda g: True)
    report["alpha_cannot_grade_as_lowest_category"] = (
        krippendorff_alpha(with_cannot_grade, labels=(*GRADES, CANNOT_GRADE))
        if len(raters) >= 2 and bool(items) and _has_pairable_item(with_cannot_grade)
        else None
    )
    report["needing_adjudication"] = items_needing_adjudication(labels)
    report["adjudicated"] = {i: adjudications[i] for i in report["needing_adjudication"] if adjudications.get(i)}
    return report


def _fmt_ci(ci: dict) -> str:
    if math.isnan(ci["point"]):
        return f"undefined -- the grades show no variation to agree or disagree about ({ci['n_items']} items)"
    text = f"{ci['point']:.3f} (95% bootstrap CI {ci['low']:.3f} to {ci['high']:.3f}, {ci['n_items']} items"
    if ci.get("n_undefined_resamples"):
        text += f"; {ci['n_undefined_resamples']} of {ci['resamples']} resamples had no variation and are left out"
    return text + ")"


def format_report(report: dict) -> str:
    """The report as markdown, with PROTOCOL.md sections 6 and 8's limits stated in the report
    itself so no reader can take the numbers for more than they are."""
    lines = [
        "# Quality-label agreement report",
        "",
        f"{report['n_raters']} rater(s), {report['n_items']} item(s) labeled, "
        f'{report["n_items_in_primary"]} in the primary analysis (no rater marked them "cannot grade").',
        "",
        "**Pooled across all species only** -- with this sample size no per-species agreement is "
        "reported or implied. **This measures whether people agree with each other on the visual "
        "condition class; it does not validate BloomLens's own `quality_grade`**, which stays an "
        "unvalidated heuristic (docs/quality/PROTOCOL.md sections 6 and 8).",
        "",
        "## Primary",
        "",
    ]
    alpha = report["alpha"]
    lines.append(
        "- Krippendorff's alpha (ordinal): "
        + (_fmt_ci(alpha) if alpha else "not computable (needs 2+ raters and an item graded by 2+ of them)")
    )
    lines.append("- Weighted (quadratic) Cohen's kappa, per rater pair:")
    if report["pairwise_kappa"]:
        lines += [f"  - {p['raters'][0]} vs {p['raters'][1]}: {_fmt_ci(p)}" for p in report["pairwise_kappa"]]
    else:
        lines.append("  - none (needs a pair of raters with 2+ shared graded items)")
    if report["mean_pairwise_kappa"] is not None:
        mean = report["mean_pairwise_kappa"]
        lines.append(f"- Mean of the pairwise kappas: {mean:.3f} (a mean, not a joint statistic)")

    lines += ["", "## Secondary", ""]
    fleiss = report["fleiss_kappa"]
    if not fleiss:
        lines.append("- Fleiss' kappa: not computable (needs every rater on the same graded items)")
    else:
        value = "undefined (no variation in the grades)" if math.isnan(fleiss["value"]) else f"{fleiss['value']:.3f}"
        lines.append(f"- Fleiss' kappa (unweighted -- cannot tell A/B from A/C): {value} ({fleiss['n_items']} items)")

    cg = report["cannot_grade"]
    lines += ["", '## "Cannot grade" (section 4)', ""]
    if cg["n_items"]:
        lines.append(
            f"- At least one rater could not grade: {cg['any_cannot_grade']:.1%} of {cg['n_items']} fully-rated items"
        )
        lines.append(f"- Every rater agreed it was ungradable: {cg['all_cannot_grade']:.1%}")
    else:
        lines.append("- No item has been labeled by every rater yet.")
    sens = report["alpha_cannot_grade_as_lowest_category"]
    lines.append(
        '- Sensitivity ("cannot grade" kept as an ordinal category below C): alpha '
        + (f"{sens:.3f}" if sens is not None and not math.isnan(sens) else "undefined or not computable")
    )

    adjudicate = report["needing_adjudication"]
    lines += ["", "## Adjudication (section 5)", ""]
    lines.append(
        f"{len(adjudicate)} item(s) differ by more than one letter and go to a third rater"
        + (": " + ", ".join(adjudicate) if adjudicate else ".")
    )
    resolved = report["adjudicated"]
    if adjudicate:
        for item in adjudicate:
            if item in resolved:
                outcome = ", ".join(f"{rater}: {grade}" for rater, grade in sorted(resolved[item].items()))
                lines.append(f"  - {item}: adjudicated ({outcome})")
            else:
                lines.append(f"  - {item}: awaiting a third rater")
        lines.append(
            "Adjudicator labels are excluded from every agreement statistic above (section 5): "
            "they resolve an item, they do not change what the original raters disagreed on."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval import quality_labels_db

    parser = argparse.ArgumentParser(description="Agreement report over the labeler's stored labels.")
    parser.add_argument(
        "--adjudicator",
        action="append",
        default=[],
        metavar="RATER_ID",
        help="a third rater (PROTOCOL.md section 5) whose labels resolve items but are kept out of the "
        "agreement statistics; repeat for several",
    )
    args = parser.parse_args(argv)

    labels = quality_labels_db.latest_labels()
    if not labels:
        print(f"No labels yet in {quality_labels_db.db_path()} -- run tools/label_quality.py first.")
        return
    print(format_report(agreement_report(labels, adjudicators=args.adjudicator)))


if __name__ == "__main__":
    main()
