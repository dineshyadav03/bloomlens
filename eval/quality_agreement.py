"""Ordinal inter-rater agreement statistics (docs/quality/PROTOCOL.md section 7).

Thin wrappers over vetted, independently-maintained libraries only -- no hand-rolled coefficient
math:
- `krippendorff.alpha(level_of_measurement="ordinal")` is the primary multi-rater statistic;
- `sklearn.metrics.cohen_kappa_score(weights="quadratic")` is the primary pairwise statistic;
- `statsmodels.stats.inter_rater.fleiss_kappa` is an optional, NOT ordinal-aware, secondary
  statistic (PROTOCOL.md section 7 explains why it's never a substitute for the alpha above);
- `scipy.stats.bootstrap` (paired resampling -- resampling ITEMS, keeping every rater's label for
  the same item together, never resampling label pairs independently) gives every CI here, using
  the "percentile" method so a bound can never fall outside the range of values these statistics
  can actually take (the default "BCa"/"basic" methods can produce out-of-range bounds by design;
  see the scipy docs for `bootstrap`).

Every function here is checked in tests/unit/test_eval_quality_agreement.py against a published
worked example with an independently-known correct answer, not only synthetic sanity checks.
**Building this module is not itself validation** -- see docs/quality/PROTOCOL.md section 8.
"""

from __future__ import annotations

import krippendorff
import numpy as np
from scipy import stats as scipy_stats
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


def krippendorff_alpha(rater_ratings: list[list[str | None]], *, labels: tuple[str, ...] = GRADES) -> float:
    """Ordinal Krippendorff's alpha across two or more raters. `rater_ratings[r]` is one rater's
    grades, one entry per item, in the SAME item order for every rater; `None` marks an item that
    rater didn't grade. Handles any number of raters and unequal per-rater coverage natively."""
    matrix = np.array([_encode(r, labels) for r in rater_ratings], dtype=float)
    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"))


def weighted_cohen_kappa(rater_a: list[str], rater_b: list[str], *, labels: tuple[str, ...] = GRADES) -> float:
    """Quadratic-weighted Cohen's kappa between exactly two raters on the same items -- drop any
    item either rater marked "cannot grade" before calling; this wrapper takes no missing marker."""
    return float(cohen_kappa_score(list(rater_a), list(rater_b), labels=list(labels), weights="quadratic"))


def fleiss_kappa(rater_ratings: list[list[str]], *, labels: tuple[str, ...] = GRADES) -> float:
    """Unweighted Fleiss' kappa across two or more raters (statsmodels). NOT ordinal-aware -- an
    A/C disagreement counts exactly the same as an A/B one -- so PROTOCOL.md section 7 reports this
    only as an optional secondary statistic, never in place of `krippendorff_alpha`. Every item
    must be graded by every rater (no missing entries; use `krippendorff_alpha` when coverage is
    incomplete)."""
    mapping = _grade_to_ordinal(labels)
    coded = np.array([[mapping[g] for g in rater] for rater in rater_ratings]).T  # (items, raters)
    table, _ = aggregate_raters(coded, n_cat=len(labels))
    return float(_sm_fleiss_kappa(table))


def bootstrap_alpha_ci(
    rater_ratings: list[list[str | None]],
    *,
    labels: tuple[str, ...] = GRADES,
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """A bootstrap CI for `krippendorff_alpha`. `scipy.stats.bootstrap`'s `paired=True` resamples
    ITEMS, keeping every rater's label for the same resampled item together -- matching
    PROTOCOL.md's stated resampling unit -- rather than resampling each rater's column
    independently, which would let raters drift apart across a resample and misstate the CI."""
    encoded = [np.array(_encode(r, labels)) for r in rater_ratings]
    n = len(encoded[0]) if encoded else 0
    if n == 0 or len(encoded) < 2:
        # Too little data even to report a point estimate: neither krippendorff_alpha nor a
        # bootstrap CI over it is defined here.
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "n_items": n, "resamples": resamples}
    point = krippendorff_alpha(rater_ratings, labels=labels)

    def statistic(*columns):
        return krippendorff.alpha(reliability_data=np.array(columns), level_of_measurement="ordinal")

    result = scipy_stats.bootstrap(
        tuple(encoded),
        statistic,
        paired=True,
        vectorized=False,
        confidence_level=confidence,
        n_resamples=resamples,
        method="percentile",
        random_state=np.random.default_rng(seed),
    )
    return {
        "point": point,
        "low": float(result.confidence_interval.low),
        "high": float(result.confidence_interval.high),
        "n_items": n,
        "resamples": resamples,
    }


def bootstrap_kappa_ci(
    rater_a: list[str],
    rater_b: list[str],
    *,
    labels: tuple[str, ...] = GRADES,
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """A bootstrap CI for `weighted_cohen_kappa`, resampling ITEMS paired across both raters
    (see `bootstrap_alpha_ci`)."""
    rater_a, rater_b = list(rater_a), list(rater_b)
    n = len(rater_a)
    if n == 0:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "n_items": 0, "resamples": resamples}
    point = weighted_cohen_kappa(rater_a, rater_b, labels=labels)

    def statistic(a, b):
        return cohen_kappa_score(list(a), list(b), labels=list(labels), weights="quadratic")

    result = scipy_stats.bootstrap(
        (np.array(rater_a, dtype=object), np.array(rater_b, dtype=object)),
        statistic,
        paired=True,
        vectorized=False,
        confidence_level=confidence,
        n_resamples=resamples,
        method="percentile",
        random_state=np.random.default_rng(seed),
    )
    return {
        "point": point,
        "low": float(result.confidence_interval.low),
        "high": float(result.confidence_interval.high),
        "n_items": n,
        "resamples": resamples,
    }


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
