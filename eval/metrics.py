"""Threshold-free and threshold-based metrics for the open-world evaluation (eval/PROTOCOL.md section 6).

Pure functions of plain lists/arrays — no dataset or file I/O, no sklearn: AUROC is the Mann-Whitney
U statistic (rank-based, exact, no threshold sweep needed), AUPR is exact step-function precision-recall
integration, both checked against brute-force reference implementations in tests.

Convention throughout: **ID is the positive class** (higher score = more in-distribution) for AUROC;
**OOD is the positive class** for AUPR-Out, matching the protocol's naming.
"""

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260921


def auroc(id_scores: list[float], ood_scores: list[float]) -> float:
    """AUROC with ID as the positive class, via the rank-sum (Mann-Whitney U) identity:
    AUROC = P(a random ID score > a random OOD score), ties counted as one half.
    Undefined (returns nan) if either class is empty."""
    if not id_scores or not ood_scores:
        return float("nan")
    n_id, n_ood = len(id_scores), len(ood_scores)
    ranks = _rank_average(np.concatenate([id_scores, ood_scores]))
    id_rank_sum = ranks[:n_id].sum()
    return float((id_rank_sum - n_id * (n_id + 1) / 2) / (n_id * n_ood))


def _rank_average(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ascending, with tied values sharing their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2  # average of the 1-based positions i+1..j+1
        i = j + 1
    return ranks


def aupr_out(id_scores: list[float], ood_scores: list[float]) -> float:
    """Average precision with OOD as the positive class and (-score) as the decision value
    (higher -score = more OOD-like). Exact step-function AP: sum of precision at each OOD
    recall step, weighted by the recall increment -- not a trapezoidal approximation."""
    if not ood_scores:
        return float("nan")
    labels = np.array([0] * len(id_scores) + [1] * len(ood_scores))  # 1 = OOD (positive)
    values = np.concatenate([-np.asarray(id_scores, dtype=float), -np.asarray(ood_scores, dtype=float)])
    order = np.argsort(-values, kind="mergesort")  # most OOD-like first; ties: stable, order-independent result
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    predicted_positives = np.arange(1, len(sorted_labels) + 1)
    precision = true_positives / predicted_positives
    total_positive = sorted_labels.sum()
    recall_increment = sorted_labels / total_positive
    return float((precision * recall_increment).sum())


def fpr_at_tpr(id_scores: list[float], ood_scores: list[float], target_tpr: float = 0.95) -> float:
    """The false-positive rate (share of OOD accepted) at the highest threshold that still keeps
    at least `target_tpr` of ID accepted. ID and OOD are both "accepted" by score >= threshold.
    The threshold is the largest actual ID score such that no more than floor((1 - target_tpr) *
    n_id) of the ID images fall below it -- an off-by-one here (e.g. via a percentile function's
    own interpolation convention) silently loosens or tightens the guarantee, so this is spelled
    out rather than delegated."""
    if not id_scores or not ood_scores:
        return float("nan")
    ordered = np.sort(np.asarray(id_scores, dtype=float))
    excluded = min(int(np.floor((1 - target_tpr) * len(ordered))), len(ordered) - 1)
    threshold = ordered[excluded]
    return float(np.mean(np.asarray(ood_scores) >= threshold))


def selective_accuracy(scores: list[float], correct: list[bool], threshold: float) -> dict:
    """On in-distribution data only: of the images with score >= threshold ("accepted"), what
    fraction were classified correctly, and what fraction of all images were accepted."""
    scores_arr, correct_arr = np.asarray(scores), np.asarray(correct)
    accepted = scores_arr >= threshold
    n_accepted = int(accepted.sum())
    return {
        "coverage": n_accepted / len(scores_arr) if len(scores_arr) else float("nan"),
        "selective_accuracy": float(correct_arr[accepted].mean()) if n_accepted else float("nan"),
        "n_accepted": n_accepted,
        "n_total": len(scores_arr),
    }


def risk_coverage_auc(scores: list[float], correct: list[bool]) -> float:
    """Area under the risk-coverage curve: sweep the threshold from accept-everything down to
    accept-nothing, plotting risk (= 1 - selective accuracy) against coverage, and integrate risk
    over coverage (trapezoidal). Lower is better (a perfect selector has area 0)."""
    scores_arr, correct_arr = np.asarray(scores, dtype=float), np.asarray(correct, dtype=bool)
    if len(scores_arr) == 0:
        return float("nan")
    order = np.argsort(-scores_arr, kind="mergesort")  # most confident first
    correct_sorted = correct_arr[order]
    cumulative_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, len(scores_arr) + 1) / len(scores_arr)
    risk = 1 - cumulative_correct / np.arange(1, len(scores_arr) + 1)
    coverage, risk = np.concatenate([[0], coverage]), np.concatenate([[risk[0]], risk])
    return float(np.trapezoid(risk, coverage))


def bootstrap_ci(
    values: list,
    statistic,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """A percentile-bootstrap CI for `statistic(resample)`, resampling individual `values` with
    replacement (the protocol's "resampling images for ID"). Used for ID-only figures (selective
    accuracy, risk-coverage AUC); family comparison metrics use `bootstrap_ci_joint` instead,
    which resamples the ID and OOD sides by their own right unit."""
    rng = np.random.default_rng(seed)
    n = len(values)
    point = statistic(values)
    if n == 0:
        return {"point": point, "low": float("nan"), "high": float("nan"), "n": 0, "resamples": resamples}
    array = np.asarray(values, dtype=object) if not _is_numeric(values) else np.asarray(values)
    draws = np.empty(resamples)
    for i in range(resamples):
        draws[i] = statistic(list(array[rng.integers(0, n, size=n)]))
    tail = (1 - confidence) / 2
    return {
        "point": point,
        "low": float(np.nanpercentile(draws, 100 * tail)),
        "high": float(np.nanpercentile(draws, 100 * (1 - tail))),
        "n": n,
        "resamples": resamples,
    }


def bootstrap_ci_grouped(
    groups: dict[str, list],
    statistic,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """A percentile-bootstrap CI for a statistic over ONE population that is naturally grouped
    (e.g. an OOD family's abstention rate, grouped by category): each resample draws
    `len(groups)` groups with replacement, pools every item of the chosen groups, and calls
    `statistic(pooled)`. Use `bootstrap_ci_joint` instead for a metric that compares this
    population against a second one (ID) resampled by its own right unit."""
    rng = np.random.default_rng(seed)
    names = list(groups)
    n_groups = len(names)
    all_items = [item for name in names for item in groups[name]]
    point = statistic(all_items)
    if n_groups == 0:
        return {"point": point, "low": float("nan"), "high": float("nan"), "n_groups": 0, "resamples": resamples}
    draws = np.empty(resamples)
    for i in range(resamples):
        chosen = rng.integers(0, n_groups, size=n_groups)
        pooled = [item for idx in chosen for item in groups[names[idx]]]
        draws[i] = statistic(pooled)
    tail = (1 - confidence) / 2
    return {
        "point": point,
        "low": float(np.nanpercentile(draws, 100 * tail)),
        "high": float(np.nanpercentile(draws, 100 * (1 - tail))),
        "n_groups": n_groups,
        "n_items": len(all_items),
        "resamples": resamples,
    }


def bootstrap_ci_joint(
    id_values: list,
    ood_groups: dict[str, list],
    statistic,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """A percentile-bootstrap CI for a family-comparison metric (AUROC, AUPR-Out, FPR@95%TPR):
    each resample independently draws `len(id_values)` ID images with replacement AND
    `len(ood_groups)` OOD categories with replacement (pooling each chosen category's images),
    then calls `statistic(id_resample, ood_resample)`. This is exactly the protocol's "resampling
    images for ID, categories then images for OOD" -- resampling a pooled list instead would let
    the ID/OOD ratio drift per draw and bias metrics (AUPR-Out especially) that depend on it."""
    rng = np.random.default_rng(seed)
    n_id, group_names = len(id_values), list(ood_groups)
    n_groups = len(group_names)
    all_ood = [item for name in group_names for item in ood_groups[name]]
    point = statistic(id_values, all_ood)
    if n_id == 0 or n_groups == 0:
        empty = {"point": point, "low": float("nan"), "high": float("nan")}
        return {**empty, "n_id": n_id, "n_groups": n_groups, "resamples": resamples}
    id_array = np.asarray(id_values, dtype=object) if not _is_numeric(id_values) else np.asarray(id_values)
    draws = np.empty(resamples)
    for i in range(resamples):
        id_sample = list(id_array[rng.integers(0, n_id, size=n_id)])
        chosen = rng.integers(0, n_groups, size=n_groups)
        ood_sample = [item for idx in chosen for item in ood_groups[group_names[idx]]]
        draws[i] = statistic(id_sample, ood_sample)
    tail = (1 - confidence) / 2
    return {
        "point": point,
        "low": float(np.nanpercentile(draws, 100 * tail)),
        "high": float(np.nanpercentile(draws, 100 * (1 - tail))),
        "n_id": n_id,
        "n_groups": n_groups,
        "n_ood_items": len(all_ood),
        "resamples": resamples,
    }


def _is_numeric(values: list) -> bool:
    return all(isinstance(v, int | float | bool | np.floating | np.integer) for v in values)
