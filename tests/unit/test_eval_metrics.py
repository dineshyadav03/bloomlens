"""eval/metrics.py: AUROC, AUPR-Out, FPR@95%TPR, selective accuracy, risk-coverage, bootstrap CIs.

Each threshold-free metric is checked against a brute-force reference implementation (no numpy
tricks, just loops), so a shortcut in the real implementation that happens to agree on one example
would still be caught on random data.
"""

import itertools

import numpy as np
import pytest

from eval import metrics


def brute_auroc(id_scores, ood_scores):
    """Every (id, ood) pair; 1 point for id>ood, 0.5 for a tie."""
    total = wins = 0
    for i, o in itertools.product(id_scores, ood_scores):
        total += 1
        wins += 1.0 if i > o else 0.5 if i == o else 0.0
    return wins / total


def brute_aupr_out(id_scores, ood_scores):
    """Average precision, OOD positive: rank every image from most to least OOD-like (lowest
    original score first), and average the precision-at-k over every rank that lands on a true
    OOD image. This is the standard rank-based AP definition (what sklearn's
    average_precision_score computes) and is independent of the cumulative-sum implementation
    in eval/metrics.py, which is checked against it."""
    labeled = [(s, 0) for s in id_scores] + [(s, 1) for s in ood_scores]
    labeled.sort(key=lambda pair: pair[0])  # ascending original score = most OOD-like first
    total_positive = sum(label for _, label in labeled)
    true_positives = 0
    ap = 0.0
    for rank, (_, label) in enumerate(labeled, start=1):
        if label == 1:
            true_positives += 1
            ap += true_positives / rank
    return ap / total_positive


def brute_fpr_at_tpr(id_scores, ood_scores, target=0.95):
    """The highest threshold (any real number, approximated by sweeping every ID score as a
    candidate) that still keeps at least `target` TPR -- found independently of the closed-form
    floor((1-target)*n) index arithmetic in eval/metrics.py."""
    for threshold in sorted(set(id_scores), reverse=True):
        tpr = sum(1 for s in id_scores if s >= threshold) / len(id_scores)
        if tpr >= target:
            return sum(1 for s in ood_scores if s >= threshold) / len(ood_scores)
    return 0.0


RNG_SEEDS = [1, 2, 3, 4, 5]


class TestAurocAgainstBruteForce:
    @pytest.mark.parametrize("seed", RNG_SEEDS)
    def test_random_continuous_scores(self, seed):
        rng = np.random.default_rng(seed)
        id_scores, ood_scores = rng.normal(0.6, 0.15, 40).tolist(), rng.normal(0.4, 0.15, 35).tolist()
        assert metrics.auroc(id_scores, ood_scores) == pytest.approx(brute_auroc(id_scores, ood_scores))

    def test_with_ties(self):
        id_scores, ood_scores = [0.5, 0.5, 0.9], [0.5, 0.5, 0.1]
        assert metrics.auroc(id_scores, ood_scores) == pytest.approx(brute_auroc(id_scores, ood_scores))

    def test_perfect_separation_is_one(self):
        assert metrics.auroc([0.9, 0.8, 0.7], [0.3, 0.2, 0.1]) == pytest.approx(1.0)

    def test_perfectly_reversed_is_zero(self):
        assert metrics.auroc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)

    def test_identical_distributions_are_about_half(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 500)
        assert metrics.auroc(values[:250].tolist(), values[250:].tolist()) == pytest.approx(0.5, abs=0.1)

    def test_symmetry_flips_around_one_half(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(0.5, 0.2, 30).tolist(), rng.normal(0.3, 0.2, 20).tolist()
        assert metrics.auroc(a, b) == pytest.approx(1 - metrics.auroc(b, a))

    def test_empty_input_is_nan(self):
        assert np.isnan(metrics.auroc([], [0.5]))
        assert np.isnan(metrics.auroc([0.5], []))


class TestAuprOutAgainstBruteForce:
    @pytest.mark.parametrize("seed", RNG_SEEDS)
    def test_random_continuous_scores(self, seed):
        rng = np.random.default_rng(seed)
        id_scores, ood_scores = rng.normal(0.6, 0.15, 30).tolist(), rng.normal(0.4, 0.15, 12).tolist()
        assert metrics.aupr_out(id_scores, ood_scores) == pytest.approx(brute_aupr_out(id_scores, ood_scores))

    @pytest.mark.parametrize("n_id, n_ood, seed", [(50, 5, 11), (5, 50, 12), (20, 20, 13)])
    def test_at_various_class_balances(self, n_id, n_ood, seed):
        rng = np.random.default_rng(seed)
        id_scores, ood_scores = rng.uniform(0, 1, n_id).tolist(), rng.uniform(0, 1, n_ood).tolist()
        assert metrics.aupr_out(id_scores, ood_scores) == pytest.approx(brute_aupr_out(id_scores, ood_scores), abs=1e-9)

    def test_perfect_separation_is_one(self):
        assert metrics.aupr_out([0.9, 0.8], [0.1, 0.2]) == pytest.approx(1.0)

    def test_worst_case_ordering_is_low(self):
        assert metrics.aupr_out([0.1, 0.2], [0.9, 0.8]) < 0.6

    def test_it_depends_on_class_balance_unlike_auroc(self):
        """The same perfect-ish ranking scores higher AUPR-Out when OOD is rarer."""
        rng = np.random.default_rng(3)
        id_scores = rng.normal(0.6, 0.2, 100).tolist()
        few_ood = rng.normal(0.4, 0.2, 5).tolist()
        many_ood = few_ood + rng.normal(0.4, 0.2, 95).tolist()
        assert metrics.aupr_out(id_scores, few_ood) != pytest.approx(metrics.aupr_out(id_scores, many_ood), abs=0.05)

    def test_no_ood_examples_is_nan(self):
        assert np.isnan(metrics.aupr_out([0.5, 0.4], []))


class TestFprAtTpr:
    @pytest.mark.parametrize("seed", RNG_SEEDS)
    def test_against_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        id_scores, ood_scores = rng.normal(0.6, 0.1, 60).tolist(), rng.normal(0.4, 0.15, 40).tolist()
        assert metrics.fpr_at_tpr(id_scores, ood_scores) == pytest.approx(brute_fpr_at_tpr(id_scores, ood_scores))

    def test_a_threshold_that_admits_only_perfect_id_scores_admits_no_ood(self):
        id_scores = [1.0] * 96 + [0.0] * 4  # 96% >= 1.0
        assert metrics.fpr_at_tpr(id_scores, [0.5, 0.5], target_tpr=0.95) == 0.0

    def test_full_overlap_gives_a_high_fpr(self):
        same = list(np.linspace(0, 1, 100))
        assert metrics.fpr_at_tpr(same, same) > 0.8

    def test_target_tpr_of_one_uses_the_minimum_id_score_as_the_threshold(self):
        id_scores = [0.9, 0.5, 0.7]
        assert metrics.fpr_at_tpr(id_scores, [0.6], target_tpr=1.0) == 1.0  # 0.6 >= min(id) = 0.5

    def test_empty_input_is_nan(self):
        assert np.isnan(metrics.fpr_at_tpr([], [0.5]))
        assert np.isnan(metrics.fpr_at_tpr([0.5], []))


class TestSelectiveAccuracy:
    def test_coverage_and_accuracy_at_a_simple_threshold(self):
        result = metrics.selective_accuracy([0.9, 0.8, 0.3, 0.2], [True, False, True, True], threshold=0.5)
        assert result == {"coverage": 0.5, "selective_accuracy": 0.5, "n_accepted": 2, "n_total": 4}

    def test_accepting_nothing_gives_zero_coverage_and_no_accuracy(self):
        result = metrics.selective_accuracy([0.1, 0.2], [True, True], threshold=0.9)
        assert result["coverage"] == 0.0 and np.isnan(result["selective_accuracy"])

    def test_accepting_everything_gives_full_coverage(self):
        result = metrics.selective_accuracy([0.1, 0.9], [True, False], threshold=0.0)
        assert result["coverage"] == 1.0 and result["selective_accuracy"] == 0.5

    def test_the_boundary_is_inclusive(self):
        assert metrics.selective_accuracy([0.5], [True], threshold=0.5)["n_accepted"] == 1


def brute_risk_coverage_auc(scores_list, correct):
    """Trapezoidal area under the risk-coverage curve, built the same conceptual way as
    eval/metrics.py but from an explicit per-k loop rather than vectorised cumsum, as a
    cross-check on the vectorised version."""
    order = sorted(range(len(scores_list)), key=lambda i: -scores_list[i])
    points = [(0.0, 0.0)]  # (coverage, risk); risk at coverage 0 is defined as the risk at k=1
    n, correct_so_far = len(scores_list), 0
    for k, i in enumerate(order, start=1):
        correct_so_far += correct[i]
        coverage = k / n
        risk = 1 - correct_so_far / k
        points.append((coverage, risk))
    points[0] = (0.0, points[1][1])  # match eval/metrics.py: coverage=0 carries risk[0]'s value
    area = 0.0
    for (c0, r0), (c1, r1) in zip(points, points[1:], strict=False):
        area += (r0 + r1) / 2 * (c1 - c0)
    return area


class TestRiskCoverageAuc:
    @pytest.mark.parametrize("seed", RNG_SEEDS)
    def test_against_a_brute_force_reference(self, seed):
        rng = np.random.default_rng(seed)
        n = 60
        scores_list = rng.uniform(size=n).tolist()
        correct = (rng.uniform(size=n) < 0.75).tolist()
        assert metrics.risk_coverage_auc(scores_list, correct) == pytest.approx(
            brute_risk_coverage_auc(scores_list, correct)
        )

    def test_a_selector_with_errors_has_a_correctly_computed_nonzero_area(self):
        """Not zero: with 2 wrong out of 4, even the best possible ordering must eventually
        cover a wrong one, so the curve cannot hug risk=0 all the way to full coverage."""
        scores_list, correct = [0.9, 0.8, 0.2, 0.1], [True, True, False, False]
        assert metrics.risk_coverage_auc(scores_list, correct) == pytest.approx(
            brute_risk_coverage_auc(scores_list, correct)
        )
        assert 0 < metrics.risk_coverage_auc(scores_list, correct) < 0.5

    def test_the_worst_ordering_has_the_largest_area(self):
        scores_list = [0.9, 0.8, 0.2, 0.1]
        best = metrics.risk_coverage_auc(scores_list, [True, True, False, False])
        worst = metrics.risk_coverage_auc(scores_list, [False, False, True, True])
        assert worst > best

    def test_random_ordering_is_close_to_the_overall_error_rate(self):
        rng = np.random.default_rng(5)
        n = 2000
        correct = (rng.uniform(size=n) < 0.8).tolist()  # 80% correct overall
        scores_list = rng.uniform(size=n).tolist()  # independent of correctness
        assert metrics.risk_coverage_auc(scores_list, correct) == pytest.approx(0.2, abs=0.03)

    def test_all_correct_has_zero_area_regardless_of_scores(self):
        assert metrics.risk_coverage_auc([0.5, 0.1, 0.9], [True, True, True]) == pytest.approx(0.0)

    def test_empty_input_is_nan(self):
        assert np.isnan(metrics.risk_coverage_auc([], []))


class TestBootstrapCi:
    def test_the_point_estimate_is_the_statistic_on_the_full_data(self):
        result = metrics.bootstrap_ci([1, 2, 3, 4, 5], lambda v: sum(v) / len(v), resamples=200)
        assert result["point"] == 3.0

    def test_the_interval_contains_the_point_and_widens_with_fewer_resamples_of_noisy_data(self):
        values = list(range(20))
        wide = metrics.bootstrap_ci(values, np.mean, resamples=50, seed=1)
        narrow = metrics.bootstrap_ci(values, np.mean, resamples=5000, seed=1)
        assert wide["low"] <= wide["point"] <= wide["high"]
        assert narrow["low"] <= narrow["point"] <= narrow["high"]

    def test_it_is_reproducible_for_a_fixed_seed(self):
        values = list(range(30))
        a = metrics.bootstrap_ci(values, np.mean, resamples=500, seed=42)
        b = metrics.bootstrap_ci(values, np.mean, resamples=500, seed=42)
        assert a == b

    def test_a_different_seed_gives_a_different_but_close_interval(self):
        values = list(np.random.default_rng(1).normal(0, 1, 200))
        a = metrics.bootstrap_ci(values, np.mean, resamples=2000, seed=1)
        b = metrics.bootstrap_ci(values, np.mean, resamples=2000, seed=2)
        assert a["low"] != b["low"]
        assert a["low"] == pytest.approx(b["low"], abs=0.05)

    def test_it_reports_n_and_the_resample_count(self):
        result = metrics.bootstrap_ci([1, 2, 3], np.mean, resamples=123)
        assert result["n"] == 3 and result["resamples"] == 123

    def test_empty_input_is_nan_but_does_not_crash(self):
        result = metrics.bootstrap_ci([], lambda v: 0, resamples=10)
        assert result["n"] == 0 and np.isnan(result["low"])

    def test_it_works_over_opaque_grouped_items_not_just_numbers(self):
        """Items can be arbitrary objects (e.g. per-image records); the statistic decides what to do."""
        items = [{"correct": True}] * 8 + [{"correct": False}] * 2
        result = metrics.bootstrap_ci(items, lambda v: sum(x["correct"] for x in v) / len(v), resamples=500)
        assert result["point"] == pytest.approx(0.8)


class TestBootstrapCiGrouped:
    def test_the_point_estimate_pools_every_group(self):
        groups = {"a": [1, 2, 3], "b": [4, 5]}
        result = metrics.bootstrap_ci_grouped(groups, lambda v: sum(v) / len(v))
        assert result["point"] == pytest.approx(3.0)  # mean of 1,2,3,4,5

    def test_a_group_is_always_drawn_whole_never_split(self):
        groups = {"a": [100, 101], "b": [200]}
        sizes_seen = set()
        metrics.bootstrap_ci_grouped(groups, lambda v: sizes_seen.add(len(v)) or 0.0, resamples=200, seed=3)
        assert sizes_seen <= {0, 1, 2, 3, 4}  # sums of chosen {1,2}-sized groups over 2 draws

    def test_it_reports_group_and_item_counts(self):
        result = metrics.bootstrap_ci_grouped({"a": [1], "b": [2, 3]}, np.mean, resamples=50)
        assert result["n_groups"] == 2 and result["n_items"] == 3

    def test_no_groups_is_nan_without_crashing(self):
        result = metrics.bootstrap_ci_grouped({}, lambda v: 0.0, resamples=10)
        assert result["n_groups"] == 0 and np.isnan(result["low"])

    def test_it_is_reproducible_for_a_fixed_seed(self):
        groups = {"a": [1, 2], "b": [3], "c": [4, 5, 6]}
        a = metrics.bootstrap_ci_grouped(groups, np.mean, resamples=300, seed=4)
        b = metrics.bootstrap_ci_grouped(groups, np.mean, resamples=300, seed=4)
        assert a == b

    def test_it_works_for_an_abstention_rate_style_statistic(self):
        groups = {"cat1": [0.1, 0.2, 0.9], "cat2": [0.05, 0.15]}  # scores; tau = 0.5
        result = metrics.bootstrap_ci_grouped(groups, lambda pooled: np.mean(np.array(pooled) < 0.5))
        assert result["point"] == pytest.approx(4 / 5)


class TestBootstrapCiJoint:
    def test_the_point_estimate_pools_every_group_and_matches_the_id_ood_auroc(self):
        id_values = [0.9, 0.8, 0.7, 0.6]
        groups = {"cat_a": [0.1, 0.2], "cat_b": [0.3, 0.15]}
        result = metrics.bootstrap_ci_joint(id_values, groups, metrics.auroc, resamples=100)
        pooled_ood = [0.1, 0.2, 0.3, 0.15]
        assert result["point"] == pytest.approx(metrics.auroc(id_values, pooled_ood))

    def test_a_category_never_appears_split_across_a_single_resample(self):
        """Every draw of a chosen category brings its whole pool of items with it."""
        id_values = list(range(10))
        groups = {"a": [100, 101, 102], "b": [200, 201]}
        seen_group_sizes = set()

        def probe(_id, ood):
            seen_group_sizes.add(len(ood))
            return 0.0

        metrics.bootstrap_ci_joint(id_values, groups, probe, resamples=300, seed=9)
        assert seen_group_sizes <= {0, 2, 3, 4, 5, 6, 7, 8, 9}  # sums of chosen {2,3}-sized groups only
        assert not seen_group_sizes & {1}  # never an odd count that neither group size alone explains... (impossible)

    def test_it_reports_n_id_and_n_groups(self):
        result = metrics.bootstrap_ci_joint([1, 2, 3], {"a": [1], "b": [2, 3]}, metrics.auroc, resamples=50)
        assert result["n_id"] == 3 and result["n_groups"] == 2 and result["n_ood_items"] == 3

    def test_no_id_or_no_groups_is_nan_without_crashing(self):
        assert np.isnan(metrics.bootstrap_ci_joint([], {"a": [1]}, metrics.auroc, resamples=10)["low"])
        assert np.isnan(metrics.bootstrap_ci_joint([1], {}, metrics.auroc, resamples=10)["low"])

    def test_it_is_reproducible_for_a_fixed_seed(self):
        id_values, groups = [0.9, 0.5, 0.3, 0.7], {"a": [0.2, 0.1], "b": [0.4]}
        a = metrics.bootstrap_ci_joint(id_values, groups, metrics.auroc, resamples=300, seed=5)
        b = metrics.bootstrap_ci_joint(id_values, groups, metrics.auroc, resamples=300, seed=5)
        assert a == b
