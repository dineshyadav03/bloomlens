"""eval/quality_agreement.py: wrappers over krippendorff/sklearn/statsmodels/scipy.

Each primary wrapper is checked against a PUBLISHED worked example with an independently-known
correct answer -- catching a library-usage mistake (wrong axis, wrong weighting, wrong missing-
data convention) that a synthetic sanity check built with the same misunderstanding would not.
Sources:
  - Krippendorff alpha (ordinal): the `krippendorff` package's own documented example, itself
    drawn from Hayes, A. F., & Krippendorff, K. (2007). "Answering the Call for a Standard
    Reliability Measure for Coding Data." Communication Methods and Measures, 1(1), 77-89, p.8.
  - Weighted (quadratic) Cohen's kappa: a two-doctor anxiety-level diagnosis contingency table
    reproduced at https://www.datanovia.com/en/lessons/weighted-kappa-in-r-for-two-ordinal-variables/,
    computed there with R's vcd::Kappa(); independently cross-checked here against sklearn and
    confirmed to reproduce the site's reported unweighted/linear/quadratic kappa values exactly
    (0.7335 / 0.7475 / 0.7664), i.e. two independently-implemented libraries (R's vcd, Python's
    sklearn) agree on the same raw data.
  - Fleiss' kappa: the canonical worked example from Fleiss, J. L. (1971). "Measuring Nominal
    Scale Agreement among Many Raters." Psychological Bulletin, 76(5), 378-382 (10 subjects, 14
    raters, 5 categories, kappa = 0.210), as reproduced on Wikipedia's "Fleiss' kappa" article.
"""

import math

import pytest

from eval import quality_agreement as qa


class TestKrippendorffAlphaAgainstThePublishedOrdinalExample:
    # From the `krippendorff` package's docstring, level_of_measurement="ordinal", reported as
    # round(alpha, 3) == 0.815. Values 1-5 here map directly onto our A-E encoding scheme (we
    # don't reuse GRADES because the source example has 5 ordered categories, not 3).
    RATER_RATINGS = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    ]
    LABELS = (1, 2, 3, 4, 5)

    def test_matches_the_published_value_to_three_decimal_places(self):
        alpha = qa.krippendorff_alpha(self.RATER_RATINGS, labels=self.LABELS)
        assert round(alpha, 3) == 0.815

    def test_a_rater_who_graded_nothing_does_not_change_alpha(self):
        # A rater with an entirely-`None` row contributes zero pairable values, so alpha must be
        # exactly unchanged by adding or removing them -- not merely "close".
        without_extra_rater = qa.krippendorff_alpha(self.RATER_RATINGS, labels=self.LABELS)
        all_missing_rater = [None] * len(self.RATER_RATINGS[0])
        with_extra_rater = qa.krippendorff_alpha([*self.RATER_RATINGS, all_missing_rater], labels=self.LABELS)
        assert with_extra_rater == pytest.approx(without_extra_rater)


class TestWeightedCohenKappaAgainstThePublishedAnxietyExample:
    # 4x4 contingency table (Doctor1 rows x Doctor2 columns), categories Normal/Moderate/High/Very
    # high, from the Datanovia worked example (see module docstring). Expanded to raw paired
    # rating lists here since our wrapper takes per-item ratings, not a contingency table.
    TABLE = [
        [11, 3, 1, 0],
        [1, 9, 0, 1],
        [0, 1, 10, 0],
        [1, 2, 0, 10],
    ]
    LABELS = ("Normal", "Moderate", "High", "Very high")

    @classmethod
    def _expand(cls):
        rater_a, rater_b = [], []
        for i, row in enumerate(cls.TABLE):
            for j, count in enumerate(row):
                rater_a.extend([cls.LABELS[i]] * count)
                rater_b.extend([cls.LABELS[j]] * count)
        return rater_a, rater_b

    def test_matches_the_published_quadratic_weighted_value(self):
        rater_a, rater_b = self._expand()
        kappa = qa.weighted_cohen_kappa(rater_a, rater_b, labels=self.LABELS)
        assert round(kappa, 4) == 0.7664

    def test_n_items_matches_the_table_total(self):
        rater_a, rater_b = self._expand()
        assert len(rater_a) == len(rater_b) == sum(sum(row) for row in self.TABLE) == 50

    def test_is_symmetric_in_the_two_raters(self):
        rater_a, rater_b = self._expand()
        assert qa.weighted_cohen_kappa(rater_a, rater_b, labels=self.LABELS) == pytest.approx(
            qa.weighted_cohen_kappa(rater_b, rater_a, labels=self.LABELS)
        )

    def test_perfect_agreement_on_the_projects_own_grades_is_kappa_one(self):
        ratings = ["A", "B", "C", "A", "B", "C", "A"]
        assert qa.weighted_cohen_kappa(ratings, ratings) == pytest.approx(1.0)

    def test_a_two_letter_miss_costs_more_than_a_one_letter_miss(self):
        # Quadratic weighting's whole point: A-vs-C must be penalized harder than A-vs-B. Truth
        # varies (A and C) so the comparison isn't the degenerate zero-variance case where both
        # raters give a single constant category and kappa collapses to an undefined 0/0.
        truth = ["A", "A", "A", "C", "C", "C"]
        one_letter_off = ["B", "B", "B", "B", "B", "B"]  # always exactly 1 away from truth
        two_letters_off = ["C", "C", "C", "A", "A", "A"]  # always exactly 2 away from truth
        kappa_near = qa.weighted_cohen_kappa(truth, one_letter_off)
        kappa_far = qa.weighted_cohen_kappa(truth, two_letters_off)
        assert kappa_far < kappa_near


class TestFleissKappaAgainstThePublishedPsychiatricExample:
    # Fleiss (1971)'s original worked example: 10 subjects, 14 raters, 5 diagnostic categories.
    # Table cell [i][j] = number of the 14 raters who assigned subject i to category j.
    COUNTS = [
        [0, 0, 0, 0, 14],
        [0, 2, 6, 4, 2],
        [0, 0, 3, 5, 6],
        [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0],
        [3, 2, 6, 3, 0],
        [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0],
        [0, 2, 2, 3, 7],
    ]
    LABELS = (1, 2, 3, 4, 5)

    @classmethod
    def _expand(cls):
        """Counts -> one row per subject, one column per (synthetic) rater slot, since our
        wrapper's `fleiss_kappa` takes raw per-rater assignments and re-aggregates them itself
        (the way real per-rater label data would arrive), not a pre-aggregated counts table."""
        rows = []
        for row in cls.COUNTS:
            expanded = [category for category, count in zip(cls.LABELS, row, strict=True) for _ in range(count)]
            assert len(expanded) == 14
            rows.append(expanded)
        # transpose: rater_ratings[r][subject] = category rater r gave that subject
        return [list(rater) for rater in zip(*rows, strict=True)]

    def test_matches_the_published_value_to_three_decimal_places(self):
        rater_ratings = self._expand()
        kappa = qa.fleiss_kappa(rater_ratings, labels=self.LABELS)
        assert round(kappa, 3) == 0.210


class TestBootstrapAlphaCi:
    RATER_RATINGS = [
        ["A", "A", "B", "C", "B", "A", "C", "A", "B", None],
        ["A", "A", "B", "C", "B", "B", "C", "A", "B", "C"],
        [None, "B", "B", "C", "B", "C", "C", "B", "B", "C"],
    ]

    def test_the_point_estimate_matches_krippendorff_alpha_directly(self):
        result = qa.bootstrap_alpha_ci(self.RATER_RATINGS, resamples=200)
        assert result["point"] == qa.krippendorff_alpha(self.RATER_RATINGS)

    def test_the_interval_contains_the_point_estimate(self):
        result = qa.bootstrap_alpha_ci(self.RATER_RATINGS, resamples=500)
        assert result["low"] <= result["point"] <= result["high"]

    def test_bounds_never_exceed_alphas_natural_range(self):
        # The "percentile" method (not the scipy default "BCa"/"basic") is chosen specifically so
        # this holds -- "basic" can reflect outside [something, 1] by construction.
        result = qa.bootstrap_alpha_ci(self.RATER_RATINGS, resamples=500)
        assert result["low"] <= 1.0
        assert result["high"] <= 1.0

    def test_is_deterministic_given_the_same_seed(self):
        first = qa.bootstrap_alpha_ci(self.RATER_RATINGS, resamples=200, seed=42)
        second = qa.bootstrap_alpha_ci(self.RATER_RATINGS, resamples=200, seed=42)
        assert first == second

    def test_empty_input_reports_nan_not_a_crash(self):
        result = qa.bootstrap_alpha_ci([[], []])
        assert math.isnan(result["point"])
        assert math.isnan(result["low"])
        assert math.isnan(result["high"])
        assert result["n_items"] == 0


class TestBootstrapKappaCi:
    RATER_A = ["A", "A", "B", "C", "B", "A", "C", "A", "B", "C"]
    RATER_B = ["A", "B", "B", "C", "B", "B", "C", "A", "B", "C"]

    def test_the_point_estimate_matches_weighted_cohen_kappa_directly(self):
        result = qa.bootstrap_kappa_ci(self.RATER_A, self.RATER_B, resamples=200)
        assert result["point"] == qa.weighted_cohen_kappa(self.RATER_A, self.RATER_B)

    def test_the_interval_contains_the_point_estimate(self):
        result = qa.bootstrap_kappa_ci(self.RATER_A, self.RATER_B, resamples=500)
        assert result["low"] <= result["point"] <= result["high"]

    def test_empty_input_reports_nan_not_a_crash(self):
        result = qa.bootstrap_kappa_ci([], [])
        assert math.isnan(result["point"])
        assert math.isnan(result["low"])
        assert result["n_items"] == 0


class TestCannotGradeRate:
    def test_counts_any_and_all_separately(self):
        ratings = [
            ["A", None, None, "B"],
            ["A", "B", None, "C"],
        ]
        result = qa.cannot_grade_rate(ratings)
        # item 0: both graded; item 1: one missing (any, not all); item 2: both missing (any+all);
        # item 3: both graded.
        assert result["any_cannot_grade"] == pytest.approx(2 / 4)
        assert result["all_cannot_grade"] == pytest.approx(1 / 4)
        assert result["n_items"] == 4

    def test_no_missing_entries_is_zero_not_nan(self):
        result = qa.cannot_grade_rate([["A", "B"], ["A", "C"]])
        assert result["any_cannot_grade"] == 0.0
        assert result["all_cannot_grade"] == 0.0

    def test_empty_input_reports_nan_not_a_crash(self):
        result = qa.cannot_grade_rate([[], []])
        assert math.isnan(result["any_cannot_grade"])
        assert result["n_items"] == 0


class TestDefaultLabelsMatchTheProjectsOwnScale:
    def test_grades_constant_is_a_b_c_in_worse_is_higher_order(self):
        assert qa.GRADES == ("A", "B", "C")

    def test_default_labels_are_used_when_not_overridden(self):
        ratings = [["A", "B", "C"], ["A", "C", "C"]]
        assert qa.weighted_cohen_kappa(*ratings) == qa.weighted_cohen_kappa(*ratings, labels=qa.GRADES)


class TestKrippendorffAlphaOnTheProjectsOwnThreeGradeScale:
    def test_perfect_agreement_is_alpha_one(self):
        ratings = [["A", "B", "C", "A", "B"], ["A", "B", "C", "A", "B"]]
        assert qa.krippendorff_alpha(ratings) == pytest.approx(1.0)

    def test_systematic_disagreement_is_alpha_near_zero_or_negative(self):
        # Every item disagrees by the maximum possible amount (A vs C) -- this is not "chance"
        # agreement, it's worse: alpha should be clearly negative, not merely below 1.
        ratings = [["A", "A", "A", "A"], ["C", "C", "C", "C"]]
        assert qa.krippendorff_alpha(ratings) < 0

    def test_a_one_letter_disagreement_scores_higher_than_a_two_letter_one(self):
        truth = ["A", "B", "C", "A", "B", "C", "A", "B"]
        one_off = ["B", "C", "B", "B", "A", "B", "B", "A"]  # every entry off by exactly one letter
        two_off = ["C", "A", "A", "C", "C", "A", "C", "A"]  # every entry off by two letters where possible
        alpha_one_off = qa.krippendorff_alpha([truth, one_off])
        alpha_two_off = qa.krippendorff_alpha([truth, two_off])
        assert alpha_one_off > alpha_two_off


class TestFleissKappaIsNotOrdinalAware:
    """Documents, with a test, exactly why PROTOCOL.md section 7 calls Fleiss' kappa optional and
    never a substitute for the ordinal alpha: it is computed purely from the per-item vote-count
    table, with no notion of which categories those votes are for -- so relabeling which category
    is "adjacent" and which is "extreme" (swapping B and C below) leaves it exactly unchanged,
    even though that swap turns every A-B (one-letter) disagreement into an A-C (two-letter) one
    and vice versa."""

    def test_swapping_which_category_is_adjacent_does_not_change_kappa(self):
        original = [["A", "B", "C", "A"], ["B", "C", "A", "B"]]
        relabel = {"A": "A", "B": "C", "C": "B"}
        swapped = [[relabel[g] for g in rater] for rater in original]
        assert qa.fleiss_kappa(original) == pytest.approx(qa.fleiss_kappa(swapped))


class TestEncodeHandlesMissingConsistently:
    def test_none_becomes_nan(self):
        encoded = qa._encode(["A", None, "C"], qa.GRADES)
        assert encoded[0] == 0.0
        assert math.isnan(encoded[1])
        assert encoded[2] == 2.0

    def test_ordinal_order_matches_grade_severity(self):
        assert qa._encode(["A"], qa.GRADES) < qa._encode(["B"], qa.GRADES) < qa._encode(["C"], qa.GRADES)
