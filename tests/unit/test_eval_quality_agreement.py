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


# --- the protocol's report -----------------------------------------------------------------
# SYNTHETIC labels only: made-up raters and item ids, chosen so each protocol rule has a case that
# exercises it. Nothing here is real rater data.

RESAMPLES = 200  # enough to exercise the bootstrap; the real default is 9,999


def synthetic_labels():
    return {
        "i1": {"r1": "A", "r2": "A", "r3": "A"},
        "i2": {"r1": "B", "r2": "B", "r3": "C"},
        "i3": {"r1": "A", "r2": "C", "r3": "B"},  # A vs C: a two-letter gap -> adjudicate
        "i4": {"r1": "CANNOT_GRADE", "r2": "B", "r3": "B"},  # excluded from the primary analysis
        "i5": {"r1": "C", "r2": "C", "r3": "C"},
        "i6": {"r1": "B", "r2": "A", "r3": "B"},
    }


# Hand-listed from synthetic_labels(): the items no rater marked "cannot grade" (i4 out), per rater.
PRIMARY_R1 = ["A", "B", "A", "C", "B"]
PRIMARY_R2 = ["A", "B", "C", "C", "A"]
PRIMARY_R3 = ["A", "C", "B", "C", "B"]


class TestItemsNeedingAdjudication:
    def test_only_a_gap_of_more_than_one_letter_is_sent_to_a_third_rater(self):
        assert qa.items_needing_adjudication(synthetic_labels()) == ["i3"]

    def test_a_one_letter_gap_is_not_adjudicated(self):
        assert qa.items_needing_adjudication({"x": {"r1": "A", "r2": "B"}, "y": {"r1": "B", "r2": "C"}}) == []

    def test_cannot_grade_is_not_a_disagreement_about_the_grade(self):
        assert qa.items_needing_adjudication({"x": {"r1": "A", "r2": "CANNOT_GRADE"}}) == []

    def test_a_cannot_grade_does_not_hide_a_real_two_letter_gap_between_the_others(self):
        assert qa.items_needing_adjudication({"x": {"r1": "A", "r2": "C", "r3": "CANNOT_GRADE"}}) == ["x"]

    def test_a_single_grade_cannot_disagree_with_itself(self):
        assert qa.items_needing_adjudication({"x": {"r1": "A"}}) == []

    def test_the_result_is_sorted_by_item_id(self):
        labels = {"b": {"r1": "A", "r2": "C"}, "a": {"r1": "A", "r2": "C"}}
        assert qa.items_needing_adjudication(labels) == ["a", "b"]


class TestAgreementReport:
    @pytest.fixture
    def report(self):
        return qa.agreement_report(synthetic_labels(), resamples=RESAMPLES)

    def test_counts(self, report):
        assert (report["n_items"], report["n_raters"], report["raters"]) == (6, 3, ["r1", "r2", "r3"])
        assert report["n_items_in_primary"] == 5  # i4 has a "cannot grade"

    def test_alpha_is_the_ordinal_alpha_of_the_primary_items(self, report):
        expected = qa.krippendorff_alpha([PRIMARY_R1, PRIMARY_R2, PRIMARY_R3])
        assert report["alpha"]["point"] == pytest.approx(expected)
        assert report["alpha"]["n_items"] == 5
        assert report["alpha"]["low"] <= expected <= report["alpha"]["high"]

    def test_there_is_one_pairwise_kappa_per_rater_pair_over_their_shared_graded_items(self, report):
        by_pair = {p["raters"]: p for p in report["pairwise_kappa"]}
        assert set(by_pair) == {("r1", "r2"), ("r1", "r3"), ("r2", "r3")}
        assert by_pair[("r1", "r2")]["point"] == pytest.approx(qa.weighted_cohen_kappa(PRIMARY_R1, PRIMARY_R2))
        assert by_pair[("r2", "r3")]["point"] == pytest.approx(qa.weighted_cohen_kappa(PRIMARY_R2, PRIMARY_R3))
        assert all(p["n_items"] == 5 for p in report["pairwise_kappa"])  # never i4

    def test_the_mean_pairwise_kappa_is_the_plain_mean_of_the_pairs(self, report):
        points = [p["point"] for p in report["pairwise_kappa"]]
        assert report["mean_pairwise_kappa"] == pytest.approx(sum(points) / len(points))

    def test_fleiss_uses_only_the_items_every_rater_graded(self, report):
        expected = qa.fleiss_kappa([PRIMARY_R1, PRIMARY_R2, PRIMARY_R3])
        assert report["fleiss_kappa"] == {"value": pytest.approx(expected), "n_items": 5}

    def test_cannot_grade_rates_count_the_item_any_rater_could_not_grade(self, report):
        assert report["cannot_grade"]["any_cannot_grade"] == pytest.approx(1 / 6)
        assert report["cannot_grade"]["all_cannot_grade"] == 0.0

    def test_the_sensitivity_alpha_keeps_cannot_grade_as_a_category_below_c(self, report):
        labels = synthetic_labels()
        rows = [[labels[i][r] for i in sorted(labels)] for r in ("r1", "r2", "r3")]
        expected = qa.krippendorff_alpha(rows, labels=("A", "B", "C", "CANNOT_GRADE"))
        assert report["alpha_cannot_grade_as_lowest_category"] == pytest.approx(expected)
        assert report["alpha_cannot_grade_as_lowest_category"] != pytest.approx(report["alpha"]["point"])

    def test_it_lists_the_items_needing_a_third_rater(self, report):
        assert report["needing_adjudication"] == ["i3"]

    def test_an_item_a_rater_has_not_reached_yet_is_not_counted_as_cannot_grade(self):
        labels = {**synthetic_labels(), "i7": {"r1": "A", "r2": "A"}}  # r3 hasn't labeled i7
        report = qa.agreement_report(labels, resamples=RESAMPLES)
        assert report["cannot_grade"]["n_items"] == 6  # i7 is not fully rated, so it is not in the rate
        assert report["cannot_grade"]["any_cannot_grade"] == pytest.approx(1 / 6)
        assert report["n_items_in_primary"] == 6  # ...but it does still count toward the primary analysis

    def test_a_single_rater_has_no_agreement_statistics(self):
        report = qa.agreement_report({"i1": {"r1": "A"}, "i2": {"r1": "B"}}, resamples=RESAMPLES)
        assert report["alpha"] is None
        assert report["pairwise_kappa"] == []
        assert report["mean_pairwise_kappa"] is None
        assert report["fleiss_kappa"] is None
        assert report["alpha_cannot_grade_as_lowest_category"] is None

    def test_the_resample_count_is_read_at_call_time_not_import_time(self, monkeypatch):
        monkeypatch.setattr(qa, "BOOTSTRAP_RESAMPLES", 37)
        report = qa.agreement_report(synthetic_labels())
        assert report["alpha"]["resamples"] == 37
        assert all(p["resamples"] == 37 for p in report["pairwise_kappa"])


class TestBootstrapResamplesAreReadAtCallTime:
    def test_alpha_and_kappa_honor_a_patched_default(self, monkeypatch):
        monkeypatch.setattr(qa, "BOOTSTRAP_RESAMPLES", 41)
        assert qa.bootstrap_alpha_ci([PRIMARY_R1, PRIMARY_R2])["resamples"] == 41
        assert qa.bootstrap_kappa_ci(PRIMARY_R1, PRIMARY_R2)["resamples"] == 41


class TestFormatReport:
    @pytest.fixture
    def text(self):
        return qa.format_report(qa.agreement_report(synthetic_labels(), resamples=RESAMPLES))

    def test_it_states_its_own_limits(self, text):
        assert "Pooled across all species only" in text
        assert "does not validate BloomLens's own `quality_grade`" in text
        assert "unvalidated heuristic" in text

    def test_it_reports_every_protocol_number(self, text):
        for expected in (
            "Krippendorff's alpha (ordinal)",
            "r1 vs r2",
            "r2 vs r3",
            "Mean of the pairwise kappas",
            "Fleiss' kappa (unweighted",
            "At least one rater could not grade: 16.7% of 6",
            "Sensitivity",
            "1 item(s) differ by more than one letter and go to a third rater: i3",
        ):
            assert expected in text, expected

    def test_a_confidence_interval_is_shown_with_every_point_estimate(self, text):
        assert text.count("95% bootstrap CI") == 1 + 3  # alpha + three rater pairs

    def test_uncomputable_numbers_say_so_instead_of_crashing(self):
        text = qa.format_report(qa.agreement_report({"i1": {"r1": "A"}}, resamples=RESAMPLES))
        assert "not computable" in text
        assert "none (needs a pair of raters" in text

    def test_no_adjudication_needed_is_stated_plainly(self):
        labels = {"i1": {"r1": "A", "r2": "A"}, "i2": {"r1": "B", "r2": "C"}}
        text = qa.format_report(qa.agreement_report(labels, resamples=RESAMPLES))
        assert "0 item(s) differ by more than one letter and go to a third rater." in text


class TestMain:
    def test_with_no_labels_it_says_so(self, capsys):
        qa.main([])
        assert "No labels yet" in capsys.readouterr().out

    def test_it_reports_on_what_the_labeler_stored(self, capsys, monkeypatch):
        from eval import quality_labels_db as db

        monkeypatch.setattr(qa, "BOOTSTRAP_RESAMPLES", 50)
        for item, by_rater in synthetic_labels().items():
            for rater, grade in by_rater.items():
                db.submit_label(rater, item, calibration=False, grade=grade)
        qa.main([])
        out = capsys.readouterr().out
        assert "# Quality-label agreement report" in out
        assert "3 rater(s), 6 item(s) labeled" in out

    def test_practice_round_labels_never_reach_the_report(self, capsys):
        from eval import quality_labels_db as db

        db.submit_label("r1", "practice-1", calibration=True, grade="A")
        db.submit_label("r2", "practice-1", calibration=True, grade="C")
        qa.main([])
        assert "No labels yet" in capsys.readouterr().out


# --- undefined statistics: a sample with no variation, or a resample that loses it ------------
# Found by running the report on a 5-item synthetic set: one bootstrap resample happened to draw
# only unanimous "A" items, and krippendorff.alpha raised instead of saying "undefined".


class TestUndefinedStatisticsAreNanNotCrashes:
    def test_alpha_is_nan_when_every_grade_is_identical(self):
        assert math.isnan(qa.krippendorff_alpha([["A", "A", "A"], ["A", "A", "A"]]))

    def test_alpha_is_nan_when_no_item_was_graded_by_two_raters(self):
        assert math.isnan(qa.krippendorff_alpha([["A", None, "B"], [None, "C", None]]))

    def test_kappa_is_nan_when_both_raters_gave_one_identical_grade(self):
        assert math.isnan(qa.weighted_cohen_kappa(["B", "B", "B"], ["B", "B", "B"]))

    def test_the_bootstrap_of_an_undefined_point_is_undefined_not_an_exception(self):
        alpha = qa.bootstrap_alpha_ci([["A", "A"], ["A", "A"]], resamples=RESAMPLES)
        kappa = qa.bootstrap_kappa_ci(["B", "B"], ["B", "B"], resamples=RESAMPLES)
        for result in (alpha, kappa):
            assert math.isnan(result["point"]) and math.isnan(result["low"]) and math.isnan(result["high"])
            assert result["n_undefined_resamples"] == 0
            assert result["n_items"] == 2

    def test_a_resample_with_no_variation_is_counted_and_left_out_not_a_crash(self):
        # Two items, unanimous but different: a resample that draws the same item twice has no
        # variation (undefined); one that draws both is fine (alpha = 1). Of many resamples about
        # half are undefined -- and every defined one is 1.0, so the interval is exactly [1, 1].
        raters = [["A", "C"], ["A", "C"]]
        result = qa.bootstrap_alpha_ci(raters, resamples=400)
        assert result["point"] == pytest.approx(1.0)
        assert 100 < result["n_undefined_resamples"] < 300
        assert result["low"] == result["high"] == pytest.approx(1.0)

    def test_the_same_holds_for_kappa(self):
        result = qa.bootstrap_kappa_ci(["A", "C"], ["A", "C"], resamples=400)
        assert result["point"] == pytest.approx(1.0)
        assert 100 < result["n_undefined_resamples"] < 300
        assert result["low"] == result["high"] == pytest.approx(1.0)

    def test_a_well_behaved_sample_has_no_undefined_resamples(self):
        rater_a = ["A", "B", "C"] * 10
        rater_b = ["A", "C", "C"] * 10
        assert qa.bootstrap_kappa_ci(rater_a, rater_b, resamples=200)["n_undefined_resamples"] == 0

    def test_no_warning_escapes_from_an_undefined_kappa(self, recwarn):
        qa.weighted_cohen_kappa(["B", "B", "B"], ["B", "B", "B"])
        qa.bootstrap_kappa_ci(["A", "C"], ["A", "C"], resamples=50)
        assert not [w for w in recwarn if "cohen_kappa" in str(w.message) or "BCa" in str(w.message)]

    def test_a_report_on_unanimous_labels_says_undefined_instead_of_crashing(self):
        labels = {"i1": {"r1": "A", "r2": "A"}, "i2": {"r1": "A", "r2": "A"}}
        text = qa.format_report(qa.agreement_report(labels, resamples=RESAMPLES))
        assert "undefined -- the grades show no variation" in text
        assert "nan" not in text.lower().replace("unanimous", "")

    def test_the_report_discloses_resamples_that_were_left_out(self):
        labels = {"i1": {"r1": "A", "r2": "A"}, "i2": {"r1": "C", "r2": "C"}}
        text = qa.format_report(qa.agreement_report(labels, resamples=400))
        assert "resamples had no variation and are left out" in text


class TestFleissUndefinedWhenUnanimous:
    def test_is_nan_and_raises_no_warning_when_every_rater_gave_one_grade(self, recwarn):
        assert math.isnan(qa.fleiss_kappa([["A", "A", "A"], ["A", "A", "A"]]))
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


class TestIntervalMatchesScipyForWellBehavedSamples:
    """With no undefined resamples, our interval must be exactly scipy's own percentile interval
    for the same data, seed and confidence level -- so a mistake in the tail arithmetic (which
    percentiles are taken) cannot hide behind a plausible-looking range."""

    RATER_A = ["A", "B", "C"] * 10
    RATER_B = ["A", "C", "C", "B", "B", "C"] * 5

    @pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95])
    def test_kappa(self, confidence):
        import numpy as np
        from scipy import stats

        ours = qa.bootstrap_kappa_ci(self.RATER_A, self.RATER_B, resamples=300, seed=5, confidence=confidence)
        theirs = stats.bootstrap(
            (np.array(self.RATER_A, dtype=object), np.array(self.RATER_B, dtype=object)),
            lambda a, b: qa.weighted_cohen_kappa(list(a), list(b)),
            paired=True,
            vectorized=False,
            n_resamples=300,
            method="percentile",
            confidence_level=confidence,
            random_state=np.random.default_rng(5),
        )
        assert ours["n_undefined_resamples"] == 0
        assert ours["low"] == pytest.approx(theirs.confidence_interval.low)
        assert ours["high"] == pytest.approx(theirs.confidence_interval.high)

    def test_a_lower_confidence_level_gives_a_narrower_interval(self):
        narrow = qa.bootstrap_kappa_ci(self.RATER_A, self.RATER_B, resamples=300, confidence=0.5)
        wide = qa.bootstrap_kappa_ci(self.RATER_A, self.RATER_B, resamples=300, confidence=0.95)
        assert narrow["high"] - narrow["low"] < wide["high"] - wide["low"]

    def test_identical_raters_stay_identical_in_every_resample(self):
        # Pairing is what keeps them identical: resampling each rater's column independently would
        # pull the lower bound well below 1.
        result = qa.bootstrap_kappa_ci(self.RATER_A, self.RATER_A, resamples=300)
        assert result["low"] == result["high"] == pytest.approx(1.0)


class TestAdjudicators:
    """PROTOCOL.md section 5: a third rater resolves a two-letter gap, but is kept OUT of the
    agreement statistics so adjudication cannot inflate them."""

    def labels_with_adjudicator(self):
        labels = synthetic_labels()
        labels["i3"] = {**labels["i3"], "adj": "B"}  # the adjudicator's call on the A-vs-C item
        return labels

    def test_without_raters_removes_them_and_drops_items_left_empty(self):
        labels = {"i1": {"r1": "A", "adj": "B"}, "i2": {"adj": "C"}}
        assert qa.without_raters(labels, ["adj"]) == {"i1": {"r1": "A"}}

    def test_the_adjudicator_is_not_counted_as_a_rater_in_any_statistic(self):
        with_adjudicator = qa.agreement_report(
            self.labels_with_adjudicator(), resamples=RESAMPLES, adjudicators=["adj"]
        )
        without = qa.agreement_report(synthetic_labels(), resamples=RESAMPLES)
        assert with_adjudicator["raters"] == ["r1", "r2", "r3"]
        assert with_adjudicator["alpha"] == without["alpha"]
        assert with_adjudicator["pairwise_kappa"] == without["pairwise_kappa"]
        assert with_adjudicator["fleiss_kappa"] == without["fleiss_kappa"]

    def test_left_in_as_an_ordinary_rater_the_adjudicator_would_change_the_numbers(self):
        # ...which is exactly why they are excluded: this shows the exclusion is doing real work.
        treated_as_a_rater = qa.agreement_report(self.labels_with_adjudicator(), resamples=RESAMPLES)
        assert treated_as_a_rater["n_raters"] == 4
        assert treated_as_a_rater["alpha"] != qa.agreement_report(synthetic_labels(), resamples=RESAMPLES)["alpha"]

    def test_the_outcome_is_reported_for_the_item_that_needed_it(self):
        report = qa.agreement_report(self.labels_with_adjudicator(), resamples=RESAMPLES, adjudicators=["adj"])
        assert report["needing_adjudication"] == ["i3"]  # still a raw disagreement between the originals
        assert report["adjudicated"] == {"i3": {"adj": "B"}}

    def test_an_item_with_no_adjudicator_label_is_awaiting_one(self):
        report = qa.agreement_report(synthetic_labels(), resamples=RESAMPLES, adjudicators=["adj"])
        assert report["adjudicated"] == {}
        text = qa.format_report(report)
        assert "i3: awaiting a third rater" in text

    def test_the_formatted_report_shows_the_outcome_and_the_exclusion_rule(self):
        report = qa.agreement_report(self.labels_with_adjudicator(), resamples=RESAMPLES, adjudicators=["adj"])
        text = qa.format_report(report)
        assert "i3: adjudicated (adj: B)" in text
        assert "Adjudicator labels are excluded from every agreement statistic" in text

    def test_an_adjudicators_label_on_an_item_that_did_not_need_one_is_not_reported_as_an_outcome(self):
        labels = synthetic_labels()
        labels["i1"] = {**labels["i1"], "adj": "A"}
        report = qa.agreement_report(labels, resamples=RESAMPLES, adjudicators=["adj"])
        assert "i1" not in report["adjudicated"]

    def test_main_takes_adjudicators_from_the_command_line(self, capsys, monkeypatch):
        from eval import quality_labels_db as db

        monkeypatch.setattr(qa, "BOOTSTRAP_RESAMPLES", 50)
        for item, by_rater in self.labels_with_adjudicator().items():
            for rater, grade in by_rater.items():
                db.submit_label(rater, item, calibration=False, grade=grade)
        qa.main(["--adjudicator", "adj"])
        out = capsys.readouterr().out
        assert "3 rater(s), 6 item(s) labeled" in out
        assert "i3: adjudicated (adj: B)" in out
