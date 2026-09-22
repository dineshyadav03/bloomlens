"""eval/scores.py: the three candidate in-distribution scores (eval/PROTOCOL.md section 3)."""

import math

import pytest

from eval import scores


class TestMaxCosine:
    def test_it_is_the_largest_value_whatever_the_order(self):
        assert scores.max_cosine([0.2, 0.9, 0.5]) == 0.9
        assert scores.max_cosine([0.9, 0.2, 0.5]) == 0.9

    def test_a_single_candidate_is_its_own_score(self):
        assert scores.max_cosine([0.42]) == 0.42

    def test_negative_similarities_are_handled_like_any_other_number(self):
        assert scores.max_cosine([-0.5, -0.1, -0.9]) == -0.1

    def test_an_empty_list_is_an_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            scores.max_cosine([])


class TestMargin:
    def test_it_is_the_gap_between_the_top_two(self):
        assert scores.margin([0.7, 0.5, 0.9]) == pytest.approx(0.2)

    def test_a_single_candidate_has_no_runner_up_so_it_is_its_own_score(self):
        assert scores.margin([0.6]) == 0.6

    def test_a_tie_for_first_gives_a_zero_margin(self):
        assert scores.margin([0.5, 0.5, 0.1]) == pytest.approx(0.0)

    def test_it_is_never_negative_for_a_correctly_sorted_top_1(self):
        assert scores.margin([0.3, 0.9, 0.9]) >= 0


class TestMaxSoftmax:
    def test_it_sums_to_one_over_all_candidates_at_every_temperature(self):
        sims = [0.6, 0.5, 0.4, 0.1]
        for t in scores.TEMPERATURES:
            total = sum(math.exp((s - max(sims)) / t) for s in sims)
            top = math.exp((max(sims) - max(sims)) / t) / total
            assert scores.max_softmax(sims, t) == pytest.approx(top)

    def test_it_is_between_the_uniform_value_and_one(self):
        sims = [0.6, 0.5, 0.4, 0.1]
        for t in scores.TEMPERATURES:
            assert 1 / len(sims) <= scores.max_softmax(sims, t) <= 1.0

    def test_a_lower_temperature_sharpens_the_distribution_toward_one(self):
        sims = [0.6, 0.55, 0.5]
        values = [scores.max_softmax(sims, t) for t in sorted(scores.TEMPERATURES)]
        assert values == sorted(values, reverse=True)  # smallest T -> largest max-softmax

    def test_equal_similarities_give_the_uniform_value(self):
        assert scores.max_softmax([0.5, 0.5, 0.5, 0.5], 0.05) == pytest.approx(0.25)

    def test_it_does_not_overflow_on_large_similarities(self):
        assert 0 < scores.max_softmax([1000.0, 999.0, 1.0], 0.01) <= 1.0

    def test_zero_or_negative_temperature_is_an_error(self):
        with pytest.raises(ValueError, match="positive"):
            scores.max_softmax([0.5, 0.4], 0.0)
        with pytest.raises(ValueError, match="positive"):
            scores.max_softmax([0.5, 0.4], -0.1)

    def test_single_candidate_is_certain(self):
        assert scores.max_softmax([0.3], 0.05) == pytest.approx(1.0)


class TestCandidateNaming:
    def test_max_softmax_carries_its_temperature(self):
        assert scores.candidate_name("max_softmax", 0.05) == "max_softmax@0.05"

    def test_the_others_do_not(self):
        assert scores.candidate_name("max_cosine") == "max_cosine"
        assert scores.candidate_name("margin") == "margin"

    def test_score_by_name_round_trips_every_candidate(self):
        sims = [0.7, 0.5, 0.3]
        for name, value in scores.all_candidates(sims).items():
            assert scores.score_by_name(name, sims) == pytest.approx(value)

    def test_an_unknown_candidate_name_is_an_error(self):
        with pytest.raises(ValueError, match="unknown candidate"):
            scores.score_by_name("min_cosine", [0.5])


class TestAllCandidates:
    def test_it_has_exactly_max_cosine_margin_and_one_softmax_per_temperature(self):
        names = set(scores.all_candidates([0.5, 0.3]))
        expected = {"max_cosine", "margin"} | {f"max_softmax@{t}" for t in scores.TEMPERATURES}
        assert names == expected
