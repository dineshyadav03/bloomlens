"""Pure decision logic in src/identify.py: no model, no network, no agent."""

import json
import math
from types import SimpleNamespace

import pytest

from src import identify as ident
from src.identify import IdentifyError


class TestClassifyConfidence:
    def test_clearly_high(self, make_candidate):
        assert ident._classify_confidence([make_candidate("A", 0.70), make_candidate("B", 0.50)]) == "high"

    def test_low_when_top_score_is_below_the_floor(self, make_candidate):
        assert ident._classify_confidence([make_candidate("A", 0.30), make_candidate("B", 0.10)]) == "low"

    def test_low_floor_is_exclusive(self, make_candidate):
        floor = ident._LOW_CONFIDENCE_MAX_SCORE
        at_floor = [make_candidate("A", floor), make_candidate("B", floor - 0.001)]
        assert ident._classify_confidence(at_floor) == "ambiguous"
        just_below = [make_candidate("A", math.nextafter(floor, 0.0)), make_candidate("B", 0.0)]
        assert ident._classify_confidence(just_below) == "low"

    def test_high_requires_both_score_and_gap(self, make_candidate):
        # high score but a tiny gap to the runner-up
        assert ident._classify_confidence([make_candidate("A", 0.60), make_candidate("B", 0.59)]) == "ambiguous"
        # big gap but the top score is under the high-confidence minimum
        assert ident._classify_confidence([make_candidate("A", 0.50), make_candidate("B", 0.10)]) == "ambiguous"

    def test_high_min_score_boundary(self, make_candidate):
        minimum = ident._HIGH_CONFIDENCE_MIN_SCORE
        assert ident._classify_confidence([make_candidate("A", minimum), make_candidate("B", 0.0)]) == "high"
        below = math.nextafter(minimum, 0.0)
        assert ident._classify_confidence([make_candidate("A", below), make_candidate("B", 0.0)]) == "ambiguous"

    def test_gap_threshold_is_where_the_constant_says(self, make_candidate):
        """Bisect for the runner-up score at which 'high' flips to 'ambiguous'; it
        must sit at top - MIN_GAP (avoids float-equality traps in a boundary test)."""
        top = 0.70
        lo, hi = 0.0, top  # runner-up 0.0 -> high; runner-up == top -> ambiguous
        for _ in range(60):
            mid = (lo + hi) / 2
            tier = ident._classify_confidence([make_candidate("A", top), make_candidate("B", mid)])
            if tier == "high":
                lo = mid
            else:
                hi = mid
        assert (top - lo) == pytest.approx(ident._HIGH_CONFIDENCE_MIN_GAP, abs=1e-9)

    def test_single_candidate_uses_its_score_as_the_gap(self, make_candidate):
        assert ident._classify_confidence([make_candidate("A", 0.60)]) == "high"
        assert ident._classify_confidence([make_candidate("A", 0.50)]) == "ambiguous"


class TestComputeConsensus:
    def test_majority_wins(self):
        species, fraction = ident._compute_consensus([("Rose", 0.6), ("Rose", 0.7), ("Tulip", 0.9)])
        assert species == "Rose"
        assert fraction == pytest.approx(2 / 3)

    def test_tie_goes_to_the_higher_summed_score(self):
        species, fraction = ident._compute_consensus([("Rose", 0.6), ("Tulip", 0.7)])
        assert (species, fraction) == ("Tulip", 0.5)

    def test_exact_tie_goes_to_the_first_seen_species(self):
        # documents current behavior: max() keeps the first maximal item
        species, _ = ident._compute_consensus([("Rose", 0.5), ("Tulip", 0.5)])
        assert species == "Rose"

    def test_unanimous_and_single(self):
        assert ident._compute_consensus([("Rose", 0.6), ("Rose", 0.7)]) == ("Rose", 1.0)
        assert ident._compute_consensus([("Rose", 0.6)]) == ("Rose", 1.0)


class TestMatchCandidate:
    NAMES = ("Rose", "Tulip", "Moth orchid")

    @pytest.fixture
    def candidates(self, make_candidate):
        return [make_candidate(n, 0.5) for n in self.NAMES]

    def test_exact_match_ignores_case_and_whitespace(self, candidates):
        assert ident._match_candidate("  rOsE ", candidates) == "Rose"

    def test_answer_with_extra_text_still_matches(self, candidates):
        assert ident._match_candidate("Moth orchid (Phalaenopsis amabilis)", candidates) == "Moth orchid"

    def test_unrelated_answer_falls_back_to_the_top_candidate(self, candidates):
        assert ident._match_candidate("a very fancy hydrangea", candidates) == "Rose"


class TestExtractFinalText:
    def test_plain_string_content(self):
        result = {"messages": [SimpleNamespace(content="hello")]}
        assert ident._extract_final_text(result) == "hello"

    def test_list_of_content_blocks_is_joined(self):
        content = [{"type": "text", "text": "he", "extras": {"x": 1}}, {"type": "text", "text": "llo"}]
        assert ident._extract_final_text({"messages": [SimpleNamespace(content=content)]}) == "hello"

    def test_non_text_and_non_dict_blocks_are_ignored(self):
        content = ["stray string", {"type": "image", "data": "..."}, {"type": "text", "text": "ok"}]
        assert ident._extract_final_text({"messages": [SimpleNamespace(content=content)]}) == "ok"

    def test_only_the_last_message_is_used(self):
        result = {"messages": [SimpleNamespace(content="first"), SimpleNamespace(content="last")]}
        assert ident._extract_final_text(result) == "last"


class TestExtractJson:
    def test_plain_object(self):
        assert ident._extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        assert ident._extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_surrounding_prose_is_ignored(self):
        assert ident._extract_json('Sure! Here you go: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}

    def test_no_object_raises_identify_error(self):
        with pytest.raises(IdentifyError, match="didn't contain a JSON object"):
            ident._extract_json("I could not decide.")

    def test_invalid_json_inside_braces_raises_a_decode_error(self):
        # not an IdentifyError on purpose: the retry loop treats it as retryable
        with pytest.raises(json.JSONDecodeError):
            ident._extract_json("{not json}")
