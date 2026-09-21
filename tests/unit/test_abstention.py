"""The machine-readable abstention signal: `abstained` and `abstain_source` on the results.

Two explicit signals only -- the retrieval policy (tier `low`) and the agent's own
`matches_a_candidate == false` flag. What the model *wrote* (its notes, its summary) is never
searched for words: eval/PROTOCOL.md section 7."""

import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from src import identify as ident
from src.identify import _abstention, _GeminiAnswer

GOOD = {
    "species": "Rose",
    "confidence_note": "clearly a rose",
    "quality_grade": "A",
    "quality_note": "fresh",
    "summary": "A fine rose.",
}


def answer(**overrides) -> _GeminiAnswer:
    return _GeminiAnswer(**{**GOOD, **overrides})


class TestPolicy:
    @pytest.mark.parametrize(
        "tier, agent, expected",
        [
            ("high", True, (False, None)),
            ("high", None, (False, None)),
            ("ambiguous", True, (False, None)),
            ("ambiguous", None, (False, None)),
            ("low", True, (True, "retrieval")),
            ("low", None, (True, "retrieval")),
            ("high", False, (True, "agent")),
            ("ambiguous", False, (True, "agent")),
            ("low", False, (True, "both")),
        ],
    )
    def test_the_truth_table(self, tier, agent, expected):
        assert _abstention(tier, agent) == expected

    def test_a_missing_opinion_from_the_agent_is_not_a_no(self):
        """None means 'gave no opinion' -- only an explicit false counts."""
        assert _abstention("high", None) == (False, None)


class TestAgentAnswerSchema:
    def test_the_flag_is_optional(self):
        assert answer().matches_a_candidate is None

    @pytest.mark.parametrize("raw, expected", [(True, True), (False, False), ("true", True), ("false", False)])
    def test_a_boolean_or_its_string_form_is_accepted(self, raw, expected):
        assert answer(matches_a_candidate=raw).matches_a_candidate is expected

    @pytest.mark.parametrize("raw", ["maybe", "not a flower", [True], {"a": 1}, 7])
    def test_anything_else_is_a_malformed_answer(self, raw):
        with pytest.raises(ValidationError):
            answer(matches_a_candidate=raw)

    def test_the_prompt_asks_for_the_flag_explicitly_and_as_a_real_boolean(self):
        prompt = ident._SYSTEM_PROMPT
        assert "matches_a_candidate" in prompt and "true or false, not a string" in prompt


class TestResults:
    @pytest.fixture
    def pipeline(self, monkeypatch, prices_csv, make_candidate):
        class Handle:
            candidates = [make_candidate("Rose", 0.70), make_candidate("Tulip", 0.50), make_candidate("Peony", 0.40)]
            per_photo = None
            agent_answer = answer()

        handle = Handle()

        def fake_search(_client, _embedding, top_k=3):
            return handle.per_photo.pop(0) if handle.per_photo else handle.candidates

        monkeypatch.setattr(ident, "embed_image", lambda _img: np.zeros(768, dtype=np.float32))
        monkeypatch.setattr(ident, "get_client", lambda: object())
        monkeypatch.setattr(ident, "search", fake_search)
        monkeypatch.setattr(ident, "_invoke_agent_with_retries", lambda _message: handle.agent_answer)
        return handle

    @pytest.fixture
    def image(self):
        return Image.new("RGB", (32, 32), (10, 200, 30))

    def test_a_confident_matching_photo_does_not_abstain(self, pipeline, image):
        result = ident.identify(image)
        assert (result.abstained, result.abstain_source) == (False, None)

    def test_a_low_similarity_photo_abstains_by_retrieval(self, pipeline, image, make_candidate):
        pipeline.candidates = [make_candidate("Rose", 0.30), make_candidate("Tulip", 0.28)]
        result = ident.identify(image)
        assert result.confidence_tier == "low"
        assert (result.abstained, result.abstain_source) == (True, "retrieval")

    def test_the_agents_explicit_no_abstains_even_when_retrieval_is_confident(self, pipeline, image):
        pipeline.agent_answer = answer(matches_a_candidate=False)
        result = ident.identify(image)
        assert result.confidence_tier == "high"
        assert (result.abstained, result.abstain_source) == (True, "agent")

    def test_both_signals_are_reported_as_both(self, pipeline, image, make_candidate):
        pipeline.candidates = [make_candidate("Rose", 0.30)]
        pipeline.agent_answer = answer(matches_a_candidate=False)
        assert ident.identify(image).abstain_source == "both"

    def test_the_agents_yes_does_not_override_low_retrieval_confidence(self, pipeline, image, make_candidate):
        pipeline.candidates = [make_candidate("Rose", 0.30)]
        pipeline.agent_answer = answer(matches_a_candidate=True)
        assert ident.identify(image).abstain_source == "retrieval"

    @pytest.mark.parametrize(
        "note",
        [
            "This is not any of the candidates; it is not a flower at all.",
            "Definitely not a match. Cannot identify. NOT A FLOWER.",
            "abstain abstain abstain",
        ],
    )
    def test_words_in_the_models_text_never_cause_an_abstention(self, pipeline, image, note):
        """No keyword detection anywhere: only the explicit flag and the retrieval policy count."""
        pipeline.agent_answer = answer(confidence_note=note, summary=note, quality_note=note)
        result = ident.identify(image)
        assert (result.abstained, result.abstain_source) == (False, None)

    def test_a_lot_abstains_from_the_consensus_photos_retrieval_or_the_agent(self, pipeline, image, make_candidate):
        pipeline.per_photo = [[make_candidate("Rose", 0.30)], [make_candidate("Rose", 0.32)]]
        result = ident.identify_lot([image, image])
        assert (result.abstained, result.abstain_source) == (True, "retrieval")

        pipeline.agent_answer = answer(matches_a_candidate=False)
        confident = ident.identify_lot([image, image])
        assert (confident.abstained, confident.abstain_source) == (True, "agent")

    def test_the_fields_are_in_the_api_json_and_default_to_not_abstained(self, make_identify_result):
        body = json.loads(make_identify_result().model_dump_json())
        assert body["abstained"] is False and body["abstain_source"] is None
        flagged = json.loads(make_identify_result(abstained=True, abstain_source="agent").model_dump_json())
        assert flagged["abstained"] is True and flagged["abstain_source"] == "agent"

    def test_a_source_outside_the_enum_is_refused(self, make_identify_result):
        with pytest.raises(ValidationError):
            make_identify_result(abstain_source="the summary said so")
