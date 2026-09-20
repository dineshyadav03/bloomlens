"""identify() / identify_lot() orchestration with the model, the vector store and
the agent faked, so only this module's own logic is under test."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from src import identify as ident
from src.identify import IdentifyError, _GeminiAnswer


def answer(species="Rose", grade="A") -> _GeminiAnswer:
    return _GeminiAnswer(
        species=species,
        confidence_note="looks right",
        quality_grade=grade,
        quality_note="fresh",
        summary="summary",
    )


@pytest.fixture
def image():
    return Image.new("RGB", (32, 32), (10, 200, 30))


@pytest.fixture
def pipeline(monkeypatch, prices_csv, make_candidate):
    """Fake everything outside identify.py. Returns a handle to script/inspect it."""

    class Handle:
        candidates = [make_candidate("Rose", 0.70), make_candidate("Tulip", 0.50), make_candidate("Carnation", 0.40)]
        per_photo_candidates = None  # identify_lot: list of candidate lists, consumed in order
        agent_answer = answer()
        agent_calls: list = []

    handle = Handle()
    handle.agent_calls = []

    def fake_search(_client, _embedding, top_k=3):
        if handle.per_photo_candidates is not None:
            return handle.per_photo_candidates.pop(0)
        return handle.candidates

    def fake_agent(message):
        handle.agent_calls.append(message)
        return handle.agent_answer

    monkeypatch.setattr(ident, "embed_image", lambda _img: np.zeros(768, dtype=np.float32))
    monkeypatch.setattr(ident, "get_client", lambda: object())
    monkeypatch.setattr(ident, "search", fake_search)
    monkeypatch.setattr(ident, "_invoke_agent_with_retries", fake_agent)
    return handle


class TestIdentify:
    def test_happy_path(self, pipeline, image):
        result = ident.identify(image)
        assert result.species == "Rose"
        assert result.scientific_name == "Rose scientificus"
        assert result.quality_grade == "A"
        assert result.confidence_tier == "high"
        assert result.price_simulated is True
        assert result.price_per_stem == pytest.approx(0.55)  # Rose grade A on 2026-09-20 in the fixture CSV
        assert result.price_trend == "up"
        assert len(pipeline.agent_calls) == 1

    def test_top_candidates_are_reported_with_rounded_scores(self, pipeline, image, make_candidate):
        pipeline.candidates = [make_candidate("Rose", 0.70123456), make_candidate("Tulip", 0.5)]
        result = ident.identify(image)
        assert [c["common_name"] for c in result.top_candidates] == ["Rose", "Tulip"]
        assert result.top_candidates[0]["score"] == 0.701

    def test_agent_answer_with_extra_text_is_mapped_to_a_candidate(self, pipeline, image):
        pipeline.agent_answer = answer(species="Rose (Rosa chinensis)")
        assert ident.identify(image).species == "Rose"

    def test_the_agent_can_override_retrieval_but_the_tier_stays_retrieval_based(self, pipeline, image):
        """Documents a design fact the README must not blur: the final species is the
        agent's pick among the retrieved candidates, while confidence_tier is derived
        from retrieval scores alone -- so retrieval accuracy is not full-system accuracy."""
        pipeline.agent_answer = answer(species="Tulip")
        result = ident.identify(image)
        assert result.species == "Tulip"
        assert result.top_candidates[0]["common_name"] == "Rose"
        assert result.confidence_tier == "high"  # from Rose's 0.70 vs 0.50, not from the agent

    def test_unknown_grade_gives_no_price_rather_than_a_wrong_one(self, pipeline, image):
        pipeline.agent_answer = answer(grade="Z")
        result = ident.identify(image)
        assert result.price_per_stem is None
        assert result.price_trend == "unknown"

    def test_an_empty_index_is_a_clear_error(self, pipeline, image):
        pipeline.candidates = []
        with pytest.raises(IdentifyError, match="index is empty"):
            ident.identify(image)
        assert pipeline.agent_calls == []  # never spend an LLM call on nothing


class TestResolveCandidate:
    def test_looks_up_the_other_candidates_price_without_the_agent(self, prices_csv):
        top = [
            {"common_name": "Rose", "scientific_name": "Rosa chinensis", "score": 0.6},
            {"common_name": "Tulip", "scientific_name": "Tulipa gesneriana", "score": 0.55},
        ]
        resolved = ident.resolve_candidate("Tulip", top, "B")
        assert resolved["scientific_name"] == "Tulipa gesneriana"
        assert resolved["price_per_stem"] == pytest.approx(0.29)  # Tulip grade B on 2026-09-20 in the fixture CSV


class TestIdentifyLot:
    def lot_setup(self, pipeline, make_candidate, top1_names):
        pipeline.per_photo_candidates = [
            [make_candidate(n, 0.7), make_candidate("Other", 0.3), make_candidate("Another", 0.2)] for n in top1_names
        ]

    def test_consensus_and_flagged_photos(self, pipeline, make_candidate, image):
        self.lot_setup(pipeline, make_candidate, ["Rose", "Rose", "Sunflower"])
        pipeline.agent_answer = answer(species="Rose")
        result = ident.identify_lot([image, image, image])
        assert result.consensus_species == "Rose"
        assert result.photo_count == 3
        assert result.agreement_fraction == pytest.approx(0.667, abs=1e-3)
        assert result.flagged_photos == [{"index": 2, "top_species": "Sunflower", "score": 0.7}]

    def test_agent_runs_exactly_once_for_the_whole_lot(self, pipeline, make_candidate, image):
        self.lot_setup(pipeline, make_candidate, ["Rose"] * 4)
        ident.identify_lot([image] * 4)
        assert len(pipeline.agent_calls) == 1

    def test_every_photo_is_sent_once_in_one_message(self, pipeline, make_candidate, image):
        self.lot_setup(pipeline, make_candidate, ["Rose"] * 3)
        ident.identify_lot([image] * 3)
        content = pipeline.agent_calls[0]["content"]
        assert [b["type"] for b in content].count("image") == 3

    def test_empty_lot_is_rejected(self, pipeline):
        with pytest.raises(IdentifyError, match="No photos"):
            ident.identify_lot([])

    def test_oversized_lot_is_rejected_before_any_work(self, pipeline, image):
        with pytest.raises(IdentifyError, match="at most"):
            ident.identify_lot([image] * (ident.LOT_MAX_PHOTOS + 1))
        assert pipeline.agent_calls == []

    def test_an_empty_index_is_a_clear_error(self, pipeline, image):
        pipeline.per_photo_candidates = [[]]
        with pytest.raises(IdentifyError, match="index is empty"):
            ident.identify_lot([image])


class TestMessageBuilding:
    def test_single_message_has_the_candidates_and_exactly_one_jpeg(self, image, make_candidate):
        message = ident._build_candidates_message(image, [make_candidate("Rose", 0.7)])
        text_block, image_block = message["content"]
        assert "Rose" in text_block["text"]
        assert image_block["mime_type"] == "image/jpeg"
        decoded = Image.open(io.BytesIO(base64.b64decode(image_block["data"])))
        assert decoded.format == "JPEG" and decoded.size == image.size

    def test_rgba_and_palette_images_are_converted_before_jpeg_encoding(self):
        rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        assert ident._image_to_jpeg_bytes(rgba)[:2] == b"\xff\xd8"  # JPEG magic
