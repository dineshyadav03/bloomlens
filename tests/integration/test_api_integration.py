"""The API end to end with REAL retrieval (BioCLIP 2 + Qdrant) and REAL inventory
logging; only the Gemini boundary is faked (api -> identify -> agent)."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api
from src import identify as ident
from src.identify import _GeminiAnswer
from tests.integration.conftest import known_good_photo


@pytest.fixture
def agent(monkeypatch):
    """Fake agent that answers with whatever `agent.species` is; records its calls."""

    class Agent:
        species = "Rose"
        calls = []

    handle = Agent()
    handle.calls = []

    def fake(message):
        handle.calls.append(message)
        return _GeminiAnswer(
            species=handle.species,
            confidence_note="looks right",
            quality_grade="A",
            quality_note="fresh",
            summary="summary",
        )

    monkeypatch.setattr(ident, "_invoke_agent_with_retries", fake)
    return handle


@pytest.fixture
def client(use_index, prices_csv, api_headers):
    return TestClient(api.app, headers=api_headers)


def photo_upload(species):
    path = known_good_photo(species)
    return path.name, path.read_bytes(), "image/jpeg"


def test_identify_with_real_retrieval_and_a_faked_agent(client, agent):
    response = client.post("/identify", files={"photo": photo_upload("Rose")})
    assert response.status_code == 200
    body = response.json()
    assert body["species"] == "Rose"
    assert body["top_candidates"][0]["common_name"] == "Rose"  # from real BioCLIP + Qdrant
    assert body["confidence_tier"] in {"high", "ambiguous", "low"}
    assert body["price_simulated"] is True
    assert len(agent.calls) == 1


def test_a_mixed_lot_reports_consensus_and_flags_the_odd_photo(client, agent):
    agent.species = "Rose"
    files = [("photos", photo_upload("Rose")), ("photos", photo_upload("Rose")), ("photos", photo_upload("Sunflower"))]
    response = client.post("/identify-lot", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["consensus_species"] == "Rose"
    assert body["photo_count"] == 3
    assert body["agreement_fraction"] == pytest.approx(0.667, abs=1e-3)
    assert [f["top_species"] for f in body["flagged_photos"]] == ["Sunflower"]
    assert body["flagged_photos"][0]["index"] == 2
    assert len(agent.calls) == 1  # one agent call for the whole lot


def test_scans_are_logged_and_visible_through_the_inventory_endpoints(client, agent):
    client.post("/identify", files={"photo": photo_upload("Rose")})
    client.post("/identify-lot", files=[("photos", photo_upload("Rose")), ("photos", photo_upload("Rose"))])
    rows = client.get("/inventory").json()
    assert [(r["mode"], r["species"]) for r in rows] == [("lot", "Rose"), ("single", "Rose")]
    assert client.get("/inventory/species-counts").json() == {"Rose": 2}


def test_explain_returns_a_png_the_size_of_the_input(client):
    name, data, mime = photo_upload("Rose")
    response = client.post("/explain", files={"photo": (name, data, mime)}, data={"species": "Rose"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    overlay = Image.open(io.BytesIO(response.content))
    assert overlay.format == "PNG"
    assert overlay.size == Image.open(io.BytesIO(data)).size


def test_a_failed_agent_call_is_a_503_and_nothing_is_logged(client, monkeypatch):
    def failing(_message):
        raise ident.IdentifyError("Gemini is unavailable")

    monkeypatch.setattr(ident, "_invoke_agent_with_retries", failing)
    response = client.post("/identify", files={"photo": photo_upload("Rose")})
    assert response.status_code == 503
    assert client.get("/inventory").json() == []
