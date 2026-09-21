"""FastAPI routes through TestClient with the pipeline (model + Gemini) faked at
api.main's module boundary. Inventory logging is real, into a temp SQLite file.

Every request here carries a valid key; the perimeter itself (401/503, size limits,
headers, docs) is tested in test_api_security.py."""

import io

import pytest
from fastapi.testclient import TestClient

import api.main as api
from src.identify import IdentifyError


@pytest.fixture
def client(api_headers):
    return TestClient(api.app, headers=api_headers)


@pytest.fixture
def upload(png_bytes):
    return lambda name="flower.png": (name, png_bytes, "image/png")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


class TestIdentify:
    def test_success_returns_the_result_and_logs_a_scan(self, client, upload, monkeypatch, make_identify_result):
        monkeypatch.setattr(api, "identify", lambda _image: make_identify_result(species="Tulip"))
        response = client.post("/identify", files={"photo": upload()})
        assert response.status_code == 200
        assert response.json()["species"] == "Tulip"
        assert client.get("/inventory").json()[0]["species"] == "Tulip"

    def test_the_upload_reaches_the_pipeline_decoded(self, client, upload, monkeypatch, make_identify_result):
        seen = {}

        def fake(image):
            seen["size"] = image.size
            return make_identify_result()

        monkeypatch.setattr(api, "identify", fake)
        client.post("/identify", files={"photo": upload()})
        assert seen["size"] == (64, 48)

    def test_an_unreadable_upload_is_a_400_and_nothing_is_logged(self, client, monkeypatch):
        monkeypatch.setattr(api, "identify", lambda _i: pytest.fail("must not run on a bad upload"))
        response = client.post("/identify", files={"photo": ("bad.png", b"not an image", "image/png")})
        assert response.status_code == 400
        assert client.get("/inventory").json() == []

    def test_a_pipeline_error_is_a_503_and_is_not_logged_as_a_scan(self, client, upload, monkeypatch):
        def failing(_image):
            raise IdentifyError("Gemini is unavailable")

        monkeypatch.setattr(api, "identify", failing)
        response = client.post("/identify", files={"photo": upload()})
        assert response.status_code == 503
        assert response.json()["detail"] == "Gemini is unavailable"
        assert client.get("/inventory").json() == []

    def test_a_missing_photo_field_is_a_422(self, client):
        assert client.post("/identify").status_code == 422


class TestIdentifyLot:
    def test_success_logs_one_lot_row(self, client, upload, monkeypatch, make_lot_result):
        monkeypatch.setattr(api, "identify_lot", lambda images: make_lot_result(photo_count=len(images)))
        files = [("photos", upload("a.png")), ("photos", upload("b.png"))]
        response = client.post("/identify-lot", files=files)
        assert response.status_code == 200
        (row,) = client.get("/inventory").json()
        assert row["mode"] == "lot" and row["photo_count"] == 2

    def test_too_many_photos_is_a_400_before_any_work(self, client, upload, monkeypatch):
        monkeypatch.setattr(api, "identify_lot", lambda _i: pytest.fail("must not run"))
        files = [("photos", upload(f"{i}.png")) for i in range(api.LOT_MAX_PHOTOS + 1)]
        assert client.post("/identify-lot", files=files).status_code == 400

    def test_no_photos_is_a_422(self, client):
        assert client.post("/identify-lot").status_code == 422

    def test_a_bad_file_anywhere_in_the_lot_is_a_400(self, client, upload, monkeypatch):
        monkeypatch.setattr(api, "identify_lot", lambda _i: pytest.fail("must not run"))
        files = [("photos", upload()), ("photos", ("bad.png", b"nope", "image/png"))]
        assert client.post("/identify-lot", files=files).status_code == 400

    def test_a_pipeline_error_is_a_503(self, client, upload, monkeypatch):
        def failing(_images):
            raise IdentifyError("rate limited")

        monkeypatch.setattr(api, "identify_lot", failing)
        assert client.post("/identify-lot", files=[("photos", upload())]).status_code == 503


class TestExplain:
    def test_an_unknown_species_is_a_400(self, client, upload):
        response = client.post("/explain", files={"photo": upload()}, data={"species": "Unicorn flower"})
        assert response.status_code == 400
        assert "curated species" in response.json()["detail"]

    def test_the_species_form_field_is_required(self, client, upload):
        assert client.post("/explain", files={"photo": upload()}).status_code == 422


class TestInventory:
    def test_it_starts_empty(self, client):
        assert client.get("/inventory").json() == []
        assert client.get("/inventory/species-counts").json() == {}

    def test_limit_and_species_counts(self, client, upload, monkeypatch, make_identify_result):
        names = iter(["Rose", "Tulip", "Rose"])
        monkeypatch.setattr(api, "identify", lambda _i: make_identify_result(species=next(names)))
        for _ in range(3):
            client.post("/identify", files={"photo": upload()})
        assert [r["species"] for r in client.get("/inventory?limit=2").json()] == ["Rose", "Tulip"]
        assert client.get("/inventory/species-counts").json() == {"Rose": 2, "Tulip": 1}


def test_a_png_and_a_jpeg_are_both_accepted(client, monkeypatch, make_identify_result):
    from PIL import Image

    monkeypatch.setattr(api, "identify", lambda _i: make_identify_result())
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), (5, 5, 5)).save(buf, format=fmt)
        response = client.post("/identify", files={"photo": (f"x.{fmt.lower()}", buf.getvalue(), mime)})
        assert response.status_code == 200
