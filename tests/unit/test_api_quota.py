"""Rate limits, quotas and the concurrency gate on the API routes that run the pipeline.

The rule under test: admission happens AFTER the upload validates and BEFORE the pipeline
runs; a bad request costs nothing, an admitted one costs a unit whether or not the
provider then fails (no refund), and any failure of the counter store refuses the request."""

import io
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api
from src import db, quota
from src.identify import IdentifyError
from tests.conftest import TEST_API_KEY


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    """Windows are wall-clock minutes; a test that happened to straddle one would flip a
    refusal into an admission. Pin 'now' (a test may override it)."""
    monkeypatch.setattr(quota, "_utc", lambda now: datetime(2026, 9, 21, 12, 0, 30, tzinfo=UTC) if now is None else now)


@pytest.fixture
def client(api_headers):
    return TestClient(api.app, headers=api_headers)


@pytest.fixture
def working_pipeline(monkeypatch, make_identify_result, make_lot_result):
    calls = []

    def fake_identify(_image):
        calls.append("identify")
        return make_identify_result()

    def fake_lot(images):
        calls.append("lot")
        return make_lot_result(photo_count=len(images))

    def fake_explain(image, _species):
        calls.append("explain")
        return image

    monkeypatch.setattr(api, "identify", fake_identify)
    monkeypatch.setattr(api, "identify_lot", fake_lot)
    monkeypatch.setattr(api, "explain_image", fake_explain)
    return calls


def png(size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 120, 60)).save(buf, format="PNG")
    return buf.getvalue()


def scan(client, **kwargs):
    return client.post("/identify", files={"photo": ("a.png", png(), "image/png")}, **kwargs)


def counter_rows():
    with closing(db.connect()) as conn:
        return conn.execute("SELECT identity, kind, used FROM quota_counters ORDER BY identity, kind").fetchall()


class TestRateLimit:
    def test_over_the_per_minute_limit_is_a_429_with_retry_after(self, client, working_pipeline, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "2")
        assert [scan(client).status_code for _ in range(2)] == [200, 200]
        refused = scan(client)
        assert refused.status_code == 429
        assert 1 <= int(refused.headers["retry-after"]) <= 60
        assert "Try again in" in refused.json()["detail"]
        assert working_pipeline == ["identify", "identify"]  # the refused request never reached the pipeline

    def test_the_daily_quota_says_when_it_resets_and_retry_after_is_the_time_to_utc_midnight(
        self, client, working_pipeline, monkeypatch
    ):
        monkeypatch.setenv("BLOOMLENS_QUOTA_PER_DAY", "1")
        monkeypatch.setattr(quota, "_utc", lambda now: datetime(2026, 9, 21, 23, 0, 0, tzinfo=UTC))
        assert scan(client).status_code == 200
        refused = scan(client)
        assert refused.status_code == 429
        assert refused.headers["retry-after"] == "3600"
        assert "00:00 UTC" in refused.json()["detail"]

    def test_the_global_ceiling_protects_the_budget_across_callers(self, api_headers, working_pipeline, monkeypatch):
        second = "second-secret-0123456789-abcdefghi"
        monkeypatch.setenv("BLOOMLENS_API_KEYS", f"tester:{TEST_API_KEY},other:{second}")
        monkeypatch.setenv("BLOOMLENS_GLOBAL_QUOTA_PER_DAY", "2")
        one = TestClient(api.app, headers={"X-API-Key": TEST_API_KEY})
        two = TestClient(api.app, headers={"X-API-Key": second})
        assert scan(one).status_code == 200 and scan(two).status_code == 200
        assert scan(one).status_code == 429 and scan(two).status_code == 429

    def test_each_key_label_has_its_own_allowance(self, api_headers, working_pipeline, monkeypatch):
        second = "second-secret-0123456789-abcdefghi"
        monkeypatch.setenv("BLOOMLENS_API_KEYS", f"tester:{TEST_API_KEY},other:{second}")
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")
        one = TestClient(api.app, headers={"X-API-Key": TEST_API_KEY})
        two = TestClient(api.app, headers={"X-API-Key": second})
        assert scan(one).status_code == 200
        assert scan(one).status_code == 429
        assert scan(two).status_code == 200  # a different label is unaffected

    def test_a_lot_and_an_explain_each_cost_one_unit(self, client, working_pipeline, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "3")
        lot = [("photos", (f"{i}.png", png(), "image/png")) for i in range(4)]
        assert client.post("/identify-lot", files=lot).status_code == 200  # 4 photos, still one unit
        explain = client.post("/explain", files={"photo": ("a.png", png(), "image/png")}, data={"species": "Rose"})
        assert explain.status_code == 200
        assert scan(client).status_code == 200
        assert scan(client).status_code == 429  # 1 (lot) + 1 (explain) + 1 (scan) = 3 used


class TestWhatDoesNotCost:
    @pytest.fixture(autouse=True)
    def one_per_minute(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")

    def test_invalid_uploads_are_free(self, client, working_pipeline):
        for bad in (b"not an image", b"", b"<svg/>"):
            assert client.post("/identify", files={"photo": ("x.png", bad, "image/png")}).status_code in (400,)
        assert counter_rows() == []
        assert scan(client).status_code == 200  # the whole allowance is still there

    def test_a_decompression_bomb_and_an_oversize_body_are_free(self, client, working_pipeline, monkeypatch):
        from src import guard
        from tests.unit.test_guard import png_bomb

        monkeypatch.setattr(guard, "MAX_UPLOAD_BYTES", 50_000)
        bomb = client.post("/identify", files={"photo": ("b.png", png_bomb(20_000, 20_000), "image/png")})
        assert bomb.status_code == 413
        assert client.post("/identify", files={"photo": ("b.png", b"x" * 300_000, "image/png")}).status_code == 413
        assert counter_rows() == []

    def test_an_unknown_species_to_explain_is_free(self, client, working_pipeline):
        response = client.post(
            "/explain", files={"photo": ("a.png", png(), "image/png")}, data={"species": "Unicorn flower"}
        )
        assert response.status_code == 400
        assert counter_rows() == []

    def test_unauthenticated_requests_are_free_and_leave_no_trace(self, api_headers, working_pipeline):
        anonymous = TestClient(api.app)
        for _ in range(5):
            assert scan(anonymous).status_code == 401
        assert counter_rows() == []

    def test_reads_and_health_are_not_metered(self, client):
        for _ in range(10):
            assert client.get("/inventory").status_code == 200
            assert client.get("/inventory/species-counts").status_code == 200
            assert client.get("/health").status_code == 200
        assert counter_rows() == []


class TestNoRefund:
    def test_a_request_that_is_admitted_and_then_fails_at_the_provider_still_costs_a_unit(
        self, client, monkeypatch
    ):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")

        def failing(_image):
            raise IdentifyError("The identification service didn't return a usable answer.")

        monkeypatch.setattr(api, "identify", failing)
        assert scan(client).status_code == 503
        assert scan(client).status_code == 429  # the unit was spent


class TestIdentityIsALabelNotASecretOrAnAddress:
    def test_only_the_label_is_stored(self, client, working_pipeline):
        assert scan(client).status_code == 200
        identities = {row[0] for row in counter_rows()}
        assert identities == {"api:tester", "*"}
        raw = b"".join(p.read_bytes() for p in Path(db.db_path()).parent.glob("inventory.db*"))
        assert TEST_API_KEY.encode() not in raw
        assert b"testclient" not in raw and b"127.0.0.1" not in raw


class TestFailClosed:
    def test_an_unreachable_counter_store_refuses_the_request(self, client, working_pipeline, tmp_path, monkeypatch):
        blocker = tmp_path / "secret-file-name"
        blocker.write_text("x")
        monkeypatch.setenv("INVENTORY_DB_PATH", str(blocker / "shared.db"))
        response = scan(client)
        assert response.status_code == 503
        assert "Rate limiting is unavailable" in response.json()["detail"]
        assert "secret-file-name" not in response.text and "sqlite" not in response.text.lower()
        assert working_pipeline == []  # it did not run unmetered

    def test_a_locked_store_is_a_fast_503_not_a_hang_and_not_an_admission(self, client, working_pipeline, monkeypatch):
        monkeypatch.setattr(quota, "ADMISSION_BUSY_TIMEOUT_SECONDS", 0.2)
        with closing(db.connect()) as holder, db.transaction(holder):
            started = time.monotonic()
            response = scan(client)
            assert time.monotonic() - started < 5
        assert response.status_code == 503
        assert working_pipeline == []
        assert counter_rows() == []


class TestConcurrencyGate:
    @pytest.fixture(autouse=True)
    def one_slot(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_MAX_CONCURRENT", "1")
        quota.reset_gate()

    def test_when_every_slot_is_busy_the_request_is_a_503_that_costs_nothing(self, client, working_pipeline):
        with quota.gate().slot():
            busy = scan(client)
        assert busy.status_code == 503
        assert busy.headers["retry-after"] == str(api.BUSY_RETRY_AFTER_SECONDS)
        assert "busy" in busy.json()["detail"]
        assert working_pipeline == [] and counter_rows() == []
        assert scan(client).status_code == 200  # and once the slot is free it goes through

    def test_a_crash_in_the_pipeline_still_frees_the_slot(self, api_headers, monkeypatch, make_identify_result):
        outcomes = [RuntimeError("boom")]

        def flaky(_image):
            if outcomes:
                raise outcomes.pop()
            return make_identify_result()

        monkeypatch.setattr(api, "identify", flaky)
        quiet = TestClient(api.app, headers=api_headers, raise_server_exceptions=False)
        assert scan(quiet).status_code == 500
        assert scan(quiet).status_code == 200  # the slot was released by the failed request
