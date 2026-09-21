"""Nothing private reaches the telemetry file.

The rule (docs/PRIVACY.md): telemetry never holds images, prompts, model answers or tool
arguments, exception messages, IP addresses, API keys or caller identities. These tests
write distinctive canary strings through EVERY path that could carry them -- model output,
tool-call arguments, provider metadata, prompt text, every kind of exception, request
headers and the API key -- then read the raw database file (and its write-ahead log) and
assert none of them is there. A test that could not fail would prove nothing, so the last
class puts a canary where telemetry *does* store text and shows the check notices it."""

import io
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import GoogleAPIError, GoogleGenerativeAIError, GoogleRateLimitError
from PIL import Image

import api.main as api
from src import db, telemetry
from src import identify as ident
from src.identify import IdentifyError
from src.telemetry import FAILURE_CATEGORIES
from tests.conftest import TEST_API_KEY
from tests.unit.test_telemetry_pipeline import GOOD, Scripted, agent_result

C_ANSWER = "CANARY-ANSWER-4471"
C_TOOL_ARG = "CANARY-TOOLARG-5582"
C_PROVIDER_META = "CANARY-PROVIDERMETA-6693"
C_PROMPT = "CANARY-PROMPT-7704"
C_EXCEPTION = "CANARY-EXCEPTION-8815"
C_ADDRESS = "203.0.113.77"
C_AGENT_HEADER = "CANARY-USERAGENT-9926"
ALL_CANARIES = [C_ANSWER, C_TOOL_ARG, C_PROVIDER_META, C_PROMPT, C_EXCEPTION, C_ADDRESS, C_AGENT_HEADER]

# What every stored text value must look like. Anything else is a leak or a bug.
ALLOWED_TEXT = {
    "recorded_at": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00",
    "mode": r"single|lot",
    "status": r"ok|error",
    "failure_category": "|".join(FAILURE_CATEGORIES),
    "bioclip_revision": r"[0-9a-f]{40}",
    "gemini_model": r"gemini-[A-Za-z0-9.-]+",
    "model_reported": r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
    "pricing_version": r"[A-Za-z0-9.-]+@\d{4}-\d{2}-\d{2}",
    "platform": r"\w+ \w+, \d+ CPUs",
}


def database_bytes() -> bytes:
    path = Path(db.db_path())
    return b"".join(p.read_bytes() for p in sorted(path.parent.glob(path.name + "*")))


def stored_metrics():
    with closing(db.connect()) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM scan_metrics")]


def assert_clean():
    raw = database_bytes()
    for canary in ALL_CANARIES:
        assert canary.encode() not in raw, f"{canary} reached the telemetry file"
    for row in stored_metrics():
        for column, value in row.items():
            if column == "id" or value is None:
                continue
            if column in ALLOWED_TEXT:
                assert re.fullmatch(ALLOWED_TEXT[column], value), f"{column}={value!r} is not an allowed value"
            else:
                assert isinstance(value, int | float), f"{column}={value!r} is not a number"


@pytest.fixture
def pipeline(monkeypatch, prices_csv, make_candidate):
    class Handle:
        agent = None
        candidates = [make_candidate("Rose", 0.70), make_candidate("Tulip", 0.50), make_candidate("Carnation", 0.40)]

    handle = Handle()
    for candidate in handle.candidates:
        candidate["payload"]["description"] = f"{C_PROMPT} {candidate['payload']['description']}"  # goes in the prompt
    monkeypatch.setattr(ident, "embed_image", lambda _image: [0.0])
    monkeypatch.setattr(ident, "get_client", lambda: object())
    monkeypatch.setattr(ident, "search", lambda *_a, **_k: handle.candidates)
    monkeypatch.setattr(ident.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ident, "_get_agent", lambda: handle.agent)
    return handle


@pytest.fixture
def image():
    return Image.new("RGB", (32, 32), (11, 222, 33))


def hostile_result(answer=None):
    """An agent result stuffed with canaries everywhere a provider or model can put text."""
    result = agent_result(answer=answer or {**GOOD, "summary": C_ANSWER, "quality_note": C_ANSWER})
    for message in result["messages"]:
        message.response_metadata = {
            "model_name": "gemini-3.1-flash-lite",
            "finish_reason": C_PROVIDER_META,
            "safety_ratings": [{"category": C_PROVIDER_META}],
        }
        message.additional_kwargs = {"function_call": {"name": "x", "arguments": C_PROVIDER_META}}
    result["messages"][0].tool_calls = [{"name": "check_price", "args": {"species": C_TOOL_ARG}, "id": "c0"}]
    result["messages"].insert(1, AIMessage(content=C_ANSWER))  # an extra turn with content and no usage
    return result


class TestSuccessPaths:
    def test_a_single_scan_whose_every_message_is_full_of_canaries(self, pipeline, image):
        pipeline.agent = Scripted(hostile_result())
        ident.identify(image)
        assert len(stored_metrics()) == 1
        assert_clean()

    def test_a_lot_scan(self, pipeline, image):
        pipeline.agent = Scripted(hostile_result())
        ident.identify_lot([image, image])
        assert_clean()

    def test_the_photo_itself_is_nowhere_in_the_file(self, pipeline, image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        pipeline.agent = Scripted(hostile_result())
        ident.identify(image)
        raw = database_bytes()
        assert buf.getvalue()[:64] not in raw and buf.getvalue()[-64:] not in raw
        assert b"JFIF" not in raw and b"\x89PNG" not in raw


class TestFailurePaths:
    """Every exception class the pipeline can meet, each carrying a canary in its message."""

    @pytest.mark.parametrize(
        "steps",
        [
            [GoogleAPIError(503, {"error": {"message": C_EXCEPTION}})] * 4,
            [GoogleGenerativeAIError(f"API key not valid: {C_EXCEPTION}")],
            [GoogleRateLimitError(f"429 RESOURCE_EXHAUSTED retryDelay: '1s' {C_EXCEPTION}")] * 3,
            [TimeoutError(C_EXCEPTION)] * 4,
            [RuntimeError(f"unexpected: {C_EXCEPTION}")] * 4,
            [ValueError(C_EXCEPTION)] * 4,
        ],
        ids=["server", "rejected", "rate-limit", "timeout", "runtime", "value"],
    )
    def test_provider_and_unexpected_error_text_is_never_stored(self, pipeline, image, steps):
        pipeline.agent = Scripted(*steps)
        with pytest.raises(IdentifyError):
            ident.identify(image)
        assert stored_metrics()[0]["status"] == "error"
        assert_clean()

    def test_a_malformed_answer_full_of_canaries_is_not_stored(self, pipeline, image):
        bad = hostile_result(answer={**GOOD, "quality_grade": C_ANSWER, "summary": C_ANSWER * 100})
        pipeline.agent = Scripted(*[bad] * 4)
        with pytest.raises(IdentifyError):
            ident.identify(image)
        assert stored_metrics()[0]["failure_category"] == "parse_error"
        assert_clean()

    def test_a_crash_outside_the_agent_is_not_stored_either(self, pipeline, image, monkeypatch):
        def explode(*_a, **_k):
            raise RuntimeError(C_EXCEPTION)

        monkeypatch.setattr(ident, "search", explode)
        with pytest.raises(RuntimeError):
            ident.identify(image)
        assert stored_metrics()[0]["failure_category"] == "other"
        assert_clean()

    def test_an_exception_that_carries_a_hostile_category_attribute_is_neutralised(self, pipeline, image):
        class Hostile(IdentifyError):
            pass

        hostile = Hostile("x", category=C_EXCEPTION)  # type: ignore[arg-type]
        pipeline.agent = Scripted(agent_result())
        with pytest.raises(Hostile), telemetry.scan("single", 1, cold_start=False):
            raise hostile
        assert stored_metrics()[0]["failure_category"] == "other"
        assert_clean()


class TestWhoAskedIsNeverRecorded:
    """Through the API, with a labeled key, a client address and a User-Agent."""

    def test_no_label_secret_address_or_header_reaches_the_telemetry_table(
        self, monkeypatch, pipeline, image, make_identify_result
    ):
        label = "CANARY-LABEL-3360"
        monkeypatch.setenv("BLOOMLENS_API_KEYS", f"{label}:{TEST_API_KEY}")
        pipeline.agent = Scripted(hostile_result())
        client = TestClient(api.app, headers={"X-API-Key": TEST_API_KEY}, client=(C_ADDRESS, 51234))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        response = client.post(
            "/identify",
            files={"photo": ("photo-CANARY-FILENAME.png", buf.getvalue(), "image/png")},
            headers={"X-Forwarded-For": C_ADDRESS, "User-Agent": C_AGENT_HEADER},
        )
        assert response.status_code == 200, response.text
        (row,) = stored_metrics()
        text = " ".join(str(v) for v in row.values())
        for private in (label, TEST_API_KEY, C_ADDRESS, C_AGENT_HEADER, "CANARY-FILENAME"):
            assert private not in text
        assert_clean()

    def test_the_metrics_endpoint_returns_no_row_and_no_identity(self, monkeypatch, pipeline, image):
        label = "CANARY-LABEL-3361"
        monkeypatch.setenv("BLOOMLENS_API_KEYS", f"{label}:{TEST_API_KEY}")
        pipeline.agent = Scripted(hostile_result())
        client = TestClient(api.app, headers={"X-API-Key": TEST_API_KEY}, client=(C_ADDRESS, 1))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        client.post("/identify", files={"photo": ("a.png", buf.getvalue(), "image/png")})
        body = client.get("/metrics").text
        for private in (label, TEST_API_KEY, C_ADDRESS, C_ANSWER, C_PROMPT, C_TOOL_ARG):
            assert private not in body


class TestTheCheckCanFail:
    """The canary check above would be worthless if it could not notice a leak."""

    def test_a_canary_placed_where_text_is_stored_is_caught(self, pipeline, image):
        pipeline.agent = Scripted(agent_result())
        ident.identify(image)
        with closing(db.connect()) as conn:
            conn.execute("UPDATE scan_metrics SET model_reported = ?", (C_EXCEPTION,))
        with pytest.raises(AssertionError, match="reached the telemetry file"):
            assert_clean()

    def test_a_value_outside_the_allowed_shape_is_caught_even_without_a_canary(self, pipeline, image):
        pipeline.agent = Scripted(agent_result())
        ident.identify(image)
        with closing(db.connect()) as conn:
            conn.execute("UPDATE scan_metrics SET platform = 'Bob Smith, laptop of bob@example.com'")
        with pytest.raises(AssertionError, match="not an allowed value"):
            assert_clean()

    def test_the_wal_is_included_in_what_is_scanned(self, pipeline, image):
        pipeline.agent = Scripted(agent_result())
        ident.identify(image)
        with closing(db.connect()) as conn:
            conn.execute("UPDATE scan_metrics SET model_reported = ?", (C_EXCEPTION,))
            names = [p.name for p in Path(db.db_path()).parent.glob(Path(db.db_path()).name + "*")]
        assert any(n.endswith("-wal") for n in names) or C_EXCEPTION.encode() in database_bytes()
