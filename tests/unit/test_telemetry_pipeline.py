"""identify()/identify_lot() through the real telemetry: one row per scan, with the timings,
attempts, tokens and failure category the pipeline actually produced. Retrieval and the
agent are faked; everything in src/identify.py and src/telemetry.py is real."""

import json
import sqlite3
from contextlib import closing

import pytest
from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import GoogleAPIError, GoogleGenerativeAIError, GoogleRateLimitError
from PIL import Image

from src import db
from src import identify as ident
from src.identify import IdentifyError

REAL_GET_AGENT = ident._get_agent  # the fixture below replaces it; one test needs the real one

GOOD = {
    "species": "Rose",
    "confidence_note": "clearly a rose",
    "quality_grade": "A",
    "quality_note": "fresh",
    "summary": "A fine rose.",
}


def agent_result(answer=GOOD, tokens=((1854, 18), (1931, 48), (2025, 24), (2098, 123)), tools=("a", "b", "c")):
    """Four model turns and three tool calls, like the real probe run (docs/telemetry_probe.md)."""
    messages = []
    for index, (tokens_in, tokens_out) in enumerate(tokens):
        last = index == len(tokens) - 1
        messages.append(
            AIMessage(
                content=json.dumps(answer) if last else "",
                usage_metadata={
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                },
                response_metadata={"model_name": "gemini-3.1-flash-lite"},
                tool_calls=[] if last else [{"name": tools[index], "args": {}, "id": f"c{index}"}],
            )
        )
    return {"messages": messages}


class Scripted:
    def __init__(self, *steps):
        self.steps, self.calls = list(steps), 0

    def invoke(self, _payload):
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture
def pipeline(monkeypatch, prices_csv, make_candidate):
    class Handle:
        agent = None
        candidates = [make_candidate("Rose", 0.70), make_candidate("Tulip", 0.50), make_candidate("Carnation", 0.40)]

    handle = Handle()
    monkeypatch.setattr(ident, "embed_image", lambda _image: [0.0])
    monkeypatch.setattr(ident, "get_client", lambda: object())
    monkeypatch.setattr(ident, "search", lambda *_a, **_k: handle.candidates)
    monkeypatch.setattr(ident.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ident, "_get_agent", lambda: handle.agent)
    return handle


@pytest.fixture
def image():
    return Image.new("RGB", (32, 32), (10, 200, 30))


def metrics():
    with closing(db.connect()) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM scan_metrics ORDER BY id")]


class TestSuccessfulScans:
    def test_one_row_with_timings_attempts_tokens_and_tool_calls(self, pipeline, image):
        pipeline.agent = Scripted(agent_result())
        ident.identify(image)
        (row,) = metrics()
        assert (row["mode"], row["photo_count"], row["status"], row["failure_category"]) == ("single", 1, "ok", None)
        assert row["attempts"] == 1
        assert (row["input_tokens"], row["output_tokens"]) == (7908, 213)
        assert (row["model_turns"], row["tool_calls"]) == (4, 3)
        assert row["model_reported"] == "gemini-3.1-flash-lite"
        for stage in ("embed_ms", "search_ms", "agent_ms"):
            assert row[stage] is not None and row[stage] >= 0
        assert row["total_ms"] >= row["embed_ms"]

    def test_the_cost_is_the_estimated_list_price_equivalent_at_the_version_in_force(self, pipeline, image):
        pipeline.agent = Scripted(agent_result())
        ident.identify(image)
        (row,) = metrics()
        assert row["pricing_version"] == "gemini-3.1-flash-lite@2026-09-16"
        assert row["est_cost_usd"] == pytest.approx((7908 * 0.25 + 213 * 1.5) / 1e6)

    def test_a_lot_is_one_row_with_its_photo_count_and_one_agent_call(self, pipeline, image):
        pipeline.agent = Scripted(agent_result())
        ident.identify_lot([image, image, image])
        (row,) = metrics()
        assert (row["mode"], row["photo_count"], row["attempts"]) == ("lot", 3, 1)

    def test_the_first_scan_of_a_process_is_flagged_cold_until_the_model_is_loaded(self, pipeline, image, monkeypatch):
        pipeline.agent = Scripted(agent_result(), agent_result())
        monkeypatch.setattr(ident, "is_loaded", lambda: False)
        ident.identify(image)
        monkeypatch.setattr(ident, "is_loaded", lambda: True)
        ident.identify(image)
        assert [r["cold_start"] for r in metrics()] == [1, 0]

    def test_a_retry_after_a_server_error_is_counted_and_the_scan_still_succeeds(self, pipeline, image):
        server = GoogleAPIError(503, {"error": {"message": "overloaded"}})
        pipeline.agent = Scripted(server, server, agent_result())
        ident.identify(image)
        (row,) = metrics()
        assert (row["status"], row["attempts"]) == ("ok", 3)
        assert row["model_turns"] == 4  # only the run that produced messages had usage

    def test_the_tokens_of_an_unusable_answer_that_was_retried_are_still_counted(self, pipeline, image):
        bad = agent_result(answer={**GOOD, "quality_grade": "Z"})  # billed, then rejected as malformed
        pipeline.agent = Scripted(bad, agent_result())
        ident.identify(image)
        (row,) = metrics()
        assert row["attempts"] == 2 and row["model_turns"] == 8 and row["input_tokens"] == 2 * 7908


class TestFailedScans:
    @pytest.mark.parametrize(
        "steps, category, attempts",
        [
            ([GoogleAPIError(503, {"error": {"message": "x"}})] * 4, "server_5xx", 4),
            ([TimeoutError("read timed out")] * 4, "timeout", 4),
            ([RuntimeError("boom")] * 4, "other", 4),
            ([GoogleRateLimitError("429 RESOURCE_EXHAUSTED.")] * 3, "rate_limit", 3),
            ([GoogleGenerativeAIError("API key not valid")], "auth", 1),
            ([agent_result(answer={**GOOD, "quality_grade": "Z"})] * 4, "parse_error", 4),
        ],
        ids=["server", "timeout", "unexpected", "rate-limit", "rejected", "malformed"],
    )
    def test_each_kind_of_failure_gets_its_category_and_its_attempts(
        self, pipeline, image, steps, category, attempts
    ):
        pipeline.agent = Scripted(*steps)
        with pytest.raises(IdentifyError):
            ident.identify(image)
        (row,) = metrics()
        assert (row["status"], row["failure_category"], row["attempts"]) == ("error", category, attempts)

    def test_a_timeout_after_a_server_error_is_still_a_timeout(self, pipeline, image):
        pipeline.agent = Scripted(
            GoogleAPIError(503, {"error": {"message": "x"}}), *[TimeoutError("t")] * 3
        )
        with pytest.raises(IdentifyError):
            ident.identify(image)
        assert metrics()[0]["failure_category"] == "timeout"  # the LAST failure names the category

    def test_an_empty_index_is_its_own_category_and_never_calls_the_agent(self, pipeline, image):
        pipeline.candidates = []
        pipeline.agent = Scripted()
        with pytest.raises(IdentifyError):
            ident.identify(image)
        (row,) = metrics()
        assert (row["failure_category"], row["attempts"], row["agent_ms"]) == ("empty_index", None, None)
        assert row["embed_ms"] is not None and row["search_ms"] is not None

    def test_a_missing_api_key_is_an_auth_failure_before_any_attempt(self, pipeline, image, monkeypatch):
        monkeypatch.setattr(ident, "_get_agent", REAL_GET_AGENT)
        monkeypatch.setattr(ident, "_agent_singleton", None)
        with pytest.raises(IdentifyError):
            ident.identify(image)
        (row,) = metrics()
        assert (row["failure_category"], row["attempts"]) == ("auth", None)

    def test_a_lot_with_an_empty_index_is_a_failed_lot_row(self, pipeline, image):
        pipeline.candidates = []
        with pytest.raises(IdentifyError):
            ident.identify_lot([image, image])
        (row,) = metrics()
        assert (row["mode"], row["photo_count"], row["failure_category"]) == ("lot", 2, "empty_index")

    def test_rejected_input_is_not_a_scan_and_records_nothing(self, pipeline, image):
        with pytest.raises(IdentifyError):
            ident.identify_lot([])
        with pytest.raises(IdentifyError):
            ident.identify_lot([image] * (ident.LOT_MAX_PHOTOS + 1))
        assert metrics() == []

    def test_a_failed_scan_still_records_the_time_it_took(self, pipeline, image):
        pipeline.agent = Scripted(*[RuntimeError("x")] * 4)
        with pytest.raises(IdentifyError):
            ident.identify(image)
        assert metrics()[0]["total_ms"] >= 0 and metrics()[0]["agent_ms"] is not None


def test_telemetry_that_cannot_be_written_does_not_fail_the_scan(pipeline, image, monkeypatch):
    pipeline.agent = Scripted(agent_result())

    def broken(**_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("src.telemetry.db.connect", broken)
    assert ident.identify(image).species == "Rose"
