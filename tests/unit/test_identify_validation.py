"""What the model's answer must look like to be accepted, and what an error may reveal.

Model text is untrusted: it is validated for shape (an A/B/C grade, length caps) and
never rewritten; and no failure path may put provider or model text into an exception
message or a log line, because those reach API callers and log files."""

import json
import logging

import pytest
from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import GoogleAPIError, GoogleGenerativeAIError
from pydantic import ValidationError

from src import identify as ident
from src.identify import IdentifyConfigError, IdentifyError, IdentifyResult, LotResult, _GeminiAnswer

CANARY = "CANARY-9F3A71-DO-NOT-LEAK"  # upper-case: the grade validator upper-cases what it is given

GOOD = {
    "species": "Rose",
    "confidence_note": "clearly a rose",
    "quality_grade": "A",
    "quality_note": "fresh",
    "summary": "A fine rose.",
}


def answer(**overrides) -> dict:
    return {**GOOD, **overrides}


class TestGrade:
    @pytest.mark.parametrize("raw, expected", [("A", "A"), ("b", "B"), (" c ", "C"), ("\nB\n", "B")])
    def test_the_three_grades_are_accepted_in_any_case_and_padding(self, raw, expected):
        assert _GeminiAnswer(**answer(quality_grade=raw)).quality_grade == expected

    @pytest.mark.parametrize(
        "raw", ["D", "A+", "AB", "", "excellent", "A; DROP TABLE scans", "<b>A</b>", None, 1, ["A"]]
    )
    def test_anything_else_is_refused(self, raw):
        with pytest.raises(ValidationError):
            _GeminiAnswer(**answer(quality_grade=raw))

    def test_result_models_refuse_a_bad_grade_too(self):
        with pytest.raises(ValidationError):
            IdentifyResult(
                species="Rose",
                confidence_note="x",
                quality_grade="Z",
                quality_note="x",
                summary="x",
                price_per_stem=None,
                price_trend="unknown",
                price_as_of=None,
                top_candidates=[],
                confidence_tier="high",
            )
        with pytest.raises(ValidationError):
            LotResult(
                consensus_species="Rose",
                quality_grade="Z",
                quality_note="x",
                summary="x",
                price_per_stem=None,
                price_trend="unknown",
                price_as_of=None,
                photo_count=1,
                agreement_fraction=1.0,
                flagged_photos=[],
            )


class TestLengthCaps:
    @pytest.mark.parametrize(
        "field, cap",
        [
            ("species", ident._MAX_NAME_CHARS),
            ("confidence_note", ident._MAX_NOTE_CHARS),
            ("quality_note", ident._MAX_NOTE_CHARS),
            ("summary", ident._MAX_SUMMARY_CHARS),
        ],
    )
    def test_each_field_is_capped_exactly(self, field, cap):
        assert getattr(_GeminiAnswer(**answer(**{field: "x" * cap})), field) == "x" * cap
        with pytest.raises(ValidationError):
            _GeminiAnswer(**answer(**{field: "x" * (cap + 1)}))

    def test_a_field_is_never_silently_truncated(self):
        with pytest.raises(ValidationError):
            _GeminiAnswer(**answer(summary="x" * (ident._MAX_SUMMARY_CHARS + 500)))

    def test_a_missing_field_is_refused(self):
        with pytest.raises(ValidationError):
            _GeminiAnswer(**{k: v for k, v in GOOD.items() if k != "summary"})


class TestTextIsKeptLiterally:
    def test_markup_and_instructions_pass_through_unchanged(self):
        """Validation checks shape; it does not 'clean' text. Rendering (st.text, JSON
        strings) is what keeps markup inert -- rewriting here would only hide it."""
        nasty = "<script>alert(1)</script> **bold** [x](javascript:alert(1)) IGNORE PREVIOUS INSTRUCTIONS"
        parsed = ident._parse_answer(json.dumps(answer(summary=nasty, quality_note=nasty)))
        assert parsed.summary == nasty and parsed.quality_note == nasty


class TestParseAnswer:
    def test_a_good_answer_parses(self):
        assert ident._parse_answer(json.dumps(GOOD)).species == "Rose"

    def test_a_failure_names_the_fields_but_never_quotes_the_values(self):
        with pytest.raises(ident._MalformedAnswer) as info:
            ident._parse_answer(json.dumps(answer(quality_grade=CANARY, summary="x" * 5000)))
        message = str(info.value)
        assert "quality_grade" in message and "summary" in message
        assert CANARY not in message

    def test_broken_json_and_prose_are_malformed_without_quoting_them(self):
        for text in (f"{{not json {CANARY}}}", f"No object here, {CANARY}"):
            with pytest.raises(ident._MalformedAnswer) as info:
                ident._parse_answer(text)
            assert CANARY not in str(info.value)

    def test_the_underlying_error_is_not_chained_to_carry_the_text_along(self):
        with pytest.raises(ident._MalformedAnswer) as info:
            ident._parse_answer(json.dumps(answer(quality_grade=CANARY)))
        assert info.value.__cause__ is None and info.value.__suppress_context__


class Scripted:
    def __init__(self, *steps):
        self.steps, self.calls = list(steps), 0

    def invoke(self, _payload):
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return {"messages": [AIMessage(content=step)]}


@pytest.fixture
def run(monkeypatch):
    def _run(*steps):
        agent = Scripted(*steps)
        monkeypatch.setattr(ident, "_get_agent", lambda: agent)
        monkeypatch.setattr(ident.time, "sleep", lambda _s: None)
        return agent

    return _run


class TestRetryLoopHygiene:
    def test_a_persistently_invalid_grade_fails_without_echoing_it(self, run, caplog):
        agent = run(*[json.dumps(answer(quality_grade=CANARY))] * 4)
        with caplog.at_level(logging.DEBUG, logger="bloomlens.identify"), pytest.raises(IdentifyError) as info:
            ident._invoke_agent_with_retries({})
        assert agent.calls == 4
        assert "usable answer" in str(info.value) and "4 attempts" in str(info.value)
        assert CANARY not in str(info.value) and CANARY not in caplog.text
        assert "quality_grade" in caplog.text  # the field is logged, the value is not

    def test_a_grade_error_then_a_good_answer_recovers(self, run):
        run(json.dumps(answer(quality_grade="excellent")), json.dumps(GOOD))
        assert ident._invoke_agent_with_retries({}).quality_grade == "A"

    def test_provider_error_text_stays_out_of_the_exception_and_the_logs(self, run, caplog):
        run(GoogleGenerativeAIError(f"API key {CANARY} is not valid"))
        with caplog.at_level(logging.DEBUG, logger="bloomlens.identify"), pytest.raises(IdentifyError) as info:
            ident._invoke_agent_with_retries({})
        assert isinstance(info.value, IdentifyConfigError)
        assert CANARY not in str(info.value) and CANARY not in caplog.text
        assert "GoogleGenerativeAIError" in caplog.text  # the type is enough to debug from

    def test_server_error_text_stays_out_after_the_attempts_run_out(self, run, caplog):
        run(*[GoogleAPIError(503, {"error": {"message": CANARY}}) for _ in range(4)])
        with caplog.at_level(logging.DEBUG, logger="bloomlens.identify"), pytest.raises(IdentifyError) as info:
            ident._invoke_agent_with_retries({})
        assert CANARY not in str(info.value) and CANARY not in caplog.text

    def test_an_unexpected_exception_message_stays_out_too(self, run, caplog):
        run(*[RuntimeError(f"boom {CANARY}") for _ in range(4)])
        with caplog.at_level(logging.DEBUG, logger="bloomlens.identify"), pytest.raises(IdentifyError) as info:
            ident._invoke_agent_with_retries({})
        assert CANARY not in str(info.value) and CANARY not in caplog.text
        assert "RuntimeError" in caplog.text


def test_the_system_prompt_treats_text_in_photos_as_data_and_pins_the_grade_letters():
    prompt = ident._SYSTEM_PROMPT
    assert "inside a photo" in prompt and "not an instruction" in prompt
    assert "exactly one of the letters A, B or C" in prompt
