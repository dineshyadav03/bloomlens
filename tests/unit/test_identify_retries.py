"""Retry/backoff behavior of identify._invoke_agent_with_retries.

The agent is a scripted fake and time.sleep is recorded instead of executed, so
these run instantly and can never touch the real Gemini API."""

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import GoogleAPIError, GoogleGenerativeAIError, GoogleRateLimitError

from src import identify as ident
from src.identify import IdentifyError

GOOD_ANSWER = {
    "species": "Rose",
    "confidence_note": "clearly a rose",
    "quality_grade": "A",
    "quality_note": "fresh",
    "summary": "A fine rose.",
}
MESSAGE = {"role": "user", "content": "unused by the fake"}


class ScriptedAgent:
    """Each invoke() consumes the next scripted step: an Exception is raised,
    a string is returned as the agent's final message."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    def invoke(self, _payload):
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return {"messages": [AIMessage(content=step)]}


@pytest.fixture
def run(monkeypatch):
    """run(*steps) -> (result_or_exception, sleeps, agent)"""

    def _run(*steps):
        agent = ScriptedAgent(*steps)
        sleeps = []
        monkeypatch.setattr(ident, "_get_agent", lambda: agent)
        monkeypatch.setattr(ident.time, "sleep", sleeps.append)
        try:
            outcome = ident._invoke_agent_with_retries(MESSAGE)
        except IdentifyError as exc:
            outcome = exc
        return outcome, sleeps, agent

    return _run


def server_error():
    return GoogleAPIError(503, {"error": {"code": 503, "message": "This model is currently experiencing high demand."}})


def rate_limited(delay: str | None = None):
    detail = f" 'retryDelay': '{delay}'" if delay else ""
    return GoogleRateLimitError(f"429 RESOURCE_EXHAUSTED.{detail}")


def test_first_try_success_never_sleeps(run):
    outcome, sleeps, agent = run(json.dumps(GOOD_ANSWER))
    assert outcome.species == "Rose" and outcome.quality_grade == "A"
    assert sleeps == [] and agent.calls == 1


def test_server_errors_back_off_linearly_then_succeed(run):
    outcome, sleeps, agent = run(server_error(), server_error(), json.dumps(GOOD_ANSWER))
    assert outcome.species == "Rose"
    assert sleeps == [2, 4]
    assert agent.calls == 3


def test_persistent_server_errors_exhaust_the_attempts(run):
    outcome, sleeps, agent = run(*[server_error() for _ in range(4)])
    assert isinstance(outcome, IdentifyError)
    assert "4 attempts" in str(outcome)
    assert sleeps == [2, 4, 6]  # no pointless sleep after the final attempt
    assert agent.calls == 4


def test_rate_limit_honors_the_servers_suggested_delay(run):
    outcome, sleeps, _ = run(rate_limited("37s"), json.dumps(GOOD_ANSWER))
    assert outcome.species == "Rose"
    assert sleeps == [37.0]


def test_rate_limit_delay_is_capped(run):
    _, sleeps, _ = run(rate_limited("600s"), json.dumps(GOOD_ANSWER))
    assert sleeps == [60]


def test_rate_limit_without_a_hint_uses_the_default_delay(run):
    _, sleeps, _ = run(rate_limited(), json.dumps(GOOD_ANSWER))
    assert sleeps == [30.0]


def test_repeated_rate_limits_give_up_with_an_actionable_message(run):
    outcome, sleeps, agent = run(rate_limited(), rate_limited(), rate_limited())
    assert isinstance(outcome, IdentifyError)
    assert "rate limit" in str(outcome).lower()
    assert sleeps == [30.0, 30.0]  # two retries allowed; the third raises immediately
    assert agent.calls == 3


def test_rate_limit_is_not_mistaken_for_an_auth_error(run):
    """GoogleRateLimitError subclasses GoogleGenerativeAIError; if the except
    clauses were reordered a 429 would be reported as 'rejected the request'."""
    assert issubclass(GoogleRateLimitError, GoogleGenerativeAIError)
    outcome, _, _ = run(rate_limited("1s"), json.dumps(GOOD_ANSWER))
    assert not isinstance(outcome, Exception)


def test_auth_or_invalid_request_errors_fail_immediately(run):
    outcome, sleeps, agent = run(GoogleGenerativeAIError("API key not valid"), json.dumps(GOOD_ANSWER))
    assert isinstance(outcome, IdentifyError)
    assert "rejected the request" in str(outcome)
    assert sleeps == [] and agent.calls == 1  # retrying can't help


def test_timeouts_are_retried(run):
    outcome, sleeps, _ = run(TimeoutError("read timed out"), TimeoutError("read timed out"), json.dumps(GOOD_ANSWER))
    assert outcome.species == "Rose"
    assert sleeps == [2, 4]


def test_unparseable_output_is_retried(run):
    outcome, sleeps, _ = run("I could not decide.", "{not json}", json.dumps(GOOD_ANSWER))
    assert outcome.species == "Rose"
    assert sleeps == [2, 4]


def test_output_missing_required_fields_is_retried_then_fails(run):
    incomplete = json.dumps({"species": "Rose"})
    outcome, _, agent = run(incomplete, incomplete, incomplete, incomplete)
    assert isinstance(outcome, IdentifyError)
    assert agent.calls == 4


def test_mixed_failures_share_one_attempt_budget(run):
    outcome, sleeps, agent = run(rate_limited("5s"), server_error(), TimeoutError("x"), json.dumps(GOOD_ANSWER))
    assert outcome.species == "Rose"
    assert agent.calls == 4
    assert sleeps == [5.0, 4, 6]
