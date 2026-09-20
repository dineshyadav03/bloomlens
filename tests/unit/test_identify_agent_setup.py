"""API-key handling and agent construction in src/identify.py, with the LangChain
model and agent factories faked: no key is real, nothing is contacted."""

import pytest

from src import identify as ident
from src import versions
from src.identify import IdentifyError


class TestRequireApiKey:
    def test_a_missing_key_is_a_clear_actionable_error(self):
        with pytest.raises(IdentifyError, match="GEMINI_API_KEY is not set"):
            ident._require_api_key()

    def test_an_empty_key_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")
        with pytest.raises(IdentifyError):
            ident._require_api_key()

    def test_a_present_key_is_returned(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        assert ident._require_api_key() == "test-key-not-real"


class TestGetAgent:
    @pytest.fixture
    def factories(self, monkeypatch):
        seen = {"models": [], "agents": []}

        def fake_model(**kwargs):
            seen["models"].append(kwargs)
            return object()

        def fake_create_agent(**kwargs):
            seen["agents"].append(kwargs)
            return object()

        monkeypatch.setattr(ident, "ChatGoogleGenerativeAI", fake_model)
        monkeypatch.setattr(ident, "create_agent", fake_create_agent)
        monkeypatch.setattr(ident, "_agent_singleton", None)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        return seen

    def test_it_is_built_once_and_cached(self, factories):
        first, second = ident._get_agent(), ident._get_agent()
        assert first is second
        assert len(factories["models"]) == len(factories["agents"]) == 1

    def test_it_uses_the_pinned_model_key_timeout_and_the_three_tools(self, factories):
        ident._get_agent()
        (model_kwargs,) = factories["models"]
        assert model_kwargs["model"] == versions.GEMINI_MODEL
        assert model_kwargs["google_api_key"] == "test-key-not-real"
        assert model_kwargs["timeout"] == 45
        (agent_kwargs,) = factories["agents"]
        assert {t.name for t in agent_kwargs["tools"]} == {"lookup_taxonomy", "assess_quality", "check_price"}
        assert agent_kwargs["system_prompt"] == ident._SYSTEM_PROMPT

    def test_a_missing_key_fails_before_any_model_is_built(self, factories, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY")
        with pytest.raises(IdentifyError, match="GEMINI_API_KEY"):
            ident._get_agent()
        assert factories["models"] == []
