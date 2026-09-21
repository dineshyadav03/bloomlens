"""Tracing must stay off: a LangSmith trace would carry the photo and the prompt to a
third party (docs/PRIVACY.md). Each case that needs a *fresh* interpreter (langsmith
caches what it reads from the environment) runs in a subprocess."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from src import identify as ident
from src import privacy

REPO_ROOT = Path(__file__).resolve().parents[2]
ON = {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true", "LANGSMITH_TRACING_V2": "true"}


def python(code: str, **env) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), **env},
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


def test_importing_the_pipeline_switches_tracing_off_even_if_the_shell_switched_it_on():
    out = python(
        """
        import src.identify
        from langsmith.utils import tracing_is_enabled
        print(tracing_is_enabled())
        """,
        **ON,
    )
    assert out == "False"


def test_a_dotenv_file_cannot_switch_it_back_on():
    """load_dotenv() runs at import time in src/identify.py; simulate a .env that sets
    tracing, and check the order of operations still ends with it off."""
    out = python(
        """
        import os, dotenv
        dotenv.load_dotenv = lambda *a, **k: os.environ.update(LANGSMITH_TRACING="true", LANGCHAIN_TRACING_V2="true")
        import src.identify
        from langsmith.utils import tracing_is_enabled
        print(tracing_is_enabled())
        """
    )
    assert out == "False"


def test_the_control_case_really_would_have_enabled_it():
    """Without the guard the same environment does enable tracing -- so the two tests
    above are not passing vacuously."""
    out = python(
        """
        from langsmith.utils import tracing_is_enabled
        print(tracing_is_enabled())
        """,
        **ON,
    )
    assert out == "True"


def test_the_model_call_runs_inside_a_tracing_off_context(monkeypatch):
    """The second guard: even if the environment is flipped after import (and langsmith's
    cache is cleared), the call itself is made with tracing forced off."""
    from langchain_core.messages import AIMessage
    from langsmith import utils as ls_utils

    seen = {}

    class Agent:
        def invoke(self, _payload):
            seen["enabled"] = ls_utils.tracing_is_enabled()
            return {
                "messages": [
                    AIMessage(
                        content='{"species":"Rose","confidence_note":"x","quality_grade":"A","quality_note":"x","summary":"x"}'
                    )
                ]
            }

    for name, value in ON.items():
        monkeypatch.setenv(name, value)
    ls_utils.get_env_var.cache_clear()
    assert ls_utils.tracing_is_enabled() is True  # the environment now says yes...
    monkeypatch.setattr(ident, "_get_agent", lambda: Agent())
    ident._invoke_agent_with_retries({})
    assert seen["enabled"] is False  # ...and the call still ran with it off
    ls_utils.get_env_var.cache_clear()


def test_disable_tracing_sets_every_switch_langsmith_reads(monkeypatch):
    for name in privacy._TRACING_VARS:
        monkeypatch.setenv(name, "true")
    privacy.disable_tracing()
    assert {os.environ[n] for n in privacy._TRACING_VARS} == {"false"}
