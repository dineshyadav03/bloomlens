"""Keep photos and prompts out of any tracing service.

LangChain reports every model call -- prompts and base64 images included -- to LangSmith
when tracing is on, and `langsmith` is a transitive dependency, so it is one environment
variable (possibly from a developer's shell or .env) away from being enabled. BloomLens
never wants that: the only place a photo may go is the Gemini request itself (see
docs/PRIVACY.md). So tracing is forced off here, whatever the environment says.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

# langsmith reads TRACING_V2 then TRACING, each under a LANGSMITH_ and a LANGCHAIN_ prefix.
_TRACING_VARS = ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING")


def disable_tracing() -> None:
    """Set every switch off. Must run before langsmith first reads the environment
    (it caches what it reads), so src/identify.py calls it at import time."""
    for name in _TRACING_VARS:
        os.environ[name] = "false"


@contextmanager
def no_tracing() -> Iterator[None]:
    """A second, independent guard for the model call itself: langsmith's own context
    override takes precedence over the environment and over a cached read of it."""
    from langsmith import tracing_context

    with tracing_context(enabled=False):
        yield
