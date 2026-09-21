"""Probe the SHAPE of one real agent run, to learn which telemetry fields actually exist.

Telemetry columns for token counts, the provider's model version and tool-call counts
are nullable until a real trace shows the provider returns them (docs/telemetry_probe.md
records what this found). This makes ONE real identify() call on a committed test photo
(a few Gemini requests, free tier) and prints only structure:

- which keys exist on each message's `usage_metadata` / `response_metadata` /
  `additional_kwargs`, with numeric values (token counts are not private);
- string values ONLY for an allow-list of non-content fields (model name/version,
  finish reason); every other string is reported as its length;
- tool-call names and the *names* of their arguments, never the values.

No message content, prompt, image or model answer is printed or stored.

    uv run python scripts/probe_agent_trace.py [path/to/photo.jpg]

Needs GEMINI_API_KEY (from .env). Never prints it.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src import identify as ident  # noqa: E402
from src import privacy  # noqa: E402
from src.embeddings import embed_image  # noqa: E402
from src.vector_store import get_client, search  # noqa: E402

DEFAULT_PHOTO = Path(__file__).resolve().parent.parent / "eval" / "test_images" / "Amaryllis_5455.jpg"

# String-valued fields that describe the call, not its content.
_SAFE_STRING_KEYS = {"model_name", "model_version", "finish_reason", "model_provider", "safety_ratings"}


def describe(value, key: str = ""):
    """A content-free description of `value`."""
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, str):
        return value if key in _SAFE_STRING_KEYS else f"<str len={len(value)}>"
    if isinstance(value, dict):
        return {k: describe(v, k) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [describe(v, key) for v in value][:5] + ([f"<{len(value) - 5} more>"] if len(value) > 5 else [])
    return f"<{type(value).__name__}>"


def describe_message(message) -> dict:
    content = message.content
    described = {
        "type": type(message).__name__,
        "content": f"<str len={len(content)}>" if isinstance(content, str) else f"<list of {len(content)} blocks>",
        "usage_metadata": describe(getattr(message, "usage_metadata", None)),
        "response_metadata": describe(getattr(message, "response_metadata", None)),
        "additional_kwargs_keys": sorted(getattr(message, "additional_kwargs", {}) or {}),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        described["tool_calls"] = [{"name": c.get("name"), "arg_names": sorted(c.get("args", {}))} for c in tool_calls]
    if getattr(message, "name", None):
        described["name"] = message.name
    return described


def main() -> None:
    photo = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PHOTO
    image = Image.open(photo).convert("RGB")
    candidates = search(get_client(), embed_image(image), top_k=3)
    message = ident._build_candidates_message(image, candidates)

    agent = ident._get_agent()
    started = time.perf_counter()
    with privacy.no_tracing():
        result = agent.invoke({"messages": [message]})
    elapsed = time.perf_counter() - started

    messages = result["messages"]
    report = {
        "photo": photo.name,
        "gemini_model_constant": ident.GEMINI_MODEL,
        "agent_wall_seconds": round(elapsed, 2),
        "result_keys": sorted(result),
        "message_count": len(messages),
        "messages": [describe_message(m) for m in messages],
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
