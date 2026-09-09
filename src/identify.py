"""Core identification pipeline: scan -> BioCLIP 2 -> Qdrant -> Gemini -> result.

Milestone 1: a single linear pass (no agent, no confidence gating, no lot
mode yet — see docs/ARCHITECTURE.md for where those land in later milestones).
"""

import io
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from src.embeddings import embed_image
from src.pricing import lookup_price
from src.vector_store import get_client, search

load_dotenv()

_GEMINI_MODEL = "gemini-3.6-flash"

# Both response_json_schema and response_schema (Gemini's constrained-decoding
# structured-output feature) returned intermittent 503s in testing — a known,
# currently-open issue: https://github.com/googleapis/python-genai/issues/1378.
# Plain generation (no response_mime_type/response_schema) was reliable, so we
# ask for JSON via the prompt instead and parse the text ourselves.
_MAX_RETRIES = 4
_RETRY_BASE_DELAY_SECONDS = 2


class _GeminiAnswer(BaseModel):
    species: str
    confidence_note: str
    quality_grade: str
    quality_note: str
    summary: str


class IdentifyResult(BaseModel):
    species: str
    scientific_name: str | None = None
    confidence_note: str
    quality_grade: str
    quality_note: str
    summary: str
    price_per_stem: float | None
    price_trend: str
    price_as_of: str | None
    price_simulated: bool = True
    top_candidates: list[dict]


class IdentifyError(Exception):
    pass


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise IdentifyError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey, then restart the app."
        )
    return genai.Client(api_key=api_key)


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _build_prompt(candidates: list[dict]) -> str:
    lines = [
        "You are helping a flower-auction buyer identify a flower from a photo.",
        "A vision similarity search has already narrowed the species down to these candidates:",
        "",
    ]
    for c in candidates:
        p = c["payload"]
        lines.append(f"- {p['common_name']} ({p['scientific_name']}): {p['description']}")
    lines.extend(
        [
            "",
            "Look at the photo and pick the best-matching candidate. The 'species' field must be "
            "ONLY the common_name exactly as written above (e.g. 'Rose') — do not add the "
            "scientific name, parentheses, or any other text to that field. If the photo clearly "
            "isn't any of these candidates or isn't a flower at all, still pick the closest "
            "candidate but say so plainly in confidence_note (e.g. 'low confidence, photo may not "
            "match any known candidate').",
            "Then assess visible quality: bloom stage, wilting, discoloration, or blemishes, and "
            "assign a quality_grade of A (excellent), B (good), or C (fair).",
            "Remember: your quality assessment is a heuristic visual read, not a calibrated "
            "agronomic grading system — keep quality_note honest about what you can and can't tell from one photo.",
            "",
            "Respond with ONLY a single JSON object, no markdown code fences and no other text, "
            "with exactly these string keys: species, confidence_note, quality_grade, quality_note, summary.",
        ]
    )
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of Gemini's text response, tolerating markdown code fences
    or stray text around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise IdentifyError(f"Gemini's response didn't contain a JSON object: {text[:200]!r}")

    return json.loads(stripped[start : end + 1])


def _match_candidate(species_text: str, candidates: list[dict]) -> str:
    """Map Gemini's free-text species answer back to one of the candidates' exact
    common_name, since the model doesn't always copy it verbatim (e.g. it may add
    the scientific name in parentheses). Falls back to the top Qdrant candidate."""
    names = [c["payload"]["common_name"] for c in candidates]

    for name in names:
        if name.lower() == species_text.strip().lower():
            return name

    for name in names:
        if name.lower() in species_text.lower():
            return name

    return names[0]


def _generate_with_retries(client: genai.Client, image: Image.Image, candidates: list[dict]) -> _GeminiAnswer:
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=[
                    _build_prompt(candidates),
                    types.Part.from_bytes(data=_image_to_jpeg_bytes(image), mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(timeout=30_000)  # milliseconds; avoid an indefinite hang
                ),
            )
            return _GeminiAnswer.model_validate(_extract_json(response.text))
        except genai_errors.ClientError as exc:
            if getattr(exc, "code", None) == 429:
                raise IdentifyError(
                    "Gemini's free-tier rate limit was reached for this key. Wait a minute and try "
                    "again, or check your quota at https://ai.dev/rate-limit."
                ) from exc
            raise IdentifyError(f"Gemini rejected the request: {exc}") from exc
        except genai_errors.ClientError:
            raise
        except Exception as exc:  # noqa: BLE001 — includes ServerError, timeouts, and any parsing failure; all retryable
            last_error = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY_SECONDS * (attempt + 1))
    raise IdentifyError(
        f"Gemini didn't return a usable answer after {_MAX_RETRIES} attempts — please try again in a moment. ({last_error})"
    )


def identify(image: Image.Image) -> IdentifyResult:
    image_embedding = embed_image(image)

    qdrant = get_client()
    candidates = search(qdrant, image_embedding, top_k=3)
    if not candidates:
        raise IdentifyError("The species index is empty — run scripts/build_index.py first.")

    client = _client()
    parsed = _generate_with_retries(client, image, candidates)

    species = _match_candidate(parsed.species, candidates)
    scientific_name = next(
        (c["payload"]["scientific_name"] for c in candidates if c["payload"]["common_name"] == species),
        None,
    )

    price = lookup_price(species, grade=parsed.quality_grade)

    return IdentifyResult(
        species=species,
        scientific_name=scientific_name,
        confidence_note=parsed.confidence_note,
        quality_grade=parsed.quality_grade,
        quality_note=parsed.quality_note,
        summary=parsed.summary,
        price_per_stem=price["price_per_stem"],
        price_trend=price["trend"],
        price_as_of=price["as_of_date"],
        top_candidates=[
            {"common_name": c["payload"]["common_name"], "score": round(c["score"], 3)} for c in candidates
        ],
    )
