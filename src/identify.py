"""Core identification pipeline: scan -> BioCLIP 2 -> Qdrant -> Gemini -> result.

Milestone 1: a single linear pass. Milestone 2: confidence gating (this file).
No agent or lot mode yet — see docs/ARCHITECTURE.md for where those land later.
"""

import io
import json
import os
import time
from typing import Literal

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

# Confidence tiers, from Qdrant's top-1/top-2 cosine-similarity gap. These are a
# heuristic starting point from Milestone 1's handful of real test photos (correct
# matches scored 0.64-0.69 with 0.06-0.14 gaps) — not yet tuned against a labeled
# dataset. Revisit once Milestone 5's evaluation harness exists.
_HIGH_CONFIDENCE_MIN_SCORE = 0.55
_HIGH_CONFIDENCE_MIN_GAP = 0.05
_LOW_CONFIDENCE_MAX_SCORE = 0.45

ConfidenceTier = Literal["high", "ambiguous", "low"]


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
    confidence_tier: ConfidenceTier


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


def _classify_confidence(candidates: list[dict]) -> ConfidenceTier:
    """Derive a confidence tier from Qdrant's top-1/top-2 similarity scores alone
    (no Gemini involvement) — see the threshold constants above for rationale."""
    top_score = candidates[0]["score"]
    if top_score < _LOW_CONFIDENCE_MAX_SCORE:
        return "low"

    gap = top_score - candidates[1]["score"] if len(candidates) > 1 else top_score
    if top_score >= _HIGH_CONFIDENCE_MIN_SCORE and gap >= _HIGH_CONFIDENCE_MIN_GAP:
        return "high"

    return "ambiguous"


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
                    http_options=types.HttpOptions(timeout=45_000)  # milliseconds; avoid an indefinite hang
                ),
            )
            return _GeminiAnswer.model_validate(_extract_json(response.text))
        except genai_errors.ClientError as exc:
            code = getattr(exc, "code", None)
            if code == 429:
                raise IdentifyError(
                    "Gemini's free-tier rate limit was reached for this key. Wait a minute and try "
                    "again, or check your quota at https://ai.dev/rate-limit."
                ) from exc
            if code == 499:
                # Our own request timeout (45s) firing under high demand — retry like any
                # other transient failure rather than treating it as a bad request.
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_BASE_DELAY_SECONDS * (attempt + 1))
                continue
            raise IdentifyError(f"Gemini rejected the request: {exc}") from exc
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
            {
                "common_name": c["payload"]["common_name"],
                "scientific_name": c["payload"]["scientific_name"],
                "score": round(c["score"], 3),
            }
            for c in candidates
        ],
        confidence_tier=_classify_confidence(candidates),
    )


def resolve_candidate(species: str, top_candidates: list[dict], quality_grade: str) -> dict:
    """Re-derive scientific_name/price for a different candidate the user picked in the
    'ambiguous' tier's switcher — no new Gemini call, just fresh lookups (app.py)."""
    scientific_name = next(
        (c["scientific_name"] for c in top_candidates if c["common_name"] == species),
        None,
    )
    price = lookup_price(species, grade=quality_grade)
    return {
        "scientific_name": scientific_name,
        "price_per_stem": price["price_per_stem"],
        "price_trend": price["trend"],
        "price_as_of": price["as_of_date"],
    }
