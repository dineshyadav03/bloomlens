"""LangChain tools available to the Milestone 3 identification agent.

The agent already sees the scanned photo directly in its own multimodal
context, so `assess_quality` does not re-look at the image — it's where the
agent formally records the visual judgment it already made, with a
consistency check rather than being a no-op pass-through. See
docs/ARCHITECTURE.md's "Agent tools" section for the reasoning.
"""

import json
import re
from pathlib import Path

from langchain_core.tools import tool

from src.pricing import lookup_price

_SPECIES_PATH = Path(__file__).resolve().parent.parent / "data" / "species_reference.json"
_SPECIES_BY_NAME = {s["common_name"]: s for s in json.loads(_SPECIES_PATH.read_text(encoding="utf-8"))}

_WILT_KEYWORDS = ("wilt", "brown", "blemish", "damage", "discolor", "droop")
_NEGATION_CUES = ("no ", "not ", "none", "free of", "free from", "without", "no signs of", "no visible")


def _mentions_unnegated_damage(note: str) -> bool:
    """True if a damage-related keyword appears in a sentence with no negation
    cue — e.g. flags "some browning on the edges" but not "completely free of
    wilting, discoloration, or pest damage" (a negation cue governs the whole
    listed clause, not just the word immediately after it, so this checks
    per-sentence rather than a fixed lookback window). Still a heuristic, not
    real NLP negation handling."""
    for sentence in re.split(r"(?<=[.!?])\s+", note.lower()):
        if any(cue in sentence for cue in _NEGATION_CUES):
            continue
        if any(keyword in sentence for keyword in _WILT_KEYWORDS):
            return True
    return False


@tool
def lookup_taxonomy(species: str) -> str:
    """Look up the botanical taxonomy and description for a curated species by its common name."""
    entry = _SPECIES_BY_NAME.get(species)
    if entry is None:
        return f"'{species}' is not in BloomLens's curated species list — no taxonomy data available."
    return f"{entry['taxonomy_string']} — {entry['description']}"


@tool
def assess_quality(quality_grade: str, quality_note: str) -> dict:
    """Record a visual quality assessment (grade A/B/C plus a short note) after looking at the photo.
    Validates the grade and flags if the note describes damage inconsistent with an A grade."""
    grade = quality_grade.strip().upper()
    if grade not in ("A", "B", "C"):
        return {"error": f"quality_grade must be A, B, or C, got {quality_grade!r}"}

    flagged = grade == "A" and _mentions_unnegated_damage(quality_note)
    return {
        "quality_grade": grade,
        "quality_note": quality_note,
        "consistency_warning": (
            "Note mentions visible damage/wilting but grade is A — consider a lower grade." if flagged else None
        ),
    }


@tool
def check_price(species: str, grade: str) -> dict:
    """Look up the simulated auction price per stem and trend for a species and quality grade (A/B/C)."""
    return lookup_price(species, grade=grade)
