"""The three candidate "in-distribution-ness" scores of eval/PROTOCOL.md section 3.

Each takes the similarities of one image to **all 30** species (any order, as `search(top_k=30)`
returns them) and returns a single float where higher means more in-distribution.
"""

import math
from typing import Literal

TEMPERATURES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1)
ScoreName = Literal["max_cosine", "margin", "max_softmax"]


def _sorted_desc(similarities: list[float]) -> list[float]:
    if not similarities:
        raise ValueError("similarities must be non-empty")
    return sorted(similarities, reverse=True)


def max_cosine(similarities: list[float]) -> float:
    """S1: the top-1 similarity."""
    return _sorted_desc(similarities)[0]


def margin(similarities: list[float]) -> float:
    """S2: top-1 minus top-2 similarity (0 if there is only one candidate)."""
    ordered = _sorted_desc(similarities)
    return ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]


def max_softmax(similarities: list[float], temperature: float) -> float:
    """S3: max_j softmax(sim_j / T). Shifted by the max before exponentiating (softmax is
    shift-invariant), so this never overflows however large the similarities are."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [s / temperature for s in similarities]
    top = max(scaled)
    weights = [math.exp(s - top) for s in scaled]
    return max(weights) / sum(weights)


def candidate_name(name: ScoreName, temperature: float | None = None) -> str:
    """The frozen-config identifier for a candidate, e.g. 'max_softmax@0.05'."""
    return f"{name}@{temperature}" if name == "max_softmax" else name


def all_candidates(similarities: list[float]) -> dict[str, float]:
    """Every candidate's value for one image, keyed by `candidate_name`."""
    values = {"max_cosine": max_cosine(similarities), "margin": margin(similarities)}
    for temperature in TEMPERATURES:
        values[candidate_name("max_softmax", temperature)] = max_softmax(similarities, temperature)
    return values


def score_by_name(candidate: str, similarities: list[float]) -> float:
    """The value of one candidate (by its `candidate_name` string) for one image."""
    if candidate == "max_cosine":
        return max_cosine(similarities)
    if candidate == "margin":
        return margin(similarities)
    if candidate.startswith("max_softmax@"):
        return max_softmax(similarities, float(candidate.split("@", 1)[1]))
    raise ValueError(f"unknown candidate {candidate!r}")
