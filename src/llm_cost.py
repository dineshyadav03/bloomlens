"""List prices for the language model, as an append-only history.

Cost in telemetry is an **estimated list-price equivalent; actual billed cost unknown**:
tokens times the published per-million-token price. It is not a bill -- on the free tier
the bill is zero -- and it is never recomputed with today's prices. Each scan stores the
`pricing_version` in force when it ran and its own cost; a price change is a *new*
version appended here (never an edit), and old rows keep the old one. The versions are
also copied into an immutable table (UPDATE/DELETE are blocked by triggers) so a stored
cost can always be traced to the price it used.

The figures come from Google's pricing page (source_url), read on `retrieved_at`
through a summarising tool: verify against the page before relying on them.
"""

from dataclasses import dataclass
from datetime import date

COST_LABEL = "estimated list-price equivalent; actual billed cost unknown"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"


@dataclass(frozen=True)
class PriceVersion:
    version: str
    model: str
    effective_from: str  # ISO date: when this price was published (the source page's "last updated")
    input_usd_per_mtok: float
    output_usd_per_mtok: float  # includes reasoning tokens, per the source
    source_url: str
    retrieved_at: str  # ISO date the page was read


# Append-only. To record a price change, add a new entry with a later effective_from.
PRICE_VERSIONS: tuple[PriceVersion, ...] = (
    PriceVersion(
        version="gemini-3.1-flash-lite@2026-09-16",
        model="gemini-3.1-flash-lite",
        effective_from="2026-09-16",
        input_usd_per_mtok=0.25,  # text / image / video (audio is $0.50; BloomLens sends none)
        output_usd_per_mtok=1.50,
        source_url=PRICING_URL,
        retrieved_at="2026-09-21",
    ),
)


def version_in_force(
    model: str, on: date, versions: tuple[PriceVersion, ...] | None = None
) -> PriceVersion | None:
    """The newest price for `model` published on or before `on`, or None -- in which case
    the run gets no cost rather than a price that did not exist yet. `versions` defaults to
    the module's history, looked up at call time (so tests, or a later edit, are honoured)."""
    versions = PRICE_VERSIONS if versions is None else versions
    candidates = [v for v in versions if v.model == model and date.fromisoformat(v.effective_from) <= on]
    return max(candidates, key=lambda v: v.effective_from, default=None)


def estimate_cost_usd(version: PriceVersion, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * version.input_usd_per_mtok + output_tokens * version.output_usd_per_mtok) / 1_000_000
