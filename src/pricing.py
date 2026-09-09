"""Simulated auction pricing.

FloraHolland does not expose a public real-time pricing API (see
docs/RESEARCH.md#floraholland). This generates a plausible-looking mock
price table instead, clearly labeled as simulated everywhere it's shown.
"""

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

GRADES = ["A", "B", "C"]
DEFAULT_CSV_PATH = str(Path(__file__).resolve().parent.parent / "data" / "simulated_prices.csv")

# Rough base price per stem (EUR), grade A. Illustrative only, not real market data.
_BASE_PRICE = {
    "Rose": 0.55,
    "Tulip": 0.35,
    "Chrysanthemum": 0.30,
    "Easter lily": 0.90,
    "Moth orchid": 1.20,
    "Carnation": 0.25,
    "Sunflower": 0.60,
    "Oxeye daisy": 0.30,
    "Bearded iris": 0.45,
    "Gerbera daisy": 0.35,
    "Peony": 1.50,
    "Bigleaf hydrangea": 1.80,
    "Freesia": 0.40,
    "Gladiolus": 0.50,
    "Daffodil": 0.30,
    "Poppy anemone": 0.45,
    "Persian buttercup": 0.55,
    "Peruvian lily": 0.35,
    "Snapdragon": 0.40,
    "Dahlia": 0.65,
    "Lisianthus": 0.60,
    "Marigold": 0.25,
    "Zinnia": 0.30,
    "Hyacinth": 0.45,
    "Amaryllis": 1.40,
    "King protea": 2.20,
    "Calla lily": 1.10,
    "Sweet pea": 0.40,
    "Statice": 0.25,
    "Bird of paradise": 2.50,
}

_GRADE_MULTIPLIER = {"A": 1.0, "B": 0.75, "C": 0.5}


def generate_simulated_prices(species_list: list[dict], days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate a small random-walk price series per species x grade, write to CSV, return it."""
    rng = random.Random(seed)
    rows = []
    today = date.today()

    for species in species_list:
        name = species["common_name"]
        base = _BASE_PRICE.get(name, 0.50)
        for grade in GRADES:
            price = base * _GRADE_MULTIPLIER[grade]
            for day_offset in range(days, -1, -1):
                price *= 1 + rng.uniform(-0.04, 0.04)
                price = max(price, 0.05)
                rows.append(
                    {
                        "species": name,
                        "grade": grade,
                        "date": (today - timedelta(days=day_offset)).isoformat(),
                        "price_per_stem": round(price, 2),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(DEFAULT_CSV_PATH, index=False)
    return df


def _load_prices() -> pd.DataFrame:
    path = Path(DEFAULT_CSV_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{DEFAULT_CSV_PATH} not found — run scripts/build_index.py first to generate simulated prices."
        )
    return pd.read_csv(path)


def lookup_price(species: str, grade: str = "B") -> dict:
    """Return {price_per_stem, trend, as_of_date, simulated: True} for a species/grade."""
    df = _load_prices()
    subset = df[(df["species"] == species) & (df["grade"] == grade)].sort_values("date")

    if subset.empty:
        return {
            "price_per_stem": None,
            "trend": "unknown",
            "as_of_date": None,
            "simulated": True,
            "note": f"No simulated price data for '{species}' grade {grade}.",
        }

    latest = subset.iloc[-1]
    trend = "flat"
    if len(subset) > 1:
        change = latest["price_per_stem"] - subset.iloc[0]["price_per_stem"]
        if change > 0.02:
            trend = "up"
        elif change < -0.02:
            trend = "down"

    return {
        "price_per_stem": float(latest["price_per_stem"]),
        "trend": trend,
        "as_of_date": latest["date"],
        "simulated": True,
    }


def price_history(species: str, grade: str = "B") -> pd.DataFrame:
    """Return the date-sorted price history for a species/grade, for charting."""
    df = _load_prices()
    subset = df[(df["species"] == species) & (df["grade"] == grade)].sort_values("date")
    return subset[["date", "price_per_stem"]]
