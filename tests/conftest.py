"""Shared, hermetic test fixtures.

Determinism rules for this suite: no real network (enforced by pytest-socket via
pyproject addopts), no live Gemini call anywhere, every database/CSV lives in
tmp_path, clocks and RNG seeds are pinned where they matter, and only committed
files are used as fixtures.
"""

import datetime
import io
import json
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_collection_modifyitems(items):
    """Everything under tests/integration is marked `integration`, so that
    `pytest -m "not integration"` is the fast, weights-free unit run."""
    for item in items:
        if "integration" in Path(str(item.fspath)).relative_to(REPO_ROOT / "tests").parts[:1]:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch, tmp_path):
    """Nothing from a developer's shell or .env may leak into a test.

    src.identify calls load_dotenv() at import time, so a developer's real
    GEMINI_API_KEY can already be in os.environ by now; remove it, so a test that
    forgot to fake the agent fails on a missing key rather than on a live call."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("BLOOMLENS_AUTH", raising=False)
    monkeypatch.delenv("BLOOMLENS_ENV", raising=False)
    monkeypatch.delenv("BLOOMLENS_API_KEYS", raising=False)
    for name in (
        "BLOOMLENS_RATE_PER_MINUTE",
        "BLOOMLENS_QUOTA_PER_DAY",
        "BLOOMLENS_GLOBAL_QUOTA_PER_DAY",
        "BLOOMLENS_MAX_CONCURRENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INVENTORY_DB_PATH", str(tmp_path / "inventory.db"))

    from src import quota, retention

    quota.reset_gate()
    monkeypatch.setattr(quota, "_last_purge", float("-inf"))
    monkeypatch.setattr(retention, "_last_purge", float("-inf"))
    monkeypatch.delenv("BLOOMLENS_METRICS_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("BLOOMLENS_INVENTORY_RETENTION_DAYS", raising=False)
    yield
    quota.reset_gate()


TEST_API_KEY = "test-secret-0123456789-abcdefghij"


@pytest.fixture
def api_headers(monkeypatch):
    """Configure one labeled API key ('tester') and return the header that presents it."""
    monkeypatch.setenv("BLOOMLENS_API_KEYS", f"tester:{TEST_API_KEY}")
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def species_list():
    return json.loads((REPO_ROOT / "data" / "species_reference.json").read_text(encoding="utf-8"))


@pytest.fixture
def png_bytes():
    """A small, valid PNG (no file needed)."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 90)).save(buf, format="PNG")
    return buf.getvalue()


class FrozenDate(datetime.date):
    @classmethod
    def today(cls):
        return cls(2026, 9, 20)


@pytest.fixture
def prices_csv(tmp_path, monkeypatch):
    """Point src.pricing at a tiny, known CSV instead of data/simulated_prices.csv
    (whose dates are baked in at whatever day it was last regenerated)."""
    from src import pricing

    # Whole cents, so expected values in tests are exact literals (e.g. Rose A on
    # 2026-09-20 is 0.55, Tulip B is 0.29), never float arithmetic. Every series rises 5c.
    cents = {"Rose": (50, 40, 25), "Tulip": (30, 24, 15), "Carnation": (20, 16, 10), "Sunflower": (40, 32, 20)}
    rows = ["species,grade,date,price_per_stem"]
    for species, by_grade in cents.items():
        for grade, first in zip("ABC", by_grade, strict=True):
            rows.append(f"{species},{grade},2026-09-19,{first / 100:.2f}")
            rows.append(f"{species},{grade},2026-09-20,{(first + 5) / 100:.2f}")
    path = tmp_path / "prices.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(pricing, "DEFAULT_CSV_PATH", str(path))
    return path


@pytest.fixture
def make_candidate():
    def _make(name: str, score: float, scientific: str | None = None) -> dict:
        return {
            "score": score,
            "payload": {
                "common_name": name,
                "scientific_name": scientific or f"{name} scientificus",
                "taxonomy_string": f"Plantae {name}",
                "description": f"A {name}.",
            },
        }

    return _make


@pytest.fixture
def make_identify_result():
    from src.identify import IdentifyResult

    def _make(**overrides) -> IdentifyResult:
        fields = dict(
            species="Rose",
            scientific_name="Rosa chinensis",
            confidence_note="clearly a rose",
            quality_grade="A",
            quality_note="fresh",
            summary="A fine rose.",
            price_per_stem=0.55,
            price_trend="up",
            price_as_of="2026-09-20",
            top_candidates=[{"common_name": "Rose", "scientific_name": "Rosa chinensis", "score": 0.7}],
            confidence_tier="high",
        )
        fields.update(overrides)
        return IdentifyResult(**fields)

    return _make


@pytest.fixture
def make_lot_result():
    from src.identify import LotResult

    def _make(**overrides) -> LotResult:
        fields = dict(
            consensus_species="Sunflower",
            scientific_name="Helianthus annuus",
            quality_grade="B",
            quality_note="mixed",
            summary="Mostly sunflowers.",
            price_per_stem=0.6,
            price_trend="flat",
            price_as_of="2026-09-20",
            photo_count=3,
            agreement_fraction=0.667,
            flagged_photos=[{"index": 2, "top_species": "Rose", "score": 0.5}],
        )
        fields.update(overrides)
        return LotResult(**fields)

    return _make
