"""src/pricing.py: the simulated price table, lookup and trend logic."""

import pandas as pd
import pytest

from src import pricing
from tests.conftest import FrozenDate


def write_csv(path, rows):
    lines = ["species,grade,date,price_per_stem"] + [f"{s},{g},{d},{p}" for s, g, d, p in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def series_csv(tmp_path, monkeypatch):
    """Write a Rose/B series from (date, price) pairs and point pricing at it."""

    def _make(prices):
        path = tmp_path / "series.csv"
        write_csv(path, [("Rose", "B", d, p) for d, p in prices])
        monkeypatch.setattr(pricing, "DEFAULT_CSV_PATH", str(path))

    return _make


class TestLookupPrice:
    def test_returns_the_latest_price_and_its_date(self, series_csv):
        series_csv([("2026-09-01", 1.00), ("2026-09-02", 1.10), ("2026-09-03", 1.20)])
        result = pricing.lookup_price("Rose", "B")
        assert result["price_per_stem"] == pytest.approx(1.20)
        assert result["as_of_date"] == "2026-09-03"
        assert result["simulated"] is True

    @pytest.mark.parametrize(
        ("last_price", "expected_trend"),
        [(1.03, "up"), (1.015, "flat"), (1.00, "flat"), (0.985, "flat"), (0.97, "down")],
    )
    def test_trend_compares_first_and_last_with_a_2_cent_dead_band(self, series_csv, last_price, expected_trend):
        series_csv([("2026-09-01", 1.00), ("2026-09-02", 1.50), ("2026-09-03", last_price)])
        assert pricing.lookup_price("Rose", "B")["trend"] == expected_trend

    def test_rows_are_sorted_by_date_not_file_order(self, series_csv):
        series_csv([("2026-09-03", 1.20), ("2026-09-01", 1.00), ("2026-09-02", 1.10)])
        assert pricing.lookup_price("Rose", "B")["as_of_date"] == "2026-09-03"

    def test_single_row_is_flat(self, series_csv):
        series_csv([("2026-09-01", 1.00)])
        assert pricing.lookup_price("Rose", "B")["trend"] == "flat"

    def test_unknown_species_or_grade_gives_no_price_not_a_wrong_one(self, series_csv):
        series_csv([("2026-09-01", 1.00)])
        for species, grade in (("Unicorn", "B"), ("Rose", "A")):
            result = pricing.lookup_price(species, grade)
            assert result["price_per_stem"] is None
            assert result["trend"] == "unknown"
            assert "No simulated price data" in result["note"]

    def test_missing_csv_says_how_to_generate_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pricing, "DEFAULT_CSV_PATH", str(tmp_path / "nope.csv"))
        with pytest.raises(FileNotFoundError, match="build_index"):
            pricing.lookup_price("Rose", "B")


class TestPriceHistory:
    def test_sorted_ascending_with_only_date_and_price(self, series_csv):
        series_csv([("2026-09-03", 1.20), ("2026-09-01", 1.00), ("2026-09-02", 1.10)])
        history = pricing.price_history("Rose", "B")
        assert list(history.columns) == ["date", "price_per_stem"]
        assert list(history["date"]) == ["2026-09-01", "2026-09-02", "2026-09-03"]

    def test_unknown_species_is_an_empty_frame(self, series_csv):
        series_csv([("2026-09-01", 1.00)])
        assert pricing.price_history("Unicorn", "B").empty


class TestGenerateSimulatedPrices:
    SPECIES = [{"common_name": "Rose"}, {"common_name": "Tulip"}, {"common_name": "Not in the base table"}]

    @pytest.fixture(autouse=True)
    def frozen(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pricing, "date", FrozenDate)
        monkeypatch.setattr(pricing, "DEFAULT_CSV_PATH", str(tmp_path / "generated.csv"))

    def test_shape_and_last_date(self):
        df = pricing.generate_simulated_prices(self.SPECIES, days=5, seed=1)
        assert len(df) == 3 * len(pricing.GRADES) * (5 + 1)
        assert df["date"].max() == "2026-09-20"
        assert df["date"].min() == "2026-09-15"

    def test_same_seed_is_reproducible_and_a_different_seed_is_not(self):
        a = pricing.generate_simulated_prices(self.SPECIES, days=5, seed=1)
        b = pricing.generate_simulated_prices(self.SPECIES, days=5, seed=1)
        c = pricing.generate_simulated_prices(self.SPECIES, days=5, seed=2)
        pd.testing.assert_frame_equal(a, b)
        assert not a["price_per_stem"].equals(c["price_per_stem"])

    def test_prices_never_fall_below_the_floor(self):
        df = pricing.generate_simulated_prices(self.SPECIES, days=200, seed=3)
        assert df["price_per_stem"].min() >= 0.05

    def test_a_better_grade_costs_more_at_the_start_of_the_walk(self):
        df = pricing.generate_simulated_prices([{"common_name": "Rose"}], days=1, seed=4)
        first_day = df[df["date"] == df["date"].min()].set_index("grade")["price_per_stem"]
        assert first_day["A"] > first_day["B"] > first_day["C"]

    def test_the_csv_is_written_and_round_trips(self):
        df = pricing.generate_simulated_prices(self.SPECIES, days=3, seed=5)
        from_disk = pd.read_csv(pricing.DEFAULT_CSV_PATH)
        assert len(from_disk) == len(df)
        assert pricing.lookup_price("Rose", "A")["price_per_stem"] is not None
