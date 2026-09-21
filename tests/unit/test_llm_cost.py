"""src/llm_cost.py: the append-only price history and the cost estimate."""

from dataclasses import replace
from datetime import date
from urllib.parse import urlparse

import pytest

from src import llm_cost
from src.llm_cost import PRICE_VERSIONS, PriceVersion, estimate_cost_usd, version_in_force
from src.versions import GEMINI_MODEL

V1 = PriceVersion("m@2026-01-01", "m", "2026-01-01", 1.0, 4.0, "https://example.com/p", "2026-01-02")
V2 = replace(V1, version="m@2026-06-01", effective_from="2026-06-01", input_usd_per_mtok=2.0, output_usd_per_mtok=8.0)


class TestVersionInForce:
    def test_before_the_first_version_there_is_no_price(self):
        assert version_in_force("m", date(2025, 12, 31), (V1, V2)) is None

    def test_a_version_applies_from_its_effective_date_inclusive(self):
        assert version_in_force("m", date(2026, 1, 1), (V1, V2)) == V1
        assert version_in_force("m", date(2026, 5, 31), (V1, V2)) == V1

    def test_a_newer_version_takes_over_on_its_own_date(self):
        assert version_in_force("m", date(2026, 6, 1), (V1, V2)) == V2
        assert version_in_force("m", date(2030, 1, 1), (V1, V2)) == V2

    def test_the_order_of_the_list_does_not_matter(self):
        assert version_in_force("m", date(2026, 7, 1), (V2, V1)) == V2

    def test_another_model_has_its_own_history(self):
        assert version_in_force("other", date(2026, 7, 1), (V1, V2)) is None


class TestEstimate:
    def test_it_is_tokens_times_the_per_million_price(self):
        assert estimate_cost_usd(V1, 1_000_000, 0) == pytest.approx(1.0)
        assert estimate_cost_usd(V1, 0, 1_000_000) == pytest.approx(4.0)
        assert estimate_cost_usd(V1, 500_000, 250_000) == pytest.approx(0.5 + 1.0)

    def test_the_real_probe_run_at_the_recorded_price(self):
        """The 4-turn run in docs/telemetry_probe.md: 7 908 tokens in, 213 out."""
        current = version_in_force(GEMINI_MODEL, date(2026, 9, 21))
        assert estimate_cost_usd(current, 7908, 213) == pytest.approx((7908 * 0.25 + 213 * 1.5) / 1e6)
        assert estimate_cost_usd(current, 7908, 213) == pytest.approx(0.0022965)

    def test_zero_tokens_cost_nothing(self):
        assert estimate_cost_usd(V1, 0, 0) == 0


class TestTheRecordedHistory:
    def test_the_model_in_use_has_a_price_from_the_day_it_was_read(self):
        assert version_in_force(GEMINI_MODEL, date(2026, 9, 21)) is not None

    def test_before_the_price_was_published_no_price_is_applied(self):
        """A scan dated before the first recorded version gets no cost, not today's price."""
        assert version_in_force(GEMINI_MODEL, date(2026, 9, 15)) is None

    def test_every_entry_is_well_formed(self):
        names = [v.version for v in PRICE_VERSIONS]
        assert len(names) == len(set(names))
        for v in PRICE_VERSIONS:
            assert v.version == f"{v.model}@{v.effective_from}"
            assert date.fromisoformat(v.retrieved_at) >= date.fromisoformat(v.effective_from)
            assert v.input_usd_per_mtok > 0 and v.output_usd_per_mtok > 0
            assert urlparse(v.source_url).scheme == "https"

    def test_the_history_is_immutable_at_the_type_level(self):
        with pytest.raises(AttributeError):
            PRICE_VERSIONS[0].input_usd_per_mtok = 0.0  # frozen dataclass

    def test_the_label_never_calls_it_a_bill_or_free(self):
        label = llm_cost.COST_LABEL
        assert "estimated" in label and "actual billed cost unknown" in label
        assert "free" not in label.lower() and "$0" not in label
