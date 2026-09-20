"""src/inventory.py against a throwaway SQLite file (INVENTORY_DB_PATH is set to
tmp_path by the autouse fixture in conftest)."""

import concurrent.futures
import datetime
import os
import sqlite3

import pytest

from src import inventory


def db_path() -> str:
    return os.environ["INVENTORY_DB_PATH"]


class TestLogging:
    def test_a_single_scan_round_trips(self, make_identify_result):
        inventory.log_scan(make_identify_result(species="Rose", confidence_tier="ambiguous", price_per_stem=0.55))
        (entry,) = inventory.list_recent()
        assert entry.mode == "single"
        assert entry.species == "Rose"
        assert entry.scientific_name == "Rosa chinensis"
        assert entry.confidence_tier == "ambiguous"
        assert entry.quality_grade == "A"
        assert entry.price_per_stem == pytest.approx(0.55)
        assert entry.photo_count == 1
        assert entry.agreement_fraction is None

    def test_a_lot_scan_uses_the_lot_only_columns(self, make_lot_result):
        inventory.log_lot_scan(make_lot_result(consensus_species="Sunflower", photo_count=3, agreement_fraction=0.667))
        (entry,) = inventory.list_recent()
        assert entry.mode == "lot"
        assert entry.species == "Sunflower"
        assert entry.confidence_tier is None
        assert entry.photo_count == 3
        assert entry.agreement_fraction == pytest.approx(0.667)

    def test_timestamps_are_utc_iso8601(self, make_identify_result):
        inventory.log_scan(make_identify_result())
        parsed = datetime.datetime.fromisoformat(inventory.list_recent()[0].scanned_at)
        assert parsed.utcoffset() == datetime.timedelta(0)

    def test_a_missing_price_is_stored_as_null(self, make_identify_result):
        inventory.log_scan(make_identify_result(price_per_stem=None, price_trend="unknown"))
        assert inventory.list_recent()[0].price_per_stem is None


class TestQueries:
    def test_newest_first_and_limit(self, make_identify_result):
        for name in ("Rose", "Tulip", "Carnation"):
            inventory.log_scan(make_identify_result(species=name))
        assert [e.species for e in inventory.list_recent()] == ["Carnation", "Tulip", "Rose"]
        assert [e.species for e in inventory.list_recent(limit=2)] == ["Carnation", "Tulip"]

    def test_species_counts_are_sorted_by_count(self, make_identify_result, make_lot_result):
        for name in ("Rose", "Tulip", "Rose"):
            inventory.log_scan(make_identify_result(species=name))
        inventory.log_lot_scan(make_lot_result(consensus_species="Rose"))
        counts = inventory.species_counts()
        assert counts == {"Rose": 3, "Tulip": 1}
        assert list(counts) == ["Rose", "Tulip"]

    def test_an_empty_log_is_empty_not_an_error(self):
        assert inventory.list_recent() == []
        assert inventory.species_counts() == {}


class TestStorage:
    def test_wal_mode_is_enabled(self, make_identify_result):
        inventory.log_scan(make_identify_result())
        with sqlite3.connect(db_path()) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    def test_the_mode_column_rejects_unknown_values(self, make_identify_result):
        inventory.log_scan(make_identify_result())
        with sqlite3.connect(db_path()) as conn, pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO scans (scanned_at, mode, species, quality_grade, price_trend) "
                "VALUES ('2026-01-01', 'bogus', 'Rose', 'A', 'flat')"
            )

    def test_the_parent_directory_is_created(self, tmp_path, monkeypatch, make_identify_result):
        nested = tmp_path / "does" / "not" / "exist" / "inventory.db"
        monkeypatch.setenv("INVENTORY_DB_PATH", str(nested))
        inventory.log_scan(make_identify_result())
        assert nested.exists()

    def test_the_default_path_is_used_when_the_env_var_is_unset(self, monkeypatch):
        monkeypatch.delenv("INVENTORY_DB_PATH")
        assert inventory._db_path() == inventory.DEFAULT_DB_PATH


class TestFailuresNeverBreakAScan:
    def test_logging_swallows_storage_errors_and_says_so(self, tmp_path, monkeypatch, capsys, make_identify_result):
        blocker = tmp_path / "a_file"
        blocker.write_text("x")
        monkeypatch.setenv("INVENTORY_DB_PATH", str(blocker / "inventory.db"))  # parent is a file
        inventory.log_scan(make_identify_result())  # must not raise
        assert "inventory: failed to log scan" in capsys.readouterr().out


class TestConcurrency:
    def test_threads_writing_at_once_lose_nothing(self, make_identify_result):
        """Milestone 10's ad-hoc check (8 threads x 10 writes), now permanent. Each
        call opens its own connection, so this exercises SQLite's own locking."""

        def worker(worker_id):
            for i in range(10):
                inventory.log_scan(make_identify_result(species=f"Species{worker_id}-{i}"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))
        assert len(inventory.list_recent(limit=1000)) == 80
