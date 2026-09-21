"""src/retention.py and scripts/purge_data.py: each kind of data is deleted exactly when its
retention ends, nothing earlier, and the purge runs opportunistically but not on every scan."""

import os
import subprocess
import sys
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from src import db, quota, retention, telemetry
from tests.conftest import REPO_ROOT
from tests.unit.test_telemetry import NOW, insert

DAY = timedelta(days=1)


def add_scan(conn, when):
    conn.execute(
        "INSERT INTO scans (scanned_at, mode, species, quality_grade, price_trend) "
        "VALUES (?, 'single', 'Rose', 'A', 'flat')",
        (when.isoformat(),),
    )


def counts():
    with closing(db.connect()) as conn:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("scan_metrics", "scans", "quota_counters")
        }


def purge(now=NOW):
    with closing(db.connect()) as conn, db.transaction(conn):
        return retention.purge_all(conn, now)


class TestSettings:
    def test_defaults_are_thirty_days_of_telemetry_and_a_year_of_inventory(self):
        assert retention.settings() == retention.Retention(metrics_days=30, inventory_days=365)

    def test_they_are_configurable(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_METRICS_RETENTION_DAYS", "7")
        monkeypatch.setenv("BLOOMLENS_INVENTORY_RETENTION_DAYS", "90")
        assert retention.settings() == retention.Retention(7, 90)

    @pytest.mark.parametrize("bad", ["0", "-1", "forever", "1.5", ""])
    def test_a_bad_value_never_means_keep_forever(self, monkeypatch, bad):
        monkeypatch.setenv("BLOOMLENS_METRICS_RETENTION_DAYS", bad)
        assert retention.settings().metrics_days == 30


class TestPurgeAll:
    def test_telemetry_older_than_the_window_goes_and_the_boundary_is_exact(self):
        with closing(db.connect()) as conn, db.transaction(conn):
            insert(conn, recorded_at=(NOW - timedelta(days=30, microseconds=1)).isoformat())  # just past
            insert(conn, recorded_at=(NOW - timedelta(days=30)).isoformat())  # exactly at the boundary: kept
            insert(conn, recorded_at=(NOW - DAY).isoformat())
            insert(conn, recorded_at=NOW.isoformat())
        assert purge()["scan_metrics"] == 1
        assert counts()["scan_metrics"] == 3

    def test_the_inventory_is_kept_for_a_year(self):
        with closing(db.connect()) as conn, db.transaction(conn):
            add_scan(conn, NOW - timedelta(days=366))
            add_scan(conn, NOW - timedelta(days=365))
            add_scan(conn, NOW - timedelta(days=100))
        assert purge()["scans"] == 1
        assert counts()["scans"] == 2

    def test_quota_counters_follow_their_own_shorter_windows(self):
        minute, day = quota._minute_window(NOW), quota._day_window(NOW)
        stale_and_live = [
            ("a", "minute", minute - 61),
            ("a", "minute", minute - 60),
            ("a", "day", day - 2),
            ("a", "day", day - 1),
        ]
        with closing(db.connect()) as conn, db.transaction(conn):
            conn.executemany("INSERT INTO quota_counters VALUES (?, ?, ?, 1)", stale_and_live)
        assert purge()["quota_counters"] == 2

    def test_a_shorter_configured_window_purges_sooner(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_METRICS_RETENTION_DAYS", "7")
        with closing(db.connect()) as conn, db.transaction(conn):
            insert(conn, recorded_at=(NOW - timedelta(days=8)).isoformat())
            insert(conn, recorded_at=(NOW - timedelta(days=6)).isoformat())
        assert purge()["scan_metrics"] == 1

    def test_nothing_else_is_touched(self):
        with closing(db.connect()) as conn, db.transaction(conn):
            insert(conn, recorded_at=NOW.isoformat())
            add_scan(conn, NOW)
        assert purge() == {"scan_metrics": 0, "scans": 0, "quota_counters": 0}
        assert counts()["scan_metrics"] == counts()["scans"] == 1

    def test_the_immutable_price_history_is_never_purged(self):
        telemetry.record(telemetry.ScanRecorder("single", 1, False), None, now=NOW)
        with closing(db.connect()) as conn, db.transaction(conn):
            conn.execute("UPDATE scan_metrics SET recorded_at = ?", ((NOW - timedelta(days=99)).isoformat(),))
        purge()
        with closing(db.connect()) as conn:
            assert conn.execute("SELECT COUNT(*) FROM pricing_versions").fetchone()[0] >= 1

    def test_timestamps_in_the_same_format_compare_correctly_across_a_year_boundary(self):
        with closing(db.connect()) as conn, db.transaction(conn):
            insert(conn, recorded_at=datetime(2025, 12, 1, tzinfo=UTC).isoformat())
            insert(conn, recorded_at=datetime(2026, 1, 15, tzinfo=UTC).isoformat())
        purge(datetime(2026, 1, 20, tzinfo=UTC))  # 30 days back = 2025-12-21
        assert counts()["scan_metrics"] == 1


class TestOpportunistic:
    def test_recording_a_scan_purges_expired_data_but_only_once_an_hour(self, monkeypatch):
        old = (NOW - timedelta(days=45)).isoformat()
        with closing(db.connect()) as conn, db.transaction(conn):
            insert(conn, recorded_at=old)
        telemetry.record(telemetry.ScanRecorder("single", 1, False), None, now=NOW)
        assert counts()["scan_metrics"] == 1  # the old row went; the new one stayed

        with closing(db.connect()) as conn, db.transaction(conn):
            insert(conn, recorded_at=old)
        telemetry.record(telemetry.ScanRecorder("single", 1, False), None, now=NOW)
        assert counts()["scan_metrics"] == 3  # within the hour: no second purge

        real = time.monotonic
        monkeypatch.setattr(retention.time, "monotonic", lambda: real() + 3601)
        telemetry.record(telemetry.ScanRecorder("single", 1, False), None, now=NOW)
        assert counts()["scan_metrics"] == 3  # the stale one went, the new one arrived


class TestPurgeScript:
    """scripts/purge_data.py as a real subprocess against the (temporary) database the
    hermetic fixture points INVENTORY_DB_PATH at."""

    def run_script(self, **env):
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "purge_data.py")],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT), **env},
            timeout=120,
        )

    def test_it_purges_a_real_database_and_reports_what_it_removed(self):
        now = datetime.now(UTC)
        with closing(db.connect()) as conn, db.transaction(conn):
            insert(conn, recorded_at=(now - timedelta(days=45)).isoformat())
            insert(conn, recorded_at=now.isoformat())
            add_scan(conn, now - timedelta(days=400))
        result = self.run_script()
        assert result.returncode == 0, result.stderr
        assert "scan_metrics: 1 rows removed" in result.stdout and "scans: 1 rows removed" in result.stdout
        assert "30 days" in result.stdout and "365 days" in result.stdout
        assert counts()["scan_metrics"] == 1 and counts()["scans"] == 0

    def test_it_honours_the_configured_retention(self):
        with closing(db.connect()) as conn, db.transaction(conn):
            insert(conn, recorded_at=(datetime.now(UTC) - timedelta(days=10)).isoformat())
        result = self.run_script(BLOOMLENS_METRICS_RETENTION_DAYS="7")
        assert "7 days" in result.stdout and "scan_metrics: 1 rows removed" in result.stdout

    def test_it_is_safe_on_an_empty_database(self):
        result = self.run_script()
        assert result.returncode == 0 and "0 rows removed" in result.stdout
