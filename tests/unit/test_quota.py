"""src/quota.py: exact limits, UTC windows and rollover, all-or-nothing admission, purge,
fail-closed behaviour and the concurrency gate. Time is injected, never slept.

The many-processes guarantee (exactly L admitted, never L+1) is in
test_quota_multiprocess.py."""

import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone

import pytest

from src import db, quota
from src.quota import Busy, ConcurrencyGate, Decision, Limits, QuotaUnavailable, admit
from tests.conftest import REPO_ROOT

T0 = datetime(2026, 9, 21, 12, 0, 30, tzinfo=UTC)
ROOMY = 10**6


def limits(per_minute=ROOMY, per_day=ROOMY, global_per_day=ROOMY):
    return Limits(per_minute=per_minute, per_day=per_day, global_per_day=global_per_day)


def counters():
    with closing(db.connect()) as conn:
        return {(who, kind, window): used for who, kind, window, used in conn.execute("SELECT * FROM quota_counters")}


@pytest.fixture(autouse=True)
def fresh_purge_clock(monkeypatch):
    monkeypatch.setattr(quota, "_last_purge", float("-inf"))


class TestExactLimits:
    def test_exactly_the_limit_is_admitted_then_it_is_refused(self):
        lim = limits(per_minute=3)
        assert [admit("a", now=T0, limits=lim).allowed for _ in range(5)] == [True, True, True, False, False]

    def test_a_refusal_names_the_limit_that_refused_it(self):
        assert admit("a", now=T0, limits=limits(per_minute=1)).allowed
        refused = admit("a", now=T0, limits=limits(per_minute=1))
        assert (refused.allowed, refused.scope) == (False, "minute")

    def test_callers_are_counted_separately(self):
        lim = limits(per_minute=1)
        assert admit("api:alice", now=T0, limits=lim).allowed
        assert not admit("api:alice", now=T0, limits=lim).allowed
        assert admit("api:bob", now=T0, limits=lim).allowed

    def test_a_refusal_consumes_nothing(self):
        lim = limits(per_minute=2)
        for _ in range(10):
            admit("a", now=T0, limits=lim)
        assert set(counters().values()) == {2}  # capped at the limit, not at ten


class TestMinuteWindow:
    LIM = limits(per_minute=3)

    def test_the_window_is_a_utc_clock_minute_not_a_sliding_one(self):
        for _ in range(3):
            admit("a", now=datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC), limits=self.LIM)
        assert not admit("a", now=datetime(2026, 9, 21, 12, 0, 59, 999999, tzinfo=UTC), limits=self.LIM).allowed
        assert admit("a", now=datetime(2026, 9, 21, 12, 1, 0, tzinfo=UTC), limits=self.LIM).allowed

    @pytest.mark.parametrize(
        "second, microsecond, expected",
        [(0, 0, 60), (30, 0, 30), (59, 0, 1), (59, 500_000, 1), (59, 999_999, 1), (1, 1, 59)],
    )
    def test_retry_after_is_the_time_left_in_the_minute_rounded_up(self, second, microsecond, expected):
        now = datetime(2026, 9, 21, 12, 0, second, microsecond, tzinfo=UTC)
        admit("a", now=now, limits=limits(per_minute=1))
        assert admit("a", now=now, limits=limits(per_minute=1)).retry_after == expected

    def test_waiting_the_stated_retry_after_really_gets_you_in(self):
        lim = limits(per_minute=1)
        admit("a", now=T0, limits=lim)
        refused = admit("a", now=T0, limits=lim)
        assert admit("a", now=T0 + timedelta(seconds=refused.retry_after), limits=lim).allowed


class TestDayWindowAndUtcRollover:
    LIM = limits(per_day=2)

    def test_the_day_quota_rolls_over_at_exactly_00_00_utc(self):
        before = datetime(2026, 9, 21, 23, 59, 59, 999999, tzinfo=UTC)
        for _ in range(2):
            admit("a", now=before, limits=self.LIM)
        assert not admit("a", now=before, limits=self.LIM).allowed
        assert admit("a", now=datetime(2026, 9, 22, 0, 0, 0, tzinfo=UTC), limits=self.LIM).allowed

    def test_retry_after_is_the_time_to_the_next_utc_midnight(self):
        now = datetime(2026, 9, 21, 23, 0, 0, tzinfo=UTC)
        for _ in range(3):
            refused = admit("a", now=now, limits=self.LIM)
        assert (refused.scope, refused.retry_after) == ("day", 3600)

    def test_the_month_and_year_boundaries_roll_over_too(self):
        end_of_year = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)
        for _ in range(2):
            admit("a", now=end_of_year, limits=self.LIM)
        refused = admit("a", now=end_of_year, limits=self.LIM)
        assert refused.retry_after == 1
        assert admit("a", now=end_of_year + timedelta(seconds=1), limits=self.LIM).allowed

    def test_a_leap_day_is_a_normal_day(self):
        leap = datetime(2028, 2, 29, 12, 0, tzinfo=UTC)
        assert admit("a", now=leap, limits=limits(per_day=1)).allowed
        assert not admit("a", now=leap, limits=limits(per_day=1)).allowed
        assert admit("a", now=datetime(2028, 3, 1, tzinfo=UTC), limits=limits(per_day=1)).allowed

    def test_the_day_is_the_utc_day_whatever_zone_the_clock_is_in(self):
        """05:29 in +05:30 is 23:59 UTC the day before; 05:30 is midnight UTC."""
        ist = timezone(timedelta(hours=5, minutes=30))
        last_local_minute_of_utc_day = datetime(2026, 9, 22, 5, 29, tzinfo=ist)  # = 2026-09-21 23:59 UTC
        for _ in range(2):
            admit("a", now=last_local_minute_of_utc_day, limits=self.LIM)
        assert not admit("a", now=last_local_minute_of_utc_day, limits=self.LIM).allowed
        assert admit("a", now=datetime(2026, 9, 22, 5, 30, tzinfo=ist), limits=self.LIM).allowed  # 00:00 UTC

    def test_a_negative_offset_zone_that_is_still_yesterday_locally_is_today_in_utc(self):
        pst = timezone(timedelta(hours=-8))
        evening_local = datetime(2026, 9, 21, 16, 0, tzinfo=pst)  # = 2026-09-22 00:00 UTC
        for _ in range(2):
            admit("a", now=evening_local, limits=self.LIM)
        assert admit("a", now=datetime(2026, 9, 21, 23, 0, tzinfo=UTC), limits=self.LIM).allowed  # previous UTC day

    def test_a_naive_clock_is_a_programming_error_not_silently_local_time(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            admit("a", now=datetime(2026, 9, 21, 12, 0), limits=limits())


class TestGlobalCeiling:
    def test_the_ceiling_is_shared_by_every_caller(self):
        lim = limits(global_per_day=3)
        results = [admit(f"ui:{n}", now=T0, limits=lim) for n in range(5)]
        assert [r.allowed for r in results] == [True, True, True, False, False]
        assert results[3].scope == "global"

    def test_the_global_counter_resets_at_utc_midnight_too(self):
        lim = limits(global_per_day=1)
        admit("a", now=T0, limits=lim)
        assert not admit("b", now=T0, limits=lim).allowed
        assert admit("b", now=datetime(2026, 9, 22, 0, 0, 1, tzinfo=UTC), limits=lim).allowed


class TestAllOrNothing:
    def test_a_refusal_by_the_day_limit_leaves_the_minute_counter_alone(self):
        lim = limits(per_day=1)
        assert admit("a", now=T0, limits=lim).allowed
        assert admit("a", now=T0, limits=lim).scope == "day"
        minute_used = next(v for (who, kind, _), v in counters().items() if (who, kind) == ("a", "minute"))
        assert minute_used == 1  # the refused attempt's minute increment was rolled back

    def test_a_refusal_by_the_global_ceiling_leaves_the_callers_counters_alone(self):
        lim = limits(global_per_day=1)
        admit("first", now=T0, limits=lim)
        assert admit("second", now=T0, limits=lim).scope == "global"
        assert not any(who == "second" for (who, _, _) in counters())

    def test_admission_touches_exactly_three_counters(self):
        admit("a", now=T0, limits=limits())
        assert {(who, kind) for (who, kind, _) in counters()} == {("a", "minute"), ("a", "day"), ("*", "day")}


class TestPurge:
    def seed(self, rows):
        with closing(db.connect()) as conn, db.transaction(conn):
            conn.executemany("INSERT INTO quota_counters VALUES (?, ?, ?, 1)", rows)

    def test_expired_windows_go_and_live_ones_stay_exactly_at_the_boundary(self):
        minute, day = quota._minute_window(T0), quota._day_window(T0)
        self.seed(
            [
                ("a", "minute", minute - 60),  # an hour old: kept
                ("a", "minute", minute - 61),  # older than an hour: purged
                ("a", "day", day - 1),  # yesterday: kept (48 h retention)
                ("a", "day", day - 2),  # the day before: purged
                ("a", "minute", minute),
                ("a", "day", day),
            ]
        )
        with closing(db.connect()) as conn, db.transaction(conn):
            assert quota.purge_expired(conn, T0) == 2
        assert set(counters()) == {
            ("a", "minute", minute - 60),
            ("a", "day", day - 1),
            ("a", "minute", minute),
            ("a", "day", day),
        }

    def test_admission_purges_opportunistically_but_at_most_once_an_hour(self, monkeypatch):
        old_minute = quota._minute_window(T0) - 500
        self.seed([("gone", "minute", old_minute)])
        admit("a", now=T0, limits=limits())
        assert ("gone", "minute", old_minute) not in counters()

        self.seed([("later", "minute", old_minute)])
        admit("a", now=T0, limits=limits())  # within the hour: no second purge
        assert ("later", "minute", old_minute) in counters()

        real = time.monotonic
        monkeypatch.setattr(quota.time, "monotonic", lambda: real() + 3601)
        admit("a", now=T0, limits=limits())
        assert ("later", "minute", old_minute) not in counters()


class TestFailClosed:
    def test_an_unusable_store_refuses_rather_than_admitting(self, tmp_path, monkeypatch):
        blocker = tmp_path / "a_file"
        blocker.write_text("x")
        monkeypatch.setenv("INVENTORY_DB_PATH", str(blocker / "shared.db"))
        with pytest.raises(QuotaUnavailable):
            admit("a", now=T0, limits=limits())

    def test_a_path_that_is_a_directory_refuses_too(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INVENTORY_DB_PATH", str(tmp_path))
        with pytest.raises(QuotaUnavailable):
            admit("a", now=T0, limits=limits())

    def test_a_lock_that_outlasts_the_timeout_is_a_fast_refusal_not_a_hang_or_an_admission(self):
        with closing(db.connect()) as holder, db.transaction(holder):
            started = time.monotonic()
            with pytest.raises(QuotaUnavailable):
                admit("a", now=T0, limits=limits(), busy_timeout=0.2)
            assert time.monotonic() - started < 3
        assert counters() == {}  # nothing was consumed
        assert admit("a", now=T0, limits=limits()).allowed  # and it recovers once the lock is gone

    def test_the_log_line_names_the_error_type_only(self, tmp_path, monkeypatch, caplog):
        blocker = tmp_path / "secret-dir-name"
        blocker.write_text("x")
        monkeypatch.setenv("INVENTORY_DB_PATH", str(blocker / "shared.db"))
        with caplog.at_level(logging.ERROR, logger="bloomlens.quota"), pytest.raises(QuotaUnavailable):
            admit("api:alice", now=T0, limits=limits())
        assert "unavailable" in caplog.text and "secret-dir-name" not in caplog.text


class TestConfiguration:
    def test_defaults(self):
        assert quota.limits_from_env() == Limits(per_minute=6, per_day=100, global_per_day=500)

    def test_values_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "12")
        monkeypatch.setenv("BLOOMLENS_QUOTA_PER_DAY", "40")
        monkeypatch.setenv("BLOOMLENS_GLOBAL_QUOTA_PER_DAY", "900")
        assert quota.limits_from_env() == Limits(12, 40, 900)

    @pytest.mark.parametrize("bad", ["0", "-3", "many", "2.5", "1e3"])
    def test_an_invalid_value_falls_back_to_the_default_and_says_so(self, monkeypatch, caplog, bad):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", bad)
        with caplog.at_level(logging.WARNING, logger="bloomlens.quota"):
            assert quota.limits_from_env().per_minute == 6
        assert "BLOOMLENS_RATE_PER_MINUTE" in caplog.text

    def test_a_blank_value_is_the_default_without_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "  ")
        with caplog.at_level(logging.WARNING, logger="bloomlens.quota"):
            assert quota.limits_from_env().per_minute == 6
        assert caplog.text == ""

    def test_admit_reads_the_environment_when_no_limits_are_passed(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")
        assert admit("a", now=T0).allowed and not admit("a", now=T0).allowed


class TestDecisionMessages:
    def test_each_scope_says_when_to_come_back_without_naming_internals(self):
        minute = Decision(False, "minute", 42).message
        assert "42 seconds" in minute
        assert "00:00 UTC" in Decision(False, "day", 100).message
        overall = Decision(False, "global", 100).message
        assert "capacity" in overall and "00:00 UTC" in overall
        assert Decision(True).message == ""
        for text in (minute, Decision(False, "day", 1).message):
            assert "sqlite" not in text.lower() and "counter" not in text.lower()


class TestConcurrencyGate:
    def test_it_admits_up_to_the_slot_count_then_says_busy(self):
        gate = ConcurrencyGate(2)
        with gate.slot(), gate.slot():
            with pytest.raises(Busy), gate.slot():
                pass

    def test_a_finished_run_frees_its_slot(self):
        gate = ConcurrencyGate(1)
        with gate.slot():
            pass
        with gate.slot():
            pass

    def test_a_failing_run_still_frees_its_slot(self):
        gate = ConcurrencyGate(1)
        with pytest.raises(RuntimeError), gate.slot():
            raise RuntimeError("pipeline blew up")
        with gate.slot():
            pass

    def test_it_never_lets_more_than_the_limit_run_at_once(self):
        gate = ConcurrencyGate(3)
        inside, peak, lock = 0, 0, threading.Lock()
        refused = []

        def run():
            nonlocal inside, peak
            try:
                with gate.slot():
                    with lock:
                        inside += 1
                        peak = max(peak, inside)
                    time.sleep(0.02)
                    with lock:
                        inside -= 1
            except Busy:
                refused.append(1)

        threads = [threading.Thread(target=run) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak <= 3 and refused  # the limit held, and some callers really were turned away

    def test_the_process_gate_is_sized_from_the_environment_once(self, monkeypatch):
        quota.reset_gate()
        monkeypatch.setenv("BLOOMLENS_MAX_CONCURRENT", "5")
        assert quota.gate().slots == 5
        monkeypatch.setenv("BLOOMLENS_MAX_CONCURRENT", "9")
        assert quota.gate() is quota.gate() and quota.gate().slots == 5
        quota.reset_gate()
        monkeypatch.delenv("BLOOMLENS_MAX_CONCURRENT")
        assert quota.gate().slots == quota.DEFAULT_CONCURRENCY
        quota.reset_gate()


def test_the_module_stays_light_enough_for_cheap_worker_processes():
    """The multi-process tests start many interpreters; importing torch in each would
    make them take minutes. quota and db must import only the standard library."""
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import src.quota; print('torch' in sys.modules or 'PIL' in sys.modules)"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
