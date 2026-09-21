"""src/db.py: the shared SQLite connection, its transaction helper and its version guard."""

import sqlite3
import time
from contextlib import closing

import pytest

from src import db


def rows(conn, sql="SELECT species FROM scans"):
    return [r[0] for r in conn.execute(sql).fetchall()]


def insert(conn, species):
    conn.execute(
        "INSERT INTO scans (scanned_at, mode, species, quality_grade, price_trend) "
        "VALUES ('t', 'single', ?, 'A', 'flat')",
        (species,),
    )


class TestConnect:
    def test_it_creates_the_file_and_its_parent_directory(self, tmp_path, monkeypatch):
        target = tmp_path / "a" / "b" / "shared.db"
        monkeypatch.setenv("INVENTORY_DB_PATH", str(target))
        with closing(db.connect()):
            pass
        assert target.exists()

    def test_wal_mode_is_on_and_both_tables_exist(self):
        with closing(db.connect()) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"scans", "quota_counters"} <= tables

    def test_the_connection_is_autocommit_so_transactions_are_always_explicit(self):
        with closing(db.connect()) as conn:
            assert conn.isolation_level is None and not conn.in_transaction

    def test_the_busy_timeout_is_applied_and_configurable(self):
        with closing(db.connect()) as default, closing(db.connect(busy_timeout=2.5)) as short:
            assert default.execute("PRAGMA busy_timeout").fetchone()[0] == db.DEFAULT_BUSY_TIMEOUT_SECONDS * 1000
            assert short.execute("PRAGMA busy_timeout").fetchone()[0] == 2500

    def test_an_unusable_path_is_an_os_error(self, tmp_path, monkeypatch):
        blocker = tmp_path / "a_file"
        blocker.write_text("x")
        monkeypatch.setenv("INVENTORY_DB_PATH", str(blocker / "shared.db"))  # parent is a file
        with pytest.raises(OSError):
            db.connect()

    def test_a_failed_initialisation_closes_the_connection_it_opened(self, monkeypatch):
        opened = []
        real_connect = sqlite3.connect

        def spy(*args, **kwargs):
            opened.append(real_connect(*args, **kwargs))
            return opened[-1]

        def failing_init(_conn):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(db.sqlite3, "connect", spy)
        monkeypatch.setattr(db, "_initialize", failing_init)
        with pytest.raises(sqlite3.OperationalError):
            db.connect()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            opened[0].execute("SELECT 1")

    def test_the_default_path_is_used_when_the_variable_is_unset(self, monkeypatch):
        monkeypatch.delenv("INVENTORY_DB_PATH")
        assert db.db_path() == db.DEFAULT_DB_PATH


class TestTransaction:
    def test_a_clean_block_commits(self):
        with closing(db.connect()) as conn:
            with db.transaction(conn):
                insert(conn, "Rose")
            assert not conn.in_transaction
        with closing(db.connect()) as other:
            assert rows(other) == ["Rose"]

    def test_an_exception_rolls_everything_back_and_is_re_raised(self):
        with closing(db.connect()) as conn:
            with pytest.raises(RuntimeError, match="boom"), db.transaction(conn):
                insert(conn, "Rose")
                raise RuntimeError("boom")
            assert not conn.in_transaction
            assert rows(conn) == []

    def test_it_takes_the_write_lock_immediately(self):
        """BEGIN IMMEDIATE: a second writer is refused as soon as the first has begun,
        before it has written anything -- that is what removes read-then-write gaps."""
        with closing(db.connect()) as first, closing(db.connect(busy_timeout=0.05)) as second:
            with db.transaction(first):
                started = time.monotonic()
                with pytest.raises(sqlite3.OperationalError, match="locked"), db.transaction(second):
                    pass
                assert time.monotonic() - started < 2
            with db.transaction(second):  # released: now it works
                insert(second, "Rose")

    def test_readers_are_not_blocked_by_a_writer_in_wal_mode(self):
        with closing(db.connect()) as writer, closing(db.connect(busy_timeout=0.05)) as reader:
            with db.transaction(writer):
                insert(writer, "Rose")
                assert rows(reader) == []  # sees the last committed state, does not wait
            assert rows(reader) == ["Rose"]


class TestSqliteVersionGuard:
    def test_the_running_sqlite_is_accepted(self):
        db.require_sqlite()

    def test_the_minimum_is_accepted_and_older_is_refused_with_a_clear_message(self):
        db.require_sqlite(db.MINIMUM_SQLITE_VERSION)
        with pytest.raises(RuntimeError, match=r"SQLite >= 3\.35\.0.*3\.34\.9"):
            db.require_sqlite((3, 34, 9))

    def test_the_installed_sqlite_really_supports_upsert_returning(self):
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE t (k TEXT PRIMARY KEY, n INTEGER)")
            row = conn.execute(
                "INSERT INTO t VALUES ('a', 1) ON CONFLICT (k) DO UPDATE SET n = n + 1 RETURNING n"
            ).fetchone()
            assert row == (1,)
