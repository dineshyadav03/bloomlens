"""Shared abuse controls: a per-caller rate limit and daily quota, a global daily
ceiling, and a per-process concurrency gate.

The counters live in the shared SQLite file (src/db.py), not in memory: the API and the
Streamlit app are separate processes, and an in-memory limiter only protects the process
it lives in. Correct for processes on ONE host sharing that file; it is not a
multi-replica store (docs/SECURITY.md says so).

Admission is one `BEGIN IMMEDIATE` transaction that raises three counters together --
the caller's current UTC minute, the caller's current UTC day, and the global day -- with
an atomic `INSERT .. ON CONFLICT DO UPDATE .. WHERE used < limit RETURNING`. Either all
three are raised and the request is admitted, or none are (a denial rolls the whole
transaction back), and there is no read-then-write gap for two processes to slip through.

Identity is an API-key *label* (`api:<label>`) or a random Streamlit session id
(`ui:<id>`): never an address, never a secret. Windows are UTC, so a daily quota rolls over
at 00:00 UTC whatever the server's local zone. Nothing here refunds: a request that was
admitted and then failed at the provider still cost a slot, and the docs say so. If the
counter store cannot be reached, admission fails CLOSED (QuotaUnavailable).
"""

import logging
import math
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from src import db

logger = logging.getLogger("bloomlens.quota")

GLOBAL_IDENTITY = "*"
_EPOCH_DAY = date(1970, 1, 1)

# Retention of expired counters (purged opportunistically, at most hourly per process).
MINUTE_WINDOW_RETENTION_MINUTES = 60
DAY_WINDOW_RETENTION_DAYS = 2  # today and yesterday: 48 h
_PURGE_EVERY_SECONDS = 3600

# Short on purpose: admission is on the request path, so a stuck lock must become a fast
# 503, not a hung request.
ADMISSION_BUSY_TIMEOUT_SECONDS = 5.0

Scope = Literal["minute", "day", "global"]


@dataclass(frozen=True)
class Limits:
    per_minute: int = 6
    per_day: int = 100
    global_per_day: int = 500


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        logger.warning("%s must be a whole number >= 1; using the default (%d).", name, default)
        return default
    return value


def limits_from_env() -> Limits:
    defaults = Limits()
    return Limits(
        per_minute=_positive_int_from_env("BLOOMLENS_RATE_PER_MINUTE", defaults.per_minute),
        per_day=_positive_int_from_env("BLOOMLENS_QUOTA_PER_DAY", defaults.per_day),
        global_per_day=_positive_int_from_env("BLOOMLENS_GLOBAL_QUOTA_PER_DAY", defaults.global_per_day),
    )


@dataclass(frozen=True)
class Decision:
    allowed: bool
    scope: Scope | None = None  # which limit refused it
    retry_after: int = 0  # whole seconds until that window ends (>= 1 when refused)

    @property
    def message(self) -> str:
        if self.allowed:
            return ""
        if self.scope == "minute":
            return f"Too many requests. Try again in {self.retry_after} seconds."
        if self.scope == "day":
            return "Your daily quota is used up. It resets at 00:00 UTC."
        return "The service's daily capacity is used up. It resets at 00:00 UTC."


class QuotaUnavailable(Exception):
    """The counters could not be read or written. Callers must refuse the request."""


class Busy(Exception):
    """Every concurrency slot in this process is in use."""


class _Refused(Exception):
    def __init__(self, scope: Scope):
        self.scope = scope


def _utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware (windows are UTC)")
    return now.astimezone(UTC)


def _minute_window(now: datetime) -> int:
    return int((now - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() // 60)


def _day_window(now: datetime) -> int:
    return (now.date() - _EPOCH_DAY).days


def _seconds_until_next_minute(now: datetime) -> int:
    seconds_into_minute = now.second + now.microsecond / 1_000_000
    return max(1, math.ceil(60 - seconds_into_minute))


def _seconds_until_next_utc_midnight(now: datetime) -> int:
    midnight = datetime(now.year, now.month, now.day, tzinfo=UTC) + timedelta(days=1)
    return max(1, math.ceil((midnight - now).total_seconds()))


_ADMIT_ONE = """
INSERT INTO quota_counters (identity, kind, window, used) VALUES (?, ?, ?, 1)
ON CONFLICT (identity, kind, window) DO UPDATE SET used = used + 1 WHERE used < ?
RETURNING used
"""

_last_purge = float("-inf")
_purge_lock = threading.Lock()


def purge_expired(conn: sqlite3.Connection, now: datetime) -> int:
    """Delete counters whose window is over (minute windows older than an hour, day
    windows older than yesterday). Returns how many rows went."""
    cursor = conn.execute(
        "DELETE FROM quota_counters WHERE (kind = 'minute' AND window < ?) OR (kind = 'day' AND window < ?)",
        (_minute_window(now) - MINUTE_WINDOW_RETENTION_MINUTES, _day_window(now) - (DAY_WINDOW_RETENTION_DAYS - 1)),
    )
    return cursor.rowcount


def _purge_if_due(conn: sqlite3.Connection, now: datetime) -> None:
    global _last_purge
    with _purge_lock:
        if time.monotonic() - _last_purge < _PURGE_EVERY_SECONDS:
            return
        _last_purge = time.monotonic()
    purge_expired(conn, now)


def admit(
    identity: str,
    *,
    now: datetime | None = None,
    limits: Limits | None = None,
    busy_timeout: float | None = None,
) -> Decision:
    """Admit one request from `identity`, or refuse it. All-or-nothing: a refusal leaves
    every counter as it was. Raises QuotaUnavailable if the store fails (fail closed)."""
    now = _utc(now)
    limits = limits or limits_from_env()
    busy_timeout = ADMISSION_BUSY_TIMEOUT_SECONDS if busy_timeout is None else busy_timeout
    minute, day = _minute_window(now), _day_window(now)
    checks: tuple[tuple[Scope, str, str, int, int], ...] = (
        ("minute", identity, "minute", minute, limits.per_minute),
        ("day", identity, "day", day, limits.per_day),
        ("global", GLOBAL_IDENTITY, "day", day, limits.global_per_day),
    )
    try:
        with closing(db.connect(busy_timeout=busy_timeout)) as conn:
            try:
                with db.transaction(conn):
                    _purge_if_due(conn, now)
                    for scope, who, kind, window, limit in checks:
                        if conn.execute(_ADMIT_ONE, (who, kind, window, limit)).fetchone() is None:
                            raise _Refused(scope)  # rolls back the counters raised so far
            except _Refused as refused:
                seconds = (
                    _seconds_until_next_minute(now)
                    if refused.scope == "minute"
                    else _seconds_until_next_utc_midnight(now)
                )
                return Decision(allowed=False, scope=refused.scope, retry_after=seconds)
    except (sqlite3.Error, OSError) as exc:
        logger.error("quota store unavailable (%s); refusing the request", type(exc).__name__)
        raise QuotaUnavailable from exc
    return Decision(allowed=True)


class ConcurrencyGate:
    """At most `slots` pipeline runs at once *in this process*. This bounds local CPU and
    memory; it is deliberately per-process and is not a quota (the counters are)."""

    def __init__(self, slots: int):
        self.slots = slots
        self._semaphore = threading.BoundedSemaphore(slots)

    @contextmanager
    def slot(self) -> Iterator[None]:
        if not self._semaphore.acquire(blocking=False):
            raise Busy
        try:
            yield
        finally:
            self._semaphore.release()


_gate: ConcurrencyGate | None = None
_gate_lock = threading.Lock()
DEFAULT_CONCURRENCY = 2


def gate() -> ConcurrencyGate:
    """This process's gate, sized once from BLOOMLENS_MAX_CONCURRENT (default 2)."""
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = ConcurrencyGate(_positive_int_from_env("BLOOMLENS_MAX_CONCURRENT", DEFAULT_CONCURRENCY))
        return _gate


def reset_gate() -> None:
    """Forget the cached gate (tests, or after changing the environment)."""
    global _gate
    with _gate_lock:
        _gate = None
