"""How long each kind of data lives, and the purge that enforces it.

| data                                      | kept                                          |
|-------------------------------------------|-----------------------------------------------|
| scan telemetry (`scan_metrics`)           | 30 days   (BLOOMLENS_METRICS_RETENTION_DAYS)  |
| inventory log (`scans`)                   | 365 days  (BLOOMLENS_INVENTORY_RETENTION_DAYS)|
| daily quota counters                      | 48 hours  (fixed, src/quota.py)               |
| per-minute rate windows                   | 1 hour    (fixed, src/quota.py)               |
| photos, prompts, model answers            | never stored                                  |

Purging runs opportunistically (at most hourly per process, when a scan is recorded) and
on demand with `scripts/purge_data.py`. Timestamps are ISO-8601 UTC strings in one format,
so comparing them as text compares them as times.
"""

import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from src import quota

_PURGE_EVERY_SECONDS = 3600
_last_purge = float("-inf")
_purge_lock = threading.Lock()


@dataclass(frozen=True)
class Retention:
    metrics_days: int = 30
    inventory_days: int = 365


def settings() -> Retention:
    defaults = Retention()
    return Retention(
        metrics_days=quota.positive_int_from_env("BLOOMLENS_METRICS_RETENTION_DAYS", defaults.metrics_days),
        inventory_days=quota.positive_int_from_env("BLOOMLENS_INVENTORY_RETENTION_DAYS", defaults.inventory_days),
    )


def purge_all(conn: sqlite3.Connection, now: datetime) -> dict[str, int]:
    """Delete everything past its retention. Run inside a transaction. Returns rows removed per table."""
    keep = settings()
    metrics = conn.execute(
        "DELETE FROM scan_metrics WHERE recorded_at < ?", ((now - timedelta(days=keep.metrics_days)).isoformat(),)
    ).rowcount
    inventory = conn.execute(
        "DELETE FROM scans WHERE scanned_at < ?", ((now - timedelta(days=keep.inventory_days)).isoformat(),)
    ).rowcount
    return {"scan_metrics": metrics, "scans": inventory, "quota_counters": quota.purge_expired(conn, now)}


def purge_if_due(conn: sqlite3.Connection, now: datetime) -> dict[str, int] | None:
    """purge_all, but at most once an hour per process. None when it was not due."""
    global _last_purge
    with _purge_lock:
        if time.monotonic() - _last_purge < _PURGE_EVERY_SECONDS:
            return None
        _last_purge = time.monotonic()
    return purge_all(conn, now)
