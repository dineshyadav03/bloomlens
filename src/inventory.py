"""Persistent inventory/traceability log: every real scan (single or lot) is
recorded automatically -- species, quality, price, timestamp -- building a
running record across sessions, rather than treating each scan as a
one-off lookup (the original tutorial's "Inventory Scanner" concept).

SQLite, not a new service: a single-table, demo-scale, mostly-append
workload doesn't need a client-server database. `INVENTORY_DB_PATH` follows
the exact same env-var-overrides-a-local-default pattern src/vector_store.py
already uses for QDRANT_URL -- unset locally, set by docker-compose.yml so
the `web` and `api` containers share one file via a named volume.

Concurrency: SQLite's default rollback-journal mode lets a writer lock out
readers -- the same class of bug Qdrant's local/embedded mode hit for real
in Milestone 1. WAL (write-ahead logging) mode is built for exactly this
and is documented to work correctly across multiple processes sharing a
real local filesystem (true for a Docker named volume on one host, and for
local dev) -- see docs/RESEARCH.md for the fuller reasoning. Every function
here opens and closes its own short-lived connection rather than sharing
one across calls/threads, since sqlite3 connections aren't safe to share
across threads without extra care and the write frequency here is low
enough that per-call connection overhead doesn't matter.
"""

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.identify import IdentifyResult, LotResult

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "inventory.db")


class InventoryEntry(BaseModel):
    id: int
    scanned_at: str
    mode: Literal["single", "lot"]
    species: str
    scientific_name: str | None
    confidence_tier: str | None
    quality_grade: str
    price_per_stem: float | None
    price_trend: str
    photo_count: int
    agreement_fraction: float | None


def _db_path() -> str:
    return os.environ.get("INVENTORY_DB_PATH", DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    path = Path(_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at TEXT NOT NULL,
            mode TEXT NOT NULL CHECK (mode IN ('single', 'lot')),
            species TEXT NOT NULL,
            scientific_name TEXT,
            confidence_tier TEXT,
            quality_grade TEXT NOT NULL,
            price_per_stem REAL,
            price_trend TEXT NOT NULL,
            photo_count INTEGER NOT NULL DEFAULT 1,
            agreement_fraction REAL
        )
        """
    )
    return conn


def _insert(
    *,
    mode: Literal["single", "lot"],
    species: str,
    scientific_name: str | None,
    confidence_tier: str | None,
    quality_grade: str,
    price_per_stem: float | None,
    price_trend: str,
    photo_count: int,
    agreement_fraction: float | None,
) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO scans (
                    scanned_at, mode, species, scientific_name, confidence_tier,
                    quality_grade, price_per_stem, price_trend, photo_count, agreement_fraction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    mode,
                    species,
                    scientific_name,
                    confidence_tier,
                    quality_grade,
                    price_per_stem,
                    price_trend,
                    photo_count,
                    agreement_fraction,
                ),
            )
    except Exception as exc:  # noqa: BLE001 -- logging must never break a real scan result
        print(f"inventory: failed to log scan ({exc})")


def log_scan(result: IdentifyResult) -> None:
    _insert(
        mode="single",
        species=result.species,
        scientific_name=result.scientific_name,
        confidence_tier=result.confidence_tier,
        quality_grade=result.quality_grade,
        price_per_stem=result.price_per_stem,
        price_trend=result.price_trend,
        photo_count=1,
        agreement_fraction=None,
    )


def log_lot_scan(result: LotResult) -> None:
    _insert(
        mode="lot",
        species=result.consensus_species,
        scientific_name=result.scientific_name,
        confidence_tier=None,
        quality_grade=result.quality_grade,
        price_per_stem=result.price_per_stem,
        price_trend=result.price_trend,
        photo_count=result.photo_count,
        agreement_fraction=result.agreement_fraction,
    )


def list_recent(limit: int = 100) -> list[InventoryEntry]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [InventoryEntry(**dict(row)) for row in rows]


def species_counts() -> dict[str, int]:
    with _connect() as conn:
        rows = conn.execute("SELECT species, COUNT(*) FROM scans GROUP BY species ORDER BY COUNT(*) DESC").fetchall()
    return dict(rows)
