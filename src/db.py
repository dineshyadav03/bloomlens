"""The one SQLite file that the inventory log and the abuse-control counters share.

Why SQLite, and why one file: docker-compose runs `web` (Streamlit) and `api` as
separate processes/containers on one shared volume, so anything that must be shared
between them -- the scan log, and the rate/quota counters -- has to live in shared
storage, not in a process's memory. A single-host WAL file does that with no new
service. Its boundary is stated plainly: it is correct for processes on ONE host
sharing the volume; it is not a multi-replica store (see docs/SECURITY.md).

Connections here are in autocommit mode (`isolation_level=None`): nothing is ever
implicitly begun, so a transaction exists only where the code writes
`with transaction(conn):`. That makes lock behaviour explicit -- `BEGIN IMMEDIATE`
takes the write lock up front, so a check-then-write sequence cannot interleave with
another writer's.
"""

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "inventory.db")

# Wait up to 30 s for a lock instead of sqlite3's 5 s default. Callers that must not
# hang a web request (quota admission) pass a shorter one.
DEFAULT_BUSY_TIMEOUT_SECONDS = 30

# `INSERT ... ON CONFLICT DO UPDATE ... RETURNING` (atomic upsert-and-read) needs 3.35.
MINIMUM_SQLITE_VERSION = (3, 35, 0)

_INIT_ATTEMPTS = 8

SCHEMA = (
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
    """,
    # One row per (caller, kind of window, which window). `window` is the UTC epoch-minute
    # for kind='minute' and the UTC day number (days since 1970-01-01) for kind='day'.
    # `identity` is an API-key label or a random session id -- never an address, never a key.
    """
    CREATE TABLE IF NOT EXISTS quota_counters (
        identity TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('minute', 'day')),
        window INTEGER NOT NULL,
        used INTEGER NOT NULL CHECK (used >= 0),
        PRIMARY KEY (identity, kind, window)
    ) WITHOUT ROWID
    """,
)


def require_sqlite(version: tuple[int, ...] = sqlite3.sqlite_version_info) -> None:
    if tuple(version) < MINIMUM_SQLITE_VERSION:
        needed = ".".join(map(str, MINIMUM_SQLITE_VERSION))
        found = ".".join(map(str, version))
        raise RuntimeError(f"BloomLens needs SQLite >= {needed} (UPSERT ... RETURNING); this Python has {found}.")


require_sqlite()  # fail at import, not on the first request


def db_path() -> str:
    return os.environ.get("INVENTORY_DB_PATH", DEFAULT_DB_PATH)


def _initialize(conn: sqlite3.Connection) -> None:
    """Switch to WAL and make sure the tables exist.

    On a brand-new file, several connections doing this at once (threads, or the
    `web` and `api` processes on their first write) race: converting to WAL needs
    exclusive access, and in that situation SQLite reports "database is locked"
    immediately, skipping the busy timeout (waiting could deadlock). Found by a
    concurrency test that lost rows only on a fresh database; retrying just this
    step with a short backoff fixes it. Once the file is in WAL mode with the
    tables present, nothing here can conflict."""
    delay = 0.05
    for attempt in range(_INIT_ATTEMPTS):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            for statement in SCHEMA:
                conn.execute(statement)
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == _INIT_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def connect(*, busy_timeout: float = DEFAULT_BUSY_TIMEOUT_SECONDS) -> sqlite3.Connection:
    """A new autocommit connection with WAL on, the schema present and a busy timeout.
    The caller closes it (`contextlib.closing`): sqlite3's own context manager only
    manages a transaction, it never closes the connection."""
    path = Path(db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=busy_timeout, isolation_level=None)
    try:
        _initialize(conn)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """`BEGIN IMMEDIATE` ... `COMMIT`, or `ROLLBACK` if the body raises. IMMEDIATE takes
    the write lock at the start (waiting up to the busy timeout), so everything inside
    is serialised against every other writer -- no read-then-write interleaving."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
