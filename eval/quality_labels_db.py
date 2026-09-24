"""Storage for tools/label_quality.py's blinded rater submissions.

Deliberately a SEPARATE SQLite file from src/db.py's shared inventory.db: that file is the
production app's runtime store (scans, quota counters, telemetry), shared between the `web` and
`api` containers; this one belongs to an offline labeling tool no deployed process touches. Kept
apart so neither can accidentally read or corrupt the other. The WAL + busy-timeout +
`BEGIN IMMEDIATE` pattern itself is copied from src/db.py rather than reinvented, since it's
already the project's proven answer to "more than one process may write this file".

Per docs/quality/PROTOCOL.md section 3 (blinding) and section 5 (adjudication): submissions are
**append-only** -- a correction is a new row, never an UPDATE of an old one, so the full history a
rater actually produced is always recoverable, and nothing here can silently overwrite what a
rater submitted. `latest_labels()` resolves "the label that counts" as the most recent row per
(rater_id, item_id) without deleting anything older.
"""

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "quality_labels.db")
DEFAULT_BUSY_TIMEOUT_SECONDS = 30
_INIT_ATTEMPTS = 8

GRADES = ("A", "B", "C", "CANNOT_GRADE")

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS raters (
        rater_id TEXT PRIMARY KEY,
        experience_note TEXT NOT NULL,
        registered_at_utc TEXT NOT NULL
    )
    """,
    # Append-only by convention (never UPDATEd/DELETEd by this module -- see module docstring).
    # `calibration` marks the fixed practice round (docs/quality/PROTOCOL.md section 2): its
    # labels are never included in agreement statistics.
    f"""
    CREATE TABLE IF NOT EXISTS quality_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rater_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        calibration INTEGER NOT NULL CHECK (calibration IN (0, 1)),
        grade TEXT NOT NULL CHECK (grade IN {GRADES!r}),
        note TEXT NOT NULL DEFAULT '',
        submitted_at_utc TEXT NOT NULL
    )
    """,
)


def db_path() -> str:
    return os.environ.get("QUALITY_LABELS_DB_PATH", DEFAULT_DB_PATH)


def _initialize(conn: sqlite3.Connection) -> None:
    """See src/db.py's `_initialize` -- same fresh-file WAL-conversion race, same fix."""
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
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def register_rater(rater_id: str, experience_note: str) -> None:
    """First-use only: `INSERT OR IGNORE` so a returning rater's second visit is a silent no-op,
    never a second row and never an overwrite of what they said the first time."""
    with closing(connect()) as conn, transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO raters (rater_id, experience_note, registered_at_utc) VALUES (?, ?, ?)",
            (rater_id, experience_note, _utc_now_iso()),
        )


def is_registered_rater(rater_id: str) -> bool:
    with closing(connect()) as conn:
        row = conn.execute("SELECT 1 FROM raters WHERE rater_id = ?", (rater_id,)).fetchone()
    return row is not None


def submit_label(rater_id: str, item_id: str, *, calibration: bool, grade: str, note: str = "") -> None:
    if grade not in GRADES:
        raise ValueError(f"grade must be one of {GRADES}, got {grade!r}")
    with closing(connect()) as conn, transaction(conn):
        conn.execute(
            """INSERT INTO quality_labels (rater_id, item_id, calibration, grade, note, submitted_at_utc)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (rater_id, item_id, int(calibration), grade, note, _utc_now_iso()),
        )


def labeled_item_ids(rater_id: str) -> set[str]:
    """Every item this rater has ALREADY submitted a label for at least once -- used only to skip
    items in their own queue; never reveals another rater's progress or labels."""
    with closing(connect()) as conn:
        rows = conn.execute("SELECT DISTINCT item_id FROM quality_labels WHERE rater_id = ?", (rater_id,)).fetchall()
    return {row[0] for row in rows}


def latest_labels(*, include_calibration: bool = False) -> dict[str, dict[str, str]]:
    """`{item_id: {rater_id: grade}}` for every item, keeping only each (rater, item) pair's most
    recent submission -- the "correction is a new row" convention resolved down to one grade per
    pair, as `eval/quality_agreement.py`'s wrappers expect. Calibration labels are excluded by
    default (docs/quality/PROTOCOL.md section 2: never part of the measured agreement)."""
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT rater_id, item_id, grade, calibration FROM quality_labels ORDER BY id ASC"
        ).fetchall()
    result: dict[str, dict[str, str]] = {}
    for rater_id, item_id, grade, calibration in rows:
        if calibration and not include_calibration:
            continue
        result.setdefault(item_id, {})[rater_id] = grade  # later rows overwrite earlier ones in-place
    return result
