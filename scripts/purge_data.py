"""Delete data past its retention (see src/retention.py for the schedule).

    uv run python scripts/purge_data.py

Prints how many rows were removed from each table. Safe to run at any time and from cron;
the API and the Streamlit app also purge opportunistically, at most hourly.
"""

import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import db, retention  # noqa: E402


def main() -> None:
    keep = retention.settings()
    with closing(db.connect()) as conn, db.transaction(conn):
        removed = retention.purge_all(conn, datetime.now(UTC))
    print(f"retention: scan telemetry {keep.metrics_days} days, inventory {keep.inventory_days} days")
    for table, count in removed.items():
        print(f"  {table}: {count} rows removed")


if __name__ == "__main__":
    main()
