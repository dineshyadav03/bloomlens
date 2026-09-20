"""The inventory log written by several REAL processes at once, on databases that
do not exist yet.

Docker Compose runs `web` and `api` as separate containers sharing one SQLite file,
and either can take the very first write. Threads (tests/unit/test_inventory.py)
share a process; this uses genuinely separate interpreters.

The cold-start race is a narrow window, so one attempt proves little (an earlier
single-shot version of this test passed even with the fix disabled). This runs many
independent rounds per run. Workers import the project once (~10 s each), then for
every round: signal "ready", wait for a shared "go" file, write into a brand-new
database, signal "done". The parent releases all workers at the same instant, so
every round is a fresh chance for the first writes to collide."""

import os
import subprocess
import sys
import time

from src import inventory
from tests.conftest import REPO_ROOT

PROCESSES = 4
ROUNDS = 25
ROWS_EACH = 3

WORKER = """
import os, sys, time
from pathlib import Path
from src import inventory
worker, base, sync = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
rounds, rows = int(sys.argv[4]), int(sys.argv[5])
for r in range(rounds):
    os.environ["INVENTORY_DB_PATH"] = str(base / f"round-{r}" / "inventory.db")
    (sync / f"ready-{r}-{worker}").write_text("x")
    while not (sync / f"go-{r}").exists():
        time.sleep(0.001)
    for i in range(rows):
        inventory._insert(
            mode="single", species=f"P{worker}-{i}", scientific_name=None, confidence_tier="high",
            quality_grade="A", price_per_stem=1.0, price_trend="flat", photo_count=1, agreement_fraction=None,
        )
    (sync / f"done-{r}-{worker}").write_text("x")
"""


def wait_for(sync, pattern, count, workers, deadline):
    while len(list(sync.glob(pattern))) < count:
        assert time.time() < deadline, f"timed out waiting for {pattern}"
        assert all(w.poll() is None for w in workers), [w.communicate() for w in workers]
        time.sleep(0.005)


def test_separate_processes_writing_to_fresh_databases_lose_no_rows(tmp_path):
    base, sync = tmp_path / "dbs", tmp_path / "sync"
    sync.mkdir()
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    env.pop("INVENTORY_DB_PATH", None)

    workers = [
        subprocess.Popen(
            [sys.executable, "-c", WORKER, str(n), str(base), str(sync), str(ROUNDS), str(ROWS_EACH)],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for n in range(PROCESSES)
    ]

    deadline = time.time() + 600
    for r in range(ROUNDS):
        wait_for(sync, f"ready-{r}-*", PROCESSES, workers, deadline)
        (sync / f"go-{r}").write_text("go")  # release every worker at once
        wait_for(sync, f"done-{r}-*", PROCESSES, workers, deadline)

    outputs = [w.communicate(timeout=120) for w in workers]
    assert [w.returncode for w in workers] == [0] * PROCESSES, outputs
    swallowed = [out for out, _ in outputs if "failed to log scan" in out]
    assert swallowed == [], f"writes were dropped (logging swallows errors): {swallowed}"

    for r in range(ROUNDS):
        os.environ["INVENTORY_DB_PATH"] = str(base / f"round-{r}" / "inventory.db")
        rows = inventory.list_recent(limit=1000)
        assert len(rows) == PROCESSES * ROWS_EACH, f"round {r} lost rows"
