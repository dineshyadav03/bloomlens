"""Quota admission from several REAL processes at once: exactly L admitted, never L+1.

The API and the Streamlit app are separate processes sharing one SQLite file, so this is
the case that matters and the one an in-memory limiter cannot pass. Threads in one process
would prove little (they share a GIL and a connection cache), so these are genuine
interpreters, released at the same instant into brand-new databases -- which also
exercises the first-write initialisation race.

Each round: every worker signals "ready", waits for a shared "go" file, hammers admit()
`ATTEMPTS` times with the clock pinned to one instant (so windows cannot roll mid-test),
and reports how many of its attempts were admitted. Across all workers the admitted total
must equal the limit exactly, and the stored counter must agree. A worker that raises
(for instance a surfaced "database is locked") fails the test."""

import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing

import pytest

from tests.conftest import REPO_ROOT

PROCESSES = 4
ROUNDS = 12
ATTEMPTS = 15
LIMIT = 20  # 4 x 15 = 60 attempts per round against a limit of 20

WORKER = """
import os, sys, time
from datetime import datetime, UTC
from pathlib import Path
from src import quota

worker, base, sync = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
mode, rounds, attempts, limit = sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)
ROOMY = 10**6

if mode == "one-caller":  # every worker is the same caller: the caller's DAY quota is what binds
    identity, limits = "api:shared", quota.Limits(per_minute=ROOMY, per_day=limit, global_per_day=ROOMY)
elif mode == "one-minute":  # the caller's MINUTE rate is what binds
    identity, limits = "api:shared", quota.Limits(per_minute=limit, per_day=ROOMY, global_per_day=ROOMY)
else:  # "global": every worker is a different caller; only the GLOBAL ceiling binds
    identity, limits = f"ui:{worker}", quota.Limits(per_minute=ROOMY, per_day=ROOMY, global_per_day=limit)

for r in range(rounds):
    os.environ["INVENTORY_DB_PATH"] = str(base / f"round-{r}" / "shared.db")
    (sync / f"ready-{r}-{worker}").write_text("x")
    while not (sync / f"go-{r}").exists():
        time.sleep(0.001)
    admitted = sum(quota.admit(identity, now=NOW, limits=limits).allowed for _ in range(attempts))
    (sync / f"done-{r}-{worker}").write_text(str(admitted))
"""


def wait_for(sync, pattern, count, workers, deadline):
    while len(list(sync.glob(pattern))) < count:
        assert time.time() < deadline, f"timed out waiting for {pattern}"
        for w in workers:
            if w.poll() not in (None, 0):
                pytest.fail(f"a worker died:\n{w.communicate()[1][-3000:]}")
        time.sleep(0.005)


def run_scenario(tmp_path, mode):
    base, sync = tmp_path / "dbs", tmp_path / "sync"
    sync.mkdir()
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    env.pop("INVENTORY_DB_PATH", None)
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", WORKER, str(n), str(base), str(sync), mode, str(ROUNDS), str(ATTEMPTS), str(LIMIT)],
            env=env,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for n in range(PROCESSES)
    ]
    deadline = time.time() + 240
    admitted_per_round = []
    try:
        for r in range(ROUNDS):
            wait_for(sync, f"ready-{r}-*", PROCESSES, workers, deadline)
            (sync / f"go-{r}").write_text("go")  # released together
            wait_for(sync, f"done-{r}-*", PROCESSES, workers, deadline)
            admitted_per_round.append(sum(int(f.read_text()) for f in sync.glob(f"done-{r}-*")))
        for w in workers:
            _, stderr = w.communicate(timeout=60)
            assert w.returncode == 0, stderr[-3000:]
    finally:
        for w in workers:
            if w.poll() is None:
                w.kill()
    return base, admitted_per_round


def stored_counters(db_file):
    with closing(sqlite3.connect(db_file)) as conn:
        return {(who, kind): used for who, kind, _, used in conn.execute("SELECT * FROM quota_counters")}


@pytest.mark.parametrize(
    "mode, key",
    [("one-caller", ("api:shared", "day")), ("one-minute", ("api:shared", "minute")), ("global", ("*", "day"))],
)
def test_exactly_the_limit_is_admitted_across_real_processes_every_round(tmp_path, mode, key):
    base, admitted_per_round = run_scenario(tmp_path, mode)

    assert admitted_per_round == [LIMIT] * ROUNDS, (
        f"admitted per round: {admitted_per_round} (limit {LIMIT}); more means the limiter let a race through, "
        "fewer means a legitimate request was lost"
    )
    for r in range(ROUNDS):
        assert stored_counters(base / f"round-{r}" / "shared.db")[key] == LIMIT
