"""The three ways aggregate telemetry is exposed -- GET /metrics, scripts/report_metrics.py
and (in tests/integration/test_app_ui.py) the Performance expander -- and that none of them
can reveal a row."""

import importlib.util
import json
import os
import re
import subprocess
import sys
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

import api.main as api
from src import db, telemetry
from tests.conftest import REPO_ROOT
from tests.unit.test_telemetry import insert

spec = importlib.util.spec_from_file_location("report_metrics", REPO_ROOT / "scripts" / "report_metrics.py")
report_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_script)


def seed(n, **overrides):
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    with closing(db.connect()) as conn, db.transaction(conn):
        for i in range(n):
            insert(conn, recorded_at=now, total_ms=1000 + i, **overrides)


@pytest.fixture
def client(api_headers):
    return TestClient(api.app, headers=api_headers)


class TestMetricsEndpoint:
    def test_it_needs_a_key_like_every_other_route(self, api_headers):
        assert TestClient(api.app).get("/metrics").status_code == 401

    def test_it_returns_the_aggregate_with_percentiles_withheld_on_a_small_sample(self, client):
        seed(5)
        body = client.get("/metrics").json()
        assert body["scans"] == {"total": 5, "ok": 5, "error": 0}
        assert body["latency_ms"]["warm"]["p50_ms"] is None and body["latency_ms"]["warm"]["n"] == 5

    def test_with_enough_scans_it_reports_p50_and_p95(self, client):
        seed(40)
        warm = client.get("/metrics").json()["latency_ms"]["warm"]
        assert warm["n"] == 40 and 1000 <= warm["p50_ms"] < warm["p95_ms"] <= 1039

    def test_it_states_what_cost_means(self, client):
        assert "actual billed cost unknown" in client.get("/metrics").json()["estimated_cost_usd"]["meaning"]

    def test_it_never_returns_a_row_a_timestamp_or_an_identifier(self, client):
        seed(30)
        text = client.get("/metrics").text
        assert not re.search(r"\d{4}-\d{2}-\d{2}T", text)
        for forbidden in ('"id"', "recorded_at", "label", "identity", "session"):
            assert forbidden not in text

    def test_reading_metrics_is_not_metered(self, client, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_RATE_PER_MINUTE", "1")
        assert all(client.get("/metrics").status_code == 200 for _ in range(10))
        with closing(db.connect()) as conn:
            assert conn.execute("SELECT COUNT(*) FROM quota_counters").fetchone()[0] == 0

    def test_an_empty_database_is_a_valid_empty_report(self, client):
        body = client.get("/metrics").json()
        assert body["scans"]["total"] == 0 and body["failure_rate"] is None

    def test_the_route_is_in_the_served_surface(self):
        assert "/metrics" in api.app.openapi()["paths"]


class TestReportScript:
    def run(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "report_metrics.py"), *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            timeout=120,
        )

    def test_the_table_for_an_empty_database_says_so_without_failing(self):
        result = self.run()
        assert result.returncode == 0, result.stderr
        assert "Scan telemetry, last 30 days" in result.stdout and "scans: 0" in result.stdout

    def test_json_mode_is_the_raw_aggregate(self):
        seed(25)
        result = self.run("--json")
        assert json.loads(result.stdout)["scans"]["total"] == 25

    def test_the_table_shows_percentiles_only_when_there_are_enough_scans(self):
        seed(5)
        assert "withheld: fewer than 20" in self.run().stdout
        seed(30)
        text = self.run().stdout
        assert "p50" in text and "p95" in text and "p99" not in text.replace("p99 is not reported", "")

    def test_the_report_names_the_hardware_the_numbers_came_from_and_the_cost_caveat(self):
        seed(3)
        text = self.run().stdout
        assert "measured on: Windows AMD64, 8 CPUs" in text
        assert "actual billed cost unknown" in text

    def test_rendering_prints_no_timestamp_and_no_identifier(self):
        seed(30)
        text = report_script.render(telemetry.summary())
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
