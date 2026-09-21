"""src/telemetry.py: the closed schema, recording, immutable prices, and aggregate-only reporting."""

import json
import logging
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from langchain_core.messages import AIMessage

from src import db, llm_cost, telemetry
from src.telemetry import FAILURE_CATEGORIES, ScanRecorder

NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)

EXPECTED_COLUMNS = [
    "id", "recorded_at", "mode", "photo_count", "status", "failure_category", "cold_start", "embed_ms",
    "search_ms", "agent_ms", "total_ms", "attempts", "model_turns", "tool_calls", "input_tokens",
    "output_tokens", "bioclip_revision", "gemini_model", "model_reported", "pricing_version", "est_cost_usd",
    "platform",
]  # fmt: skip


def rows(sql="SELECT * FROM scan_metrics"):
    with closing(db.connect()) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql).fetchall()]


def turn(tokens_in, tokens_out, *, calls=(), model="gemini-3.1-flash-lite"):
    return AIMessage(
        content="",
        usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out},
        response_metadata={"model_name": model},
        tool_calls=[{"name": n, "args": {}, "id": f"c{i}"} for i, n in enumerate(calls)],
    )


def run_scan(mode="single", photos=1, cold=False, failure=None, **recorder_fields):
    recorder = ScanRecorder(mode, photos, cold)
    for key, value in recorder_fields.items():
        setattr(recorder, key, value)
    telemetry.record(recorder, failure, now=NOW)
    return recorder


class TestTheSchemaIsClosed:
    def test_the_columns_are_exactly_the_documented_set(self):
        with closing(db.connect()) as conn:
            columns = [r[1] for r in conn.execute("PRAGMA table_info(scan_metrics)")]
        assert columns == EXPECTED_COLUMNS

    def test_no_column_can_hold_free_text_from_a_user_or_a_provider(self):
        """Every TEXT column is an enum, a version, a timestamp or the fixed hardware string;
        anything that could carry a message, prompt, address or identity is simply absent."""
        forbidden = {"message", "error", "exception", "prompt", "answer", "response", "ip", "address", "key",
                     "label", "identity", "user", "session", "image", "filename", "summary", "note"}  # fmt: skip
        assert not [c for c in EXPECTED_COLUMNS if forbidden & set(c.split("_"))]

    @pytest.mark.parametrize(
        "mode, status, category, photos",
        [
            ("batch", "ok", None, 1),  # not single/lot
            ("single", "ok", "boom", 1),  # a category outside the enum
            ("single", "error", None, 1),  # an error with no category
            ("single", "ok", "other", 1),  # a success with a category
            ("single", "ok", None, 0),  # no photos
        ],
    )
    def test_the_database_itself_refuses_a_malformed_row(self, mode, status, category, photos):
        with closing(db.connect()) as conn, pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO scan_metrics (recorded_at, mode, photo_count, status, failure_category, cold_start, "
                "total_ms, bioclip_revision, gemini_model, platform) VALUES ('t', ?, ?, ?, ?, 0, 1, 'r', 'm', 'p')",
                (mode, photos, status, category),
            )

    def test_the_failure_categories_are_exactly_the_planned_enum(self):
        assert set(FAILURE_CATEGORIES) == {
            "rate_limit", "server_5xx", "timeout", "parse_error", "auth", "empty_index", "other",
        }  # fmt: skip


class TestRecording:
    def test_a_successful_scan_is_one_ok_row_with_no_failure_category(self):
        run_scan(attempts=1, input_tokens=100, output_tokens=10, model_turns=1, tool_calls=0)
        (row,) = rows()
        assert row["status"] == "ok" and row["failure_category"] is None
        assert (row["mode"], row["photo_count"], row["attempts"]) == ("single", 1, 1)
        assert row["recorded_at"] == NOW.isoformat()

    def test_the_versions_in_play_are_recorded(self):
        from src.versions import BIOCLIP_REVISION, GEMINI_MODEL

        run_scan()
        (row,) = rows()
        assert (row["bioclip_revision"], row["gemini_model"]) == (BIOCLIP_REVISION, GEMINI_MODEL)
        assert re.fullmatch(r"\w+ .+, \d+ CPUs", row["platform"])

    def test_a_failure_is_an_error_row_with_its_category(self):
        run_scan(failure="rate_limit")
        (row,) = rows()
        assert (row["status"], row["failure_category"]) == ("error", "rate_limit")

    def test_unset_measurements_are_null_not_zero(self):
        run_scan()  # nothing measured: the agent never ran
        (row,) = rows()
        unmeasured = ("embed_ms", "search_ms", "agent_ms", "attempts", "model_turns", "tool_calls", "input_tokens",
                      "output_tokens", "model_reported", "pricing_version", "est_cost_usd")  # fmt: skip
        assert {column: row[column] for column in unmeasured} == dict.fromkeys(unmeasured)

    def test_cold_start_is_recorded(self):
        run_scan(cold=True)
        run_scan(cold=False)
        assert [r["cold_start"] for r in rows()] == [1, 0]

    def test_recording_never_breaks_a_scan_and_logs_only_the_error_type(self, monkeypatch, caplog):
        def broken(**_kwargs):
            raise sqlite3.OperationalError("disk I/O error at /secret/path/inv.db")

        monkeypatch.setattr(telemetry.db, "connect", broken)
        with caplog.at_level(logging.WARNING, logger="bloomlens.telemetry"):
            run_scan()  # must not raise
        assert "OperationalError" in caplog.text
        assert "/secret/path" not in caplog.text and "disk I/O" not in caplog.text


class TestScanContext:
    def test_it_records_exactly_one_row_on_success(self):
        with telemetry.scan("single", 1, cold_start=False):
            pass
        (row,) = rows()
        assert row["status"] == "ok" and row["total_ms"] >= 0

    def test_it_records_the_category_the_pipeline_attached_and_reraises(self):
        class Failure(Exception):
            category = "timeout"

        with pytest.raises(Failure), telemetry.scan("lot", 4, cold_start=True):
            raise Failure("provider said: super secret detail")
        (row,) = rows()
        assert (row["mode"], row["photo_count"], row["failure_category"], row["cold_start"]) == ("lot", 4, "timeout", 1)

    def test_an_exception_with_no_category_is_other(self):
        with pytest.raises(RuntimeError), telemetry.scan("single", 1, cold_start=False):
            raise RuntimeError("boom")
        assert rows()[0]["failure_category"] == "other"

    def test_a_category_that_is_not_in_the_enum_is_never_stored(self):
        class Sneaky(Exception):
            category = "the user's password is hunter2"

        with pytest.raises(Sneaky), telemetry.scan("single", 1, cold_start=False):
            raise Sneaky
        assert rows()[0]["failure_category"] == "other"

    def test_a_keyboard_interrupt_is_not_a_scan_outcome(self):
        with pytest.raises(KeyboardInterrupt), telemetry.scan("single", 1, cold_start=False):
            raise KeyboardInterrupt
        assert rows() == []

    def test_helpers_are_no_ops_outside_a_scan(self):
        with telemetry.timed("embed"):
            pass
        telemetry.note_attempt()
        telemetry.note_agent_result({"messages": [turn(1, 1)]})
        assert rows() == []

    def test_stage_timings_and_attempts_accumulate(self):
        with telemetry.scan("single", 1, cold_start=False):
            with telemetry.timed("embed"):
                time.sleep(0.03)
            with telemetry.timed("agent"):
                time.sleep(0.03)
            with telemetry.timed("agent"):  # a second call adds to the same stage
                time.sleep(0.03)
            telemetry.note_attempt()
            telemetry.note_attempt()
        (row,) = rows()
        assert 25 <= row["embed_ms"] < 2000 and 55 <= row["agent_ms"] < 3000
        assert row["search_ms"] is None and row["attempts"] == 2
        assert row["total_ms"] >= row["embed_ms"] + row["agent_ms"]

    def test_a_scan_that_fails_before_the_agent_has_no_attempts(self):
        with pytest.raises(RuntimeError), telemetry.scan("single", 1, cold_start=False):
            with telemetry.timed("embed"):
                raise RuntimeError
        assert rows()[0]["attempts"] is None

    def test_concurrent_scans_do_not_mix_their_measurements(self):
        import threading

        def one(n):
            with telemetry.scan("single", 1, cold_start=False):
                for _ in range(n):
                    telemetry.note_attempt()
                time.sleep(0.02)

        threads = [threading.Thread(target=one, args=(n,)) for n in (1, 2, 3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(r["attempts"] for r in rows()) == [1, 2, 3]


class Fake:
    """A stand-in message exposing only what telemetry may read. Its `content` fails the
    test if anything touches it."""

    def __init__(self, usage=None, calls=None, metadata=None):
        self.usage_metadata, self.tool_calls, self.response_metadata = usage, calls, metadata or {}

    @property
    def content(self):
        raise AssertionError("telemetry read a message's content")


class TestAgentResult:
    def record_result(self, *messages):
        with telemetry.scan("single", 1, cold_start=False):
            telemetry.note_agent_result({"messages": list(messages)})
        return rows()[0]

    def test_tokens_turns_and_tool_calls_are_summed_over_the_run(self):
        """The shape of the real probe run (docs/telemetry_probe.md): four turns, three tool calls."""
        row = self.record_result(
            turn(1854, 18, calls=["lookup_taxonomy"]),
            turn(1931, 48, calls=["assess_quality"]),
            turn(2025, 24, calls=["check_price"]),
            turn(2098, 123),
        )
        assert (row["input_tokens"], row["output_tokens"]) == (7908, 213)
        assert (row["model_turns"], row["tool_calls"]) == (4, 3)
        assert row["model_reported"] == "gemini-3.1-flash-lite"

    def test_messages_without_usage_leave_the_token_columns_null(self):
        row = self.record_result(AIMessage(content="no usage here"))
        assert row["input_tokens"] is None and row["output_tokens"] is None and row["model_turns"] is None
        assert row["est_cost_usd"] is None

    @pytest.mark.parametrize("bad", ["12", 1.5, True, -3, 10**12, None])
    def test_a_token_count_that_is_not_a_sane_integer_is_ignored(self, bad):
        row = self.record_result(Fake(usage={"input_tokens": bad, "output_tokens": 5}))
        assert row["input_tokens"] is None and row["model_turns"] is None

    @pytest.mark.parametrize("name", ["gemini x; DROP TABLE scan_metrics", "x" * 200, "", "has space", "ünï"])
    def test_a_reported_model_name_is_stored_only_if_it_is_a_plain_identifier(self, name):
        assert self.record_result(turn(1, 1, model=name))["model_reported"] is None

    def test_the_content_of_messages_and_tool_arguments_is_never_read(self):
        message = Fake(
            usage={"input_tokens": 2, "output_tokens": 3},
            calls=[{"name": "check_price", "args": {"species": "SECRET-ARG"}, "id": "c1"}],
        )
        row = self.record_result(message)
        assert (row["input_tokens"], row["tool_calls"]) == (2, 1)
        assert "SECRET-ARG" not in json.dumps(row)

    def test_a_result_that_is_not_an_agent_result_is_ignored(self):
        for junk in (None, {}, {"messages": None}, "text", 5):
            with telemetry.scan("single", 1, cold_start=False):
                telemetry.note_agent_result(junk)
        assert len(rows()) == 5 and all(r["input_tokens"] is None for r in rows())

    def test_two_agent_runs_in_one_scan_add_up(self):
        """A retry after a malformed answer: the tokens of BOTH runs were spent."""
        with telemetry.scan("single", 1, cold_start=False):
            telemetry.note_agent_result({"messages": [turn(100, 10)]})
            telemetry.note_agent_result({"messages": [turn(200, 20)]})
        assert (rows()[0]["input_tokens"], rows()[0]["output_tokens"]) == (300, 30)


class TestPriceHistory:
    V1 = llm_cost.PriceVersion("gemini-3.1-flash-lite@2026-09-16", "gemini-3.1-flash-lite", "2026-09-16", 0.25, 1.5,
                               "https://example.com", "2026-09-21")  # fmt: skip
    V2 = replace(V1, version="gemini-3.1-flash-lite@2026-10-01", effective_from="2026-10-01",
                 input_usd_per_mtok=0.50, output_usd_per_mtok=3.0)  # fmt: skip

    def scan_at(self, when, versions, tokens=(1_000_000, 1_000_000)):
        telemetry.llm_cost.PRICE_VERSIONS = versions
        recorder = ScanRecorder("single", 1, False)
        recorder.input_tokens, recorder.output_tokens = tokens
        telemetry.record(recorder, None, now=when)

    @pytest.fixture(autouse=True)
    def restore_prices(self):
        original = llm_cost.PRICE_VERSIONS
        yield
        llm_cost.PRICE_VERSIONS = original

    def test_each_scan_keeps_the_price_it_ran_under_when_prices_change_later(self):
        self.scan_at(datetime(2026, 9, 21, tzinfo=UTC), (self.V1,))
        self.scan_at(datetime(2026, 10, 5, tzinfo=UTC), (self.V1, self.V2))
        first, second = rows()
        assert (first["pricing_version"], first["est_cost_usd"]) == (self.V1.version, pytest.approx(1.75))
        assert (second["pricing_version"], second["est_cost_usd"]) == (self.V2.version, pytest.approx(3.5))

    def test_the_old_cost_is_stored_not_recomputed_with_todays_price(self):
        self.scan_at(datetime(2026, 9, 21, tzinfo=UTC), (self.V1,))
        self.scan_at(datetime(2026, 10, 5, tzinfo=UTC), (self.V1, self.V2))
        report = telemetry.summary(now=datetime(2026, 10, 6, tzinfo=UTC))
        assert report["estimated_cost_usd"]["total"] == pytest.approx(1.75 + 3.5)  # not 2 x 3.5
        assert report["estimated_cost_usd"]["pricing_versions"] == [self.V1.version, self.V2.version]

    def test_a_scan_dated_before_any_price_existed_gets_no_cost(self):
        self.scan_at(datetime(2026, 9, 1, tzinfo=UTC), (self.V1,))
        (row,) = rows()
        assert row["pricing_version"] is None and row["est_cost_usd"] is None

    def test_the_price_table_is_immutable(self):
        run_scan(input_tokens=1, output_tokens=1)
        with closing(db.connect()) as conn:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute("UPDATE pricing_versions SET input_usd_per_mtok = 0")
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute("DELETE FROM pricing_versions")
            assert conn.execute("SELECT COUNT(*) FROM pricing_versions").fetchone()[0] >= 1

    def test_editing_a_price_in_code_instead_of_adding_a_version_is_refused(self, caplog):
        self.scan_at(datetime(2026, 9, 21, tzinfo=UTC), (self.V1,))
        edited = replace(self.V1, input_usd_per_mtok=99.0)  # same version name, different number
        with caplog.at_level(logging.WARNING, logger="bloomlens.telemetry"):
            self.scan_at(datetime(2026, 9, 22, tzinfo=UTC), (edited,))
        assert len(rows()) == 1  # nothing was recorded under a rewritten price
        assert "RuntimeError" in caplog.text

    def test_the_committed_price_versions_match_what_gets_stored(self):
        run_scan()
        with closing(db.connect()) as conn:
            stored = {r[0]: tuple(r) for r in conn.execute("SELECT * FROM pricing_versions")}
        for v in llm_cost.PRICE_VERSIONS:
            assert stored[v.version][1:] == (
                v.model, v.effective_from, v.input_usd_per_mtok, v.output_usd_per_mtok, v.source_url, v.retrieved_at,
            )  # fmt: skip


def insert(conn, **overrides):
    values = dict(
        recorded_at=NOW.isoformat(), mode="single", photo_count=1, status="ok", failure_category=None, cold_start=0,
        embed_ms=100, search_ms=5, agent_ms=900, total_ms=1000, attempts=1, model_turns=4, tool_calls=3,
        input_tokens=8000, output_tokens=200, bioclip_revision="rev1", gemini_model="gemini-3.1-flash-lite",
        model_reported="gemini-3.1-flash-lite", pricing_version=None, est_cost_usd=None,
        platform="Windows AMD64, 8 CPUs",
    )  # fmt: skip
    values.update(overrides)
    columns, marks = ", ".join(values), ", ".join("?" * len(values))
    conn.execute(f"INSERT INTO scan_metrics ({columns}) VALUES ({marks})", tuple(values.values()))


def seed(**overrides_each):
    """Insert one row per (kwargs) dict given as a list under key 'rows'."""
    with closing(db.connect()) as conn, db.transaction(conn):
        for row in overrides_each["rows"]:
            insert(conn, **row)


class TestSummary:
    def test_an_empty_database_reports_zero_and_no_rate(self):
        report = telemetry.summary(now=NOW)
        assert report["scans"] == {"total": 0, "ok": 0, "error": 0}
        assert report["failure_rate"] is None
        assert report["latency_ms"]["warm"]["p50_ms"] is None
        assert report["estimated_cost_usd"]["total"] is None

    def test_percentiles_are_withheld_below_twenty_scans_and_say_how_many(self):
        seed(rows=[{"total_ms": 1000 + i} for i in range(19)])
        warm = telemetry.summary(now=NOW)["latency_ms"]["warm"]
        assert warm["n"] == 19 and warm["p50_ms"] is None and warm["p95_ms"] is None
        assert "fewer than 20" in warm["note"]

    def test_at_twenty_scans_they_appear(self):
        seed(rows=[{"total_ms": 1000 + i} for i in range(20)])
        warm = telemetry.summary(now=NOW)["latency_ms"]["warm"]
        assert warm["n"] == 20 and warm["p50_ms"] is not None and warm["p95_ms"] is not None

    @pytest.mark.parametrize("n, seed_value", [(20, 1), (37, 2), (200, 3), (1001, 4)])
    def test_percentiles_agree_with_numpy(self, n, seed_value):
        values = [int(v) for v in np.random.default_rng(seed_value).integers(200, 60_000, size=n)]
        seed(rows=[{"total_ms": v} for v in values])
        warm = telemetry.summary(now=NOW)["latency_ms"]["warm"]
        assert warm["p50_ms"] == round(float(np.percentile(values, 50)))
        assert warm["p95_ms"] == round(float(np.percentile(values, 95)))

    def test_there_is_no_p99_anywhere(self):
        seed(rows=[{"total_ms": 1000 + i} for i in range(50)])

        def keys(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    yield from keys(value)

        report_keys = set(keys(telemetry.summary(now=NOW)))
        assert {k for k in report_keys if re.fullmatch(r"p\d+(_ms)?", k)} == {"p50_ms", "p95_ms"}

    def test_cold_scans_are_reported_apart_from_warm_ones(self):
        seed(rows=[{"total_ms": 1000}] * 25 + [{"total_ms": 30_000, "cold_start": 1}] * 25)
        latency = telemetry.summary(now=NOW)["latency_ms"]
        assert latency["warm"]["p50_ms"] == 1000 and latency["cold"]["p50_ms"] == 30_000

    def test_failed_scans_are_kept_out_of_the_latency_figures(self):
        failed = {"status": "error", "failure_category": "timeout", "total_ms": 90_000}
        seed(rows=[{"total_ms": 1000}] * 20 + [failed] * 5)
        report = telemetry.summary(now=NOW)
        assert report["latency_ms"]["warm"]["p95_ms"] == 1000
        assert report["scans"] == {"total": 25, "ok": 20, "error": 5}
        assert report["failure_rate"] == 0.2

    def test_failures_are_counted_by_category_with_every_category_present(self):
        seed(rows=[{"status": "error", "failure_category": "rate_limit"}] * 3
             + [{"status": "error", "failure_category": "parse_error"}])  # fmt: skip
        by_category = telemetry.summary(now=NOW)["failures_by_category"]
        assert set(by_category) == set(FAILURE_CATEGORIES)
        assert by_category["rate_limit"] == 3 and by_category["parse_error"] == 1 and by_category["auth"] == 0

    def test_retries_per_scan_is_the_mean_of_attempts_minus_one(self):
        seed(rows=[{"attempts": 1}, {"attempts": 1}, {"attempts": 3}, {"attempts": 4}, {"attempts": None}])
        retries = telemetry.summary(now=NOW)["retries_per_scan"]
        assert retries == {"n": 4, "mean": 1.25}  # (0 + 0 + 2 + 3) / 4; the NULL is not counted

    def test_scans_without_token_counts_are_reported_as_such_not_as_zero(self):
        seed(rows=[{"input_tokens": 100, "output_tokens": 10}, {"input_tokens": 300, "output_tokens": 30},
                   {"input_tokens": None, "output_tokens": None}])  # fmt: skip
        tokens = telemetry.summary(now=NOW)["tokens"]
        assert (tokens["scans_with_counts"], tokens["scans_without_counts"]) == (2, 1)
        assert (tokens["mean_input"], tokens["mean_output"]) == (200.0, 20.0)

    def test_cost_sums_only_priced_scans_and_says_what_it_means(self):
        seed(rows=[{"est_cost_usd": 0.002, "pricing_version": "v"}, {"est_cost_usd": 0.003, "pricing_version": "v"},
                   {"est_cost_usd": None}])  # fmt: skip
        cost = telemetry.summary(now=NOW)["estimated_cost_usd"]
        assert (cost["total"], cost["scans_priced"], cost["scans_unpriced"]) == (0.005, 2, 1)
        assert cost["meaning"] == llm_cost.COST_LABEL and "actual billed cost unknown" in cost["meaning"]

    def test_versions_and_hardware_are_listed(self):
        seed(rows=[{"bioclip_revision": "a"}, {"bioclip_revision": "b", "platform": "Linux x86_64, 2 CPUs"},
                   {"model_reported": None}])  # fmt: skip
        report = telemetry.summary(now=NOW)
        assert set(report["versions"]["bioclip_revision"]) == {"a", "b", "rev1"}
        assert report["platforms"] == ["Linux x86_64, 2 CPUs", "Windows AMD64, 8 CPUs"]
        assert report["versions"]["gemini_model_reported"] == ["gemini-3.1-flash-lite"]

    def test_rows_outside_the_window_are_ignored(self):
        old = (NOW - timedelta(days=45)).isoformat()
        seed(rows=[{"recorded_at": old}, {"recorded_at": NOW.isoformat()}])
        assert telemetry.summary(now=NOW)["scans"]["total"] == 1
        assert telemetry.summary(now=NOW, window_days=60)["scans"]["total"] == 2

    def test_the_default_window_is_the_metrics_retention(self, monkeypatch):
        monkeypatch.setenv("BLOOMLENS_METRICS_RETENTION_DAYS", "7")
        assert telemetry.summary(now=NOW)["window_days"] == 7

    def test_the_report_contains_no_row_level_data(self):
        """Aggregates only: no timestamps, no ids, nothing shaped like a row."""
        seed(rows=[{"total_ms": 1000 + i} for i in range(30)])
        text = json.dumps(telemetry.summary(now=NOW))
        assert not re.search(r"\d{4}-\d{2}-\d{2}T", text)  # no timestamp
        assert '"id"' not in text and "recorded_at" not in text
