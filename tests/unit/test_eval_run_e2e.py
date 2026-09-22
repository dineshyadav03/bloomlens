"""eval/run_e2e.py: the resumable cache, one real-identify() call wrapped for the pilot, the
summary/report, and main()'s --max-calls / resume behaviour -- with identify() faked throughout
(a live run is exercised for real separately, outside the test suite)."""

import json
import sys
from contextlib import closing
from pathlib import Path

import pytest
from PIL import Image

from eval import run_e2e
from src import db
from src.identify import IdentifyError


def write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, format="JPEG")


def pilot_entry(source_id="oxford102:image_00001", family="id", label="Rose", path="a.jpg"):
    return {"source_id": source_id, "family": family, "label": label, "path": path}


class TestCacheEntryKey:
    def test_it_changes_with_any_of_its_three_inputs(self):
        base = run_e2e.cache_entry_key("h1", "model-a", "rev1")
        assert base != run_e2e.cache_entry_key("h2", "model-a", "rev1")
        assert base != run_e2e.cache_entry_key("h1", "model-b", "rev1")
        assert base != run_e2e.cache_entry_key("h1", "model-a", "rev2")

    def test_it_is_deterministic(self):
        assert run_e2e.cache_entry_key("h", "m", "r") == run_e2e.cache_entry_key("h", "m", "r")


class TestCacheRoundTrip:
    def test_an_absent_file_is_an_empty_cache(self, tmp_path):
        assert run_e2e.load_cache(tmp_path / "missing.jsonl") == {}

    def test_appended_records_are_loaded_back_keyed_by_cache_key(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        run_e2e.append_result({"cache_key": "a", "x": 1}, path)
        run_e2e.append_result({"cache_key": "b", "x": 2}, path)
        cache = run_e2e.load_cache(path)
        assert cache == {"a": {"cache_key": "a", "x": 1}, "b": {"cache_key": "b", "x": 2}}

    def test_a_repeated_key_keeps_the_later_record(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        run_e2e.append_result({"cache_key": "a", "x": 1}, path)
        run_e2e.append_result({"cache_key": "a", "x": 2}, path)
        assert run_e2e.load_cache(path)["a"]["x"] == 2

    def test_a_truncated_trailing_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        path.write_text('{"cache_key": "a", "x": 1}\n{"cache_key": "b", "x"', encoding="utf-8")
        assert run_e2e.load_cache(path) == {"a": {"cache_key": "a", "x": 1}}

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        path.write_text('{"cache_key": "a", "x": 1}\n\n\n', encoding="utf-8")
        assert run_e2e.load_cache(path) == {"a": {"cache_key": "a", "x": 1}}

    def test_appending_creates_the_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "cache.jsonl"
        run_e2e.append_result({"cache_key": "a"}, path)
        assert path.exists()

    def test_the_default_path_follows_a_monkeypatched_cache_path(self, tmp_path, monkeypatch):
        """The same default-argument-bound-at-import trap fixed elsewhere in this project
        (src/guard.py, src/quota.py, src/llm_cost.py, eval/frozen_config.py, eval/open_world_data.py)."""
        monkeypatch.setattr(run_e2e, "CACHE_PATH", tmp_path / "cache.jsonl")
        run_e2e.append_result({"cache_key": "a"})  # no explicit path
        assert (tmp_path / "cache.jsonl").exists()
        assert run_e2e.load_cache() == {"a": {"cache_key": "a"}}  # no explicit path either


class TestRunOne:
    @pytest.fixture
    def image_file(self, tmp_path, monkeypatch):
        path = tmp_path / "a.jpg"
        write_image(path)
        monkeypatch.setattr(run_e2e, "RAW_ROOT", tmp_path)
        return path

    def test_a_successful_call_records_the_result_and_agreement_with_retrieval(
        self, image_file, monkeypatch, make_identify_result
    ):
        result = make_identify_result(
            species="Rose",
            top_candidates=[{"common_name": "Rose", "scientific_name": "x", "score": 0.9}],
            abstained=False,
            abstain_source=None,
        )
        monkeypatch.setattr(run_e2e, "identify", lambda _image: result)
        record = run_e2e.run_one(pilot_entry(), model="gemini-x", code_revision="rev1")
        assert record["status"] == "ok"
        assert record["final_species"] == "Rose" and record["retrieval_top1_species"] == "Rose"
        assert record["agrees_with_retrieval"] is True
        assert record["abstained"] is False and record["schema_valid"] is True
        assert record["family"] == "id" and record["label"] == "Rose"

    def test_disagreement_with_retrieval_is_recorded(self, image_file, monkeypatch, make_identify_result):
        result = make_identify_result(
            species="Tulip", top_candidates=[{"common_name": "Rose", "scientific_name": "x", "score": 0.9}]
        )
        monkeypatch.setattr(run_e2e, "identify", lambda _image: result)
        record = run_e2e.run_one(pilot_entry(), model="gemini-x", code_revision="rev1")
        assert record["agrees_with_retrieval"] is False

    def test_a_pipeline_failure_is_recorded_not_raised(self, image_file, monkeypatch):
        def failing(_image):
            raise IdentifyError("boom", category="timeout")

        monkeypatch.setattr(run_e2e, "identify", failing)
        record = run_e2e.run_one(pilot_entry(), model="gemini-x", code_revision="rev1")
        assert record["status"] == "error" and record["failure_category"] == "timeout"
        assert record["schema_valid"] is False and record["final_species"] is None

    def test_the_cache_key_is_a_function_of_entry_identity_model_and_revision(
        self, image_file, monkeypatch, make_identify_result
    ):
        monkeypatch.setattr(run_e2e, "identify", lambda _image: make_identify_result())
        a = run_e2e.run_one(pilot_entry(), model="model-a", code_revision="rev1")
        b = run_e2e.run_one(pilot_entry(), model="model-b", code_revision="rev1")
        c = run_e2e.run_one(pilot_entry(source_id="oxford102:image_99999"), model="model-a", code_revision="rev1")
        assert a["cache_key"] != b["cache_key"] != c["cache_key"] and a["cache_key"] != c["cache_key"]

    def test_two_entries_sharing_byte_identical_image_content_get_different_cache_keys(
        self, image_file, monkeypatch, make_identify_result
    ):
        """Regression: this dataset really does contain two different pilot entries that point at
        byte-identical Oxford photos (found in M15a's leakage test) -- content hashing would give
        them one cache slot and only one of the two would ever get its own identify() call."""
        monkeypatch.setattr(run_e2e, "identify", lambda _image: make_identify_result())
        first = run_e2e.run_one(pilot_entry(source_id="oxford102:image_08067"), model="m", code_revision="r")
        second = run_e2e.run_one(pilot_entry(source_id="oxford102:image_08077"), model="m", code_revision="r")
        assert first["cache_key"] != second["cache_key"]

    def test_telemetry_fields_come_from_the_real_scan_metrics_row_identify_just_wrote(
        self, image_file, monkeypatch, make_identify_result
    ):
        def fake_identify(_image):
            with closing(db.connect()) as conn, db.transaction(conn):
                conn.execute(
                    "INSERT INTO scan_metrics (recorded_at, mode, photo_count, status, failure_category, "
                    "cold_start, total_ms, attempts, input_tokens, output_tokens, est_cost_usd, "
                    "bioclip_revision, gemini_model, model_reported, platform) VALUES "
                    "('t','single',1,'ok',NULL,0,1234,2,100,20,0.001,'rev','m','m-reported','p')"
                )
            return make_identify_result()

        monkeypatch.setattr(run_e2e, "identify", fake_identify)
        record = run_e2e.run_one(pilot_entry(), model="gemini-x", code_revision="rev1")
        assert record["total_ms"] == 1234 and record["attempts"] == 2
        assert record["input_tokens"] == 100 and record["output_tokens"] == 20
        assert record["model_reported"] == "m-reported"

    def test_no_telemetry_row_leaves_those_fields_null(self, image_file, monkeypatch, make_identify_result):
        monkeypatch.setattr(run_e2e, "identify", lambda _i: make_identify_result())
        monkeypatch.setattr(run_e2e, "latest_telemetry_row", lambda _after_id: None)
        record = run_e2e.run_one(pilot_entry(), model="gemini-x", code_revision="rev1")
        assert record["total_ms"] is None and record["attempts"] is None


def outcome(family="id", status="ok", abstained=False, agrees=True, total_ms=1000, attempts=1,
            tokens_in=100, tokens_out=10, cost=0.001, category=None):  # fmt: skip
    return {
        "family": family, "status": status, "abstained": abstained, "agrees_with_retrieval": agrees,
        "total_ms": total_ms, "attempts": attempts, "input_tokens": tokens_in, "output_tokens": tokens_out,
        "est_cost_usd": cost, "failure_category": category,
    }  # fmt: skip


class TestSummarize:
    def test_an_empty_list_is_a_valid_empty_summary(self):
        summary = run_e2e.summarize([])
        assert summary["n_total"] == 0 and summary["schema_valid_rate"] is None

    def test_schema_valid_rate_is_the_fraction_that_succeeded(self):
        records = [outcome(status="ok"), outcome(status="ok"), outcome(status="error", category="timeout")]
        assert run_e2e.summarize(records)["schema_valid_rate"] == pytest.approx(2 / 3)

    def test_abstention_and_agreement_are_computed_per_family_over_successes_only(self):
        records = [
            outcome(family="id", abstained=False, agrees=True),
            outcome(family="id", abstained=True, agrees=False),
            outcome(family="id", status="error", category="other"),  # excluded from the rates
            outcome(family="near_ood", abstained=True, agrees=False),
        ]
        by_family = run_e2e.summarize(records)["by_family"]
        assert by_family["id"]["n"] == 3 and by_family["id"]["n_ok"] == 2
        assert by_family["id"]["abstained_rate"] == pytest.approx(0.5)
        assert by_family["near_ood"]["abstained_rate"] == pytest.approx(1.0)
        assert by_family["far_ood"] == {"n": 0, "n_ok": 0, "abstained_rate": None, "agreement_with_retrieval": None}

    def test_failures_are_counted_by_category(self):
        records = [outcome(status="error", category="timeout"), outcome(status="error", category="timeout"),
                   outcome(status="error", category="rate_limit")]  # fmt: skip
        assert run_e2e.summarize(records)["failures_by_category"] == {"timeout": 2, "rate_limit": 1}

    def test_latency_percentiles_are_over_successful_calls_only(self):
        records = [outcome(total_ms=100), outcome(total_ms=200), outcome(status="error", total_ms=99999)]
        latency = run_e2e.summarize(records)["latency_ms"]
        assert latency["n"] == 2 and latency["p50"] == pytest.approx(150)

    def test_tokens_and_cost_report_how_many_were_priced(self):
        priced = outcome(tokens_in=100, tokens_out=10, cost=0.001)
        unpriced = outcome(tokens_in=None, tokens_out=None, cost=None)
        records = [priced, unpriced]
        summary = run_e2e.summarize(records)
        assert summary["tokens"]["n_with_counts"] == 1 and summary["tokens"]["mean_input"] == pytest.approx(100)
        assert summary["estimated_cost_usd"] == {"total": pytest.approx(0.001), "n_priced": 1}


class TestRender:
    def test_a_partial_run_says_so(self):
        summary = run_e2e.summarize([outcome()])
        text = run_e2e.render(summary, sample_size=5)
        assert "1 of 5" in text and "partial" in text

    def test_a_complete_run_does_not_say_partial(self):
        summary = run_e2e.summarize([outcome(), outcome()])
        text = run_e2e.render(summary, sample_size=2)
        assert "partial" not in text

    def test_it_states_plainly_that_this_is_not_a_benchmark(self):
        text = run_e2e.render(run_e2e.summarize([]), sample_size=0)
        assert "NOT a benchmark" in text or "not a headline claim" in text.lower()

    def test_every_family_row_appears_even_with_no_data(self):
        text = run_e2e.render(run_e2e.summarize([]), sample_size=0)
        for family in ("id", "near_ood", "far_ood"):
            assert f"| {family} |" in text

    def test_cost_is_labelled_as_an_estimate(self):
        summary = run_e2e.summarize([outcome(cost=0.01)])
        assert "actual billed cost unknown" in run_e2e.render(summary, 1)

    def test_no_p99_anywhere(self):
        text = run_e2e.render(run_e2e.summarize([outcome(), outcome()]), 2)
        assert "p99" not in text.lower()

    def test_blank_line_separators_survive_even_when_a_conditional_line_is_omitted(self):
        """Regression: the final join used to filter out EVERY blank string, including the
        intentional paragraph breaks, not just the one conditionally-omitted line (mean attempts
        when there is no data) -- so the whole report ran together with no spacing at all."""
        text = run_e2e.render(run_e2e.summarize([]), sample_size=0)  # mean_attempts is None here
        assert "\n\n" in text  # at least one real blank line survived
        assert text.count("\n\n") >= 3  # title, summary and disclaimer are each followed by one


class TestMainResumability:
    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir()
        id_doc = {
            "entries": [
                {"source_id": f"oxford102:image_{i:05d}", "label": "Rose", "group": "rose",
                 "split": "pilot", "path": "a.jpg"}
                for i in range(3)
            ]
        }
        empty = {"entries": []}
        (manifest_dir / "id.json").write_text(json.dumps(id_doc), encoding="utf-8")
        (manifest_dir / "near_ood.json").write_text(json.dumps(empty), encoding="utf-8")
        (manifest_dir / "far_ood.json").write_text(json.dumps(empty), encoding="utf-8")
        write_image(tmp_path / "a.jpg")
        monkeypatch.setattr(run_e2e, "MANIFEST_DIR", manifest_dir)
        monkeypatch.setattr(run_e2e, "RAW_ROOT", tmp_path)
        monkeypatch.setattr(run_e2e, "CACHE_PATH", tmp_path / "cache.jsonl")
        monkeypatch.setattr(run_e2e, "RESULTS_PATH", tmp_path / "results.md")
        monkeypatch.setattr(run_e2e, "git_revision", lambda: "fixedrev")

    def test_max_calls_bounds_how_many_new_identify_calls_happen(self, monkeypatch, make_identify_result):
        calls = []
        monkeypatch.setattr(run_e2e, "identify", lambda _i: calls.append(1) or make_identify_result())
        monkeypatch.setattr(sys, "argv", ["run_e2e.py", "--max-calls", "1"])
        run_e2e.main()
        assert len(calls) == 1

    def test_a_second_run_resumes_without_recalling_already_cached_images(self, monkeypatch, make_identify_result):
        calls = []
        monkeypatch.setattr(run_e2e, "identify", lambda _i: calls.append(1) or make_identify_result())
        monkeypatch.setattr(sys, "argv", ["run_e2e.py", "--max-calls", "1"])
        run_e2e.main()
        run_e2e.main()
        run_e2e.main()
        assert len(calls) == 3  # one new call per run, three distinct pilot images total

        monkeypatch.setattr(sys, "argv", ["run_e2e.py", "--max-calls", "10"])
        run_e2e.main()
        assert len(calls) == 3  # nothing left to do: no new calls

    def test_a_crash_mid_run_leaves_prior_progress_cached(self, monkeypatch, make_identify_result):
        outcomes = [make_identify_result(), RuntimeError("network blip")]

        def flaky(_image):
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(run_e2e, "identify", flaky)
        monkeypatch.setattr(sys, "argv", ["run_e2e.py", "--max-calls", "5"])
        run_e2e.main()  # first image succeeds, second raises -> caught, loop stops
        assert len(run_e2e.load_cache()) == 1

    def test_the_results_file_reflects_a_partial_run(self, monkeypatch, make_identify_result):
        monkeypatch.setattr(run_e2e, "identify", lambda _i: make_identify_result())
        monkeypatch.setattr(sys, "argv", ["run_e2e.py", "--max-calls", "1"])
        run_e2e.main()
        text = (run_e2e.RESULTS_PATH).read_text(encoding="utf-8")
        assert "1 of 3" in text and "partial" in text
