"""eval/run_open_world.py: the pure per-family/per-image metric assembly, the guard clauses
(no frozen config, already locked, BioCLIP revision drift, a changed manifest), and a full fake run."""

import json

import pytest

from eval import build_datasets, frozen_config, open_world_data, run_open_world, select_threshold
from src.versions import BIOCLIP_REVISION


def key(entry):
    return open_world_data.cache_key(entry)


def sim(top_score, top_name="Rose", n=30):
    return [{"common_name": top_name, "score": top_score}] + [
        {"common_name": f"Other{i}", "score": top_score - 0.3 - i * 0.001} for i in range(n - 1)
    ]


class TestGroupBy:
    def test_it_groups_values_by_the_named_key_preserving_order_within_a_group(self):
        entries = [{"group": "a"}, {"group": "b"}, {"group": "a"}]
        assert run_open_world.group_by(entries, "group", [1, 2, 3]) == {"a": [1, 3], "b": [2]}

    def test_an_empty_input_is_an_empty_mapping(self):
        assert run_open_world.group_by([], "group", []) == {}

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            run_open_world.group_by([{"group": "a"}], "group", [1, 2])


class TestScoresOf:
    def test_it_reads_the_named_candidate_for_every_entry(self):
        similarities = {"s1": sim(0.8), "s2": sim(0.6)}
        entries = [{"source_id": "s1"}, {"source_id": "s2"}]
        assert run_open_world.scores_of(similarities, entries, "max_cosine") == pytest.approx([0.8, 0.6])


class TestFamilyMetrics:
    def test_it_reports_the_shape_expected_by_render(self):
        id_scores = [0.9] * 20
        ood_entries = [{"group": "catA"}] * 10 + [{"group": "catB"}] * 10
        ood_scores = [0.1] * 20
        result = run_open_world.family_metrics(id_scores, ood_entries, ood_scores, tau=0.5)
        assert result["n_images"] == 20 and result["n_categories"] == 2
        for key in ("auroc", "aupr_out", "fpr_at_95tpr", "abstention_rate"):
            assert "point" in result[key] and "low" in result[key] and "high" in result[key]

    def test_perfect_separation_gives_auroc_near_one_and_full_abstention(self):
        id_scores = [0.9] * 30
        ood_entries = [{"group": f"cat{i}"} for i in range(5) for _ in range(6)]
        ood_scores = [0.1] * 30
        result = run_open_world.family_metrics(id_scores, ood_entries, ood_scores, tau=0.5)
        assert result["auroc"]["point"] == pytest.approx(1.0)
        assert result["abstention_rate"]["point"] == pytest.approx(1.0)

    def test_abstention_rate_uses_the_given_tau_not_a_fixed_one(self):
        id_scores = [0.9] * 10
        ood_entries = [{"group": "a"}] * 10
        low_tau = run_open_world.family_metrics(id_scores, ood_entries, [0.3] * 10, tau=0.2)
        high_tau = run_open_world.family_metrics(id_scores, ood_entries, [0.3] * 10, tau=0.5)
        assert low_tau["abstention_rate"]["point"] == pytest.approx(0.0)
        assert high_tau["abstention_rate"]["point"] == pytest.approx(1.0)


class TestIdMetrics:
    def entries_and_similarities(self, correct_flags, tau):
        entries, similarities = [], {}
        for i, correct in enumerate(correct_flags):
            source_id = f"oxford102:image_{i:05d}"
            entries.append({"source_id": source_id, "label": "Rose"})
            top_score = 0.9 if correct else 0.9  # score is independent of correctness in this fixture
            name = "Rose" if correct else "NotRose"
            similarities[source_id] = sim(top_score, top_name=name)
        return entries, similarities

    def test_top1_correctness_is_read_from_the_similarities_not_assumed(self):
        entries, similarities = self.entries_and_similarities([True, True, False, False], tau=0.5)
        scores = run_open_world.scores_of(similarities, entries, "max_cosine")
        result = run_open_world.id_metrics(entries, similarities, scores, tau=0.5)
        assert result["top1_accuracy_unfiltered"] == pytest.approx(0.5)

    def test_coverage_and_false_abstain_rate_at_a_given_tau(self):
        entries, similarities = self.entries_and_similarities([True] * 8 + [False] * 2, tau=0.5)
        scores = [0.9] * 5 + [0.1] * 5  # half above tau, half below
        result = run_open_world.id_metrics(entries, similarities, scores, tau=0.5)
        assert result["coverage_at_tau"]["point"] == pytest.approx(0.5)
        assert result["false_abstain_rate"]["point"] == pytest.approx(0.5)

    def test_it_reports_a_risk_coverage_auc(self):
        entries, similarities = self.entries_and_similarities([True, True, False, False], tau=0.5)
        scores = [0.9, 0.8, 0.2, 0.1]
        result = run_open_world.id_metrics(entries, similarities, scores, tau=0.5)
        assert 0 <= result["risk_coverage_auc"]["point"] <= 1


class TestCorruptedMetrics:
    def test_it_groups_by_variant_and_reports_accuracy_and_abstention_per_variant(self):
        entries = [
            {"source_id": "a1", "label": "Rose", "variant": "dark"},
            {"source_id": "a2", "label": "Rose", "variant": "dark"},
            {"source_id": "a3", "label": "Rose", "variant": "rotate"},
        ]
        similarities = {
            key(entries[0]): sim(0.9, top_name="Rose"),  # correct, above tau
            key(entries[1]): sim(0.1, top_name="NotRose"),  # incorrect, below tau
            key(entries[2]): sim(0.9, top_name="Rose"),  # correct, above tau
        }
        result = run_open_world.corrupted_metrics(entries, similarities, "max_cosine", tau=0.5)
        assert set(result) == {"dark", "rotate"}
        assert result["dark"]["n_images"] == 2
        assert result["dark"]["top1_accuracy"]["point"] == pytest.approx(0.5)
        assert result["dark"]["abstention_rate"]["point"] == pytest.approx(0.5)
        assert result["rotate"]["top1_accuracy"]["point"] == pytest.approx(1.0)

    def test_variants_are_reported_in_a_stable_sorted_order(self):
        variants = ["rotate", "dark", "crop"]
        entries = [{"source_id": f"s{i}", "label": "Rose", "variant": v} for i, v in enumerate(variants)]
        similarities = {key(e): sim(0.9) for e in entries}
        result = run_open_world.corrupted_metrics(entries, similarities, "max_cosine", tau=0.1)
        assert list(result) == ["crop", "dark", "rotate"]


class TestRender:
    def make_ci(self, point):
        return {"point": point, "low": point - 0.1, "high": point + 0.1}

    def test_it_includes_the_frozen_rule_and_every_family(self):
        config = frozen_config.FrozenConfig(
            candidate="max_cosine", tau=0.55, adopted=True, dev_near_ood_gain_pp=0.2, dev_id_abstention=0.03,
            candidate_auroc={}, manifest_hashes={}, bioclip_revision="rev1", code_revision="abc123",
            created_at="2026-09-22T00:00:00+00:00",
        )  # fmt: skip
        family_results = {
            "near_ood": {"n_images": 10, "n_categories": 2, "auroc": self.make_ci(0.9), "aupr_out": self.make_ci(0.8),
                         "fpr_at_95tpr": self.make_ci(0.1), "abstention_rate": self.make_ci(0.7)},  # fmt: skip
        }
        id_results = {
            "n_images": 50, "top1_accuracy_unfiltered": 0.87, "false_abstain_rate": self.make_ci(0.05),
            "coverage_at_tau": self.make_ci(0.95), "selective_accuracy_at_tau": self.make_ci(0.99),
            "risk_coverage_auc": self.make_ci(0.02),
        }  # fmt: skip
        corrupted_results = {
            "dark": {"n_images": 20, "top1_accuracy": self.make_ci(0.6), "abstention_rate": self.make_ci(0.3)}
        }
        text = run_open_world.render(config, family_results, id_results, corrupted_results, "deadbeef" * 8)
        assert "max_cosine" in text and "0.550000" in text
        assert "near_ood" in text and "dark" in text
        assert "87.0%" in text
        assert "quality (no labels)" in text  # the not-evaluated statement is present

    def test_an_unadopted_config_says_so_and_names_the_tuning_report(self):
        config = frozen_config.FrozenConfig(
            candidate="margin", tau=0.4, adopted=False, dev_near_ood_gain_pp=0.02, dev_id_abstention=0.2,
            candidate_auroc={}, manifest_hashes={}, bioclip_revision="rev1", code_revision="abc123",
            created_at="2026-09-22T00:00:00+00:00",
        )  # fmt: skip
        empty_id_results = {
            "n_images": 0, "top1_accuracy_unfiltered": 0, "false_abstain_rate": self.make_ci(0),
            "coverage_at_tau": self.make_ci(0), "selective_accuracy_at_tau": self.make_ci(0),
            "risk_coverage_auc": self.make_ci(0),
        }  # fmt: skip
        text = run_open_world.render(config, {}, empty_id_results, {}, "hash")
        assert "baseline kept" in text
        assert frozen_config.BASELINE_CANDIDATE in text


class TestMainGuardClauses:
    """Every test here points MANIFEST_DIR at empty manifests (not the ~2 600-image real ones) and
    never fakes the model: if a guard clause under test were accidentally removed, main() must
    fail fast on the next thing (empty data) rather than silently kick off several minutes of real
    BioCLIP/Qdrant work against the committed manifests -- which is what happened the one time this
    suite ran with `already_run`'s check disabled, and is exactly the failure mode a guard-clause
    test must not depend on someone else (the fake-model fixture) to prevent."""

    @pytest.fixture(autouse=True)
    def empty_manifests(self, tmp_path, monkeypatch):
        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir()
        for name in ("id", "near_ood", "far_ood", "corrupted"):
            (manifest_dir / f"{name}.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
        monkeypatch.setattr(run_open_world, "MANIFEST_DIR", manifest_dir)
        monkeypatch.setattr(frozen_config, "CONFIG_PATH", tmp_path / "v1.json")
        monkeypatch.setattr(frozen_config, "LOCK_PATH", tmp_path / "v1.lock")
        monkeypatch.setattr(run_open_world, "frozen_config", frozen_config)
        monkeypatch.setattr(open_world_data, "CACHE_PATH", tmp_path / "cache.json")  # never touch the real cache

    def test_no_frozen_config_refuses_with_a_clear_message(self):
        with pytest.raises(SystemExit, match="Run eval/select_threshold.py first"):
            run_open_world.main()

    def make_config(self, **overrides):
        fields = dict(
            candidate="max_cosine", tau=0.5, adopted=True, dev_near_ood_gain_pp=0.2, dev_id_abstention=0.02,
            candidate_auroc={}, manifest_hashes={}, bioclip_revision=BIOCLIP_REVISION, code_revision="abc",
            created_at="2026-09-22T00:00:00+00:00",
        )  # fmt: skip
        fields.update(overrides)
        return frozen_config.FrozenConfig(**fields)

    def test_an_already_locked_config_refuses(self):
        digest = frozen_config.write_frozen(self.make_config())
        frozen_config.write_lock(digest, results_path="r.md")
        with pytest.raises(SystemExit, match="already records a completed run"):
            run_open_world.main()

    def test_a_bioclip_revision_mismatch_refuses(self):
        frozen_config.write_frozen(self.make_config(bioclip_revision="a-different-revision"))
        with pytest.raises(SystemExit, match="Re-run eval/select_threshold.py first"):
            run_open_world.main()

    def test_a_changed_manifest_since_freezing_refuses(self):
        frozen_config.write_frozen(self.make_config(manifest_hashes={"id": "not-the-real-hash"}))
        with pytest.raises(SystemExit, match="has changed since the config was frozen"):
            run_open_world.main()

    def test_removing_the_lock_check_is_still_caught_even_with_empty_manifests(self, monkeypatch):
        """The mutation this class exists to catch, made explicit: with `already_run` short-circuited,
        main() must still fail -- here, on the empty ID split -- rather than proceed to fabricate a
        result. Whatever it fails with, it must not be a silent success."""
        monkeypatch.setattr(frozen_config, "already_run", lambda *a, **k: False)
        digest = frozen_config.write_frozen(self.make_config())
        frozen_config.write_lock(digest, results_path="r.md")
        with pytest.raises(BaseException):  # noqa: B017 -- deliberately broad: ANY failure is acceptable, a clean run is not
            run_open_world.main()


class TestMainFakeFullRun:
    def entry(self, i, label, split, **extra):
        return {"source_id": f"oxford102:image_{i:05d}", "label": label, "group": label.lower(), "split": split,
                "path": "a.jpg", **extra}  # fmt: skip

    def build_manifests(self, tmp_path, monkeypatch):
        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir()
        id_doc = {"entries": [self.entry(i, "Rose", "test") for i in range(10)]}
        near_doc = {"entries": [self.entry(i, "orchidx", "test") for i in range(100, 105)]}
        far_doc = {"entries": [self.entry(i, "airplane", "test") for i in range(200, 205)]}
        corrupted_doc = {
            "entries": [self.entry(i, "Rose", "test", variant=v) for i, v in enumerate(["dark", "crop"], start=300)]
        }
        for name, doc in (("id", id_doc), ("near_ood", near_doc), ("far_ood", far_doc), ("corrupted", corrupted_doc)):
            (manifest_dir / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
        monkeypatch.setattr(run_open_world, "MANIFEST_DIR", manifest_dir)
        return {"id": build_datasets.manifest_hash(id_doc), "near_ood": build_datasets.manifest_hash(near_doc)}

    @pytest.fixture(autouse=True)
    def faked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(frozen_config, "CONFIG_PATH", tmp_path / "v1.json")
        monkeypatch.setattr(frozen_config, "LOCK_PATH", tmp_path / "v1.lock")
        monkeypatch.setattr(run_open_world, "frozen_config", frozen_config)
        monkeypatch.setattr(run_open_world, "RESULTS_PATH", tmp_path / "open_world_results.md")
        monkeypatch.setattr(open_world_data, "CACHE_PATH", tmp_path / "cache.json")  # belt and braces

        def fake_similarities_for(entries, cache_path=None, progress=None):
            result = {}
            for entry in entries:
                is_id_like = entry["label"] == "Rose"
                result[open_world_data.cache_key(entry)] = sim(0.9 if is_id_like else 0.1, top_name=entry["label"])
            return result

        monkeypatch.setattr(open_world_data, "similarities_for", fake_similarities_for)

    def test_it_writes_results_and_locks_the_config(self, tmp_path, monkeypatch):
        hashes = self.build_manifests(tmp_path, monkeypatch)
        digest = frozen_config.write_frozen(
            select_threshold.build_config(
                {"max_cosine": [0.9] * 20}, {"max_cosine": [0.1] * 20}, manifests=hashes
            )
        )
        run_open_world.main()
        assert (tmp_path / "open_world_results.md").exists()
        assert frozen_config.already_run(digest)

    def test_running_it_again_for_the_same_config_refuses(self, tmp_path, monkeypatch):
        hashes = self.build_manifests(tmp_path, monkeypatch)
        frozen_config.write_frozen(
            select_threshold.build_config({"max_cosine": [0.9] * 20}, {"max_cosine": [0.1] * 20}, manifests=hashes)
        )
        run_open_world.main()
        with pytest.raises(SystemExit, match="already records a completed run"):
            run_open_world.main()

    def test_the_results_file_mentions_every_family_and_the_config_hash(self, tmp_path, monkeypatch):
        hashes = self.build_manifests(tmp_path, monkeypatch)
        digest = frozen_config.write_frozen(
            select_threshold.build_config({"max_cosine": [0.9] * 20}, {"max_cosine": [0.1] * 20}, manifests=hashes)
        )
        run_open_world.main()
        text = (tmp_path / "open_world_results.md").read_text(encoding="utf-8")
        assert "near_ood" in text and "far_ood" in text and "Corrupted-in-set" in text and digest[:12] in text
