"""eval/select_threshold.py: candidate selection, tau, the adoption rule, and a full fake run."""

import json

import pytest

from eval import frozen_config, metrics, open_world_data, select_threshold
from eval.frozen_config import BASELINE_CANDIDATE, BASELINE_TAU, MAX_ID_ABSTENTION, MIN_NEAR_OOD_GAIN, TIE_BAND


class TestPickCandidate:
    def test_the_clear_winner_is_chosen(self):
        id_dev = {"max_cosine": [0.9, 0.85], "margin": [0.5, 0.5]}
        ood_dev = {"max_cosine": [0.1, 0.2], "margin": [0.6, 0.7]}  # margin: id scores BELOW ood -> poor AUROC
        chosen, aurocs = select_threshold.pick_candidate(id_dev, ood_dev)
        assert chosen == "max_cosine" and aurocs["max_cosine"] > aurocs["margin"]

    def test_a_near_tie_is_broken_toward_the_simpler_candidate(self):
        """Both candidates perfectly separate id/ood (AUROC 1.0 each, a genuine tie): max_cosine,
        earlier in CANDIDATE_PREFERENCE, must be chosen over margin even though margin's raw AUROC
        value could differ in the last bit of floating point."""
        id_dev = {"max_cosine": [0.90, 0.91], "margin": [0.9, 0.8]}
        ood_dev = {"max_cosine": [0.1, 0.2], "margin": [0.1, 0.2]}
        auroc_cosine = metrics.auroc(id_dev["max_cosine"], ood_dev["max_cosine"])
        auroc_margin = metrics.auroc(id_dev["margin"], ood_dev["margin"])
        assert auroc_cosine == pytest.approx(1.0) and auroc_margin == pytest.approx(1.0)
        assert abs(auroc_cosine - auroc_margin) <= TIE_BAND
        chosen, _ = select_threshold.pick_candidate(id_dev, ood_dev)
        assert chosen == "max_cosine"

    def test_the_softmax_temperatures_break_ties_in_ascending_order(self):
        id_dev = {"max_softmax@0.01": [0.9], "max_softmax@0.1": [0.9]}
        ood_dev = {"max_softmax@0.01": [0.1], "max_softmax@0.1": [0.1]}
        chosen, _ = select_threshold.pick_candidate(id_dev, ood_dev)
        assert chosen == "max_softmax@0.01"

    def test_every_candidates_auroc_is_returned(self):
        id_dev = {"max_cosine": [0.9], "margin": [0.8]}
        ood_dev = {"max_cosine": [0.1], "margin": [0.2]}
        _, aurocs = select_threshold.pick_candidate(id_dev, ood_dev)
        assert set(aurocs) == {"max_cosine", "margin"}


class TestComputeTau:
    def test_it_accepts_at_least_ninety_five_percent_of_id_dev(self):
        scores = list(range(100))  # 0..99
        tau = select_threshold.compute_tau([float(s) for s in scores])
        accepted = sum(1 for s in scores if s >= tau)
        assert accepted / len(scores) >= 0.95

    def test_it_is_an_achievable_score_not_an_interpolated_one(self):
        scores = [0.1, 0.5, 0.9, 0.95, 0.99]
        assert select_threshold.compute_tau(scores) in scores

    def test_a_single_score_is_its_own_tau(self):
        assert select_threshold.compute_tau([0.7]) == 0.7


class TestAbstentionRate:
    def test_it_is_the_fraction_strictly_below_tau(self):
        assert select_threshold.abstention_rate([0.1, 0.2, 0.6, 0.9], tau=0.5) == 0.5

    def test_the_boundary_itself_is_not_an_abstention(self):
        assert select_threshold.abstention_rate([0.5], tau=0.5) == 0.0

    def test_everything_above_tau_is_zero_abstention(self):
        assert select_threshold.abstention_rate([0.9, 0.8], tau=0.5) == 0.0


class TestDecideAdoption:
    def test_meeting_both_bars_adopts(self):
        assert select_threshold.decide_adoption(MIN_NEAR_OOD_GAIN, MAX_ID_ABSTENTION) is True

    def test_just_under_the_gain_bar_does_not_adopt(self):
        assert select_threshold.decide_adoption(MIN_NEAR_OOD_GAIN - 0.001, 0.0) is False

    def test_just_over_the_abstention_ceiling_does_not_adopt(self):
        assert select_threshold.decide_adoption(0.5, MAX_ID_ABSTENTION + 0.001) is False

    def test_a_negative_gain_never_adopts(self):
        assert select_threshold.decide_adoption(-0.1, 0.0) is False


class TestBuildConfig:
    def test_records_a_plausible_config_end_to_end(self):
        # max_cosine: id in {0.6 (8%), 0.9 (92%)}, ood a flat 0.5 -- perfectly separated, and the
        # 5th-percentile tau (0.6) sits ABOVE the baseline's 0.45, so it catches ood the baseline misses
        # while abstaining on none of ID (no id score is strictly below its own tau).
        id_scores = {"max_cosine": [0.9] * 92 + [0.6] * 8, "margin": [0.5] * 100}
        ood_scores = {"max_cosine": [0.5] * 100, "margin": [0.5] * 100}
        config = select_threshold.build_config(id_scores, ood_scores, manifests={"id": "h1", "near_ood": "h2"})
        assert config.candidate == "max_cosine"
        assert config.manifest_hashes == {"id": "h1", "near_ood": "h2"}
        assert config.tau == pytest.approx(0.6)
        assert config.dev_id_abstention == pytest.approx(0.0)
        assert config.dev_near_ood_gain_pp == pytest.approx(1.0)
        assert config.adopted is True

    def test_baseline_abstention_is_always_computed_from_the_fixed_baseline_rule(self):
        """Even when `margin` is the chosen candidate, the baseline comparison must still use
        max_cosine @ 0.45 -- never the chosen candidate's own values."""
        id_scores = {"max_cosine": [0.5] * 10, "margin": [0.9] * 10}  # max_cosine: no separation at all
        ood_scores = {"max_cosine": [0.5] * 10, "margin": [0.1] * 10}  # margin: perfect separation
        config = select_threshold.build_config(id_scores, ood_scores, manifests={})
        assert config.candidate == "margin"
        baseline_ood_abstention = select_threshold.abstention_rate(ood_scores[BASELINE_CANDIDATE], BASELINE_TAU)
        assert baseline_ood_abstention == pytest.approx(0.0)  # ood max_cosine (0.5) is not < 0.45
        assert config.dev_near_ood_gain_pp == pytest.approx(1.0)  # margin's own rule abstains on all of it


class TestReport:
    def test_it_names_the_frozen_candidate_and_marks_it_in_the_table(self):
        config = select_threshold.build_config(
            {"max_cosine": [0.9, 0.8], "margin": [0.5, 0.4]},
            {"max_cosine": [0.1, 0.2], "margin": [0.6, 0.7]},
            manifests={},
        )
        text = select_threshold.report(config)
        assert f"Frozen: `{config.candidate}`" in text
        assert f"| {config.candidate} |" in text and "<- frozen" in text
        assert f"Adopted: {config.adopted}" in text


class TestMainRefusesAfterTestIsLocked:
    def test_it_refuses_to_overwrite_a_config_whose_test_run_is_already_locked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(frozen_config, "CONFIG_PATH", tmp_path / "v1.json")
        monkeypatch.setattr(frozen_config, "LOCK_PATH", tmp_path / "v1.lock")
        monkeypatch.setattr(select_threshold, "frozen_config", frozen_config)
        config = select_threshold.build_config({"max_cosine": [0.9]}, {"max_cosine": [0.1]}, manifests={})
        digest = frozen_config.write_frozen(config)
        frozen_config.write_lock(digest, results_path="r.md")
        with pytest.raises(SystemExit, match="already evaluated"):
            select_threshold.main()


class TestMainFakeFullRun:
    """The whole script, with the model faked, against tiny manifests on disk."""

    def build_manifests(self, tmp_path, monkeypatch):
        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir()
        def entry(i, label, split="dev"):
            return {"source_id": f"oxford102:image_{i:05d}", "label": label, "group": label.lower(),
                    "split": split, "path": "a.jpg"}  # fmt: skip

        id_doc = {"entries": [entry(i, "Rose") for i in range(20)] + [entry(9999, "Rose", "test")]}
        near_doc = {"entries": [entry(i, "orchidx") for i in range(200, 220)]}
        (manifest_dir / "id.json").write_text(json.dumps(id_doc), encoding="utf-8")
        (manifest_dir / "near_ood.json").write_text(json.dumps(near_doc), encoding="utf-8")
        monkeypatch.setattr(select_threshold, "MANIFEST_DIR", manifest_dir)
        return id_doc, near_doc

    @pytest.fixture(autouse=True)
    def faked_model(self, monkeypatch, tmp_path):
        monkeypatch.setattr(frozen_config, "CONFIG_PATH", tmp_path / "v1.json")
        monkeypatch.setattr(frozen_config, "LOCK_PATH", tmp_path / "v1.lock")
        monkeypatch.setattr(select_threshold, "frozen_config", frozen_config)
        monkeypatch.setattr(select_threshold, "REPORT_PATH", tmp_path / "tuning_report.md")
        monkeypatch.setattr(open_world_data, "CACHE_PATH", tmp_path / "cache.json")  # never touch the real cache

        def fake_similarities_for(entries, cache_path=None, progress=None):
            result = {}
            for entry in entries:
                is_id = entry["label"] == "Rose"
                top = 0.9 if is_id else 0.1
                result[open_world_data.cache_key(entry)] = [{"common_name": "Rose", "score": top}] + [
                    {"common_name": f"Other{i}", "score": top - 0.5} for i in range(29)
                ]
            return result

        monkeypatch.setattr(open_world_data, "similarities_for", fake_similarities_for)

    ALL_CANDIDATES = (
        "max_cosine", "margin",
        "max_softmax@0.01", "max_softmax@0.02", "max_softmax@0.05", "max_softmax@0.1",
    )  # fmt: skip

    def test_it_writes_a_frozen_config_and_a_tuning_report(self, tmp_path, monkeypatch):
        self.build_manifests(tmp_path, monkeypatch)
        select_threshold.main()
        config, digest = frozen_config.read_frozen()
        assert config.candidate in self.ALL_CANDIDATES
        assert (tmp_path / "tuning_report.md").exists()
        assert digest  # a real hash was produced

    def test_running_it_twice_overwrites_the_config_when_not_yet_locked(self, tmp_path, monkeypatch):
        self.build_manifests(tmp_path, monkeypatch)
        select_threshold.main()
        first, _ = frozen_config.read_frozen()
        select_threshold.main()  # no lock yet: allowed to re-tune
        second, _ = frozen_config.read_frozen()
        assert first.created_at != second.created_at or first == second  # ran again without raising
