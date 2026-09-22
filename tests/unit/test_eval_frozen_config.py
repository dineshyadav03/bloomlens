"""eval/frozen_config.py: the frozen file's self-verifying hash and the lock (eval/PROTOCOL.md
sections 4-5): a config edited after the fact is caught, and a completed run is never repeated."""

import json

import pytest

from eval import frozen_config
from eval.frozen_config import FrozenConfig, TamperedConfigError


def make_config(**overrides) -> FrozenConfig:
    fields = dict(
        candidate="max_cosine",
        tau=0.5,
        adopted=True,
        dev_near_ood_gain_pp=0.12,
        dev_id_abstention=0.04,
        candidate_auroc={"max_cosine": 0.9, "margin": 0.85},
        manifest_hashes={"id": "aaa", "near_ood": "bbb"},
        bioclip_revision="rev123",
        code_revision="deadbeef",
        created_at="2026-09-22T00:00:00+00:00",
    )
    fields.update(overrides)
    return FrozenConfig(**fields)


class TestContentHash:
    def test_key_order_does_not_matter(self):
        assert frozen_config.content_hash({"a": 1, "b": 2}) == frozen_config.content_hash({"b": 2, "a": 1})

    def test_a_different_value_changes_the_hash(self):
        assert frozen_config.content_hash({"a": 1}) != frozen_config.content_hash({"a": 2})

    def test_it_is_a_hex_sha256(self):
        digest = frozen_config.content_hash({"a": 1})
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


class TestWriteAndReadFrozen:
    def test_round_trips_exactly(self, tmp_path):
        path = tmp_path / "v1.json"
        config = make_config()
        digest = frozen_config.write_frozen(config, path)
        loaded, loaded_hash = frozen_config.read_frozen(path)
        assert loaded == config and loaded_hash == digest

    def test_the_file_carries_its_own_hash_field(self, tmp_path):
        path = tmp_path / "v1.json"
        frozen_config.write_frozen(make_config(), path)
        assert "self_sha256" in json.loads(path.read_text(encoding="utf-8"))

    def test_a_hand_edit_to_a_value_is_caught(self, tmp_path):
        path = tmp_path / "v1.json"
        frozen_config.write_frozen(make_config(), path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["tau"] = 0.99  # edited without updating self_sha256
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(TamperedConfigError, match="does not match its own recorded hash"):
            frozen_config.read_frozen(path)

    def test_a_hand_edit_to_the_recorded_hash_itself_is_also_caught(self, tmp_path):
        path = tmp_path / "v1.json"
        frozen_config.write_frozen(make_config(), path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["self_sha256"] = "0" * 64
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(TamperedConfigError):
            frozen_config.read_frozen(path)

    def test_a_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            frozen_config.read_frozen(tmp_path / "nope.json")

    def test_two_different_configs_have_different_hashes(self, tmp_path):
        a = frozen_config.write_frozen(make_config(tau=0.1), tmp_path / "a.json")
        b = frozen_config.write_frozen(make_config(tau=0.2), tmp_path / "b.json")
        assert a != b

    def test_the_default_path_follows_a_monkeypatched_config_path_not_the_one_at_import_time(
        self, tmp_path, monkeypatch
    ):
        """Regression: a default argument bound to the module-level Path object at import time
        (the same trap found in src/guard.py, src/quota.py and src/llm_cost.py earlier in this
        project) would silently keep writing to the ORIGINAL location no matter what a caller
        monkeypatches CONFIG_PATH to."""
        monkeypatch.setattr(frozen_config, "CONFIG_PATH", tmp_path / "v1.json")
        frozen_config.write_frozen(make_config())  # no explicit path: must use the patched CONFIG_PATH
        assert (tmp_path / "v1.json").exists()
        loaded, _ = frozen_config.read_frozen()  # same for the read side
        assert loaded == make_config()

    def test_already_run_and_write_lock_also_follow_a_monkeypatched_lock_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(frozen_config, "LOCK_PATH", tmp_path / "v1.lock")
        frozen_config.write_lock("abc123", results_path="r.md")
        assert (tmp_path / "v1.lock").exists()
        assert frozen_config.already_run("abc123") is True


class TestEffectiveRule:
    def test_when_adopted_the_tuned_candidate_and_tau_are_in_force(self):
        config = make_config(adopted=True, candidate="margin", tau=0.3)
        assert (config.effective_candidate, config.effective_tau) == ("margin", 0.3)

    def test_when_not_adopted_the_baseline_is_in_force(self):
        config = make_config(adopted=False, candidate="margin", tau=0.3)
        assert (config.effective_candidate, config.effective_tau) == (
            frozen_config.BASELINE_CANDIDATE,
            frozen_config.BASELINE_TAU,
        )


class TestLock:
    def test_no_lock_file_means_not_yet_run(self, tmp_path):
        assert frozen_config.already_run("anyhash", tmp_path / "v1.lock") is False

    def test_a_matching_lock_means_already_run(self, tmp_path):
        path = tmp_path / "v1.lock"
        frozen_config.write_lock("abc123", results_path="open_world_results.md", path=path)
        assert frozen_config.already_run("abc123", path) is True

    def test_a_lock_for_a_different_config_hash_does_not_block_this_one(self, tmp_path):
        path = tmp_path / "v1.lock"
        frozen_config.write_lock("some-other-hash", results_path="r.md", path=path)
        assert frozen_config.already_run("abc123", path) is False

    def test_the_lock_records_when_it_was_written_and_where_the_results_are(self, tmp_path):
        path = tmp_path / "v1.lock"
        frozen_config.write_lock("abc123", results_path="open_world_results.md", path=path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["results_path"] == "open_world_results.md" and "written_at" in payload

    def test_writing_the_lock_creates_its_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "v1.lock"
        frozen_config.write_lock("abc", results_path="r.md", path=path)
        assert path.exists()


def test_git_revision_returns_a_real_looking_hash_in_this_repo():
    revision = frozen_config.git_revision()
    assert revision == "unknown" or (len(revision) == 40 and all(c in "0123456789abcdef" for c in revision))


def test_now_iso_is_a_parseable_utc_timestamp():
    import datetime

    parsed = datetime.datetime.fromisoformat(frozen_config.now_iso())
    assert parsed.tzinfo is not None
