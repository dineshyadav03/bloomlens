"""eval/open_world_data.py: the similarity cache and the corruption-aware image loader.

BioCLIP and Qdrant are faked throughout -- these tests are about the caching and plumbing, not
about real embeddings (that is covered, on real data, by tests/integration/test_eval_data.py and
by actually running the scripts)."""

import json
from pathlib import Path

import pytest
from PIL import Image

from eval import open_world_data
from src.versions import BIOCLIP_REVISION


def make_image(path: Path, colour=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(path, format="JPEG")


class TestCacheKey:
    def test_a_plain_entry_is_keyed_by_its_source_id(self):
        assert open_world_data.cache_key({"source_id": "oxford102:image_00001"}) == "oxford102:image_00001"

    def test_a_variant_entry_includes_the_variant(self):
        entry = {"source_id": "oxford102:image_00001", "variant": "dark"}
        assert open_world_data.cache_key(entry) == "oxford102:image_00001|dark"

    def test_different_variants_of_the_same_source_have_different_keys(self):
        base = {"source_id": "x"}
        keys = {open_world_data.cache_key({**base, "variant": v}) for v in ("dark", "rotate", None)}
        assert len(keys) == 3


class TestCacheRoundTrip:
    def test_an_absent_file_is_an_empty_cache(self, tmp_path):
        assert open_world_data.load_cache(tmp_path / "missing.json") == {}

    def test_save_then_load_round_trips(self, tmp_path):
        path = tmp_path / "cache.json"
        open_world_data.save_cache({"a": [{"common_name": "Rose", "score": 0.5}]}, path)
        assert open_world_data.load_cache(path) == {"a": [{"common_name": "Rose", "score": 0.5}]}

    def test_a_cache_from_a_different_bioclip_revision_is_discarded(self, tmp_path):
        path = tmp_path / "cache.json"
        stale = {"bioclip_revision": "some-other-revision", "similarities": {"a": []}}
        path.write_text(json.dumps(stale), encoding="utf-8")
        assert open_world_data.load_cache(path) == {}

    def test_a_cache_from_the_current_revision_is_kept(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text(
            json.dumps({"bioclip_revision": BIOCLIP_REVISION, "similarities": {"a": [1]}}), encoding="utf-8"
        )
        assert open_world_data.load_cache(path) == {"a": [1]}

    def test_saving_creates_the_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "cache.json"
        open_world_data.save_cache({}, path)
        assert path.exists()

    def test_the_default_path_follows_a_monkeypatched_cache_path_not_the_one_at_import_time(
        self, tmp_path, monkeypatch
    ):
        """Regression: the same default-argument-bound-at-import trap found (and fixed) in
        src/guard.py, src/quota.py, src/llm_cost.py and eval/frozen_config.py earlier in this
        project. Without the fix, save_cache()/load_cache() called with no explicit path would
        silently keep using the ORIGINAL eval/cache/similarities.json no matter what a caller
        monkeypatches CACHE_PATH to -- which is exactly how a guard-clause test that forgot to
        pass cache_path once wrote 2.8 MB into the real project cache during this session."""
        monkeypatch.setattr(open_world_data, "CACHE_PATH", tmp_path / "cache.json")
        open_world_data.save_cache({"a": [1]})  # no explicit path
        assert (tmp_path / "cache.json").exists()
        assert open_world_data.load_cache() == {"a": [1]}  # no explicit path either


class TestLoadImage:
    def test_a_plain_entry_loads_the_image_unmodified(self, tmp_path, monkeypatch):
        monkeypatch.setattr(open_world_data, "RAW_ROOT", tmp_path)
        make_image(tmp_path / "a.jpg", (10, 20, 30))
        image = open_world_data.load_image({"source_id": "s", "path": "a.jpg"})
        assert image.getpixel((0, 0)) == (10, 20, 30)

    def test_a_variant_entry_applies_its_corruption(self, tmp_path, monkeypatch):
        monkeypatch.setattr(open_world_data, "RAW_ROOT", tmp_path)
        make_image(tmp_path / "a.jpg", (200, 200, 200))
        plain = open_world_data.load_image({"source_id": "s", "path": "a.jpg"})
        dark = open_world_data.load_image({"source_id": "s", "path": "a.jpg", "variant": "dark"})
        assert dark.getpixel((5, 5)) != plain.getpixel((5, 5))

    def test_the_variant_is_applied_deterministically_by_source_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(open_world_data, "RAW_ROOT", tmp_path)
        make_image(tmp_path / "a.jpg")
        entry = {"source_id": "oxford102:image_00042", "path": "a.jpg", "variant": "occlusion"}
        assert open_world_data.load_image(entry).tobytes() == open_world_data.load_image(entry).tobytes()


class TestAllSimilarities:
    def test_it_returns_every_species_with_its_score(self, monkeypatch):
        fake_results = [{"score": 0.9 - 0.01 * i, "payload": {"common_name": f"Species{i}"}} for i in range(30)]
        monkeypatch.setattr(open_world_data, "get_client", lambda: object())
        monkeypatch.setattr(open_world_data, "embed_image", lambda _img: [0.0])
        monkeypatch.setattr(open_world_data, "search", lambda *_a, **_k: fake_results)
        result = open_world_data.all_similarities(Image.new("RGB", (4, 4)))
        assert len(result) == 30 and result[0] == {"common_name": "Species0", "score": 0.9}

    def test_fewer_than_thirty_results_is_an_error(self, monkeypatch):
        monkeypatch.setattr(open_world_data, "get_client", lambda: object())
        monkeypatch.setattr(open_world_data, "embed_image", lambda _img: [0.0])
        one_result = [{"score": 0.5, "payload": {"common_name": "x"}}]
        monkeypatch.setattr(open_world_data, "search", lambda *_a, **_k: one_result)
        with pytest.raises(RuntimeError, match="expected 30"):
            open_world_data.all_similarities(Image.new("RGB", (4, 4)))


class TestSimilaritiesFor:
    @pytest.fixture
    def faked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(open_world_data, "RAW_ROOT", tmp_path)
        make_image(tmp_path / "a.jpg")
        calls = []

        def fake_all_similarities(image):
            calls.append(image)
            return [{"common_name": f"S{i}", "score": 1.0 - i * 0.01} for i in range(30)]

        monkeypatch.setattr(open_world_data, "all_similarities", fake_all_similarities)
        return calls

    def entries(self, n=3):
        return [{"source_id": f"oxford102:image_{i:05d}", "path": "a.jpg"} for i in range(n)]

    def test_it_computes_and_caches_every_missing_entry(self, faked, tmp_path):
        cache_path = tmp_path / "cache.json"
        result = open_world_data.similarities_for(self.entries(), cache_path=cache_path)
        assert len(faked) == 3 and len(result) == 3
        assert open_world_data.load_cache(cache_path)  # persisted to disk

    def test_a_second_call_uses_the_cache_and_embeds_nothing_new(self, faked, tmp_path):
        cache_path = tmp_path / "cache.json"
        entries = self.entries()
        open_world_data.similarities_for(entries, cache_path=cache_path)
        open_world_data.similarities_for(entries, cache_path=cache_path)
        assert len(faked) == 3  # not 6

    def test_a_partially_cached_batch_only_embeds_the_missing_entries(self, faked, tmp_path):
        cache_path = tmp_path / "cache.json"
        open_world_data.similarities_for(self.entries(2), cache_path=cache_path)
        assert len(faked) == 2
        open_world_data.similarities_for(self.entries(4), cache_path=cache_path)
        assert len(faked) == 2 + 2  # only the 2 new ones

    def test_progress_is_reported_once_per_new_computation(self, faked, tmp_path):
        seen = []
        open_world_data.similarities_for(
            self.entries(3), cache_path=tmp_path / "cache.json", progress=lambda d, t: seen.append((d, t))
        )
        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_no_progress_calls_when_everything_is_already_cached(self, faked, tmp_path):
        cache_path = tmp_path / "cache.json"
        entries = self.entries()
        open_world_data.similarities_for(entries, cache_path=cache_path)
        seen = []
        open_world_data.similarities_for(entries, cache_path=cache_path, progress=lambda d, t: seen.append((d, t)))
        assert seen == []

    def test_a_variant_entry_gets_its_own_cache_slot(self, faked, tmp_path):
        cache_path = tmp_path / "cache.json"
        plain = {"source_id": "oxford102:image_00001", "path": "a.jpg"}
        variant = {**plain, "variant": "dark"}
        open_world_data.similarities_for([plain, variant], cache_path=cache_path)
        assert len(faked) == 2

    def test_the_result_covers_every_requested_entry_by_its_cache_key(self, faked, tmp_path):
        entries = self.entries(2)
        result = open_world_data.similarities_for(entries, cache_path=tmp_path / "cache.json")
        assert set(result) == {open_world_data.cache_key(e) for e in entries}

    def test_with_no_explicit_cache_path_it_follows_a_monkeypatched_cache_path(self, faked, tmp_path, monkeypatch):
        """The same regression as TestCacheRoundTrip's, exercised through the real call site
        (eval/run_open_world.py's embed()) that has no explicit cache_path of its own."""
        monkeypatch.setattr(open_world_data, "CACHE_PATH", tmp_path / "cache.json")
        open_world_data.similarities_for(self.entries(2))  # no cache_path kwarg at all
        assert (tmp_path / "cache.json").exists()


class TestScoresOnlyAndTop1:
    SIMILARITIES = [{"common_name": "Tulip", "score": 0.5}, {"common_name": "Rose", "score": 0.9}]

    def test_scores_only_extracts_the_score_column(self):
        assert open_world_data.scores_only(self.SIMILARITIES) == [0.5, 0.9]

    def test_top1_species_is_the_highest_scoring_one_whatever_the_order(self):
        assert open_world_data.top1_species(self.SIMILARITIES) == "Rose"
        assert open_world_data.top1_species(list(reversed(self.SIMILARITIES))) == "Rose"
