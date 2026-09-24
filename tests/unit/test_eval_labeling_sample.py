"""eval/labeling_sample.py: the deterministic calibration/study split.

`build_sample` is tested on plain synthetic lists. `default_sample_source` is tested on a
synthetic manifest written to tmp_path, plus a few structural checks against the already-committed
ID manifest (metadata only -- no images, no rater data), like the other eval/manifest tests.
"""

import json
from pathlib import Path

from eval import labeling_sample as ls


def synthetic_items(n):
    return [{"id": f"synthetic-{i:03d}", "path": f"/nonexistent/{i}.png"} for i in range(n)]


class TestBuildSample:
    def test_the_first_n_items_are_calibration_the_rest_are_study(self):
        sample = ls.build_sample(synthetic_items(6), calibration_count=2)
        assert [s["calibration"] for s in sample] == [True, True, False, False, False, False]

    def test_order_and_fields_are_preserved(self):
        items = synthetic_items(4)
        sample = ls.build_sample(items, calibration_count=1)
        assert [s["id"] for s in sample] == [i["id"] for i in items]
        assert all(s["path"] == i["path"] for s, i in zip(sample, items, strict=True))

    def test_the_input_items_are_not_mutated(self):
        items = synthetic_items(3)
        ls.build_sample(items, calibration_count=1)
        assert all("calibration" not in i for i in items)

    def test_is_deterministic(self):
        assert ls.build_sample(synthetic_items(20)) == ls.build_sample(synthetic_items(20))

    def test_fewer_items_than_the_calibration_count_are_all_calibration(self):
        sample = ls.build_sample(synthetic_items(3), calibration_count=15)
        assert all(s["calibration"] for s in sample)

    def test_zero_calibration_marks_everything_as_study(self):
        sample = ls.build_sample(synthetic_items(3), calibration_count=0)
        assert not any(s["calibration"] for s in sample)

    def test_the_default_calibration_count_is_the_protocols_fifteen(self):
        assert ls.DEFAULT_CALIBRATION_COUNT == 15  # docs/quality/PROTOCOL.md section 2
        sample = ls.build_sample(synthetic_items(20))
        assert sum(s["calibration"] for s in sample) == 15

    def test_extra_fields_on_an_item_are_carried_through(self):
        # build_sample is a slicer, not a filter: blinding is the SOURCE's job (see below), so a
        # caller that hands it extra keys gets them back unchanged.
        sample = ls.build_sample([{"id": "x", "path": "p", "extra": 1}], calibration_count=0)
        assert sample == [{"id": "x", "path": "p", "extra": 1, "calibration": False}]


class TestBuildSampleReadsTheCountAtCallTime:
    def test_a_patched_default_is_honored(self, monkeypatch):
        monkeypatch.setattr(ls, "DEFAULT_CALIBRATION_COUNT", 1)
        assert [s["calibration"] for s in ls.build_sample(synthetic_items(3))] == [True, False, False]


def synthetic_manifest(tmp_path, *, dev=20, pilot=8, test=5):
    entries = []
    for split, n in (("dev", dev), ("pilot", pilot), ("test", test)):
        for i in range(n):
            entries.append(
                {
                    "source_id": f"synthetic:{split}-{i:02d}",
                    "path": f"synthetic/{split}-{i:02d}.png",
                    "split": split,
                    "label": f"SPECIES-CANARY-{i}",
                }
            )
    path = tmp_path / "id.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


class TestDefaultSampleSourceOnASyntheticManifest:
    def test_practice_items_come_from_dev_then_measured_items_from_pilot(self, tmp_path):
        items = ls.default_sample_source(synthetic_manifest(tmp_path), calibration_count=5)
        assert [i["id"].split(":")[1].split("-")[0] for i in items] == ["dev"] * 5 + ["pilot"] * 8

    def test_the_test_split_is_never_used(self, tmp_path):
        items = ls.default_sample_source(synthetic_manifest(tmp_path), calibration_count=5)
        assert not any("test-" in i["id"] for i in items)

    def test_practice_and_measured_items_are_disjoint(self, tmp_path):
        items = ls.default_sample_source(synthetic_manifest(tmp_path), calibration_count=5)
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))

    def test_only_an_id_and_a_path_are_exposed(self, tmp_path):
        items = ls.default_sample_source(synthetic_manifest(tmp_path), calibration_count=5)
        assert all(set(i) == {"id", "path"} for i in items)
        assert "SPECIES-CANARY" not in json.dumps(items)

    def test_the_order_is_deterministic_but_not_the_manifests_own_order(self, tmp_path):
        manifest = synthetic_manifest(tmp_path)
        first = ls.default_sample_source(manifest, calibration_count=5)
        assert first == ls.default_sample_source(manifest, calibration_count=5)
        pilot_ids = [i["id"] for i in first[5:]]
        assert pilot_ids != sorted(pilot_ids)  # a fixed shuffle, not an alphabetical run

    def test_the_order_does_not_depend_on_the_order_entries_appear_in_the_manifest(self, tmp_path):
        manifest = synthetic_manifest(tmp_path)
        forward = ls.default_sample_source(manifest, calibration_count=5)
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["entries"].reverse()
        manifest.write_text(json.dumps(document), encoding="utf-8")
        assert ls.default_sample_source(manifest, calibration_count=5) == forward

    def test_fewer_dev_entries_than_the_count_gives_a_shorter_practice_round(self, tmp_path):
        items = ls.default_sample_source(synthetic_manifest(tmp_path, dev=3), calibration_count=15)
        assert sum(1 for i in items if ":dev-" in i["id"]) == 3

    def test_the_manifest_path_and_count_are_read_at_call_time(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ls, "MANIFEST_PATH", synthetic_manifest(tmp_path))
        monkeypatch.setattr(ls, "DEFAULT_CALIBRATION_COUNT", 2)
        items = ls.default_sample_source()
        assert sum(1 for i in items if ":dev-" in i["id"]) == 2
        assert all(i["id"].startswith("synthetic:") for i in items)

    def test_paths_are_absolute_and_under_the_raw_data_root(self, tmp_path):
        for item in ls.default_sample_source(synthetic_manifest(tmp_path), calibration_count=5):
            assert Path(item["path"]).is_absolute()
            assert ls.RAW_ROOT in Path(item["path"]).parents


class TestDefaultSampleSourceOnTheCommittedManifest:
    """Structural checks only, on metadata already in the repo."""

    def test_it_yields_fifteen_practice_items_then_the_pilot_split_and_never_test(self):
        manifest = json.loads(ls.MANIFEST_PATH.read_text(encoding="utf-8"))
        by_split = {}
        for e in manifest["entries"]:
            by_split.setdefault(e["split"], set()).add(e["source_id"])
        items = ls.default_sample_source()
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids)) == 15 + len(by_split["pilot"])
        assert set(ids[:15]) <= by_split["dev"]
        assert set(ids[15:]) == by_split["pilot"]
        assert not set(ids) & by_split["test"]

    def test_the_first_practice_items_are_not_all_one_species(self):
        # Oxford numbers its images in one-species blocks, so an unshuffled order would fail this.
        manifest = json.loads(ls.MANIFEST_PATH.read_text(encoding="utf-8"))
        label = {e["source_id"]: e["label"] for e in manifest["entries"]}
        practice = [label[i["id"]] for i in ls.default_sample_source()[:15]]
        assert len(set(practice)) >= 5
