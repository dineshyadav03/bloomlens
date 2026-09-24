"""eval/labeling_sample.py: the deterministic calibration/study split.

`build_sample` is tested on plain synthetic lists. `default_sample_source` reads only the
already-committed ID manifest (metadata, no images, no rater data) -- the same kind of real-file
read the other eval/manifest tests do.
"""

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


class TestDefaultSampleSource:
    def test_only_pilot_entries_are_returned(self):
        import json

        manifest = json.loads(ls.MANIFEST_PATH.read_text(encoding="utf-8"))
        pilot_ids = {e["source_id"] for e in manifest["entries"] if e["split"] == "pilot"}
        assert {item["id"] for item in ls.default_sample_source()} == pilot_ids

    def test_items_carry_only_an_id_and_a_path_so_the_rater_is_blind_to_everything_else(self):
        items = ls.default_sample_source()
        assert items  # the committed manifest has a pilot split
        assert all(set(item) == {"id", "path"} for item in items)

    def test_the_order_is_sorted_by_id_so_every_rater_sees_the_same_sequence(self):
        ids = [item["id"] for item in ls.default_sample_source()]
        assert ids == sorted(ids)

    def test_paths_are_absolute_and_under_the_raw_data_root(self):
        from pathlib import Path

        for item in ls.default_sample_source()[:5]:
            path = Path(item["path"])
            assert path.is_absolute()
            assert ls.RAW_ROOT in path.parents

    def test_no_dev_or_test_split_image_leaks_into_the_labeling_pool(self):
        import json

        manifest = json.loads(ls.MANIFEST_PATH.read_text(encoding="utf-8"))
        held_out = {e["source_id"] for e in manifest["entries"] if e["split"] in ("dev", "test")}
        assert not held_out & {item["id"] for item in ls.default_sample_source()}
