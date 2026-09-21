"""eval/splits.py: the deterministic, salt-based split assignment behind the open-world protocol."""

import os
import random
import subprocess
import sys

import pytest

from eval import splits
from tests.conftest import REPO_ROOT

KEYS = [f"oxford102:image_{i:05d}" for i in range(1, 121)]


class TestBucket:
    def test_it_is_pinned_so_a_change_of_hash_or_salt_is_noticed(self):
        """These literals are the protocol: changing the salt or the hash changes every split."""
        assert splits.bucket("oxford102:image_00001") == pytest.approx(0.6104667885708821, abs=1e-15)
        assert splits.bucket("rose") == pytest.approx(0.10532923615976841, abs=1e-15)

    def test_it_is_in_the_unit_interval(self):
        assert all(0 <= splits.bucket(k) < 1 for k in KEYS)

    def test_the_salt_changes_it_and_the_same_salt_repeats_it(self):
        assert splits.bucket("x", "salt-a") != splits.bucket("x", "salt-b")
        assert splits.bucket("x", "salt-a") == splits.bucket("x", "salt-a")

    def test_it_does_not_depend_on_pythons_hash_randomisation(self):
        code = "from eval import splits; print(splits.bucket('oxford102:image_00001'))"
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONHASHSEED": seed},
            ).stdout.strip()
            for seed in ("0", "1", "12345")
        }
        assert outputs == {"0.6104667885708821"}

    def test_the_buckets_are_spread_out(self):
        values = sorted(splits.bucket(k) for k in KEYS)
        assert 0.35 < sum(values) / len(values) < 0.65 and values[0] < 0.1 and values[-1] > 0.9


class TestSplitSizes:
    @pytest.mark.parametrize("n", range(0, 301))
    def test_they_always_add_up(self, n):
        dev, test, pilot = splits.split_sizes(n)
        assert dev + test + pilot == n and min(dev, test, pilot) >= 0

    @pytest.mark.parametrize("n, expected", [(20, (8, 8, 4)), (5, (2, 2, 1)), (1, (0, 1, 0)), (77, (31, 31, 15))])
    def test_known_sizes(self, n, expected):
        assert splits.split_sizes(n) == expected

    @pytest.mark.parametrize("n", [50, 100, 150, 1000])
    def test_they_are_about_forty_forty_twenty(self, n):
        dev, test, pilot = splits.split_sizes(n)
        assert abs(dev / n - 0.4) < 0.02 and abs(test / n - 0.4) < 0.02 and abs(pilot / n - 0.2) < 0.02


class TestAssignSplits:
    def test_every_key_gets_exactly_one_split_of_the_right_size(self):
        assignment = splits.assign_splits(KEYS)
        assert set(assignment) == set(KEYS)
        counts = {s: sum(1 for v in assignment.values() if v == s) for s in splits.SPLITS}
        assert (counts["dev"], counts["test"], counts["pilot"]) == splits.split_sizes(len(KEYS))

    def test_the_input_order_and_duplicates_do_not_matter(self):
        shuffled = KEYS[:]
        random.Random(3).shuffle(shuffled)
        assert splits.assign_splits(shuffled + KEYS[:10]) == splits.assign_splits(KEYS)

    def test_it_is_repeatable(self):
        assert splits.assign_splits(KEYS) == splits.assign_splits(KEYS)

    def test_a_different_salt_gives_a_different_assignment(self):
        assert splits.assign_splits(KEYS) != splits.assign_splits(KEYS, salt="bloomlens-eval-v2")

    def test_the_split_follows_bucket_order(self):
        ordered = splits.by_bucket(KEYS)
        assignment = splits.assign_splits(KEYS)
        dev, test, pilot = splits.split_sizes(len(KEYS))
        assert [assignment[k] for k in ordered] == ["dev"] * dev + ["test"] * test + ["pilot"] * pilot

    def test_an_empty_input_is_an_empty_assignment(self):
        assert splits.assign_splits([]) == {}


class TestStratified:
    GROUPS = {"Rose": [f"r{i}" for i in range(40)], "Tulip": [f"t{i}" for i in range(15)], "Tiny": ["x1", "x2", "x3"]}

    def test_each_group_is_split_on_its_own_and_every_key_appears_once(self):
        assignment = splits.assign_stratified(self.GROUPS)
        assert set(assignment) == {k for keys in self.GROUPS.values() for k in keys}
        for keys in self.GROUPS.values():
            counts = [sum(1 for k in keys if assignment[k] == s) for s in splits.SPLITS]
            assert tuple(counts) == splits.split_sizes(len(keys))

    def test_a_group_of_five_or_more_appears_in_every_split(self):
        assignment = splits.assign_stratified(self.GROUPS)
        for name in ("Rose", "Tulip"):
            assert {assignment[k] for k in self.GROUPS[name]} == set(splits.SPLITS)

    def test_the_order_of_the_groups_does_not_matter(self):
        reordered = dict(reversed(list(self.GROUPS.items())))
        assert splits.assign_stratified(reordered) == splits.assign_stratified(self.GROUPS)


class TestCapPerCategory:
    IMAGES = {"a": [f"a{i}" for i in range(50)], "b": ["b0", "b1", "b2"]}

    def test_no_category_exceeds_the_cap_and_small_ones_are_untouched(self):
        capped = splits.cap_per_category(self.IMAGES)
        assert len(capped["a"]) == splits.MAX_IMAGES_PER_CATEGORY and sorted(capped["b"]) == ["b0", "b1", "b2"]

    def test_the_kept_images_are_the_smallest_buckets(self):
        kept = splits.cap_per_category(self.IMAGES, cap=5)["a"]
        assert kept == splits.by_bucket(self.IMAGES["a"])[:5]

    def test_it_is_a_fixed_sample_independent_of_input_order(self):
        shuffled = {"a": list(reversed(self.IMAGES["a"])), "b": self.IMAGES["b"]}
        assert splits.cap_per_category(shuffled) == splits.cap_per_category(self.IMAGES)


class TestCorruptionSources:
    def test_at_most_five_per_species_and_split_and_a_subset(self):
        pool = {("Rose", "dev"): [f"d{i}" for i in range(30)], ("Rose", "test"): ["t1", "t2"]}
        chosen = splits.corruption_sources(pool)
        assert len(chosen[("Rose", "dev")]) == 5 and set(chosen[("Rose", "dev")]) <= set(pool[("Rose", "dev")])
        assert sorted(chosen[("Rose", "test")]) == ["t1", "t2"]
