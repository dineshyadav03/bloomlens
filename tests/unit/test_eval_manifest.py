"""The committed evaluation manifests (eval/manifest/) and the builder that produces them.

These are the leakage rules of eval/PROTOCOL.md section 2.6, checked on the actual committed files:
one split per source image, categories never shared between splits, the legacy 360 disjoint from
everything new, every corrupted variant in its source's split -- plus that the builder is a pure,
order-independent function, so the manifests can be rebuilt exactly."""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict

import pytest

from eval import build_datasets as build
from eval import corruptions, splits
from eval.oxford102 import GRAY_ZONE_CLASSES
from eval.species_mapping import OXFORD_TO_BLOOMLENS
from tests.conftest import REPO_ROOT

MANIFEST_DIR = REPO_ROOT / "eval" / "manifest"
NAMES = ("legacy", "id", "near_ood", "far_ood", "corrupted")


@pytest.fixture(scope="module")
def docs():
    return {n: json.loads((MANIFEST_DIR / f"{n}.json").read_text(encoding="utf-8")) for n in NAMES}


def entries(docs, name):
    return docs[name]["entries"]


class TestEveryManifest:
    @pytest.mark.parametrize("name", NAMES)
    def test_the_header_names_the_protocol_and_the_salt(self, docs, name):
        assert docs[name]["protocol"] == "v1" and docs[name]["salt"] == splits.SALT

    @pytest.mark.parametrize("name", NAMES)
    def test_splits_are_from_the_documented_set(self, docs, name):
        allowed = {"dev_legacy"} if name == "legacy" else {"dev", "test", "pilot"}
        assert {e["split"] for e in entries(docs, name)} <= allowed

    @pytest.mark.parametrize("name", ["legacy", "id", "near_ood", "far_ood"])
    def test_a_source_id_appears_once_per_manifest_and_has_a_hash_and_a_path(self, docs, name):
        ids = [e["source_id"] for e in entries(docs, name)]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) for e in entries(docs, name))
        assert all(e["path"] and not e["path"].startswith(("/", "..")) for e in entries(docs, name))

    def test_the_manifest_files_are_canonical_json(self, docs):
        for name in NAMES:
            raw = (MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8")
            assert raw.replace("\r\n", "\n") == build.canonical(docs[name])


class TestNoLeakage:
    def test_every_id_source_has_exactly_one_split(self, docs):
        split_of = defaultdict(set)
        for e in entries(docs, "id"):
            split_of[e["source_id"]].add(e["split"])
        assert all(len(v) == 1 for v in split_of.values())

    def test_the_legacy_360_share_nothing_with_any_new_split(self, docs):
        legacy = {e["source_id"] for e in entries(docs, "legacy")}
        assert len(legacy) == 360
        for name in ("id", "near_ood", "far_ood"):
            assert not legacy & {e["source_id"] for e in entries(docs, name)}, name

    def test_no_image_is_shared_between_families(self, docs):
        pools = {n: {e["source_id"] for e in entries(docs, n)} for n in ("legacy", "id", "near_ood", "far_ood")}
        names = sorted(pools)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                assert not pools[a] & pools[b], (a, b)

    @pytest.mark.parametrize("name", ["near_ood", "far_ood"])
    def test_a_whole_category_lives_in_exactly_one_split(self, docs, name):
        """Open-world means the test categories are ones the thresholds never saw."""
        split_of = defaultdict(set)
        for e in entries(docs, name):
            split_of[e["group"]].add(e["split"])
        assert all(len(v) == 1 for v in split_of.values())

    def test_no_image_content_is_repeated_across_splits(self, docs):
        """Duplicate files under different names would leak; the SHA-256 of every image is unique."""
        for name in ("id", "near_ood", "far_ood"):
            hashes = [e["sha256"] for e in entries(docs, name)]
            duplicated = [h for h, n in Counter(hashes).items() if n > 1]
            for h in duplicated:
                splits_of = {e["split"] for e in entries(docs, name) if e["sha256"] == h}
                assert len(splits_of) == 1, f"{name}: identical image content in {splits_of}"

    def test_no_image_content_is_repeated_at_all_within_a_family(self, docs):
        """Stricter than 'one split per content': byte-identical copies are collapsed to one."""
        for name in ("id", "near_ood", "far_ood"):
            hashes = [e["sha256"] for e in entries(docs, name)]
            assert len(hashes) == len(set(hashes)), name

    def test_no_image_content_is_shared_between_families_or_with_the_legacy_360(self, docs):
        content = {n: {e["sha256"] for e in entries(docs, n)} for n in ("legacy", "id", "near_ood", "far_ood")}
        names = sorted(content)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                assert not content[a] & content[b], (a, b)


class TestContent:
    def test_the_id_pool_is_every_non_legacy_image_of_the_mapped_categories(self, docs):
        assert len(entries(docs, "id")) == 1181  # the figure PROTOCOL.md states
        assert {e["group"] for e in entries(docs, "id")} == set(OXFORD_TO_BLOOMLENS)
        assert {e["label"] for e in entries(docs, "id")} == set(OXFORD_TO_BLOOMLENS.values())
        for e in entries(docs, "id"):
            assert OXFORD_TO_BLOOMLENS[e["group"]] == e["label"]

    def test_every_species_appears_in_every_split(self, docs):
        seen = defaultdict(set)
        for e in entries(docs, "id"):
            seen[e["label"]].add(e["split"])
        assert all(s == set(splits.SPLITS) for s in seen.values())

    def test_id_split_sizes_are_stratified_forty_forty_twenty_per_species(self, docs):
        per_species = defaultdict(Counter)
        for e in entries(docs, "id"):
            per_species[e["label"]][e["split"]] += 1
        for counts in per_species.values():
            assert (counts["dev"], counts["test"], counts["pilot"]) == splits.split_sizes(sum(counts.values()))

    def test_near_ood_is_77_categories_none_of_them_covered_or_gray_zone(self, docs):
        groups = {e["group"] for e in entries(docs, "near_ood")}
        assert len(groups) == 77
        assert not groups & (set(OXFORD_TO_BLOOMLENS) | set(GRAY_ZONE_CLASSES))

    def test_the_gray_zone_is_in_neither_the_id_nor_the_near_ood_side(self, docs):
        everywhere = {e["group"] for n in ("id", "near_ood") for e in entries(docs, n)}
        assert not everywhere & set(GRAY_ZONE_CLASSES)

    @pytest.mark.parametrize("name", ["near_ood", "far_ood"])
    def test_no_category_exceeds_the_cap(self, docs, name):
        counts = Counter(e["group"] for e in entries(docs, name))
        assert max(counts.values()) <= splits.MAX_IMAGES_PER_CATEGORY

    def test_far_ood_holds_no_flowers_no_people_and_no_clutter(self, docs):
        groups = {e["group"] for e in entries(docs, "far_ood")}
        assert len(groups) == 96
        assert not groups & build.CALTECH_EXCLUDED
        assert build.CALTECH_EXCLUDED >= {"sunflower", "water_lilly", "lotus", "Faces", "Faces_easy"}
        assert all(e["source_id"].startswith("caltech101:") for e in entries(docs, "far_ood"))

    def test_the_legacy_manifest_is_the_20_by_18_of_the_existing_test_set(self, docs):
        legacy = json.loads((REPO_ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
        assert Counter(e["label"] for e in entries(docs, "legacy")) == Counter(i["species"] for i in legacy)
        assert set(Counter(e["label"] for e in entries(docs, "legacy")).values()) == {20}

    def test_the_legacy_hashes_match_the_committed_images(self, docs):
        """Content-hash matching, not file-name arithmetic: each committed image IS its Oxford source."""
        items = json.loads((REPO_ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
        committed = {hashlib.sha256((REPO_ROOT / "eval" / i["image_path"]).read_bytes()).hexdigest() for i in items}
        assert committed == {e["sha256"] for e in entries(docs, "legacy")}

    def test_the_protocol_states_the_sizes_the_manifests_have(self, docs):
        text = (REPO_ROOT / "eval" / "PROTOCOL.md").read_text(encoding="utf-8")
        assert "1 181 images" in text and "77 categories" in text
        assert f"{len({e['group'] for e in entries(docs, 'far_ood')})} categories" in text


class TestCorrupted:
    def test_variants_are_exactly_the_eight_protocol_corruptions(self, docs):
        assert docs["corrupted"]["variants"] == list(corruptions.CORRUPTIONS)
        assert {e["variant"] for e in entries(docs, "corrupted")} == set(corruptions.CORRUPTIONS)

    def test_every_variants_source_is_an_id_image_in_the_same_split_with_the_same_label(self, docs):
        by_id = {e["source_id"]: e for e in entries(docs, "id")}
        for e in entries(docs, "corrupted"):
            source = by_id[e["source_id"]]
            assert (source["split"], source["label"]) == (e["split"], e["label"])

    def test_every_entry_carries_the_original_uncorrupted_files_path_and_hash(self, docs):
        """Regression: the corruption is applied at LOAD time from the original file, so the
        manifest must still point at that original -- a first version omitted `path` entirely,
        which only broke (KeyError) when eval/run_open_world.py actually tried to load an image."""
        by_id = {e["source_id"]: e for e in entries(docs, "id")}
        for e in entries(docs, "corrupted"):
            source = by_id[e["source_id"]]
            assert e["path"] == source["path"] and e["sha256"] == source["sha256"]

    def test_only_dev_and_test_sources_are_used_and_never_the_pilot(self, docs):
        assert {e["split"] for e in entries(docs, "corrupted")} == {"dev", "test"}

    def test_five_sources_per_species_and_split_each_in_every_variant(self, docs):
        sources = defaultdict(set)
        variants = defaultdict(set)
        for e in entries(docs, "corrupted"):
            sources[(e["label"], e["split"])].add(e["source_id"])
            variants[e["source_id"]].add(e["variant"])
        assert len(sources) == 18 * 2 and all(len(v) == 5 for v in sources.values())
        assert all(v == set(corruptions.CORRUPTIONS) for v in variants.values())
        assert len(entries(docs, "corrupted")) == 18 * 2 * 5 * 8


# --- the builder is a pure function of its inputs ------------------------------------------


def synthetic_oxford():
    records = []
    for cls, per in (("rose", 30), ("carnation", 12), ("bird of paradise", 9), ("sword lily", 6), ("tiger lily", 25),
                     ("fire lily", 10), ("buttercup", 8), ("bishop of llandaff", 7), ("moon orchid", 20)):  # fmt: skip
        for _ in range(per):
            sid = f"oxford102:image_{len(records) + 1:05d}"
            records.append({"source_id": sid, "class": cls, "oxford_split": "test", "path": f"o/{sid}.jpg",
                            "sha256": hashlib.sha256(sid.encode()).hexdigest()})  # fmt: skip
    return records


def synthetic_caltech():
    return [
        {"source_id": f"caltech101:{cat}/image_{i:04d}", "class": cat, "path": f"c/{cat}/{i}.jpg",
         "sha256": hashlib.sha256(f"{cat}{i}".encode()).hexdigest()}
        for cat in ("airplanes", "ant", "brain", "camera", "crab", "cup") for i in range(25)
    ]  # fmt: skip


def with_duplicates(oxford):
    """Add: a byte-identical twin of a rose (same content, new id), a photo filed under two different
    categories, and a twin of a legacy image."""
    rose = [r for r in oxford if r["class"] == "rose"]
    twin = {**rose[-1], "source_id": "oxford102:image_09001"}
    conflicted = [
        {**rose[0], "source_id": "oxford102:image_09002", "sha256": "c" * 64},
        {**rose[0], "source_id": "oxford102:image_09003", "sha256": "c" * 64, "class": "tiger lily"},
    ]
    legacy_twin = {**oxford[0], "source_id": "oxford102:image_09004"}
    return [*oxford, twin, *conflicted, legacy_twin]


def synthetic_legacy(oxford):
    pick = [r for r in oxford if r["class"] in ("rose", "carnation")][:6]
    return [{"source_id": r["source_id"], "label": build.OXFORD_TO_BLOOMLENS[r["class"]], "group": r["class"],
             "oxford_split": r["oxford_split"], "path": r["path"], "sha256": r["sha256"]} for r in pick]  # fmt: skip


class TestBuilderIsPure:
    def test_the_same_inputs_give_the_same_documents(self):
        oxford, caltech = synthetic_oxford(), synthetic_caltech()
        legacy = synthetic_legacy(oxford)
        assert build.assemble(oxford, caltech, legacy) == build.assemble(oxford, caltech, legacy)

    def test_the_order_of_the_input_records_does_not_matter(self):
        oxford, caltech = synthetic_oxford(), synthetic_caltech()
        legacy = synthetic_legacy(oxford)
        shuffled_o, shuffled_c = oxford[:], caltech[:]
        random.Random(1).shuffle(shuffled_o)
        random.Random(2).shuffle(shuffled_c)
        assert build.assemble(shuffled_o, shuffled_c, legacy[::-1]) == build.assemble(oxford, caltech, legacy)

    def test_legacy_images_never_enter_the_new_pool(self):
        oxford = synthetic_oxford()
        legacy = synthetic_legacy(oxford)
        pool = {e["source_id"] for e in build.assemble(oxford, synthetic_caltech(), legacy)["id"]["entries"]}
        assert not pool & {e["source_id"] for e in legacy}

    def test_ood_categories_stay_whole_in_one_split_and_are_capped(self):
        documents = build.assemble(synthetic_oxford(), synthetic_caltech(), [])
        for name in ("near_ood", "far_ood"):
            by_group = defaultdict(set)
            counts = Counter()
            for e in documents[name]["entries"]:
                by_group[e["group"]].add(e["split"])
                counts[e["group"]] += 1
            assert all(len(v) == 1 for v in by_group.values()) and max(counts.values()) <= 20

    def test_gray_zone_and_covered_categories_are_kept_out_of_near_ood(self):
        near = build.assemble(synthetic_oxford(), synthetic_caltech(), [])["near_ood"]["entries"]
        groups = {e["group"] for e in near}
        assert groups == {"tiger lily", "fire lily"}  # rose/carnation/... are covered; buttercup/bishop are gray-zone

    def test_a_different_input_changes_the_content_hash(self):
        a = build.assemble(synthetic_oxford(), synthetic_caltech(), [])
        b = build.assemble(synthetic_oxford()[:-1], synthetic_caltech(), [])
        assert build.manifest_hash(a["id"]) != build.manifest_hash(b["id"])

    def test_the_content_hash_ignores_key_order_and_whitespace(self):
        document = {"b": [1, 2], "a": {"y": 1, "x": 2}}
        assert build.manifest_hash(document) == build.manifest_hash(json.loads(json.dumps(document, indent=4)))


class TestUniqueByContent:
    def rec(self, sid, sha, cls="rose"):
        return {"source_id": sid, "class": cls, "sha256": sha}

    def test_byte_identical_twins_collapse_to_the_smallest_source_id(self):
        kept = build.unique_by_content([self.rec("b", "1" * 64), self.rec("a", "1" * 64), self.rec("c", "2" * 64)])
        assert [r["source_id"] for r in kept] == ["a", "c"]

    def test_content_under_two_categories_is_dropped_as_ambiguous(self):
        kept = build.unique_by_content([self.rec("a", "1" * 64, "rose"), self.rec("b", "1" * 64, "tiger lily")])
        assert kept == []

    def test_excluded_content_is_dropped_including_its_copies(self):
        kept = build.unique_by_content([self.rec("a", "1" * 64), self.rec("b", "2" * 64)], exclude={"1" * 64})
        assert [r["source_id"] for r in kept] == ["b"]

    def test_the_result_does_not_depend_on_input_order(self):
        records = [self.rec(f"s{i}", str(i % 3) * 64) for i in range(9)]
        assert build.unique_by_content(records) == build.unique_by_content(records[::-1])


class TestDuplicatesNeverLeakThroughTheBuilder:
    def test_a_twin_cannot_land_in_a_different_split_because_only_one_copy_survives(self):
        oxford = with_duplicates(synthetic_oxford())
        documents = build.assemble(oxford, synthetic_caltech(), synthetic_legacy(oxford))
        hashes = [e["sha256"] for e in documents["id"]["entries"]]
        assert len(hashes) == len(set(hashes))

    def test_a_photo_filed_under_two_categories_is_kept_out_of_every_pool(self):
        oxford = with_duplicates(synthetic_oxford())
        documents = build.assemble(oxford, synthetic_caltech(), [])
        for name in ("id", "near_ood"):
            assert not any(e["sha256"] == "c" * 64 for e in documents[name]["entries"])

    def test_a_twin_of_a_legacy_image_is_kept_out_of_every_pool(self):
        oxford = with_duplicates(synthetic_oxford())
        legacy = synthetic_legacy(oxford)
        legacy_hashes = {e["sha256"] for e in legacy}
        documents = build.assemble(oxford, synthetic_caltech(), legacy)
        for name in ("id", "near_ood", "far_ood"):
            assert not any(e["sha256"] in legacy_hashes for e in documents[name]["entries"])

    def test_caltech_content_that_is_also_in_oxford_is_dropped(self):
        oxford = synthetic_oxford()
        clash = {"source_id": "caltech101:ant/dup", "class": "ant", "path": "c/x.jpg", "sha256": oxford[0]["sha256"]}
        caltech = [*synthetic_caltech(), clash]
        far = build.assemble(oxford, caltech, [])["far_ood"]["entries"]
        assert not any(e["source_id"] == "caltech101:ant/dup" for e in far)
