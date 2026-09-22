"""eval/pilot_sample.py: the fixed, deterministic pilot sample (eval/PROTOCOL.md section 9)."""

import pytest

from eval import pilot_sample, splits


def entry(sid, label, group=None, split="pilot"):
    return {"source_id": sid, "label": label, "group": group or label, "split": split, "path": f"{sid}.jpg"}


def id_doc(species_counts: dict[str, int], split="pilot") -> dict:
    entries = []
    for species, n in species_counts.items():
        entries += [entry(f"{species}-{i}", species, split=split) for i in range(n)]
    return {"entries": entries}


def ood_doc(category_counts: dict[str, int], split="pilot") -> dict:
    entries = []
    for category, n in category_counts.items():
        entries += [entry(f"{category}-{i}", category, group=category, split=split) for i in range(n)]
    return {"entries": entries}


class TestSelectIdSample:
    def test_it_takes_up_to_per_species_from_the_pilot_split_only(self):
        doc = id_doc({"Rose": 10, "Tulip": 3})
        sample = pilot_sample.select_id_sample(doc, per_species=5)
        counts = {}
        for e in sample:
            counts[e["label"]] = counts.get(e["label"], 0) + 1
        assert counts == {"Rose": 5, "Tulip": 3}  # Tulip has fewer than 5 available

    def test_non_pilot_entries_are_never_selected(self):
        doc = {"entries": [entry("a", "Rose", split="dev"), entry("b", "Rose", split="test")]}
        assert pilot_sample.select_id_sample(doc) == []

    def test_every_entry_is_tagged_with_its_family(self):
        sample = pilot_sample.select_id_sample(id_doc({"Rose": 2}))
        assert all(e["family"] == "id" for e in sample)

    def test_it_is_deterministic_and_order_independent(self):
        doc = id_doc({"Rose": 8, "Tulip": 8})
        shuffled = {"entries": list(reversed(doc["entries"]))}
        assert pilot_sample.select_id_sample(doc) == pilot_sample.select_id_sample(shuffled)

    def test_the_chosen_images_are_the_smallest_bucket_per_species(self):
        doc = id_doc({"Rose": 10})
        chosen_ids = {e["source_id"] for e in pilot_sample.select_id_sample(doc, per_species=3)}
        ordered = sorted(doc["entries"], key=lambda e: splits.bucket(e["source_id"]))
        assert chosen_ids == {e["source_id"] for e in ordered[:3]}

    def test_species_are_returned_in_alphabetical_order(self):
        doc = id_doc({"Zinnia": 2, "Amaryllis": 2})
        assert [e["label"] for e in pilot_sample.select_id_sample(doc, per_species=2)] == ["Amaryllis"] * 2 + [
            "Zinnia"
        ] * 2


class TestSelectOodSample:
    def test_it_spreads_across_categories_before_repeating_one(self):
        doc = ood_doc({"cat_a": 5, "cat_b": 5, "cat_c": 5})
        sample = pilot_sample.select_ood_sample(doc, "near_ood", count=3)
        assert {e["group"] for e in sample} == {"cat_a", "cat_b", "cat_c"}  # one from each, not 3 from one

    def test_it_stops_at_the_requested_count(self):
        doc = ood_doc({"cat_a": 5, "cat_b": 5})
        assert len(pilot_sample.select_ood_sample(doc, "far_ood", count=3)) == 3

    def test_it_uses_every_available_image_if_the_count_exceeds_the_pool(self):
        doc = ood_doc({"cat_a": 2, "cat_b": 1})
        sample = pilot_sample.select_ood_sample(doc, "near_ood", count=100)
        assert len(sample) == 3

    def test_a_second_round_only_touches_categories_with_a_second_image(self):
        doc = ood_doc({"cat_a": 2, "cat_b": 1})
        sample = pilot_sample.select_ood_sample(doc, "near_ood", count=3)
        assert {e["source_id"] for e in sample} == {e["source_id"] for e in doc["entries"]}

    def test_every_entry_is_tagged_with_the_given_family(self):
        sample = pilot_sample.select_ood_sample(ood_doc({"cat_a": 2}), "far_ood", count=2)
        assert all(e["family"] == "far_ood" for e in sample)

    def test_it_is_deterministic_and_order_independent(self):
        doc = ood_doc({"cat_a": 4, "cat_b": 4, "cat_c": 4})
        shuffled = {"entries": list(reversed(doc["entries"]))}
        first = pilot_sample.select_ood_sample(doc, "near_ood", count=6)
        second = pilot_sample.select_ood_sample(shuffled, "near_ood", count=6)
        assert first == second

    def test_non_pilot_entries_are_excluded(self):
        doc = {"entries": [entry("a", "x", group="cat", split="test")]}
        assert pilot_sample.select_ood_sample(doc, "near_ood", count=5) == []

    def test_an_empty_pool_returns_nothing(self):
        assert pilot_sample.select_ood_sample({"entries": []}, "near_ood", count=5) == []


class TestSelectPilotSample:
    def test_it_concatenates_id_then_near_ood_then_far_ood(self):
        id_document = id_doc({"Rose": 5})
        near_document = ood_doc({"cat_a": 5})
        far_document = ood_doc({"cat_x": 5})
        sample = pilot_sample.select_pilot_sample(
            id_document, near_document, far_document, id_per_species=2, near_ood_count=2, far_ood_count=2
        )
        assert [e["family"] for e in sample] == ["id", "id", "near_ood", "near_ood", "far_ood", "far_ood"]

    def test_the_default_sizes_match_the_protocol(self):
        assert (pilot_sample.ID_PER_SPECIES, pilot_sample.NEAR_OOD_COUNT, pilot_sample.FAR_OOD_COUNT) == (5, 15, 15)

    def test_no_entry_is_duplicated_across_families(self):
        id_document = id_doc({"Rose": 5})
        near_document = ood_doc({"cat_a": 5})
        far_document = ood_doc({"cat_a": 5})  # same category name, different document/family
        sample = pilot_sample.select_pilot_sample(id_document, near_document, far_document)
        source_family_pairs = [(e["source_id"], e["family"]) for e in sample]
        assert len(source_family_pairs) == len(set(source_family_pairs))


@pytest.mark.parametrize("count", [-1, 0])
def test_a_non_positive_count_selects_nothing(count):
    assert pilot_sample.select_ood_sample(ood_doc({"cat_a": 3}), "near_ood", count=count) == []
