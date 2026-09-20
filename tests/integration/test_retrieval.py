"""Real BioCLIP 2 + real Qdrant, no LLM: the retrieval stage the README's accuracy
number describes."""

import numpy as np
import pytest
from PIL import Image

from src import model_pin, versions
from src.embeddings import embed_image, embed_text
from src.vector_store import COLLECTION_NAME, EMBEDDING_DIM, search
from tests.integration.conftest import KNOWN_GOOD, known_good_photo


@pytest.mark.parametrize("species", sorted(KNOWN_GOOD))
def test_top1_is_the_right_species_for_a_committed_photo(species, retrieval_index):
    embedding = embed_image(Image.open(known_good_photo(species)))
    results = search(retrieval_index, embedding, top_k=3)
    assert results[0]["payload"]["common_name"] == species
    assert species in [r["payload"]["common_name"] for r in results]


def test_results_are_ranked_and_top_k_long(retrieval_index):
    results = search(retrieval_index, embed_image(Image.open(known_good_photo("Rose"))), top_k=3)
    scores = [r["score"] for r in results]
    assert len(results) == 3
    assert scores == sorted(scores, reverse=True)


def test_the_index_holds_every_curated_species(retrieval_index, species_list):
    assert retrieval_index.count(COLLECTION_NAME).count == len(species_list) == 30


def test_embeddings_are_unit_length_768_and_deterministic(bioclip):
    image = Image.open(known_good_photo("Sunflower"))
    first, second = embed_image(image), embed_image(image)
    assert first.shape == (EMBEDDING_DIM,)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(first, second)


def test_an_image_matches_its_own_species_text_better_than_an_unrelated_one(bioclip, species_list):
    by_name = {s["common_name"]: s["taxonomy_string"] for s in species_list}
    image_vec = embed_image(Image.open(known_good_photo("Rose")))
    own = float(image_vec @ embed_text(by_name["Rose"]))
    unrelated = float(image_vec @ embed_text(by_name["Bird of paradise"]))
    assert own > unrelated


def test_the_loaded_weights_are_the_pinned_revision(bioclip):
    paths = model_pin.fetch_pinned_bioclip()
    for path in paths.values():
        assert versions.BIOCLIP_REVISION in path.parts  # served from the pinned snapshot directory
