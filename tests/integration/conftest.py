"""Fixtures for the integration tests: the real pinned BioCLIP 2 weights and a real
Qdrant index in a temp directory. CI runs these once (Ubuntu / Python 3.13) after
scripts/build_index.py has fetched the weights. Nothing here calls Gemini."""

import json

import pytest

from tests.conftest import REPO_ROOT

TEST_IMAGES = REPO_ROOT / "eval" / "test_images"

# One committed photo per species from the species BioCLIP gets 20/20 on in
# eval/results.md, so a wrong top-1 here means something broke, not "an unlucky photo".
KNOWN_GOOD = {
    "Rose": "Rose_3763.jpg",
    "Sunflower": "Sunflower_2612.jpg",
    "Carnation": "Carnation_1140.jpg",
    "Calla lily": "Calla_lily_*.jpg",
    "Bird of paradise": "Bird_of_paradise_*.jpg",
    "Moth orchid": "Moth_orchid_*.jpg",
}


def known_good_photo(species: str):
    pattern = KNOWN_GOOD[species]
    matches = sorted(TEST_IMAGES.glob(pattern))
    assert matches, f"no committed test photo matches {pattern}"
    return matches[0]


@pytest.fixture(scope="session")
def bioclip():
    """(model, preprocess, tokenizer), loaded once from the pinned files."""
    from src.embeddings import load_model

    return load_model()


@pytest.fixture(scope="session")
def retrieval_index(tmp_path_factory, bioclip):
    """A real Qdrant collection of the 30 curated species in a temp directory."""
    from src import vector_store

    mp = pytest.MonkeyPatch()
    mp.delenv("QDRANT_URL", raising=False)
    path = str(tmp_path_factory.mktemp("qdrant"))
    client = vector_store.get_client(path)
    species = json.loads((REPO_ROOT / "data" / "species_reference.json").read_text(encoding="utf-8"))
    vector_store.build_collection(client, species)
    yield client
    client.close()
    vector_store._clients.pop(path, None)
    mp.undo()


@pytest.fixture
def use_index(monkeypatch, retrieval_index):
    """Make identify()/identify_lot() search the temp index instead of ./qdrant_data."""
    from src import identify

    monkeypatch.setattr(identify, "get_client", lambda: retrieval_index)
    return retrieval_index
