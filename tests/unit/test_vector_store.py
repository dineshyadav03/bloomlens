"""src/vector_store.py with the Qdrant client faked: the local-vs-server switch
(QDRANT_URL) and the lock-retry loop behind Milestone 1's real
"Storage folder is already accessed by another instance" bug."""

import numpy as np
import pytest
from qdrant_client.models import Distance

from src import vector_store


class FakeQdrant:
    """Stands in for qdrant_client.QdrantClient. `script` lists what each successive
    construction does: an Exception is raised, anything else means 'succeed'."""

    script: list = []
    constructed: list = []

    def __init__(self, **kwargs):
        type(self).constructed.append(kwargs)
        step = type(self).script.pop(0) if type(self).script else None
        if isinstance(step, Exception):
            raise step


@pytest.fixture
def fake_qdrant(monkeypatch):
    FakeQdrant.script, FakeQdrant.constructed = [], []
    sleeps = []
    monkeypatch.setattr(vector_store, "QdrantClient", FakeQdrant)
    monkeypatch.setattr(vector_store, "_clients", {})
    monkeypatch.setattr(vector_store.time, "sleep", sleeps.append)
    FakeQdrant.sleeps = sleeps
    return FakeQdrant


class TestGetClient:
    def test_qdrant_url_selects_server_mode_and_ignores_the_path(self, fake_qdrant, monkeypatch, tmp_path):
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        client = vector_store.get_client(str(tmp_path))
        assert fake_qdrant.constructed == [{"url": "http://qdrant:6333"}]
        assert vector_store.get_client(str(tmp_path)) is client  # cached per URL

    def test_without_the_env_var_it_uses_local_embedded_mode_on_the_given_path(self, fake_qdrant, tmp_path):
        vector_store.get_client(str(tmp_path / "q"))
        assert fake_qdrant.constructed == [{"path": str(tmp_path / "q")}]

    def test_clients_are_cached_per_path(self, fake_qdrant, tmp_path):
        a = vector_store.get_client(str(tmp_path / "a"))
        assert vector_store.get_client(str(tmp_path / "a")) is a
        assert vector_store.get_client(str(tmp_path / "b")) is not a
        assert len(fake_qdrant.constructed) == 2

    def test_a_lock_held_briefly_by_another_process_is_retried(self, fake_qdrant, tmp_path):
        lock = RuntimeError("Storage folder is already accessed by another instance of Qdrant client")
        fake_qdrant.script = [lock, lock, None]
        assert vector_store.get_client(str(tmp_path)) is not None
        assert len(fake_qdrant.constructed) == 3
        assert fake_qdrant.sleeps == [vector_store._OPEN_RETRY_DELAY_SECONDS] * 2

    def test_a_lock_that_never_clears_raises_an_actionable_error(self, fake_qdrant, tmp_path):
        fake_qdrant.script = [RuntimeError("locked")] * vector_store._OPEN_RETRIES
        with pytest.raises(RuntimeError, match="another process may be holding it"):
            vector_store.get_client(str(tmp_path))
        assert len(fake_qdrant.constructed) == vector_store._OPEN_RETRIES
        assert len(fake_qdrant.sleeps) == vector_store._OPEN_RETRIES - 1  # no sleep after the last try

    def test_a_failed_open_is_not_cached(self, fake_qdrant, tmp_path):
        fake_qdrant.script = [RuntimeError("locked")] * vector_store._OPEN_RETRIES
        with pytest.raises(RuntimeError):
            vector_store.get_client(str(tmp_path))
        assert vector_store._clients == {}


class FakeCollectionClient:
    def __init__(self, points=None):
        self.points_to_return = points or []
        self.recreated, self.upserted, self.queried = None, None, None

    def recreate_collection(self, **kwargs):
        self.recreated = kwargs

    def upsert(self, **kwargs):
        self.upserted = kwargs

    def query_points(self, **kwargs):
        self.queried = kwargs

        class Response:
            points = self.points_to_return

        return Response()


class TestBuildCollection:
    def test_it_recreates_the_collection_and_upserts_one_point_per_species(self, monkeypatch, species_list):
        ones = np.ones(vector_store.EMBEDDING_DIM, dtype=np.float32)
        monkeypatch.setattr(vector_store, "embed_text", lambda _text: ones)
        client = FakeCollectionClient()
        count = vector_store.build_collection(client, species_list)

        assert count == len(species_list) == 30
        vectors = client.recreated["vectors_config"]
        assert vectors.size == vector_store.EMBEDDING_DIM
        assert vectors.distance == Distance.COSINE
        points = client.upserted["points"]
        assert [p.id for p in points] == list(range(30))
        assert set(points[0].payload) == {"common_name", "scientific_name", "taxonomy_string", "description"}
        assert len(points[0].vector) == vector_store.EMBEDDING_DIM

    def test_each_species_is_embedded_from_its_taxonomy_string(self, monkeypatch, species_list):
        embedded = []
        monkeypatch.setattr(
            vector_store, "embed_text", lambda text: embedded.append(text) or np.zeros(vector_store.EMBEDDING_DIM)
        )
        vector_store.build_collection(FakeCollectionClient(), species_list)
        assert embedded == [s["taxonomy_string"] for s in species_list]


class TestSearch:
    def test_it_returns_score_and_payload_and_passes_top_k(self):
        class Point:
            def __init__(self, score, payload):
                self.score, self.payload = score, payload

        client = FakeCollectionClient([Point(0.7, {"common_name": "Rose"}), Point(0.5, {"common_name": "Tulip"})])
        results = vector_store.search(client, np.array([0.1, 0.2]), top_k=2)
        assert results == [
            {"score": 0.7, "payload": {"common_name": "Rose"}},
            {"score": 0.5, "payload": {"common_name": "Tulip"}},
        ]
        assert client.queried["limit"] == 2
        assert client.queried["query"] == pytest.approx([0.1, 0.2])
        assert client.queried["collection_name"] == vector_store.COLLECTION_NAME
