"""Local Qdrant collection of curated species' taxonomy-string embeddings."""

import threading
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.embeddings import embed_text

COLLECTION_NAME = "species_taxonomy"
EMBEDDING_DIM = 768
DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "qdrant_data")

# Qdrant's local (embedded) storage mode takes an exclusive file lock, so only one
# QdrantClient may be open on a given path per process at a time. Cache it as a
# singleton rather than opening a fresh client on every call (e.g. every Streamlit
# rerun / every identify() call), which would leave the previous instance's lock held.
_lock = threading.Lock()
_clients: dict[str, QdrantClient] = {}

# Some launchers (including the one used for local dev here) start more than one
# process for a single Streamlit app at boot; both briefly race to open the local
# Qdrant lock. Retrying a few times covers that startup race rather than failing
# the first request outright.
_OPEN_RETRIES = 5
_OPEN_RETRY_DELAY_SECONDS = 1.5


def get_client(path: str = DEFAULT_DB_PATH) -> QdrantClient:
    with _lock:
        if path not in _clients:
            last_error = None
            for attempt in range(_OPEN_RETRIES):
                try:
                    _clients[path] = QdrantClient(path=path)
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if attempt < _OPEN_RETRIES - 1:
                        time.sleep(_OPEN_RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(
                    f"Could not open the local Qdrant storage at {path} after "
                    f"{_OPEN_RETRIES} attempts — another process may be holding it. "
                    f"({last_error})"
                )
        return _clients[path]


def build_collection(client: QdrantClient, species_list: list[dict]) -> int:
    """(Re)build the taxonomy collection from curated species entries. Returns count embedded."""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = []
    for idx, species in enumerate(species_list):
        vector = embed_text(species["taxonomy_string"])
        points.append(
            PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    "common_name": species["common_name"],
                    "scientific_name": species["scientific_name"],
                    "taxonomy_string": species["taxonomy_string"],
                    "description": species["description"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search(client: QdrantClient, image_embedding, top_k: int = 3) -> list[dict]:
    """Return up to top_k {score, payload} matches for an image embedding."""
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=image_embedding.tolist(),
        limit=top_k,
    ).points
    return [{"score": r.score, "payload": r.payload} for r in results]
