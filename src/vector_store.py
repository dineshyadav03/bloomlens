"""Qdrant collection of curated species' taxonomy-string embeddings.

Local dev: Qdrant's local/embedded mode (a file lock, one process at a time --
this bit us for real in Milestone 1, "Storage folder is already accessed by
another instance"). Docker Compose (Milestone 7): a real Qdrant *server*
container instead, since running both app.py and api/main.py at once is
exactly what local mode can't handle. Controlled by the QDRANT_URL env var so
callers (identify.py, app.py, api/main.py, build_index.py) don't need to know
or care which mode is active -- they just call get_client().
"""

import os
import threading
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.embeddings import embed_text

COLLECTION_NAME = "species_taxonomy"
EMBEDDING_DIM = 768
DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "qdrant_data")

# Cache the client as a singleton rather than opening a fresh one on every call
# (e.g. every Streamlit rerun / every identify() call) -- in local mode that
# would leave the previous instance's file lock held; in server mode it's just
# wasteful to reconnect constantly.
_lock = threading.Lock()
_clients: dict[str, QdrantClient] = {}

# Some launchers (including the one used for local dev here) start more than one
# process for a single Streamlit app at boot; both briefly race to open the local
# Qdrant lock. Retrying a few times covers that startup race rather than failing
# the first request outright. Only relevant to local mode -- server mode has no
# such lock to race for.
_OPEN_RETRIES = 5
_OPEN_RETRY_DELAY_SECONDS = 1.5


def get_client(path: str = DEFAULT_DB_PATH) -> QdrantClient:
    qdrant_url = os.environ.get("QDRANT_URL")
    key = qdrant_url or path

    with _lock:
        if key in _clients:
            return _clients[key]

        if qdrant_url:
            _clients[key] = QdrantClient(url=qdrant_url)
            return _clients[key]

        last_error = None
        for attempt in range(_OPEN_RETRIES):
            try:
                _clients[key] = QdrantClient(path=path)
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
        return _clients[key]


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
