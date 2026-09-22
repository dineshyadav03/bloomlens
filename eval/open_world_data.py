"""Turn a manifest entry into similarities against all 30 species, for the open-world scripts
(eval/select_threshold.py, eval/run_open_world.py).

A disk cache (`eval/cache/similarities.json`, git-ignored) keys each entry's similarities by its
source id (plus corruption variant, if any) and the BioCLIP revision the index was built with, so
re-running a script during development does not re-embed images it has already seen. This changes
nothing about what is measured -- the cached value is the same deterministic model output the
image would produce again -- it only avoids paying for it twice. The frozen config and the results
file both record the BioCLIP revision, so a model change is never silently served from a stale cache.
"""

import json
import threading
from pathlib import Path

from PIL import Image

from eval import corruptions
from eval.oxford102 import RAW_ROOT
from src.embeddings import embed_image
from src.versions import BIOCLIP_REVISION
from src.vector_store import get_client, search

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "similarities.json"
N_SPECIES = 30  # every curated species: search(top_k=N_SPECIES) returns a similarity to each

_lock = threading.Lock()


def cache_key(entry: dict) -> str:
    """A stable key for one (possibly corrupted) manifest entry."""
    variant = entry.get("variant")
    return f"{entry['source_id']}|{variant}" if variant else entry["source_id"]


def load_cache(path: Path | None = None) -> dict:
    """`path` defaults to the CURRENT value of CACHE_PATH, resolved when called rather than
    when this module was imported -- so monkeypatching CACHE_PATH (in tests) or changing it at
    runtime is honoured by every caller that doesn't pass its own path explicitly."""
    path = CACHE_PATH if path is None else path
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("similarities", {}) if data.get("bioclip_revision") == BIOCLIP_REVISION else {}


def save_cache(cache: dict, path: Path | None = None) -> None:
    """`path` defaults to the current CACHE_PATH, resolved at call time (see `load_cache`)."""
    path = CACHE_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"bioclip_revision": BIOCLIP_REVISION, "similarities": cache}, indent=0), encoding="utf-8"
    )


def load_image(entry: dict) -> Image.Image:
    """The entry's image, with its corruption applied if it names one."""
    image = Image.open(RAW_ROOT / entry["path"]).convert("RGB")
    variant = entry.get("variant")
    return corruptions.apply(image, variant, entry["source_id"]) if variant else image


def all_similarities(image: Image.Image) -> list[dict]:
    """`[{"common_name": ..., "score": ...}, ...]` for all 30 species, highest score first."""
    client = get_client()
    embedding = embed_image(image)
    results = search(client, embedding, top_k=N_SPECIES)
    if len(results) != N_SPECIES:
        raise RuntimeError(f"expected {N_SPECIES} species in the index, found {len(results)}")
    return [{"common_name": r["payload"]["common_name"], "score": r["score"]} for r in results]


def similarities_for(entries: list[dict], *, cache_path: Path | None = None, progress=None) -> dict[str, list[dict]]:
    """cache_key(entry) -> its all_similarities(), computing and caching whatever is missing.
    `progress(done, total)` is called after each new computation, if given. `cache_path` defaults
    to the current CACHE_PATH, resolved at call time (see `load_cache`)."""
    cache_path = CACHE_PATH if cache_path is None else cache_path
    with _lock:
        cache = load_cache(cache_path)
    missing = [e for e in entries if cache_key(e) not in cache]
    for done, entry in enumerate(missing, start=1):
        cache[cache_key(entry)] = all_similarities(load_image(entry))
        if progress:
            progress(done, len(missing))
        if done % 25 == 0:
            with _lock:
                save_cache(cache, cache_path)
    if missing:
        with _lock:
            save_cache(cache, cache_path)
    return {cache_key(e): cache[cache_key(e)] for e in entries}


def scores_only(similarities: list[dict]) -> list[float]:
    return [item["score"] for item in similarities]


def top1_species(similarities: list[dict]) -> str:
    return max(similarities, key=lambda item: item["score"])["common_name"]
