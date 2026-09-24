"""Time the pieces of a cold start, in one fresh process: what the README's "Performance" numbers are.

    uv run python scripts/measure_startup.py

Run it in a NEW process each time (that is the point -- import and model load are once-per-process
costs) and with no Streamlit/API process running, since local-mode Qdrant allows one opener.
Needs the pinned BioCLIP 2 weights already on disk (`scripts/build_index.py` fetches them) and the
built index: this measures loading and running the model, not downloading it, and says so.

It prints hardware and timings only. It reads one committed test photo and makes no network call
(no Gemini), so it measures the retrieval stage -- the `embed` and `search` parts of a scan; the
agent stage is measured by `scripts/sample_scans.py` and the telemetry it fills.
"""

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHOTO = Path(__file__).resolve().parent.parent / "eval" / "test_images" / "Amaryllis_5455.jpg"


def ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--warm-runs", type=int, default=10, help="warm embeds/searches to time after the first")
    args = parser.parse_args()

    started = time.perf_counter()
    import torch  # noqa: PLC0415 -- imports are timed on purpose
    from PIL import Image  # noqa: PLC0415

    t_torch = time.perf_counter() - started

    started = time.perf_counter()
    from src import embeddings, vector_store  # noqa: PLC0415

    t_project_imports = time.perf_counter() - started
    assert not embeddings.is_loaded()

    started = time.perf_counter()
    embeddings.load_model()
    t_load = time.perf_counter() - started

    image = Image.open(PHOTO).convert("RGB")
    started = time.perf_counter()
    vector = embeddings.embed_image(image)
    t_first_embed = time.perf_counter() - started
    warm_embed = []
    for _ in range(args.warm_runs):
        started = time.perf_counter()
        embeddings.embed_image(image)
        warm_embed.append(time.perf_counter() - started)

    started = time.perf_counter()
    client = vector_store.get_client()
    t_open = time.perf_counter() - started
    started = time.perf_counter()
    vector_store.search(client, vector, top_k=3)
    t_first_search = time.perf_counter() - started
    warm_search = []
    for _ in range(args.warm_runs):
        started = time.perf_counter()
        vector_store.search(client, vector, top_k=3)
        warm_search.append(time.perf_counter() - started)

    def warm(label, samples):
        return f"{label} (n={args.warm_runs})  median {ms(statistics.median(samples))} ms, max {ms(max(samples))} ms"

    total = t_torch + t_project_imports + t_load + t_first_embed + t_open + t_first_search
    print(f"python {platform.python_version()} | torch {torch.__version__} | {platform.platform()}")
    print(
        f"processor: {platform.processor()} | logical CPUs: {os.cpu_count()} | torch threads: {torch.get_num_threads()}"
    )
    print(f"photo: eval/test_images/{PHOTO.name} {image.size}  (weights and index already on disk)")
    print(f"import torch + PIL          {ms(t_torch)} ms")
    print(f"import src.embeddings/vs    {ms(t_project_imports)} ms")
    print(f"load BioCLIP 2 (CPU)        {ms(t_load)} ms")
    print(f"first embed                 {ms(t_first_embed)} ms")
    print(warm("warm embed ", warm_embed))
    print(f"open Qdrant (local mode)    {ms(t_open)} ms")
    print(f"first search                {ms(t_first_search)} ms")
    print(warm("warm search", warm_search))
    print(f"cold total, process start to first result: {ms(total)} ms")


if __name__ == "__main__":
    main()
