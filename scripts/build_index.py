"""One-time setup: embed curated species' taxonomy strings into Qdrant,
and generate the simulated price table. Re-run any time
data/species_reference.json changes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pricing import generate_simulated_prices  # noqa: E402
from src.vector_store import build_collection, get_client  # noqa: E402

SPECIES_PATH = Path(__file__).resolve().parent.parent / "data" / "species_reference.json"


def main() -> None:
    species_list = json.loads(SPECIES_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(species_list)} species from {SPECIES_PATH.name}")

    print("Embedding taxonomy strings with BioCLIP 2 and upserting into Qdrant...")
    client = get_client()
    count = build_collection(client, species_list)
    print(f"Indexed {count} species into the local Qdrant collection.")

    print("Generating simulated price table...")
    prices = generate_simulated_prices(species_list)
    print(f"Wrote {len(prices)} simulated price rows to data/simulated_prices.csv")

    print("Done. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
