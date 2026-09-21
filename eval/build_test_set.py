"""One-time: sample Oxford 102 Flowers test-split images for BloomLens's matched
species (eval/species_mapping.py) into eval/test_images/ + eval/test_set.json.

Only run this once; re-run only if species_mapping.py changes. Downloads the
Oxford 102 dataset (~345MB, public, no auth) via torchvision on first use.
"""

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchvision.datasets import Flowers102  # noqa: E402

from eval.oxford102 import CAT_TO_NAME  # noqa: E402
from eval.species_mapping import OXFORD_TO_BLOOMLENS  # noqa: E402

MAX_PER_SPECIES = 20
SEED = 42
DATA_ROOT = Path(__file__).resolve().parent / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent / "test_images"
OUTPUT_MANIFEST = Path(__file__).resolve().parent / "test_set.json"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading Oxford 102 Flowers (test split) via torchvision...")
    dataset = Flowers102(root=str(DATA_ROOT), split="test", download=True)

    # Confirmed via torchvision's Flowers102 source: it already stores labels as
    # `labels["labels"] - 1`, i.e. dataset._labels is 0-indexed. cat_to_name.json's
    # keys are 1-indexed (the original, published numbering), so the "-1" below
    # is the correct and only offset needed — verified by reading torchvision's
    # source rather than assumed.
    # Also: "barbeton daisy" in cat_to_name.json is a typo for "barberton daisy" —
    # species_mapping.py uses the correct spelling; corrected here to match.
    label_to_oxford_name = {}
    for cat_str, name in CAT_TO_NAME.items():
        fixed_name = "barberton daisy" if name == "barbeton daisy" else name
        label_to_oxford_name[int(cat_str) - 1] = fixed_name

    by_species: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(dataset._labels):
        oxford_name = label_to_oxford_name.get(label)
        species = OXFORD_TO_BLOOMLENS.get(oxford_name) if oxford_name else None
        if species:
            by_species[species].append(idx)

    if not by_species:
        raise RuntimeError(
            "No matched species found — the label indexing assumption is probably wrong. "
            "Inspect dataset._labels and CAT_TO_NAME before proceeding."
        )

    rng = random.Random(SEED)
    manifest = []
    for species, indices in sorted(by_species.items()):
        sample = rng.sample(indices, min(MAX_PER_SPECIES, len(indices)))
        for i in sample:
            src_path = dataset._image_files[i]
            filename = f"{species.replace(' ', '_')}_{i}.jpg"
            shutil.copy2(src_path, OUTPUT_DIR / filename)
            manifest.append({"image_path": f"test_images/{filename}", "species": species})
        print(f"{species}: {len(sample)} images sampled (of {len(indices)} available)")

    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(manifest)} images across {len(by_species)} species to {OUTPUT_MANIFEST}")
    print(f"Species matched: {sorted(by_species.keys())}")


if __name__ == "__main__":
    main()
