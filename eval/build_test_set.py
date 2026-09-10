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

from eval.species_mapping import OXFORD_TO_BLOOMLENS  # noqa: E402

# The standard Oxford 102 category-name mapping (1-indexed, as published) —
# https://github.com/udacity/aipnd-project/blob/master/cat_to_name.json
CAT_TO_NAME = {
    "21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose",
    "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower",
    "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger",
    "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle",
    "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood",
    "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower",
    "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata",
    "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan",
    "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort",
    "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil",
    "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum",
    "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold",
    "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose",
    "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium",
    "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania",
    "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise",
    "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine",
    "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary",
    "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea",
    "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower",
    "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium",
    "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily",
    "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani",
    "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower",
    "51": "petunia",
}

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
