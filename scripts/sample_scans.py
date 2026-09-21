"""Run N REAL scans on committed test photos, to fill scan telemetry with real numbers.

    INVENTORY_DB_PATH=/some/scratch.db uv run python scripts/sample_scans.py --count 24

Each scan is a genuine identify() call: BioCLIP 2 on this machine plus the Gemini agent
(a few requests each, free tier; needs GEMINI_API_KEY). Point INVENTORY_DB_PATH at a
scratch file so benchmark rows don't mix with real usage; then read the result with
`scripts/report_metrics.py` (same INVENTORY_DB_PATH).

Photos are taken round-robin across species from eval/test_images/ (Oxford 102 Flowers
samples), in a fixed order, so a rerun measures the same inputs. The first scan of a run
is the cold one (it loads the model). Prints progress only: no photo names beyond the
species, no model text.
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.identify import IdentifyError, identify  # noqa: E402

PHOTO_DIR = Path(__file__).resolve().parent.parent / "eval" / "test_images"


def pick_photos(count: int) -> list[Path]:
    by_species: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(PHOTO_DIR.glob("*.jpg")):
        by_species[path.stem.rsplit("_", 1)[0]].append(path)
    order, depth = [], 0
    while len(order) < count and any(len(v) > depth for v in by_species.values()):
        order += [v[depth] for _, v in sorted(by_species.items()) if len(v) > depth]
        depth += 1
    return order[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--pause", type=float, default=3.0, help="seconds between scans (free-tier rate limits)")
    args = parser.parse_args()

    photos = pick_photos(args.count)
    for number, path in enumerate(photos, start=1):
        started = time.perf_counter()
        try:
            with Image.open(path) as opened:
                identify(opened.convert("RGB"))
            outcome = "ok"
        except IdentifyError as exc:
            outcome = f"failed ({exc.category})"
        print(f"scan {number}/{len(photos)} [{path.stem.rsplit('_', 1)[0]}]: {outcome} in "
              f"{time.perf_counter() - started:.1f}s", flush=True)
        time.sleep(args.pause)


if __name__ == "__main__":
    main()
