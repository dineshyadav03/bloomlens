"""Deterministic calibration/study split for the blinded quality labeler (tools/label_quality.py).

Pure functions only -- no Streamlit -- so `build_sample` is exercised directly, and only against
clearly-marked synthetic fixtures per docs/quality/PROTOCOL.md section 8 ("building the tool is
not validation"). `default_sample_source` is the one function that touches real project data (the
already-committed ID manifest); it is never called from a test.
"""

import json
from pathlib import Path

from eval.oxford102 import RAW_ROOT

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest" / "id.json"
DEFAULT_CALIBRATION_COUNT = 15


def build_sample(items: list[dict], *, calibration_count: int = DEFAULT_CALIBRATION_COUNT) -> list[dict]:
    """Tag the first `calibration_count` of `items`, IN THE GIVEN ORDER, as the practice round
    (docs/quality/PROTOCOL.md section 2) and the rest as the measured study round. `items` order
    is the caller's to fix deterministically (e.g. sorted by id) -- this function only slices it,
    so a rater who reloads the tool always sees the same item in the same position."""
    return [{**item, "calibration": index < calibration_count} for index, item in enumerate(items)]


def default_sample_source(manifest_path: Path = MANIFEST_PATH) -> list[dict]:
    """The real labeling pool: every `pilot`-split entry of the committed ID manifest, sorted by
    `source_id` for determinism, reduced to just `id` and an absolute `path` -- no species, no
    candidate list, no BloomLens output of any kind (docs/quality/PROTOCOL.md section 3, blinding:
    raters see only the photo)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    pilot = sorted((e for e in manifest["entries"] if e.get("split") == "pilot"), key=lambda e: e["source_id"])
    return [{"id": e["source_id"], "path": str(RAW_ROOT / e["path"])} for e in pilot]
