"""Deterministic calibration/study split for the blinded quality labeler (tools/label_quality.py).

Pure functions only -- no Streamlit -- so `build_sample` is exercised directly, and only against
clearly-marked synthetic fixtures per docs/quality/PROTOCOL.md section 8 ("building the tool is
not validation"). `default_sample_source` is the one function that touches real project data (the
already-committed ID manifest, metadata only); it is never called from the labeler's own tests.

**The photos it points at are a stand-in pool** (docs/quality/PROTOCOL.md section 6): Oxford 102
garden and wild flowers, nearly all in fine condition, covering 18 of the 30 species. They exist to
exercise the tools. A real study replaces this source with cut-flower photos that actually vary in
condition -- agreement on a set that is almost all "A" would be undefined or meaningless.
"""

import hashlib
import json
from pathlib import Path

from eval.oxford102 import RAW_ROOT

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest" / "id.json"
DEFAULT_CALIBRATION_COUNT = 15


def build_sample(items: list[dict], *, calibration_count: int | None = None) -> list[dict]:
    """Tag the first `calibration_count` of `items`, IN THE GIVEN ORDER, as the practice round
    (docs/quality/PROTOCOL.md section 2) and the rest as the measured study round. `items` order
    is the caller's to fix deterministically -- this function only slices it, so a rater who
    reloads the tool always sees the same item in the same position."""
    count = DEFAULT_CALIBRATION_COUNT if calibration_count is None else calibration_count
    return [{**item, "calibration": index < count} for index, item in enumerate(items)]


def _shuffled(entries) -> list[dict]:
    """A fixed pseudo-random order (SHA-256 of the source id): the same on every run and machine,
    but not the manifest's own order -- Oxford image numbers run in blocks of one species, so a
    plain sort would show a rater the same flower twenty times in a row."""
    return sorted(entries, key=lambda e: hashlib.sha256(e["source_id"].encode()).hexdigest())


def default_sample_source(manifest_path: Path | None = None, *, calibration_count: int | None = None) -> list[dict]:
    """The stand-in labeling pool, in the order a rater sees it: first the practice round drawn
    from the manifest's `dev` split (docs/quality/PROTOCOL.md section 2: practice images come
    from `dev` and are never reused in the measured set), then the measured items, the `pilot`
    split. Each is reduced to just `id` and an absolute `path` -- no species, no candidate list, no
    BloomLens output of any kind (section 3, blinding: raters see only the photo). The `test`
    split is never used. Paths default to the current MANIFEST_PATH, resolved when called."""
    manifest_path = MANIFEST_PATH if manifest_path is None else manifest_path
    count = DEFAULT_CALIBRATION_COUNT if calibration_count is None else calibration_count
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)["entries"]
    practice = _shuffled(e for e in entries if e.get("split") == "dev")[:count]
    measured = _shuffled(e for e in entries if e.get("split") == "pilot")
    return [{"id": e["source_id"], "path": str(RAW_ROOT / e["path"])} for e in (*practice, *measured)]
