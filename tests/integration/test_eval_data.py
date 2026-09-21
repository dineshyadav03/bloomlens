"""The evaluation manifests against the raw datasets, when those are on disk.

The raw images (eval/raw/, git-ignored) are 800 MB and are not in CI, so these skip there and run
on a machine that has built the datasets. They check that what the manifests claim about a file is
true of the file: its SHA-256, that it opens, and that the corruptions work on real photos."""

import hashlib
import json
import random

import pytest
from PIL import Image

from eval import corruptions
from eval.oxford102 import RAW_ROOT
from tests.conftest import REPO_ROOT

MANIFEST_DIR = REPO_ROOT / "eval" / "manifest"
NAMES = ("id", "near_ood", "far_ood")

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "flowers-102" / "jpg").is_dir(), reason="eval/raw is not present (see eval/build_datasets.py)"
)


def sample(name, n=120):
    entries = json.loads((MANIFEST_DIR / f"{name}.json").read_text(encoding="utf-8"))["entries"]
    return random.Random(7).sample(entries, min(n, len(entries)))


@pytest.mark.parametrize("name", NAMES)
def test_a_sample_of_each_manifests_files_match_their_recorded_hashes(name):
    if name == "far_ood" and not (RAW_ROOT / "caltech-101").is_dir():
        pytest.skip("Caltech-101 has not been extracted")
    for entry in sample(name):
        data = (RAW_ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["source_id"]


def test_every_corruption_works_on_real_photos_and_changes_them():
    entries = sample("id", 3)
    for entry in entries:
        image = Image.open(RAW_ROOT / entry["path"]).convert("RGB")
        for name in corruptions.CORRUPTIONS:
            result = corruptions.apply(image, name, entry["source_id"])
            assert result.size == image.size and result.tobytes() != image.tobytes(), (entry["source_id"], name)
