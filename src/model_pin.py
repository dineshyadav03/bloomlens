"""Fetch and verify the pinned BioCLIP 2 files (see src/versions.py).

`fetch_pinned_bioclip()` returns local paths to the exact files at the pinned
Hugging Face commit, after checking each one's size and SHA-256. It tries the
local cache first (no network, so a cached install starts even if DNS is down)
and only downloads if the pinned revision isn't cached yet.

What the check protects against: upstream drift, truncated/corrupted downloads,
and a cache populated from a different revision. It is not a defense against
someone who can already write to your cache directory (they could edit the
verification stamp too).
"""

import hashlib
import json
from pathlib import Path

from huggingface_hub import constants as hf_constants
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from src.versions import BIOCLIP_FILES, BIOCLIP_REPO, BIOCLIP_REVISION

_CHUNK = 8 * 1024 * 1024


class ModelIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp_file() -> Path:
    # Lives beside the cached weights so it persists with the same volume.
    return Path(hf_constants.HF_HOME) / "bloomlens-verified.json"


def _read_stamps() -> dict:
    try:
        return json.loads(_stamp_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_stamps(stamps: dict) -> None:
    try:
        stamp = _stamp_file()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(stamps), encoding="utf-8")
    except OSError:
        pass  # a missing stamp only means re-hashing next time


def verify_file(path: Path, expected_size: int, expected_sha256: str, *, use_stamp: bool = True) -> None:
    """Raise ModelIntegrityError unless `path` has the pinned size and SHA-256.

    Hashing a 1.7 GB file takes seconds, so a passing result is remembered
    (keyed on path + size + mtime) and not repeated until the file changes."""
    stat = path.stat()
    if stat.st_size != expected_size:
        raise ModelIntegrityError(
            f"{path.name}: size {stat.st_size} != pinned {expected_size} "
            f"(truncated download, or a different revision than {BIOCLIP_REVISION[:8]})"
        )

    fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": expected_sha256}
    stamps = _read_stamps() if use_stamp else {}
    if stamps.get(str(path)) == fingerprint:
        return

    actual = _sha256(path)
    if actual != expected_sha256:
        raise ModelIntegrityError(
            f"{path.name}: sha256 {actual} != pinned {expected_sha256}. Refusing to load unverified model weights."
        )

    if use_stamp:
        stamps[str(path)] = fingerprint
        _write_stamps(stamps)


def _resolve(filename: str) -> Path:
    kwargs = {"repo_id": BIOCLIP_REPO, "filename": filename, "revision": BIOCLIP_REVISION}
    try:
        return Path(hf_hub_download(local_files_only=True, **kwargs))
    except LocalEntryNotFoundError:
        return Path(hf_hub_download(**kwargs))


def fetch_pinned_bioclip() -> dict[str, Path]:
    """Return {filename: verified local path} for the pinned BioCLIP 2 revision."""
    paths = {}
    for filename, meta in BIOCLIP_FILES.items():
        path = _resolve(filename)
        verify_file(path, meta["size"], meta["sha256"])
        paths[filename] = path
    return paths
