"""The frozen threshold-selection config (eval/PROTOCOL.md section 4 step 5) and the lock that
enforces "test evaluated once" (section 5).

`FrozenConfig` is written by eval/select_threshold.py and read by eval/run_open_world.py. Its file
carries its own content hash so an edit made outside the tuning script -- by hand, or by a future
change to select_threshold.py that forgets to rewrite it -- is caught rather than silently trusted.
The lock file, written only after eval/run_open_world.py has successfully produced its results,
is what makes a second run for the same config refuse instead of quietly re-computing.
"""

import datetime
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
CONFIG_PATH = FROZEN_DIR / "v1.json"
LOCK_PATH = FROZEN_DIR / "v1.lock"

# The order candidates are preferred in when their dev AUROC is within TIE_BAND of the best
# (eval/PROTOCOL.md section 4 step 1: "simplest first").
TIE_BAND = 0.005
CANDIDATE_PREFERENCE = (
    "max_cosine", "margin",
    "max_softmax@0.01", "max_softmax@0.02", "max_softmax@0.05", "max_softmax@0.1",
)  # fmt: skip

BASELINE_CANDIDATE = "max_cosine"
BASELINE_TAU = 0.45  # the current tier-`low` gate (src/identify.py's _LOW_CONFIDENCE_MAX_SCORE)

# The adoption rule (eval/PROTOCOL.md section 4 step 4), fixed before any data was seen.
MIN_NEAR_OOD_GAIN = 0.10  # the new rule must abstain on >= 10 more percentage points of near-OOD-dev
MAX_ID_ABSTENTION = 0.08  # ...while abstaining on at most 8% of ID-dev


class TamperedConfigError(ValueError):
    """The config file's content does not match its own recorded hash."""


@dataclass(frozen=True)
class FrozenConfig:
    candidate: str  # e.g. "max_cosine", "margin", "max_softmax@0.05"
    tau: float
    adopted: bool  # whether `candidate`/`tau` replace the baseline tier-low gate
    dev_near_ood_gain_pp: float  # (new near-OOD-dev abstention %) - (baseline near-OOD-dev abstention %)
    dev_id_abstention: float  # new rule's false-abstain rate on ID-dev
    candidate_auroc: dict  # every candidate's dev AUROC, for the record
    manifest_hashes: dict  # {"id": ..., "near_ood": ...} -- eval.build_datasets.manifest_hash of the dev pools
    bioclip_revision: str
    code_revision: str
    created_at: str
    protocol: str = "v1"

    @property
    def effective_candidate(self) -> str:
        """The rule actually in force: the tuned one if adopted, the baseline otherwise."""
        return self.candidate if self.adopted else BASELINE_CANDIDATE

    @property
    def effective_tau(self) -> float:
        return self.tau if self.adopted else BASELINE_TAU


def content_hash(payload: dict) -> str:
    """SHA-256 of `payload` as canonical JSON (sorted keys, no incidental whitespace)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_frozen(config: FrozenConfig, path: Path | None = None) -> str:
    """Write `config` plus its own content hash. Returns the hash. `path` defaults to the
    CURRENT value of CONFIG_PATH (resolved when called, not when this module was imported --
    so tests can monkeypatch CONFIG_PATH and have every default-using call follow it)."""
    path = CONFIG_PATH if path is None else path
    payload = asdict(config)
    digest = content_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**payload, "self_sha256": digest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def read_frozen(path: Path | None = None) -> tuple[FrozenConfig, str]:
    """(the config, its verified content hash). Raises TamperedConfigError if the file's content
    no longer matches the hash it was written with, or FileNotFoundError if it does not exist.
    `path` defaults to the current CONFIG_PATH, resolved at call time (see `write_frozen`)."""
    path = CONFIG_PATH if path is None else path
    raw = json.loads(path.read_text(encoding="utf-8"))
    recorded = raw.pop("self_sha256", None)
    digest = content_hash(raw)
    if recorded != digest:
        raise TamperedConfigError(f"{path} does not match its own recorded hash: edited after being written?")
    return FrozenConfig(**raw), digest


def already_run(config_hash: str, path: Path | None = None) -> bool:
    """Whether `run_open_world.py` has already completed for this exact config. `path` defaults
    to the current LOCK_PATH, resolved at call time (see `write_frozen`)."""
    path = LOCK_PATH if path is None else path
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("config_sha256") == config_hash


def write_lock(config_hash: str, *, results_path: str, path: Path | None = None) -> None:
    """`path` defaults to the current LOCK_PATH, resolved at call time (see `write_frozen`)."""
    path = LOCK_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config_sha256": config_hash, "results_path": results_path, "written_at": now_iso()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()
