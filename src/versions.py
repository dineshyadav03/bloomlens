"""Pinned identities of the external models BloomLens depends on.

One place to read (and, later, record in telemetry) exactly which model versions
produced a result. Changing any value here changes model behavior and must be
followed by re-running the evaluation harness.

BioCLIP 2 is pinned to an exact Hugging Face commit *and* to per-file sizes and
SHA-256 checksums. Without this, `hf-hub:imageomics/bioclip-2` silently resolves
whatever `main` is on the day of the first download (open_clip 2.32 never passes a
revision), so two installs could quietly disagree. The checksums were taken from
the locally cached files every evaluation so far used, and the safetensors hash
was cross-checked against the Hub's own LFS record for this commit. At the time
of pinning this commit was also the head of `main`, so pinning changed no
behavior.
"""

BIOCLIP_REPO = "imageomics/bioclip-2"
BIOCLIP_REVISION = "2957b322090f9cb17ae72c71981c7218a28d81e0"  # commit dated 2026-05-20

BIOCLIP_CONFIG_FILE = "open_clip_config.json"
BIOCLIP_WEIGHTS_FILE = "open_clip_model.safetensors"
BIOCLIP_FILES = {
    BIOCLIP_CONFIG_FILE: {
        "size": 534,
        "sha256": "1bf947e96e943fe50efd5c3e26c37f843a2fa3c358967719a68c8a6d17ce68c8",
    },
    BIOCLIP_WEIGHTS_FILE: {
        "size": 1710517724,
        "sha256": "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef",
    },
}

# Named Gemini model, deliberately not a `-latest` alias (see docs/RESEARCH.md's
# Gemini section for why aliases bit this project in Milestone 1).
GEMINI_MODEL = "gemini-3.1-flash-lite"
