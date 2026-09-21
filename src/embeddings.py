"""BioCLIP 2 image/text embeddings.

Classification is zero-shot: cosine similarity between a photo's image
embedding and candidate species' taxonomic-name text embeddings (see
docs/RESEARCH.md#bioclip) rather than a reference-photo database.

The weights are loaded from a pinned, checksum-verified Hugging Face revision
(src/versions.py, src/model_pin.py), not from whatever `main` currently is.
"""

import json
import threading

import numpy as np
import open_clip
import torch
from PIL import Image

from src.model_pin import fetch_pinned_bioclip
from src.versions import BIOCLIP_CONFIG_FILE, BIOCLIP_WEIGHTS_FILE

# open_clip 2.32 can't load a local hub snapshot by revision, but BioCLIP 2's
# architecture is exactly its built-in ViT-L-14 (checked below, so a future
# open_clip upgrade that changed that config fails loudly instead of quietly
# changing the model). Weights come from the pinned local file.
_ARCHITECTURE = "ViT-L-14"

_lock = threading.Lock()
_state = {"model": None, "preprocess": None, "tokenizer": None}


def _assert_architecture_matches_pinned_config(config: dict) -> None:
    builtin = open_clip.get_model_config(_ARCHITECTURE)
    if builtin != config["model_cfg"]:
        raise RuntimeError(
            f"open_clip's built-in {_ARCHITECTURE} config no longer matches the pinned BioCLIP 2 "
            f"config: {builtin} != {config['model_cfg']}. Inspect before loading."
        )


def _assert_preprocess_matches_pinned_config(config: dict, preprocess) -> None:
    normalize = next(t for t in preprocess.transforms if type(t).__name__ == "Normalize")
    expected = config["preprocess_cfg"]
    if tuple(normalize.mean) != tuple(expected["mean"]) or tuple(normalize.std) != tuple(expected["std"]):
        raise RuntimeError(
            f"Image normalization {normalize.mean}/{normalize.std} no longer matches the pinned BioCLIP 2 "
            f"preprocessing {expected['mean']}/{expected['std']}. Inspect before loading."
        )


def is_loaded() -> bool:
    """Whether this process has already loaded the model (the first scan pays for that)."""
    return _state["model"] is not None


def load_model():
    """Load (once) and return (model, preprocess, tokenizer). CPU inference."""
    with _lock:
        if _state["model"] is None:
            paths = fetch_pinned_bioclip()
            config = json.loads(paths[BIOCLIP_CONFIG_FILE].read_text(encoding="utf-8"))
            _assert_architecture_matches_pinned_config(config)

            model, _, preprocess = open_clip.create_model_and_transforms(
                _ARCHITECTURE, pretrained=str(paths[BIOCLIP_WEIGHTS_FILE])
            )
            _assert_preprocess_matches_pinned_config(config, preprocess)
            model.eval()
            tokenizer = open_clip.get_tokenizer(_ARCHITECTURE)
            _state["model"] = model
            _state["preprocess"] = preprocess
            _state["tokenizer"] = tokenizer
        return _state["model"], _state["preprocess"], _state["tokenizer"]


def _normalize(vec: torch.Tensor) -> np.ndarray:
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.squeeze(0).cpu().numpy().astype(np.float32)


def embed_image(image: Image.Image) -> np.ndarray:
    """Embed a PIL image, returning an L2-normalized 768-dim vector."""
    model, preprocess, _ = load_model()
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(tensor)
    return _normalize(features)


def embed_text(text: str) -> np.ndarray:
    """Embed a text string (e.g. a taxonomic name), returning an L2-normalized 768-dim vector."""
    model, _, tokenizer = load_model()
    tokens = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(tokens)
    return _normalize(features)
