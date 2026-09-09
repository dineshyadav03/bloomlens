"""BioCLIP 2 image/text embeddings.

Classification is zero-shot: cosine similarity between a photo's image
embedding and candidate species' taxonomic-name text embeddings (see
docs/RESEARCH.md#bioclip) rather than a reference-photo database.
"""

import threading

import numpy as np
import open_clip
import torch
from PIL import Image

_MODEL_NAME = "hf-hub:imageomics/bioclip-2"

_lock = threading.Lock()
_state = {"model": None, "preprocess": None, "tokenizer": None}


def load_model():
    """Load (once) and return (model, preprocess, tokenizer). CPU inference."""
    with _lock:
        if _state["model"] is None:
            model, _, preprocess = open_clip.create_model_and_transforms(_MODEL_NAME)
            model.eval()
            tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
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
