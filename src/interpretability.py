"""Interpretability overlay: which part of a scanned photo drove BioCLIP 2's
species match, shown as a heatmap over the original image.

Not literal Grad-CAM: the original Grad-CAM paper backprops a class score
into a CNN's last convolutional feature map. BioCLIP 2 (src/embeddings.py)
is a Vision Transformer doing zero-shot classification (cosine similarity
to a text embedding, no conv feature maps and no classifier head to take a
class score from) -- confirmed against the actual loaded model, not assumed
from the model name. Uses **Grad-ECLIP** instead ("Gradient-based Visual
Explanation for Transformer-based CLIP", Zhao et al., ICML 2024;
arXiv:2502.18816), a published technique built for exactly this
CLIP-style dual-encoder case. See docs/RESEARCH.md#grad-eclip for the full
research trail, including the paper's own reference implementation this
was checked against.

Scoped to the paper's own best-fidelity finding: gradients + values from
ONLY the last transformer block (they found this beats aggregating all
layers for images). No ground truth exists to unit-test a "correct"
heatmap against -- this was validated by eye against real photos during
development (see the plan/commit history), not by an automated metric.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.embeddings import embed_text, load_model

_SPECIES_PATH = Path(__file__).resolve().parent.parent / "data" / "species_reference.json"
_SPECIES_BY_NAME = {s["common_name"]: s for s in json.loads(_SPECIES_PATH.read_text(encoding="utf-8"))}

# Cool (low importance) -> warm (high importance) 3-stop gradient. A flat
# single-hue overlay was tried first and found illegible on already-red/
# orange flowers (confirmed on real Rose/Carnation photos) -- hue-shifting
# stays legible regardless of the photo's own colors.
_COLOR_STOPS = np.array([[40, 60, 200], [255, 220, 60], [230, 30, 30]], dtype=np.float32)
_MAX_OVERLAY_ALPHA = 200  # out of 255 -- low-importance regions stay near-transparent


class ExplainError(Exception):
    pass


def _colorize(norm: np.ndarray) -> np.ndarray:
    """norm: array in [0, 1]. Returns an RGB uint8 array via the cool->warm gradient."""
    t = np.clip(norm * 2, 0, 2)
    t0 = np.clip(t, 0, 1)[..., None]
    t1 = np.clip(t - 1, 0, 1)[..., None]
    color = np.where(
        (t <= 1)[..., None],
        _COLOR_STOPS[0] * (1 - t0) + _COLOR_STOPS[1] * t0,
        _COLOR_STOPS[1] * (1 - t1) + _COLOR_STOPS[2] * t1,
    )
    return color.astype(np.uint8)


def _heatmap_for_last_layer(pixel_tensor: torch.Tensor, text_embedding: np.ndarray) -> np.ndarray:
    """Grad-ECLIP for the last ViT block, adapted for open_clip's batch_first=True
    layout (the reference implementation targets OpenAI CLIP's [seq, batch, dim]).
    Returns a (grid, grid) numpy array of per-patch importance, un-normalized.

    Manually replicates VisionTransformer.forward() (embed -> blocks -> pool ->
    proj) instead of calling `model.visual(...)` directly, unrolling only the
    LAST block by hand to expose its pre-out_proj attention output (the basis
    the raw value vectors live in -- gradient against the post-out_proj output
    was tried first and produced heatmaps concentrated on background noise,
    fixed by checking the paper's reference implementation, which takes the
    gradient pre-out_proj). Deliberately reads weights off the shared model's
    submodules rather than monkey-patching any of its bound methods -- the
    model is a process-wide singleton (src/embeddings.py's load_model()) that
    a concurrent identify()/embed_image() call on another thread could be
    using at the same moment (Streamlit/FastAPI can both serve concurrent
    requests, see api/main.py); mutating shared state here would corrupt
    that other call. Every tensor below is local to this call.
    """
    model, _, _ = load_model()
    visual = model.visual
    blocks = visual.transformer.resblocks
    last_block = blocks[-1]
    attn = last_block.attn
    embed_dim, num_heads = attn.embed_dim, attn.num_heads
    head_dim = embed_dim // num_heads

    x = visual._embeds(pixel_tensor)  # patchify + CLS + positional embed + ln_pre
    for block in blocks[:-1]:
        x = block(x)  # normal forward, safe (read-only) for all but the last block

    q_x = last_block.ln_1(x)  # self-attention: k_x = v_x = q_x
    w_q, w_k, w_v = attn.in_proj_weight.chunk(3)
    b_q, b_k, b_v = attn.in_proj_bias.chunk(3)
    q = F.linear(q_x, w_q, b_q)
    k = F.linear(q_x, w_k, b_k)
    v = F.linear(q_x, w_v, b_v)

    batch, seq, _ = q.shape

    def split_heads(t):
        return t.view(batch, seq, num_heads, head_dim).transpose(1, 2)

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
    scores = (qh @ kh.transpose(-2, -1)) / (head_dim**0.5)
    weights = F.softmax(scores, dim=-1)
    attn_output = (weights @ vh).transpose(1, 2).reshape(batch, seq, embed_dim)  # pre-out_proj

    out = F.linear(attn_output, attn.out_proj.weight, attn.out_proj.bias)
    x = x + last_block.ls_1(out)
    x = x + last_block.ls_2(last_block.mlp(last_block.ln_2(x)))

    pooled, _tokens = visual._pool(x)
    if visual.proj is not None:
        pooled = pooled @ visual.proj

    image_embedding = pooled / pooled.norm(dim=-1, keepdim=True)
    text_vec = torch.from_numpy(text_embedding).unsqueeze(0)
    similarity = (image_embedding * text_vec).sum()

    grad = torch.autograd.grad(similarity, attn_output)[0]

    # batch_first=True: [batch, seq, dim]. Index 0 = CLS, 1: = patches.
    grad_cls = grad[0, 0, :]
    q, k, v = q[0], k[0], v[0]

    q_cls = F.normalize(q[0:1, :], dim=-1)
    k_patch = F.normalize(k[1:, :], dim=-1)
    cosine_qk = (q_cls * k_patch).sum(-1)
    cosine_qk = (cosine_qk - cosine_qk.min()) / (cosine_qk.max() - cosine_qk.min() + 1e-8)

    v_patch = v[1:, :]
    emap = F.relu((grad_cls * v_patch * cosine_qk[:, None]).sum(-1)).detach()

    grid = int(round(emap.shape[0] ** 0.5))
    return emap.reshape(grid, grid).numpy()


def explain_image(image: Image.Image, species: str) -> Image.Image:
    """Returns `image` with a heatmap overlay showing which regions most drove
    its match to `species` (a curated common_name). Raises ExplainError if
    `species` isn't in the curated list."""
    entry = _SPECIES_BY_NAME.get(species)
    if entry is None:
        raise ExplainError(f"'{species}' is not in BloomLens's curated species list.")

    model, preprocess, _ = load_model()
    rgb_image = image.convert("RGB")
    pixel_tensor = preprocess(rgb_image).unsqueeze(0)
    text_embedding = embed_text(entry["taxonomy_string"])

    heatmap = _heatmap_for_last_layer(pixel_tensor, text_embedding)

    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    norm_img = Image.fromarray((norm * 255).astype(np.uint8)).resize(rgb_image.size, Image.BICUBIC)
    norm_up = np.asarray(norm_img).astype(np.float32) / 255.0

    color_img = Image.fromarray(_colorize(norm_up))
    alpha = Image.fromarray((norm_up * _MAX_OVERLAY_ALPHA).astype(np.uint8))
    return Image.composite(color_img, rgb_image, alpha)
