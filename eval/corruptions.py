"""Deterministic image corruptions for the robustness family (eval/PROTOCOL.md section 2.5).

Covariate shift on covered species, not out-of-distribution data: the label stays the species.
Each corruption is a pure function of (image, source id): the random parts (where the occluding
square sits, which 40 % crop) come from a generator seeded by SHA-256 of "<source id>|<name>", so a
variant is reproducible on any machine and never has to be stored. Output is always an RGB image the
same size as the input.
"""

import hashlib
import random
from collections.abc import Callable

from PIL import Image, ImageEnhance, ImageFilter

BLUR_SIGMA_FRACTION = 0.02  # of the short side
LOW_RES_SHORT_SIDE = 48
DARK_FACTOR = 0.25
OCCLUDED_AREA = 0.30
CROP_AREA = 0.40
JPEG_QUALITY = 8
ROTATE_DEGREES = 45
HUE_SHIFT = 40  # out of 256
SATURATION_FACTOR = 0.6


def _rng(source_id: str, name: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{source_id}|{name}".encode()).hexdigest()[:16], 16))


def gaussian_blur(image: Image.Image, source_id: str) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(BLUR_SIGMA_FRACTION * min(image.size)))


def low_res(image: Image.Image, source_id: str) -> Image.Image:
    width, height = image.size
    scale = LOW_RES_SHORT_SIDE / min(width, height)
    small = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    return small.resize((width, height), Image.BICUBIC)


def dark(image: Image.Image, source_id: str) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(DARK_FACTOR)


def occlusion(image: Image.Image, source_id: str) -> Image.Image:
    """A black square covering OCCLUDED_AREA of the image at a seeded position."""
    width, height = image.size
    side = min(round((OCCLUDED_AREA * width * height) ** 0.5), width, height)
    rng = _rng(source_id, "occlusion")
    left, top = rng.randint(0, width - side), rng.randint(0, height - side)
    out = image.copy()
    out.paste((0, 0, 0), (left, top, left + side, top + side))
    return out


def crop(image: Image.Image, source_id: str) -> Image.Image:
    """A seeded crop keeping CROP_AREA of the image (same aspect ratio), resized back up."""
    width, height = image.size
    factor = CROP_AREA**0.5
    crop_w, crop_h = max(1, round(width * factor)), max(1, round(height * factor))
    rng = _rng(source_id, "crop")
    left, top = rng.randint(0, width - crop_w), rng.randint(0, height - crop_h)
    return image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.BICUBIC)


def jpeg(image: Image.Image, source_id: str) -> Image.Image:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def rotate(image: Image.Image, source_id: str) -> Image.Image:
    return image.rotate(ROTATE_DEGREES, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def color_shift(image: Image.Image, source_id: str) -> Image.Image:
    """Hue +40/256 and saturation x 0.6."""
    hue, saturation, value = image.convert("HSV").split()
    hue = hue.point(lambda h: (h + HUE_SHIFT) % 256)
    saturation = saturation.point(lambda s: round(s * SATURATION_FACTOR))
    return Image.merge("HSV", (hue, saturation, value)).convert("RGB")


CORRUPTIONS: dict[str, Callable[[Image.Image, str], Image.Image]] = {
    "gaussian_blur": gaussian_blur,
    "low_res": low_res,
    "dark": dark,
    "occlusion": occlusion,
    "crop": crop,
    "jpeg": jpeg,
    "rotate": rotate,
    "color_shift": color_shift,
}


def apply(image: Image.Image, name: str, source_id: str) -> Image.Image:
    """`image` under the named corruption, as RGB of the original size."""
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; known: {sorted(CORRUPTIONS)}")
    result = CORRUPTIONS[name](image.convert("RGB"), source_id)
    assert result.mode == "RGB" and result.size == image.size, name
    return result
