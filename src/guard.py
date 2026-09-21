"""Upload validation shared by the FastAPI service and the Streamlit app.

Everything a user can send that reaches the model pipeline goes through here, so
there is one implementation of the rules, not one per front end:

- a hard byte cap, enforced while *reading* (an oversized body is never held whole);
- the format is decided from the file's actual content, never from its name or the
  client-supplied Content-Type, and only JPEG / PNG / WebP are accepted;
- a pixel cap checked from the header *before* any pixel data is decoded, so a small
  file that declares a gigantic canvas (a decompression bomb) costs nothing;
- animated images are refused;
- the decoded image is rebuilt from pixels alone, which drops EXIF (GPS position,
  camera serial, timestamps), XMP and ICC profiles. The camera's orientation is
  applied first so the photo isn't left sideways once that tag is gone.

Errors carry a fixed, generic public message: never a filename and never library
internals. The specific reason is logged server-side.
"""

import io
import logging
import os
import warnings

from PIL import Image, ImageOps

logger = logging.getLogger("bloomlens.guard")


def _megabytes_from_env(name: str, default: float) -> int:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        value = default
    return int(max(value, 0.1) * 1024 * 1024)


MAX_UPLOAD_BYTES = _megabytes_from_env("BLOOMLENS_MAX_UPLOAD_MB", 8)
MAX_LOT_BYTES = _megabytes_from_env("BLOOMLENS_MAX_LOT_MB", 40)
MAX_IMAGE_PIXELS = 40_000_000  # 40 megapixels: more than any phone camera in normal mode
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})

# Pillow warns above this and errors above twice it; we enforce it ourselves, exactly.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_READ_CHUNK = 64 * 1024
_PIXELS_MESSAGE = f"That photo has too many pixels (the limit is {MAX_IMAGE_PIXELS // 1_000_000} megapixels)."


class UploadRejected(Exception):
    """A user-caused rejection. `status` is an HTTP status; `message` is safe to show."""

    def __init__(self, status: int, message: str, *, reason: str):
        super().__init__(message)
        self.status = status
        self.message = message
        self.reason = reason


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 * 1024):g} MB"


def read_limited(stream, limit: int) -> bytes:
    """Read a file-like object, refusing as soon as it exceeds `limit` bytes."""
    buffer = bytearray()
    while True:
        chunk = stream.read(min(_READ_CHUNK, limit + 1 - len(buffer)))
        if not chunk:
            return bytes(buffer)
        buffer += chunk
        if len(buffer) > limit:
            raise UploadRejected(413, f"That photo is too large (the limit is {_mb(limit)}).", reason="too_large")


def _reject(status: int, message: str, reason: str) -> UploadRejected:
    logger.warning("upload rejected: %s", reason)
    return UploadRejected(status, message, reason=reason)


def decode_image(data: bytes) -> Image.Image:
    """Decode untrusted bytes into a clean RGB image, or raise UploadRejected."""
    if not data:
        raise _reject(400, "That file is empty.", "empty")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))

            if image.format not in ALLOWED_FORMATS:
                raise _reject(415, "Unsupported image type. Please use a JPEG, PNG or WebP photo.", "format")
            if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) > 1:
                raise _reject(415, "Animated images aren't supported.", "animated")
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise _reject(413, _PIXELS_MESSAGE, "pixels")

            image.load()
    except UploadRejected:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        # Pillow's own bomb check fires inside Image.open for anything over 2x the limit.
        raise _reject(413, _PIXELS_MESSAGE, "pixels") from exc
    except Exception as exc:  # noqa: BLE001 -- decoders raise many types; any failure is a bad upload
        raise _reject(400, "That file isn't a readable image.", type(exc).__name__) from exc

    return _clean(image)


def _clean(image: Image.Image) -> Image.Image:
    """Upright, opaque RGB pixels and nothing else (no EXIF/XMP/ICC/info)."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        rgb = flattened
    else:
        rgb = image.convert("RGB")

    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    return clean


def validate_upload(stream, limit: int | None = None) -> Image.Image:
    """Read (bounded) and decode one uploaded file."""
    return decode_image(read_limited(stream, MAX_UPLOAD_BYTES if limit is None else limit))


def validate_lot(streams, per_file_limit: int | None = None, total_limit: int | None = None) -> list[Image.Image]:
    """Validate every file of a lot; the whole lot has its own, smaller-than-N-times cap."""
    per_file_limit = MAX_UPLOAD_BYTES if per_file_limit is None else per_file_limit
    total_limit = MAX_LOT_BYTES if total_limit is None else total_limit
    images, total = [], 0
    for stream in streams:
        data = read_limited(stream, per_file_limit)
        total += len(data)
        if total > total_limit:
            raise _reject(413, f"That lot is too large (the limit is {_mb(total_limit)} in total).", "lot_too_large")
        images.append(decode_image(data))
    return images
