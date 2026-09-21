"""src/guard.py: every rule an untrusted upload must pass before the pipeline sees it."""

import io
import os
import struct
import tracemalloc
import zlib

import pytest
from PIL import Image

from src import guard
from src.guard import UploadRejected, decode_image, read_limited, validate_lot, validate_upload


def encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def png_bomb(width: int, height: int) -> bytes:
    """A REAL PNG that declares width x height pixels, built by streaming zeros
    through zlib so it stays small (~hundreds of KB) however huge the canvas."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    compressor = zlib.compressobj(1)
    row = b"\x00" * (width + 1)  # filter byte + one zero byte per pixel (8-bit greyscale)
    parts = [compressor.compress(row) for _ in range(height)] + [compressor.flush()]
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    signature = b"\x89PNG\r\n\x1a\n"
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", b"".join(parts)) + chunk(b"IEND", b"")


@pytest.fixture
def photo():
    return Image.new("RGB", (40, 30), (200, 30, 90))


def rejected(data: bytes) -> UploadRejected:
    with pytest.raises(UploadRejected) as info:
        decode_image(data)
    return info.value


class TestReadLimited:
    def test_a_file_exactly_at_the_limit_is_accepted(self):
        assert read_limited(io.BytesIO(b"x" * 100), 100) == b"x" * 100

    def test_one_byte_over_is_a_413(self):
        with pytest.raises(UploadRejected) as info:
            read_limited(io.BytesIO(b"x" * 101), 100)
        assert info.value.status == 413

    def test_it_stops_reading_once_the_limit_is_exceeded(self):
        """A huge body must never be pulled into memory just to be refused."""

        class Endless:
            reads = 0

            def read(self, n):
                self.reads += n
                return b"x" * n

        stream = Endless()
        with pytest.raises(UploadRejected):
            read_limited(stream, 1000)
        assert stream.reads < 2 * guard._READ_CHUNK + 2000

    def test_an_empty_stream_reads_as_empty(self):
        assert read_limited(io.BytesIO(b""), 100) == b""


class TestFormats:
    @pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
    def test_the_three_allowed_formats_decode_to_rgb(self, photo, fmt):
        image = decode_image(encode(photo, fmt))
        assert image.mode == "RGB" and image.size == photo.size

    @pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
    def test_other_real_image_formats_are_refused_with_415(self, photo, fmt):
        error = rejected(encode(photo, fmt))
        assert error.status == 415 and error.reason == "format"

    def test_the_format_comes_from_the_content_not_the_name(self, photo):
        """A GIF renamed .jpg is still a GIF; a JPEG named .txt is still a JPEG.
        (decode_image never sees a filename or Content-Type at all.)"""
        assert rejected(encode(photo, "GIF")).status == 415
        assert decode_image(encode(photo, "JPEG")).size == photo.size

    @pytest.mark.parametrize(
        "payload",
        [b"not an image at all", b"<svg xmlns='http://www.w3.org/2000/svg'><script>x()</script></svg>", b"%PDF-1.7"],
    )
    def test_non_images_are_a_400_not_a_crash(self, payload):
        assert rejected(payload).status == 400

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
    def test_an_image_cut_off_mid_stream_is_a_400(self, fmt):
        noisy = Image.frombytes("RGB", (96, 96), os.urandom(96 * 96 * 3))  # incompressible, so cutting bites
        data = encode(noisy, fmt)
        assert rejected(data[: len(data) * 6 // 10]).status == 400

    def test_an_empty_upload_is_a_400(self):
        assert rejected(b"").reason == "empty"

    def test_an_animated_image_is_refused(self):
        frames = [Image.new("RGB", (16, 16), (i * 40, 0, 0)) for i in range(3)]
        animated = encode(frames[0], "WEBP", save_all=True, append_images=frames[1:], duration=50)
        error = rejected(animated)
        assert error.status == 415 and error.reason == "animated"


class TestDecompressionBombs:
    def test_a_400_megapixel_png_is_refused_from_its_header_without_decoding_it(self):
        bomb = png_bomb(20_000, 20_000)  # declares 400 MP (~400 MB of pixels)
        assert len(bomb) < guard.MAX_UPLOAD_BYTES  # small enough to pass the byte cap
        tracemalloc.start()
        try:
            error = rejected(bomb)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert error.status == 413 and error.reason == "pixels"
        assert peak < 50 * 1024 * 1024  # nothing like the ~400 MB it declares

    def test_the_pixel_cap_is_exact_at_the_boundary(self, monkeypatch, photo):
        monkeypatch.setattr(guard, "MAX_IMAGE_PIXELS", 40 * 30)  # exactly this photo
        assert decode_image(encode(photo, "PNG")).size == (40, 30)
        monkeypatch.setattr(guard, "MAX_IMAGE_PIXELS", 40 * 30 - 1)
        assert rejected(encode(photo, "PNG")).status == 413


class TestMetadataIsStripped:
    @pytest.fixture
    def geotagged(self):
        """A JPEG with EXIF: a GPS position, a camera serial and an orientation tag."""
        exif = Image.Exif()
        exif[0x0110] = "SecretCameraModel-XYZ"  # Model
        exif[0xA431] = "SERIAL-0123456789"  # BodySerialNumber
        exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display
        exif[0x8825] = {1: "N", 2: (51.0, 30.0, 26.0), 3: "W", 4: (0.0, 7.0, 39.0)}  # GPSInfo: 51.5N 0.1W
        return encode(Image.new("RGB", (60, 20), (10, 200, 30)), "JPEG", exif=exif)

    def test_the_fixture_really_contains_the_sensitive_metadata(self, geotagged):
        assert b"SecretCameraModel-XYZ" in geotagged and b"SERIAL-0123456789" in geotagged
        assert len(Image.open(io.BytesIO(geotagged)).getexif().get_ifd(0x8825)) > 0

    def test_the_decoded_image_carries_no_metadata_at_all(self, geotagged):
        image = decode_image(geotagged)
        assert image.info == {}
        assert len(image.getexif()) == 0

    def test_the_orientation_is_applied_before_it_is_dropped(self, geotagged):
        """60x20 tagged 'rotate 90' displays as 20x60; stripping the tag without
        rotating would send the model a sideways photo."""
        assert decode_image(geotagged).size == (20, 60)

    def test_what_the_pipeline_re_encodes_for_gemini_contains_none_of_it(self, geotagged):
        from src.identify import _image_to_jpeg_bytes

        sent = _image_to_jpeg_bytes(decode_image(geotagged))
        for secret in (b"SecretCameraModel-XYZ", b"SERIAL-0123456789", b"Exif"):
            assert secret not in sent
        reopened = Image.open(io.BytesIO(sent))
        assert len(reopened.getexif()) == 0
        assert reopened.getexif().get_ifd(0x8825) == {}  # no GPS block survives


class TestPixelHandling:
    def test_transparent_pixels_are_flattened_onto_white_not_black(self):
        rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 0))  # fully transparent red
        assert decode_image(encode(rgba, "PNG")).getpixel((5, 5)) == (255, 255, 255)

    def test_opaque_colors_survive(self):
        opaque = Image.new("RGBA", (10, 10), (10, 20, 30, 255))
        assert decode_image(encode(opaque, "PNG")).getpixel((5, 5)) == (10, 20, 30)

    def test_palette_and_greyscale_images_become_rgb(self):
        assert decode_image(encode(Image.new("L", (8, 8), 128), "PNG")).mode == "RGB"
        assert decode_image(encode(Image.new("P", (8, 8)), "PNG")).mode == "RGB"


class TestPublicMessages:
    def test_messages_never_leak_library_internals_or_filenames(self):
        for payload in (b"garbage", b"", png_bomb(20_000, 20_000)[:500]):
            error = rejected(payload)
            for leak in ("PIL", "Pillow", "cannot identify", "Traceback", "UnidentifiedImageError", "BytesIO"):
                assert leak not in error.message

    def test_lot_and_single_limits_are_stated_in_megabytes(self):
        with pytest.raises(UploadRejected, match="MB"):
            read_limited(io.BytesIO(b"x" * 20), 10)


class TestValidateHelpers:
    def test_validate_upload_reads_then_decodes(self, photo):
        assert validate_upload(io.BytesIO(encode(photo, "PNG"))).size == (40, 30)

    def test_validate_upload_enforces_the_byte_cap_before_decoding(self, photo):
        with pytest.raises(UploadRejected) as info:
            validate_upload(io.BytesIO(encode(photo, "PNG")), limit=10)
        assert info.value.reason == "too_large"

    def test_a_lot_has_its_own_total_cap(self, photo):
        data = encode(photo, "PNG")
        files = [io.BytesIO(data) for _ in range(3)]
        with pytest.raises(UploadRejected) as info:
            validate_lot(files, per_file_limit=len(data) + 1, total_limit=len(data) * 2)
        assert info.value.reason == "lot_too_large" and info.value.status == 413

    def test_a_valid_lot_decodes_every_photo(self, photo):
        files = [io.BytesIO(encode(photo, "JPEG")) for _ in range(3)]
        assert [i.size for i in validate_lot(files)] == [(40, 30)] * 3

    def test_one_bad_file_in_a_lot_rejects_the_lot(self, photo):
        files = [io.BytesIO(encode(photo, "PNG")), io.BytesIO(b"nope")]
        with pytest.raises(UploadRejected):
            validate_lot(files)
