"""eval/corruptions.py: deterministic, size-preserving image corruptions (eval/PROTOCOL.md section 2.5)."""

import colorsys
import re

import numpy as np
import pytest
from PIL import Image

from eval import corruptions
from tests.conftest import REPO_ROOT

NAMES = ["gaussian_blur", "low_res", "dark", "occlusion", "crop", "jpeg", "rotate", "color_shift"]


def gradient(width=120, height=90) -> Image.Image:
    """A photo-like image with structure everywhere: distinct values at every position."""
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(height, 0)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None].repeat(width, 1)
    return Image.fromarray(np.stack([x, y, (x // 2 + y // 2).astype(np.uint8)], axis=-1), "RGB")


def flat(colour, size=(100, 100)) -> Image.Image:
    return Image.new("RGB", size, colour)


def pixels(image) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)


class TestTheSet:
    def test_the_eight_corruptions_of_the_protocol_are_exactly_those_implemented(self):
        assert sorted(corruptions.CORRUPTIONS) == sorted(NAMES)

    def test_the_protocol_document_names_every_one_of_them(self):
        text = (REPO_ROOT / "eval" / "PROTOCOL.md").read_text(encoding="utf-8")
        section = text[text.index("### 2.5") : text.index("### 2.6")]
        assert sorted(re.findall(r"`(\w+)`\s+\(", section)) == sorted(NAMES)

    def test_an_unknown_name_is_an_error_not_a_silent_no_op(self):
        with pytest.raises(KeyError, match="unknown corruption"):
            corruptions.apply(gradient(), "sepia", "s")


class TestEveryCorruption:
    @pytest.mark.parametrize("name", NAMES)
    def test_it_returns_rgb_of_the_same_size(self, name):
        for mode in ("RGB", "RGBA", "L", "P"):
            source = gradient(77, 53).convert(mode)
            result = corruptions.apply(source, name, "src")
            assert result.mode == "RGB" and result.size == (77, 53), (name, mode)

    @pytest.mark.parametrize("name", NAMES)
    def test_it_handles_tiny_and_extreme_shapes(self, name):
        for size in ((8, 8), (1, 1), (300, 20), (20, 300), (47, 49)):
            assert corruptions.apply(gradient(*size), name, "src").size == size

    @pytest.mark.parametrize("name", NAMES)
    def test_it_is_deterministic(self, name):
        first = corruptions.apply(gradient(), name, "oxford102:image_00042")
        second = corruptions.apply(gradient(), name, "oxford102:image_00042")
        assert first.tobytes() == second.tobytes()

    @pytest.mark.parametrize("name", NAMES)
    def test_it_actually_changes_the_image(self, name):
        source = gradient()
        assert corruptions.apply(source, name, "s").tobytes() != source.tobytes()

    @pytest.mark.parametrize("name", NAMES)
    def test_it_leaves_its_input_untouched(self, name):
        source = gradient()
        before = source.tobytes()
        corruptions.apply(source, name, "s")
        assert source.tobytes() == before


class TestSeededRandomness:
    @pytest.mark.parametrize("name", ["occlusion", "crop"])
    def test_the_random_part_depends_on_the_source_id(self, name):
        outputs = {corruptions.apply(gradient(), name, f"oxford102:image_{i:05d}").tobytes() for i in range(1, 9)}
        assert len(outputs) > 1

    @pytest.mark.parametrize("name", ["gaussian_blur", "low_res", "dark", "jpeg", "rotate", "color_shift"])
    def test_the_deterministic_ones_ignore_the_source_id(self, name):
        assert corruptions.apply(gradient(), name, "a").tobytes() == corruptions.apply(gradient(), name, "b").tobytes()


class TestEffects:
    def test_dark_scales_brightness_to_a_quarter(self):
        assert pixels(corruptions.apply(flat((200, 100, 40)), "dark", "s")).mean() == pytest.approx(
            pixels(flat((200, 100, 40))).mean() * corruptions.DARK_FACTOR, abs=1.5
        )

    def test_occlusion_blacks_out_thirty_percent_of_the_area(self):
        result = pixels(corruptions.apply(flat((255, 255, 255), (200, 150)), "occlusion", "s"))
        black = (result.sum(axis=2) == 0).mean()
        assert black == pytest.approx(corruptions.OCCLUDED_AREA, abs=0.02)

    def test_the_occluded_region_is_one_solid_square_inside_the_image(self):
        result = pixels(corruptions.apply(flat((255, 255, 255), (120, 120)), "occlusion", "s"))
        rows, cols = np.where(result.sum(axis=2) == 0)
        assert (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1) == len(rows)  # a filled rectangle

    def test_blur_removes_high_frequency_detail(self):
        checker = Image.fromarray((np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)).convert("RGB")
        assert pixels(corruptions.apply(checker, "gaussian_blur", "s")).std() < pixels(checker).std() * 0.5

    def test_low_res_flattens_fine_detail_but_keeps_the_size(self):
        checker = Image.fromarray((np.indices((200, 200)).sum(axis=0) % 2 * 255).astype(np.uint8)).convert("RGB")
        result = corruptions.apply(checker, "low_res", "s")
        assert result.size == (200, 200) and pixels(result).std() < pixels(checker).std() * 0.3

    def test_crop_keeps_forty_percent_of_the_area_scaled_back_up(self):
        """A vertical ramp cropped to 40 % of its area then stretched: the value range shrinks to about
        sqrt(0.4) of the original, and it is a genuine sub-window (not the whole image)."""
        ramp = Image.fromarray(np.linspace(0, 255, 200, dtype=np.uint8)[:, None].repeat(200, 1)).convert("RGB")
        result = pixels(corruptions.apply(ramp, "crop", "s"))[:, :, 0]
        assert result.max() - result.min() == pytest.approx(255 * (corruptions.CROP_AREA**0.5), abs=8)

    def test_jpeg_at_quality_eight_visibly_degrades_smooth_content(self):
        source = gradient()
        assert np.abs(pixels(corruptions.apply(source, "jpeg", "s")) - pixels(source)).mean() > 1.0

    def test_rotation_fills_the_corners_with_black_and_keeps_the_centre(self):
        result = pixels(corruptions.apply(flat((200, 200, 200)), "rotate", "s"))
        assert result[0, 0].sum() == 0 and result[-1, -1].sum() == 0
        assert result[50, 50].tolist() == [200, 200, 200]

    def test_color_shift_moves_the_hue_and_lowers_the_saturation(self):
        result = corruptions.apply(flat((255, 0, 0)), "color_shift", "s").getpixel((5, 5))
        hue, saturation, _ = colorsys.rgb_to_hsv(*(c / 255 for c in result))
        assert hue == pytest.approx(corruptions.HUE_SHIFT / 256, abs=0.02)
        assert saturation == pytest.approx(corruptions.SATURATION_FACTOR, abs=0.03)
