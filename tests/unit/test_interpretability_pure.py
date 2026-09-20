"""The parts of src/interpretability.py that need no model: the colormap and the
input validation that must happen before any heavy work."""

import numpy as np
import pytest
from PIL import Image

from src import interpretability


class TestColorize:
    def rgb(self, value):
        return tuple(interpretability._colorize(np.array([[value]], dtype=np.float32))[0, 0])

    def test_the_three_stops_are_cool_yellow_warm(self):
        stops = interpretability._COLOR_STOPS
        assert self.rgb(0.0) == tuple(stops[0].astype(np.uint8))
        assert self.rgb(0.5) == tuple(stops[1].astype(np.uint8))
        assert self.rgb(1.0) == tuple(stops[2].astype(np.uint8))

    def test_low_is_bluer_and_high_is_redder(self):
        low, high = self.rgb(0.0), self.rgb(1.0)
        assert low[2] > low[0]  # blue channel dominates at the cool end
        assert high[0] > high[2]  # red channel dominates at the warm end

    def test_values_outside_zero_one_are_clamped_not_wrapped(self):
        assert self.rgb(-3.0) == self.rgb(0.0)
        assert self.rgb(7.0) == self.rgb(1.0)

    def test_shape_and_dtype(self):
        out = interpretability._colorize(np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4))
        assert out.shape == (3, 4, 3)
        assert out.dtype == np.uint8


class TestExplainInputValidation:
    def test_an_unknown_species_is_rejected_before_the_model_is_touched(self, monkeypatch):
        def must_not_load():
            raise AssertionError("the model must not be loaded for an invalid species")

        monkeypatch.setattr(interpretability, "load_model", must_not_load)
        with pytest.raises(interpretability.ExplainError, match="not in BloomLens's curated species list"):
            interpretability.explain_image(Image.new("RGB", (8, 8)), "Unicorn flower")

    def test_every_curated_species_is_explainable(self, species_list):
        assert set(interpretability._SPECIES_BY_NAME) == {s["common_name"] for s in species_list}
