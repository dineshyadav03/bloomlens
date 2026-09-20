"""The Grad-ECLIP overlay against the real model. There is no ground truth for
"the heatmap is right", so these pin what CAN be checked: shape/finite/non-negative,
determinism, dependence on the target species, and -- the Milestone 9 regression --
that explaining a photo never mutates the shared model, even under concurrency."""

import concurrent.futures

import numpy as np
import pytest
import torch
from PIL import Image

from src import interpretability
from src.embeddings import embed_image, embed_text
from tests.integration.conftest import known_good_photo


@pytest.fixture
def taxonomy(species_list):
    by_name = {s["common_name"]: s["taxonomy_string"] for s in species_list}
    return lambda name: by_name[name]


@pytest.fixture
def rose():
    return Image.open(known_good_photo("Rose")).convert("RGB")


def heatmap(bioclip, image, taxonomy_string):
    _, preprocess, _ = bioclip
    tensor = preprocess(image).unsqueeze(0)
    return interpretability._heatmap_for_last_layer(tensor, embed_text(taxonomy_string))


class TestHeatmap:
    def test_it_is_a_16x16_finite_non_negative_map_with_signal(self, bioclip, rose, taxonomy):
        h = heatmap(bioclip, rose, taxonomy("Rose"))
        assert h.shape == (16, 16)
        assert np.isfinite(h).all()
        assert (h >= 0).all()  # the method ends in a ReLU
        assert h.max() > 0

    def test_it_is_deterministic(self, bioclip, rose, taxonomy):
        assert np.array_equal(heatmap(bioclip, rose, taxonomy("Rose")), heatmap(bioclip, rose, taxonomy("Rose")))

    def test_it_depends_on_which_species_is_being_explained(self, bioclip, rose, taxonomy):
        """Gradients (not plain attention) are the point: the map must change with the target."""
        right = heatmap(bioclip, rose, taxonomy("Rose"))
        wrong = heatmap(bioclip, rose, taxonomy("Bird of paradise"))
        assert not np.array_equal(right, wrong)
        assert np.abs(right - wrong).mean() > 1e-4


class TestExplainImage:
    def test_it_returns_an_rgb_overlay_of_the_same_size_that_differs_from_the_input(self, bioclip, rose):
        overlay = interpretability.explain_image(rose, "Rose")
        assert overlay.mode == "RGB" and overlay.size == rose.size
        assert np.abs(np.asarray(overlay, dtype=int) - np.asarray(rose, dtype=int)).max() > 0

    def test_non_square_and_tiny_inputs_are_supported(self, bioclip):
        for size in ((300, 120), (40, 40)):
            image = Image.new("RGB", size, (180, 40, 60))
            assert interpretability.explain_image(image, "Rose").size == size


class TestSharedModelIsNeverMutated:
    """Milestone 9's first version monkey-patched the shared model's last attention
    block for the duration of an explain call, which would have corrupted a concurrent
    identify()/embed_image() on another thread. It was rewritten to read weights only."""

    @staticmethod
    def fingerprint(model):
        last = model.visual.transformer.resblocks[-1]
        return (
            float(last.attn.in_proj_weight.abs().sum()),
            float(model.visual.proj.abs().sum()),
            model.training,
            [p.requires_grad for p in model.parameters()][:5],
        )

    def test_no_block_gets_an_instance_level_override(self, bioclip, rose):
        model, _, _ = bioclip
        interpretability.explain_image(rose, "Rose")
        for block in model.visual.transformer.resblocks:
            assert "attention" not in vars(block), "a block's attention method was monkey-patched"
            assert "forward" not in vars(block)

    def test_weights_mode_gradients_and_global_grad_state_are_untouched(self, bioclip, rose):
        model, _, _ = bioclip
        before = self.fingerprint(model)
        interpretability.explain_image(rose, "Rose")
        assert self.fingerprint(model) == before
        assert all(p.grad is None for p in model.parameters())  # torch.autograd.grad leaves .grad alone
        assert torch.is_grad_enabled()

    def test_concurrent_embeddings_stay_bit_identical_while_explanations_run(self, bioclip, rose):
        sunflower = Image.open(known_good_photo("Sunflower")).convert("RGB")
        baseline = embed_image(sunflower)

        def explain():
            interpretability.explain_image(rose, "Rose")
            return None

        def embed():
            return embed_image(sunflower)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            explanations = [pool.submit(explain) for _ in range(2)]
            embeddings = [pool.submit(embed) for _ in range(6)]
            for f in explanations:
                f.result()
            results = [f.result() for f in embeddings]
        assert all(np.array_equal(baseline, r) for r in results)
