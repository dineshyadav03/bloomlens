"""Model pinning (src/versions.py, src/model_pin.py) and the loader's drift guards
in src/embeddings.py. Uses tiny temp files; never needs the real weights."""

import copy
import hashlib
import os
import re

import pytest
from huggingface_hub import constants as hf_constants
from huggingface_hub.errors import LocalEntryNotFoundError
from torchvision import transforms

from src import embeddings, model_pin, versions
from src.model_pin import ModelIntegrityError, verify_file

CONTENT = b'{"pretend": "model config"}'
SHA = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def pinned_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(CONTENT)
    return path


@pytest.fixture(autouse=True)
def isolated_stamp_dir(tmp_path, monkeypatch):
    """The verification stamp lives under HF_HOME; keep it out of the real cache."""
    monkeypatch.setattr(hf_constants, "HF_HOME", str(tmp_path / "hf_home"))


class TestPinConstants:
    def test_the_revision_is_a_full_commit_hash(self):
        assert re.fullmatch(r"[0-9a-f]{40}", versions.BIOCLIP_REVISION)

    def test_every_pinned_file_has_a_size_and_sha256(self):
        assert set(versions.BIOCLIP_FILES) == {versions.BIOCLIP_CONFIG_FILE, versions.BIOCLIP_WEIGHTS_FILE}
        for meta in versions.BIOCLIP_FILES.values():
            assert meta["size"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256"])

    def test_the_gemini_model_is_a_named_version_not_a_moving_alias(self):
        assert versions.GEMINI_MODEL and "latest" not in versions.GEMINI_MODEL


class TestVerifyFile:
    def test_an_untouched_file_passes(self, pinned_file):
        verify_file(pinned_file, len(CONTENT), SHA, use_stamp=False)

    def test_same_size_tampering_is_refused_on_the_checksum(self, pinned_file):
        tampered = bytearray(CONTENT)
        tampered[5] ^= 0x01
        pinned_file.write_bytes(bytes(tampered))
        assert len(tampered) == len(CONTENT)
        with pytest.raises(ModelIntegrityError, match="sha256"):
            verify_file(pinned_file, len(CONTENT), SHA, use_stamp=False)

    def test_a_truncated_file_is_refused_on_its_size(self, pinned_file):
        pinned_file.write_bytes(CONTENT[:-3])
        with pytest.raises(ModelIntegrityError, match="size"):
            verify_file(pinned_file, len(CONTENT), SHA, use_stamp=False)

    def test_the_error_names_the_pinned_revision_prefix(self, pinned_file):
        pinned_file.write_bytes(b"x")
        with pytest.raises(ModelIntegrityError, match=versions.BIOCLIP_REVISION[:8]):
            verify_file(pinned_file, len(CONTENT), SHA, use_stamp=False)


class TestVerificationStamp:
    @pytest.fixture
    def hash_calls(self, monkeypatch):
        calls = []
        real = model_pin._sha256
        monkeypatch.setattr(model_pin, "_sha256", lambda path: calls.append(path) or real(path))
        return calls

    def test_a_passing_check_is_remembered_so_big_files_are_not_rehashed(self, pinned_file, hash_calls):
        verify_file(pinned_file, len(CONTENT), SHA)
        verify_file(pinned_file, len(CONTENT), SHA)
        assert len(hash_calls) == 1

    def test_a_modified_file_is_rehashed_and_caught_despite_the_stamp(self, pinned_file, hash_calls):
        verify_file(pinned_file, len(CONTENT), SHA)
        tampered = bytearray(CONTENT)
        tampered[5] ^= 0x01
        pinned_file.write_bytes(bytes(tampered))
        os.utime(pinned_file, ns=(1, 1))  # a different mtime, as any real edit would give
        with pytest.raises(ModelIntegrityError):
            verify_file(pinned_file, len(CONTENT), SHA)
        assert len(hash_calls) == 2

    def test_an_unwritable_stamp_location_only_costs_a_rehash(self, pinned_file, tmp_path, monkeypatch):
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x")
        monkeypatch.setattr(hf_constants, "HF_HOME", str(blocker))  # stamp path would be inside a file
        verify_file(pinned_file, len(CONTENT), SHA)  # must not raise


class TestFetchPinned:
    def test_the_local_cache_is_tried_first_and_no_network_call_is_made_when_cached(self, monkeypatch, tmp_path):
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return str(tmp_path / "cached.bin")

        monkeypatch.setattr(model_pin, "hf_hub_download", fake_download)
        model_pin._resolve("open_clip_config.json")
        assert len(calls) == 1
        assert calls[0]["local_files_only"] is True
        assert calls[0]["revision"] == versions.BIOCLIP_REVISION
        assert calls[0]["repo_id"] == versions.BIOCLIP_REPO

    def test_a_cache_miss_falls_back_to_a_download_of_the_same_pinned_revision(self, monkeypatch, tmp_path):
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise LocalEntryNotFoundError("not cached")
            return str(tmp_path / "downloaded.bin")

        monkeypatch.setattr(model_pin, "hf_hub_download", fake_download)
        model_pin._resolve("open_clip_config.json")
        assert [bool(c.get("local_files_only")) for c in calls] == [True, False]
        assert calls[1]["revision"] == versions.BIOCLIP_REVISION

    def test_fetch_refuses_a_cached_file_that_does_not_match_the_pin(self, monkeypatch, pinned_file):
        monkeypatch.setattr(model_pin, "_resolve", lambda _name: pinned_file)
        monkeypatch.setattr(model_pin, "BIOCLIP_FILES", {"open_clip_config.json": {"size": 999, "sha256": "0" * 64}})
        with pytest.raises(ModelIntegrityError):
            model_pin.fetch_pinned_bioclip()


class TestLoaderDriftGuards:
    """If an open_clip upgrade changed its built-in ViT-L-14 config or default
    normalization, load_model() must refuse rather than quietly change the model."""

    PINNED = {
        "model_cfg": {
            "embed_dim": 768,
            "vision_cfg": {"image_size": 224, "layers": 24, "width": 1024, "patch_size": 14},
            "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 768, "heads": 12, "layers": 12},
        },
        "preprocess_cfg": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
    }

    def test_the_real_builtin_architecture_matches_the_pinned_config(self):
        embeddings._assert_architecture_matches_pinned_config(self.PINNED)

    def test_a_changed_architecture_is_refused(self):
        drifted = copy.deepcopy(self.PINNED)
        drifted["model_cfg"]["vision_cfg"]["layers"] = 12
        with pytest.raises(RuntimeError, match="no longer matches the pinned"):
            embeddings._assert_architecture_matches_pinned_config(drifted)

    def compose(self, mean, std):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    def test_matching_normalization_passes(self):
        cfg = self.PINNED["preprocess_cfg"]
        embeddings._assert_preprocess_matches_pinned_config(self.PINNED, self.compose(cfg["mean"], cfg["std"]))

    def test_changed_normalization_is_refused(self):
        cfg = self.PINNED["preprocess_cfg"]
        wrong_mean = [0.5, 0.5, 0.5]
        with pytest.raises(RuntimeError, match="normalization"):
            embeddings._assert_preprocess_matches_pinned_config(self.PINNED, self.compose(wrong_mean, cfg["std"]))
