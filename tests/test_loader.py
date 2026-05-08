"""Tests for loader.py — compared against safetensors library as golden reference."""

import os
import sys
import struct
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loader import load_config, load_weights

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "model.safetensors")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_successfully(self):
        config = load_config(CONFIG_PATH)
        assert isinstance(config, dict)

    def test_has_required_keys(self):
        config = load_config(CONFIG_PATH)
        for key in ["n_layer", "n_head", "n_embd", "vocab_size", "n_positions"]:
            assert key in config, f"missing key: {key}"

    def test_gpt2_values(self):
        config = load_config(CONFIG_PATH)
        assert config["n_layer"] == 12
        assert config["n_head"] == 12
        assert config["n_embd"] == 768
        assert config["vocab_size"] == 50257
        assert config["n_positions"] == 1024

    def test_layer_norm_epsilon(self):
        config = load_config(CONFIG_PATH)
        assert config["layer_norm_epsilon"] == 1e-5

    def test_token_ids(self):
        config = load_config(CONFIG_PATH)
        assert config["bos_token_id"] == 50256
        assert config["eos_token_id"] == 50256


# ---------------------------------------------------------------------------
# Weight loading tests
# ---------------------------------------------------------------------------

class TestLoadWeights:
    @pytest.fixture(scope="class")
    def weights(self):
        """Load weights once for the whole test class."""
        return load_weights(WEIGHTS_PATH)

    def test_loads_all_tensors(self, weights):
        # GPT-2 has 160 tensors total
        assert len(weights) == 160

    def test_has_embedding_weights(self, weights):
        assert "wte.weight" in weights
        assert "wpe.weight" in weights

    def test_has_final_layer_norm(self, weights):
        assert "ln_f.weight" in weights
        assert "ln_f.bias" in weights

    def test_has_all_layers(self, weights):
        for i in range(12):
            prefix = f"h.{i}"
            assert f"{prefix}.ln_1.weight" in weights
            assert f"{prefix}.ln_1.bias" in weights
            assert f"{prefix}.ln_2.weight" in weights
            assert f"{prefix}.ln_2.bias" in weights
            assert f"{prefix}.attn.c_attn.weight" in weights
            assert f"{prefix}.attn.c_attn.bias" in weights
            assert f"{prefix}.attn.c_proj.weight" in weights
            assert f"{prefix}.attn.c_proj.bias" in weights
            assert f"{prefix}.mlp.c_fc.weight" in weights
            assert f"{prefix}.mlp.c_fc.bias" in weights
            assert f"{prefix}.mlp.c_proj.weight" in weights
            assert f"{prefix}.mlp.c_proj.bias" in weights

    def test_embedding_shapes(self, weights):
        assert weights["wte.weight"].shape == [50257, 768]
        assert weights["wpe.weight"].shape == [1024, 768]

    def test_layer_shapes(self, weights):
        # Check layer 0 shapes
        assert weights["h.0.attn.c_attn.weight"].shape == [768, 2304]
        assert weights["h.0.attn.c_attn.bias"].shape == [2304]
        assert weights["h.0.attn.c_proj.weight"].shape == [768, 768]
        assert weights["h.0.attn.c_proj.bias"].shape == [768]
        assert weights["h.0.mlp.c_fc.weight"].shape == [768, 3072]
        assert weights["h.0.mlp.c_fc.bias"].shape == [3072]
        assert weights["h.0.mlp.c_proj.weight"].shape == [3072, 768]
        assert weights["h.0.mlp.c_proj.bias"].shape == [768]
        assert weights["h.0.ln_1.weight"].shape == [768]
        assert weights["h.0.ln_1.bias"].shape == [768]

    def test_final_layer_norm_shapes(self, weights):
        assert weights["ln_f.weight"].shape == [768]
        assert weights["ln_f.bias"].shape == [768]

    def test_values_not_all_zero(self, weights):
        """Sanity check: loaded tensors should contain non-zero values."""
        wte = np.array(weights["wte.weight"].to_vec())
        assert np.any(wte != 0.0)

        w0 = np.array(weights["h.0.attn.c_attn.weight"].to_vec())
        assert np.any(w0 != 0.0)


# ---------------------------------------------------------------------------
# Compare against safetensors library (golden reference)
# ---------------------------------------------------------------------------

class TestVsGoldenReference:
    """Load the same file with the safetensors library and compare values."""

    @pytest.fixture(scope="class")
    def our_weights(self):
        return load_weights(WEIGHTS_PATH)

    @pytest.fixture(scope="class")
    def ref_weights(self):
        """Load with safetensors library as golden reference."""
        safetensors = pytest.importorskip("safetensors")
        from safetensors.numpy import load_file
        return load_file(WEIGHTS_PATH)

    SPOT_CHECK_TENSORS = [
        "wte.weight",
        "wpe.weight",
        "ln_f.weight",
        "ln_f.bias",
        "h.0.attn.c_attn.weight",
        "h.0.attn.c_attn.bias",
        "h.0.attn.c_proj.weight",
        "h.0.mlp.c_fc.weight",
        "h.0.mlp.c_fc.bias",
        "h.0.mlp.c_proj.weight",
        "h.5.attn.c_attn.weight",
        "h.11.mlp.c_proj.weight",
    ]

    @pytest.mark.parametrize("name", SPOT_CHECK_TENSORS)
    def test_tensor_matches(self, our_weights, ref_weights, name):
        ours = np.array(our_weights[name].to_vec(), dtype=np.float32).reshape(
            our_weights[name].shape
        )
        ref = ref_weights[name].astype(np.float32)

        assert ours.shape == ref.shape, f"{name}: shape {ours.shape} != {ref.shape}"
        np.testing.assert_allclose(
            ours, ref, atol=0, rtol=0,
            err_msg=f"Tensor {name} values don't match"
        )

    def test_all_tensors_present(self, our_weights, ref_weights):
        """Every tensor in the golden reference should be in our output."""
        for name in ref_weights:
            assert name in our_weights, f"Missing tensor: {name}"

    def test_total_parameter_count(self, our_weights):
        """GPT-2 124M should have ~124M parameters (excluding attn.bias masks)."""
        total = sum(
            t.numel()
            for name, t in our_weights.items()
            if not name.endswith("attn.bias")
        )
        # GPT-2 "124M" actually has ~124.4M params
        assert 120_000_000 < total < 130_000_000, f"Unexpected param count: {total}"
