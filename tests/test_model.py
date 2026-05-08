"""Tests for model.py — compared against HuggingFace transformers as golden reference."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loader import load_config, load_weights
from model import GPT2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "model.safetensors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np(t) -> np.ndarray:
    """Convert our Tensor to numpy."""
    return np.array(t.to_vec(), dtype=np.float32).reshape(t.shape)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def our_model():
    """Load our GPT-2 model (once for all tests)."""
    config = load_config(CONFIG_PATH)
    weights = load_weights(WEIGHTS_PATH)
    return GPT2(config, weights)


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace GPT-2 model as golden reference."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel.from_pretrained(
        MODELS_DIR, local_files_only=True
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelInit:
    def test_loads_successfully(self, our_model):
        assert our_model.n_layer == 12
        assert our_model.n_head == 12
        assert our_model.n_embd == 768

    def test_output_shape(self, our_model):
        """Forward pass output should be [seq_len, vocab_size]."""
        logits = our_model.forward([15496])  # "Hello"
        assert logits.shape == [1, 50257]


class TestVsHuggingFace:
    """Compare forward pass output against HuggingFace transformers."""

    PROMPTS = {
        "single_token": [15496],                          # "Hello"
        "hello_world": [15496, 11, 995, 0],              # "Hello, world!"
        "the": [464],                                     # "The"
        "short_sentence": [464, 3139, 286, 4881, 318],   # "The capital of France is"
    }

    @pytest.mark.parametrize("name,token_ids", list(PROMPTS.items()))
    def test_logits_match(self, our_model, hf_model, name, token_ids):
        """Our logits should match HuggingFace within tolerance."""
        import torch

        # Our forward pass
        our_logits = to_np(our_model.forward(token_ids))  # [seq_len, vocab_size]

        # HuggingFace forward pass
        with torch.no_grad():
            hf_input = torch.tensor([token_ids])
            hf_output = hf_model(hf_input)
            hf_logits = hf_output.logits[0].numpy()  # [seq_len, vocab_size]

        assert our_logits.shape == hf_logits.shape, (
            f"Shape mismatch: {our_logits.shape} vs {hf_logits.shape}"
        )

        np.testing.assert_allclose(
            our_logits, hf_logits, atol=1e-3, rtol=1e-3,
            err_msg=f"Logits mismatch for prompt '{name}'"
        )

    @pytest.mark.parametrize("name,token_ids", list(PROMPTS.items()))
    def test_greedy_next_token_matches(self, our_model, hf_model, name, token_ids):
        """The greedy (argmax) next-token prediction should be identical."""
        import torch

        # Our prediction
        our_logits = to_np(our_model.forward(token_ids))
        our_next = int(np.argmax(our_logits[-1]))

        # HuggingFace prediction
        with torch.no_grad():
            hf_input = torch.tensor([token_ids])
            hf_output = hf_model(hf_input)
            hf_next = int(torch.argmax(hf_output.logits[0, -1]).item())

        assert our_next == hf_next, (
            f"Greedy next token mismatch for '{name}': "
            f"ours={our_next} vs hf={hf_next}"
        )

    def test_top5_match(self, our_model, hf_model):
        """Top-5 predictions for 'The capital of France is' should match."""
        import torch

        token_ids = [464, 3139, 286, 4881, 318]  # "The capital of France is"

        our_logits = to_np(our_model.forward(token_ids))
        our_top5 = list(np.argsort(our_logits[-1])[-5:][::-1])

        with torch.no_grad():
            hf_input = torch.tensor([token_ids])
            hf_output = hf_model(hf_input)
            hf_top5 = torch.topk(hf_output.logits[0, -1], 5).indices.tolist()

        assert our_top5 == hf_top5, (
            f"Top-5 mismatch: ours={our_top5} vs hf={hf_top5}"
        )


class TestEdgeCases:
    def test_single_token(self, our_model):
        """Should work with a single token."""
        logits = our_model.forward([50256])  # <|endoftext|>
        assert logits.shape == [1, 50257]

    def test_max_length(self, our_model):
        """Should accept up to 1024 tokens (but this test uses a shorter seq
        to keep test runtime reasonable)."""
        logits = our_model.forward(list(range(50)))
        assert logits.shape == [50, 50257]
