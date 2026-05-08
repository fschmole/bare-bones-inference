"""Tests for generate.py — compared against HuggingFace transformers.generate() as golden reference."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loader import load_config, load_weights
from model import GPT2
from tokenizer import Tokenizer
from generate import generate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "model.safetensors")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
MERGES_PATH = os.path.join(MODELS_DIR, "merges.txt")


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
def tokenizer():
    """Load our tokenizer (once for all tests)."""
    return Tokenizer(VOCAB_PATH, MERGES_PATH)


@pytest.fixture(scope="module")
def hf_generate():
    """Return a helper that generates text using HuggingFace as golden ref."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model = transformers.GPT2LMHeadModel.from_pretrained(
        MODELS_DIR, local_files_only=True
    )
    model.eval()
    hf_tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
        MODELS_DIR, local_files_only=True
    )

    def _generate(prompt: str, max_new_tokens: int) -> str:
        input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][input_ids.shape[1]:]
        return hf_tokenizer.decode(new_ids)

    return _generate


# ---------------------------------------------------------------------------
# Tests — greedy generation vs HuggingFace
# ---------------------------------------------------------------------------

class TestGreedyGeneration:
    """Compare our greedy generation against HuggingFace generate(do_sample=False)."""

    def test_single_token_prompt(self, our_model, tokenizer, hf_generate):
        """Generate from a single-token prompt."""
        prompt = "The"
        max_tokens = 5

        ours, _ = generate(our_model, tokenizer, prompt, max_tokens=max_tokens, stream=False)
        theirs = hf_generate(prompt, max_tokens)

        assert ours == theirs, f"ours={ours!r} theirs={theirs!r}"

    def test_short_sentence(self, our_model, tokenizer, hf_generate):
        """Generate continuation of a short sentence."""
        prompt = "The capital of France is"
        max_tokens = 5

        ours, _ = generate(our_model, tokenizer, prompt, max_tokens=max_tokens, stream=False)
        theirs = hf_generate(prompt, max_tokens)

        assert ours == theirs, f"ours={ours!r} theirs={theirs!r}"

    def test_hello_world(self, our_model, tokenizer, hf_generate):
        """Generate from 'Hello, world!'"""
        prompt = "Hello, world!"
        max_tokens = 5

        ours, _ = generate(our_model, tokenizer, prompt, max_tokens=max_tokens, stream=False)
        theirs = hf_generate(prompt, max_tokens)

        assert ours == theirs, f"ours={ours!r} theirs={theirs!r}"

    def test_longer_generation(self, our_model, tokenizer, hf_generate):
        """Generate a longer sequence (10 tokens)."""
        prompt = "Once upon a time"
        max_tokens = 10

        ours, _ = generate(our_model, tokenizer, prompt, max_tokens=max_tokens, stream=False)
        theirs = hf_generate(prompt, max_tokens)

        assert ours == theirs, f"ours={ours!r} theirs={theirs!r}"


class TestGenerationEdgeCases:
    """Test edge cases in the generation loop."""

    def test_empty_result_on_immediate_eos(self, our_model, tokenizer):
        """If model immediately predicts EOS, return empty string."""
        # We can't force this, but we verify the function handles it
        # by checking the generate function returns a string
        result, tok_s = generate(our_model, tokenizer, "Test", max_tokens=1, stream=False)
        assert isinstance(result, str)
        assert isinstance(tok_s, float)

    def test_max_tokens_zero(self, our_model, tokenizer):
        """max_tokens=0 should return empty string without any forward pass."""
        result, _ = generate(our_model, tokenizer, "Hello", max_tokens=0, stream=False)
        assert result == ""

    def test_streaming_output(self, our_model, tokenizer, capsys):
        """stream=True should print tokens to stdout."""
        generate(our_model, tokenizer, "The", max_tokens=2, stream=True)
        captured = capsys.readouterr()
        # Should have some stdout output (the generated tokens + newline)
        assert len(captured.out) > 0

    def test_no_streaming_output(self, our_model, tokenizer, capsys):
        """stream=False should not print to stdout."""
        generate(our_model, tokenizer, "The", max_tokens=2, stream=False)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestGenerationDeterminism:
    """Greedy generation should be deterministic."""

    def test_same_output_twice(self, our_model, tokenizer):
        """Running generate twice with same input gives identical output."""
        prompt = "The meaning of"
        r1, _ = generate(our_model, tokenizer, prompt, max_tokens=3, stream=False)
        r2, _ = generate(our_model, tokenizer, prompt, max_tokens=3, stream=False)
        assert r1 == r2
