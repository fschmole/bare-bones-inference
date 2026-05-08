"""Tests for tokenizer.py — compared against tiktoken as golden reference."""

import os
import sys
import pytest

# Add project root to path so we can import tokenizer.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2")


@pytest.fixture(scope="module")
def tok():
    """Our from-scratch tokenizer."""
    return Tokenizer(
        os.path.join(MODELS_DIR, "vocab.json"),
        os.path.join(MODELS_DIR, "merges.txt"),
    )


@pytest.fixture(scope="module")
def tiktoken_enc():
    """tiktoken GPT-2 encoding (golden reference)."""
    tiktoken = pytest.importorskip("tiktoken")
    return tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------------------------
# Encode tests — compare against tiktoken
# ---------------------------------------------------------------------------

class TestEncode:
    """Every test encodes the same string with both tokenizers and compares IDs."""

    CASES = [
        "Hello, world!",
        "The capital of France is Paris.",
        "GPT-2 is a transformer-based language model.",
        " leading space",
        "multiple   spaces   here",
        "Hello\nworld",            # newline
        "Tab\there",               # tab
        "UPPERCASE lowercase MiXeD",
        "Numbers: 42, 3.14, -1",
        "Special chars: @#$%^&*()",
        "Unicode: café résumé naïve",
        "Contractions: I'm can't won't they're we've he'd she'll",
        "a",                       # single char
        "I",                       # single letter word
        " ",                       # single space
        "  ",                      # double space
        "The quick brown fox jumps over the lazy dog.",
        "1234567890",
        "def hello():\n    print('hi')\n",  # code-like
        "http://example.com/path?q=hello&lang=en",
        # Longer text
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet "
        "hole, filled with the ends of worms and an oozy smell, nor yet a dry, "
        "bare, sandy hole with nothing in it to sit down on or to eat: it was a "
        "hobbit-hole, and that means comfort.",
    ]

    @pytest.mark.parametrize("text", CASES, ids=lambda t: t[:40])
    def test_encode_matches_tiktoken(self, tok, tiktoken_enc, text):
        ours = tok.encode(text)
        expected = tiktoken_enc.encode(text)
        assert ours == expected, (
            f"Mismatch for {text!r}:\n"
            f"  ours:     {ours}\n"
            f"  tiktoken: {expected}"
        )


# ---------------------------------------------------------------------------
# Decode tests
# ---------------------------------------------------------------------------

class TestDecode:
    """Verify that decode(encode(text)) == text (round-trip)."""

    CASES = [
        "Hello, world!",
        "The capital of France is Paris.",
        " leading space",
        "café résumé",
        "Contractions: I'm can't won't",
        "def f(x):\n    return x + 1\n",
        "1234567890",
    ]

    @pytest.mark.parametrize("text", CASES, ids=lambda t: t[:40])
    def test_roundtrip(self, tok, text):
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_decode_matches_tiktoken(self, tok, tiktoken_enc):
        """Encode with tiktoken, decode with ours — should match."""
        text = "The quick brown fox jumps over the lazy dog."
        ids = tiktoken_enc.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self, tok):
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_vocab_size(self, tok):
        assert tok.vocab_size == 50257

    def test_single_token_decode(self, tok, tiktoken_enc):
        """Verify single-token encode/decode for common words."""
        for word in ["the", "The", " the", " The", "is", " is"]:
            ours = tok.encode(word)
            expected = tiktoken_enc.encode(word)
            assert ours == expected
            assert tok.decode(ours) == word
