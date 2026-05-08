"""
tokenizer.py — BPE tokenizer for GPT-2 (encode/decode only, no training).

Implements the same algorithm as OpenAI's GPT-2 tokenizer:
  1. Pre-tokenize text into words using a regex pattern.
  2. Convert each word to a sequence of byte-level tokens.
  3. Iteratively merge the highest-priority adjacent byte pairs.

Golden reference: tiktoken.get_encoding("gpt2")
"""

import json
import re
import sys

# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

_trace = False


def set_trace(enabled: bool):
    global _trace
    _trace = enabled


def _log(msg: str):
    if _trace:
        print(f"[tokenizer] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# GPT-2 pre-tokenization regex
# ---------------------------------------------------------------------------

# This regex splits text into "words" before BPE is applied.
# It matches contractions, letters, numbers, non-whitespace, or whitespace.
# The key insight: leading spaces are attached to the next word ("Ġ" convention).
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Byte <-> Unicode mapping
# ---------------------------------------------------------------------------

# GPT-2 represents raw bytes as Unicode characters to keep everything as
# clean text. Printable ASCII bytes map to themselves; other bytes map to
# Unicode chars starting at U+0100. This is the "bytes_to_unicode" table
# from the original OpenAI implementation.


def _bytes_to_unicode() -> dict[int, str]:
    """Build the byte-value → unicode-character mapping used by GPT-2."""
    # Printable byte ranges that map to themselves:
    #   ! to ~ (33-126), ¡ to ¬ (161-172), ® to ÿ (174-255)
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_UNICODE = _bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}


# ---------------------------------------------------------------------------
# BPE core algorithm
# ---------------------------------------------------------------------------


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return all adjacent pairs in a word (tuple of tokens)."""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def _bpe(word: tuple[str, ...], merge_ranks: dict[tuple[str, str], int]) -> tuple[str, ...]:
    """
    Apply BPE merges to a word (tuple of single-char tokens) until no more
    merges are possible. Returns the final tuple of merged tokens.

    This is the core BPE algorithm:
      1. Find all adjacent pairs in the current word.
      2. Pick the pair with the lowest rank (highest priority merge).
      3. Merge all occurrences of that pair in the word.
      4. Repeat until no pairs have a merge rule.
    """
    if len(word) < 2:
        return word

    while True:
        pairs = _get_pairs(word)
        if not pairs:
            break

        # Find the pair with the lowest merge rank
        best_pair = min(
            pairs,
            key=lambda p: merge_ranks.get(p, float("inf")),
        )
        if best_pair not in merge_ranks:
            break  # No more merges possible

        first, second = best_pair
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)

        if len(word) == 1:
            break

    return word


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------


class Tokenizer:
    """
    GPT-2 BPE tokenizer.

    Usage:
        tok = Tokenizer("models/gpt2/vocab.json", "models/gpt2/merges.txt")
        ids = tok.encode("Hello, world!")
        text = tok.decode(ids)
    """

    def __init__(self, vocab_path: str, merges_path: str):
        # Load vocab: token_string -> token_id
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id: dict[str, int] = json.load(f)
        self.id_to_token: dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        # Load merges: list of (token_a, token_b) pairs, ordered by priority.
        # The first merge in the file has rank 0 (highest priority).
        # The first line is a version header (#version: 0.2) — skip it.
        # All subsequent lines are merges (even if they start with '#', which
        # is a valid BPE token for the hash character).
        self.merge_ranks: dict[tuple[str, str], int] = {}
        with open(merges_path, "r", encoding="utf-8") as f:
            rank = 0
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if i == 0 and line.startswith("#"):
                    continue  # skip version header
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    self.merge_ranks[(parts[0], parts[1])] = rank
                    rank += 1

        # Cache: word_string -> bpe_tokens (avoids re-running BPE on repeat words)
        self._cache: dict[str, tuple[str, ...]] = {}

        _log(f"loaded vocab={len(self.token_to_id)} merges={len(self.merge_ranks)}")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        _log(f"encode input: {text!r}")

        token_ids: list[int] = []

        # Step 1: Pre-tokenize into "words" using the GPT-2 regex.
        words = GPT2_PATTERN.findall(text)
        _log(f"  pre-tokenized into {len(words)} words: {words[:10]}{'...' if len(words) > 10 else ''}")

        for word in words:
            # Step 2: Convert each byte of the word to its unicode representation.
            byte_tokens = tuple(BYTE_TO_UNICODE[b] for b in word.encode("utf-8"))

            # Step 3: Apply BPE merges (with caching).
            if byte_tokens in self._cache:
                bpe_tokens = self._cache[byte_tokens]
            else:
                bpe_tokens = _bpe(byte_tokens, self.merge_ranks)
                self._cache[byte_tokens] = bpe_tokens

            # Step 4: Look up token IDs.
            for token in bpe_tokens:
                token_ids.append(self.token_to_id[token])

        _log(f"  encoded {len(token_ids)} tokens: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        # Step 1: Look up each token ID -> unicode token string.
        token_strings = [self.id_to_token[tid] for tid in token_ids]

        # Step 2: Concatenate and convert unicode chars back to bytes.
        text_bytes = bytes(UNICODE_TO_BYTE[c] for token in token_strings for c in token)

        # Step 3: Decode UTF-8.
        text = text_bytes.decode("utf-8", errors="replace")

        _log(f"decoded {len(token_ids)} tokens -> {text!r}")
        return text
