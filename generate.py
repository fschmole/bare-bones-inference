"""
generate.py — Greedy text generation for GPT-2.

Autoregressive loop: encode prompt → forward → argmax → append → repeat.
No KV cache — full forward pass recomputed each step.
Tokens are printed as they are generated (streaming effect).

Golden reference: transformers.GPT2LMHeadModel.generate(do_sample=False)
"""

import math
import sys
import time

from compute import Tensor
from model import GPT2
from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

# Verbosity: 0=off, 1=low (summary only), 2=high (per-token detail)
_trace_level = 0
_trace_file = None


def set_trace(level: int):
    """Set trace verbosity: 0=off, 1=low, 2=high."""
    global _trace_level
    _trace_level = level


def set_trace_file(path: str | None):
    """Set a file path to also log trace output to. None to disable."""
    global _trace_file
    if path is None:
        _trace_file = None
    else:
        _trace_file = open(path, "a")


def _log(msg: str, level: int = 1):
    """Write trace message to log file only (not console)."""
    if _trace_level >= level and _trace_file:
        line = f"[generate] {msg}"
        print(line, file=_trace_file, flush=True)


def _log_token(token_text: str):
    """Log a generated token to the trace file."""
    if _trace_file:
        print(f"[generated_token] {token_text!r}", file=_trace_file, flush=True)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    model: GPT2,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 20,
    stream: bool = True,
    eos_token_id: int = 50256,
) -> tuple[str, float]:
    """
    Generate text by greedily picking the most probable next token.

    Args:
        model:        GPT2 model instance.
        tokenizer:    Tokenizer instance.
        prompt:       Input text to continue from.
        max_tokens:   Maximum number of new tokens to generate.
        stream:       If True, print each token as it's generated.
        eos_token_id: Stop when this token is generated (default: <|endoftext|>).

    Returns:
        Tuple of (generated_text, tokens_per_second).
    """
    # Encode the prompt
    token_ids = tokenizer.encode(prompt)
    prompt_len = len(token_ids)

    if prompt_len >= model.n_positions:
        raise ValueError(
            f"Prompt ({prompt_len} tokens) exceeds context window ({model.n_positions})"
        )

    _log(f"prompt: {prompt!r} → {prompt_len} tokens")
    _log(f"generating up to {max_tokens} new tokens")

    t_total_start = time.time()
    generated_ids: list[int] = []

    for step in range(max_tokens):
        # Check context window
        if len(token_ids) >= model.n_positions:
            _log(f"  hit context window limit ({model.n_positions})")
            break

        t_step = time.time()

        # Forward pass (full recompute)
        logits = model.forward(token_ids)

        # Get logits for the last position: [vocab_size]
        last_logits = logits.slice_row(-1)

        # Greedy: pick the highest-probability token
        next_token = last_logits.argmax()

        # High verbosity: show top-5 candidates per step
        if _trace_level >= 2:
            logits_data = last_logits.to_vec()
            # Compute softmax for probabilities
            max_val = max(logits_data)
            exps = [math.exp(x - max_val) for x in logits_data]
            total = sum(exps)
            probs = [e / total for e in exps]

            # Top 5
            indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
            top5_str = ", ".join(
                f"{tokenizer.decode([tid])!r}({prob:.3f})"
                for tid, prob in indexed
            )

            elapsed = time.time() - t_step
            tok_per_sec = 1.0 / elapsed if elapsed > 0 else 0
            _log(
                f"  step {step}: token={next_token} "
                f"{tokenizer.decode([next_token])!r} "
                f"top5=[{top5_str}] "
                f"({elapsed:.1f}s, {tok_per_sec:.2f} tok/s)",
                level=2,
            )

        # Stop on end-of-text token
        if next_token == eos_token_id:
            _log(f"  hit EOS token at step {step}")
            break

        # Append and stream
        token_ids.append(next_token)
        generated_ids.append(next_token)

        token_text = tokenizer.decode([next_token])
        _log_token(token_text)

        if stream:
            print(token_text, end="", flush=True)

    if stream:
        print()  # Final newline

    generated_text = tokenizer.decode(generated_ids)

    elapsed_total = time.time() - t_total_start
    tokens_generated = len(generated_ids)
    tok_per_sec = tokens_generated / elapsed_total if elapsed_total > 0 else 0
    _log(
        f"done: {tokens_generated} tokens in {elapsed_total:.1f}s "
        f"({tok_per_sec:.2f} tok/s)"
    )

    return generated_text, tok_per_sec
