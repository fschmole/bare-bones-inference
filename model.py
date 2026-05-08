"""
model.py — GPT-2 transformer (124M) implemented from scratch.

The entire model is a single class with a single forward() method.
Attention and FFN are inline helper functions, not separate classes.

No KV cache — the full forward pass is recomputed for every token.
This is slower but dramatically simpler (no mutable state).

GPT-2 Conv1D convention:
  All linear layers use x @ W + b (weight shape [in, out])
  instead of the standard PyTorch W @ x + b (weight shape [out, in]).

Golden reference: transformers.GPT2LMHeadModel
"""

import math
import sys
import time

from compute import Tensor

# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

_trace = False


def set_trace(enabled: bool):
    global _trace
    _trace = enabled


def _log(msg: str):
    if _trace:
        print(f"[model] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """
    Linear projection using GPT-2's Conv1D convention: x @ W + b.
    x: [seq_len, n_embd]  or  [n_heads, seq_len, head_dim]
    W: [in_features, out_features]  (Conv1D layout)
    b: [out_features]
    """
    return x.matmul(weight).add(bias)


def attention(
    x: Tensor,
    c_attn_weight: Tensor,
    c_attn_bias: Tensor,
    c_proj_weight: Tensor,
    c_proj_bias: Tensor,
    n_head: int,
    layer_idx: int,
) -> Tensor:
    """
    Multi-head causal self-attention.

    x: [seq_len, n_embd]

    Steps:
      1. Project to Q, K, V:  x @ W_qkv + b_qkv → [seq_len, 3*n_embd]
      2. Split into Q, K, V:  each [seq_len, n_embd]
      3. Reshape to heads:    [n_head, seq_len, head_dim]
      4. Scores:              Q @ K.T / sqrt(head_dim) → [n_head, seq_len, seq_len]
      5. Causal mask:         add upper-triangular -1e9 mask
      6. Softmax:             normalize scores
      7. Attend:              scores @ V → [n_head, seq_len, head_dim]
      8. Merge heads:         [seq_len, n_embd]
      9. Output projection:   x @ W_proj + b_proj
    """
    seq_len = x.shape[0]
    n_embd = x.shape[1]
    head_dim = n_embd // n_head

    t0 = time.time()

    # 1. Project to Q, K, V concatenated: [seq_len, 3*n_embd]
    qkv = linear(x, c_attn_weight, c_attn_bias)

    # 2. Split into Q, K, V: each [seq_len, n_embd]
    q = qkv.slice(1, 0, n_embd)
    k = qkv.slice(1, n_embd, 2 * n_embd)
    v = qkv.slice(1, 2 * n_embd, 3 * n_embd)

    # 3. Reshape to [n_head, seq_len, head_dim]
    q = q.reshape([seq_len, n_head, head_dim]).transpose(0, 1)
    k = k.reshape([seq_len, n_head, head_dim]).transpose(0, 1)
    v = v.reshape([seq_len, n_head, head_dim]).transpose(0, 1)

    # 4. Attention scores: Q @ K^T / sqrt(head_dim) → [n_head, seq_len, seq_len]
    k_t = k.transpose(1, 2)  # [n_head, head_dim, seq_len]
    scores = q.matmul(k_t).mul_scalar(1.0 / math.sqrt(head_dim))

    # 5. Causal mask: prevent attending to future tokens
    mask = Tensor.tri_mask(seq_len)
    scores = scores.add(mask)

    # 6. Softmax over the last dimension
    attn_weights = scores.softmax(-1)

    # 7. Attend: [n_head, seq_len, seq_len] @ [n_head, seq_len, head_dim]
    #          → [n_head, seq_len, head_dim]
    out = attn_weights.matmul(v)

    # 8. Merge heads: [n_head, seq_len, head_dim] → [seq_len, n_embd]
    out = out.transpose(0, 1).reshape([seq_len, n_embd])

    # 9. Output projection
    out = linear(out, c_proj_weight, c_proj_bias)

    if _trace:
        elapsed = (time.time() - t0) * 1000
        # Get attention stats from last head
        score_data = scores.to_vec()
        score_min = min(v for v in score_data if v > -1e8)  # ignore masked
        score_max = max(v for v in score_data if v > -1e8)
        _log(
            f"  layer {layer_idx} attn: seq_len={seq_len} "
            f"scores=[{score_min:.3f}, {score_max:.3f}] "
            f"({elapsed:.1f}ms)"
        )

    return out


def ffn(
    x: Tensor,
    c_fc_weight: Tensor,
    c_fc_bias: Tensor,
    c_proj_weight: Tensor,
    c_proj_bias: Tensor,
    layer_idx: int,
) -> Tensor:
    """
    Feed-forward network (MLP).

    x: [seq_len, n_embd]

    Steps:
      1. Up-project:   x @ W1 + b1 → [seq_len, 4*n_embd]  (768→3072)
      2. GELU:         activation
      3. Down-project: h @ W2 + b2 → [seq_len, n_embd]     (3072→768)
    """
    t0 = time.time()

    # Up-project + GELU
    h = linear(x, c_fc_weight, c_fc_bias).gelu()

    # Down-project
    out = linear(h, c_proj_weight, c_proj_bias)

    if _trace:
        elapsed = (time.time() - t0) * 1000
        h_data = h.to_vec()
        h_mean = sum(h_data) / len(h_data)
        h_max = max(h_data)
        _log(
            f"  layer {layer_idx} ffn: hidden_mean={h_mean:.4f} "
            f"hidden_max={h_max:.4f} ({elapsed:.1f}ms)"
        )

    return out


# ---------------------------------------------------------------------------
# GPT-2 Model
# ---------------------------------------------------------------------------


class GPT2:
    """
    GPT-2 124M language model.

    Usage:
        config = load_config("models/gpt2/config.json")
        weights = load_weights("models/gpt2/model.safetensors")
        model = GPT2(config, weights)
        logits = model.forward([15496, 11, 995, 0])  # "Hello, world!"
    """

    def __init__(self, config: dict, weights: dict[str, Tensor]):
        self.config = config
        self.weights = weights

        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.n_positions = config["n_positions"]
        self.vocab_size = config["vocab_size"]
        self.eps = config.get("layer_norm_epsilon", 1e-5)

        _log(
            f"GPT2 initialized: {self.n_layer} layers, {self.n_head} heads, "
            f"{self.n_embd} embd, {self.vocab_size} vocab"
        )

    def forward(self, token_ids: list[int]) -> Tensor:
        """
        Run the full GPT-2 forward pass.

        Args:
            token_ids: list of token IDs (e.g., from tokenizer.encode())

        Returns:
            Tensor of shape [seq_len, vocab_size] — logits for each position.
            The logits at position [-1] predict the next token.
        """
        seq_len = len(token_ids)
        if seq_len > self.n_positions:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max {self.n_positions}"
            )

        t_start = time.time()
        w = self.weights

        _log(f"forward: {seq_len} tokens")

        # ---- Embeddings ----
        # Token embeddings: look up each token in the embedding table
        # wte.weight: [vocab_size, n_embd]
        tok_emb = w["wte.weight"].embedding_lookup(token_ids)  # [seq_len, n_embd]

        # Position embeddings: look up positions 0..seq_len
        # wpe.weight: [n_positions, n_embd]
        pos_ids = list(range(seq_len))
        pos_emb = w["wpe.weight"].embedding_lookup(pos_ids)  # [seq_len, n_embd]

        # Combine
        x = tok_emb.add(pos_emb)  # [seq_len, n_embd]

        _log(f"  embeddings: {x.shape}")

        # ---- Transformer blocks ----
        for i in range(self.n_layer):
            t_layer = time.time()
            p = f"h.{i}"

            # Pre-norm + attention + residual
            ln1 = x.layer_norm(w[f"{p}.ln_1.weight"], w[f"{p}.ln_1.bias"], self.eps)
            attn_out = attention(
                ln1,
                w[f"{p}.attn.c_attn.weight"],
                w[f"{p}.attn.c_attn.bias"],
                w[f"{p}.attn.c_proj.weight"],
                w[f"{p}.attn.c_proj.bias"],
                self.n_head,
                i,
            )
            x = x.add(attn_out)

            # Pre-norm + FFN + residual
            ln2 = x.layer_norm(w[f"{p}.ln_2.weight"], w[f"{p}.ln_2.bias"], self.eps)
            ffn_out = ffn(
                ln2,
                w[f"{p}.mlp.c_fc.weight"],
                w[f"{p}.mlp.c_fc.bias"],
                w[f"{p}.mlp.c_proj.weight"],
                w[f"{p}.mlp.c_proj.bias"],
                i,
            )
            x = x.add(ffn_out)

            if _trace:
                elapsed = (time.time() - t_layer) * 1000
                _log(f"  layer {i} total: {elapsed:.1f}ms")

        # ---- Final layer norm ----
        x = x.layer_norm(w["ln_f.weight"], w["ln_f.bias"], self.eps)

        # ---- Language model head (weight tying) ----
        # Logits = x @ wte.T → [seq_len, vocab_size]
        # wte.weight is [vocab_size, n_embd], we need [n_embd, vocab_size]
        wte_t = w["wte.weight"].transpose(0, 1)  # [n_embd, vocab_size]
        logits = x.matmul(wte_t)  # [seq_len, vocab_size]

        elapsed = (time.time() - t_start) * 1000
        _log(f"  forward done: {logits.shape} ({elapsed:.1f}ms)")

        return logits
