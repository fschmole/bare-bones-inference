"""
loader.py — Load GPT-2 model weights (safetensors) and config (JSON).

Safetensors binary format:
  - 8 bytes: little-endian u64 header length
  - N bytes: JSON header (tensor name → {dtype, shape, data_offsets})
  - remainder: raw tensor data (contiguous, referenced by offsets)

Golden reference: safetensors.torch.load_file()
"""

import json
import struct
import sys

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
        print(f"[loader] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Dtype sizes (bytes per element)
# ---------------------------------------------------------------------------

DTYPE_SIZES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """
    Load GPT-2 config.json and return a dict with the keys we need.

    Returns dict with keys:
        n_layer, n_head, n_embd, vocab_size, n_positions,
        layer_norm_epsilon, bos_token_id, eos_token_id
    """
    with open(path, "r") as f:
        raw = json.load(f)

    config = {
        "n_layer": raw["n_layer"],
        "n_head": raw["n_head"],
        "n_embd": raw["n_embd"],
        "vocab_size": raw["vocab_size"],
        "n_positions": raw["n_positions"],
        "layer_norm_epsilon": raw.get("layer_norm_epsilon", 1e-5),
        "bos_token_id": raw.get("bos_token_id", 50256),
        "eos_token_id": raw.get("eos_token_id", 50256),
    }

    _log(
        f"config: n_layer={config['n_layer']} n_head={config['n_head']} "
        f"n_embd={config['n_embd']} vocab_size={config['vocab_size']} "
        f"n_positions={config['n_positions']}"
    )
    return config


# ---------------------------------------------------------------------------
# Safetensors loader
# ---------------------------------------------------------------------------


def load_weights(path: str) -> dict[str, Tensor]:
    """
    Load all tensors from a safetensors file.

    Returns a dict mapping tensor name → our Rust Tensor.
    Only F32 is supported (GPT-2 is all F32).

    Note on GPT-2 Conv1D weights:
    GPT-2 stores linear layer weights in Conv1D format [in_features, out_features]
    instead of the standard PyTorch [out_features, in_features]. We do NOT
    transpose here — the model code will handle the Convention1D layout by doing
    x @ W instead of W @ x. This keeps the loaded weights identical to the
    original file for easier testing.
    """
    _log(f"loading weights from {path}")

    with open(path, "rb") as f:
        # Read header length (8 bytes, little-endian u64)
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("File too small — not a valid safetensors file")
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        _log(f"  header length: {header_len} bytes")

        # Read and parse JSON header
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)

        # Data starts right after the header
        data_start = 8 + header_len

        # Read all tensor data at once
        f.seek(data_start)
        data = f.read()

    _log(f"  raw data: {len(data)} bytes")

    tensors: dict[str, Tensor] = {}
    skipped = 0

    for name, meta in sorted(header.items()):
        if name == "__metadata__":
            continue

        dtype = meta["dtype"]
        shape = meta["shape"]
        offset_start, offset_end = meta["data_offsets"]

        if dtype != "F32":
            _log(f"  SKIP {name} (dtype={dtype}, only F32 supported)")
            skipped += 1
            continue

        # Extract raw bytes and convert to f32 list
        raw = data[offset_start:offset_end]
        num_elements = (offset_end - offset_start) // 4
        floats = list(struct.unpack(f"<{num_elements}f", raw))

        tensor = Tensor(floats, shape)
        tensors[name] = tensor

        _log(
            f"  {name:45s} {dtype} {str(shape):20s} "
            f"({offset_end - offset_start:>10,} bytes)"
        )

    _log(f"loaded {len(tensors)} tensors, skipped {skipped}")
    return tensors
