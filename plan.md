# Plan: Local ChatGPT Clone from Scratch

**TL;DR**: Build a local text-generation system in 5 core phases, bottom-up. Rust provides a minimal tensor math library exposed to Python via PyO3. Python implements everything else: tokenizer, transformer, generation, tracing, and CLI. Each component has a golden-reference library for correctness testing. The target model is GPT-2 124M (`openai-community/gpt2` from HuggingFace, safetensors format). This is an educational project — **clarity over performance**.

**Hardware**: Intel Core Ultra 7 258V (Lunar Lake), 32GB RAM, Ubuntu 24 in WSL2.

**Performance expectation**: With AVX2 SIMD acceleration, expect ~1–2 tokens/sec for GPT-2 124M. Without AVX2 (naive scalar ops), ~0.05–0.1 tok/s. This is fine — the goal is understanding, not speed.

## Status

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 1: Rust Tensor Library | ✅ Complete | 49/49 |
| Phase 2: Tokenizer (BPE) | ✅ Complete | 32/32 |
| Phase 3: Model Loader | ✅ Complete | 27/27 |
| Phase 4: GPT-2 Transformer | ✅ Complete | 13/13 |
| Phase 5: Generation + CLI | ✅ Complete | 9/9 |
| Phase 9: SIMD (AVX2) | ✅ Complete | (all 130 pass) |
| Tracing & Logging | ✅ Complete | — |

**Total: 130 tests passing.**

### Completed Enhancements (beyond core MVP)

- **AVX2+FMA SIMD acceleration** (Phase 9): Runtime-dispatched AVX2 kernels for matmul (tiled GEMM), elementwise ops (add, mul, gelu, softmax, layer_norm). ~9× speedup over naive ops. Togglable at runtime via `compute.set_avx2(True/False)` and `--no-avx2` CLI flag.
- **Trace verbosity levels**: `0`=off, `1`=low (op name + time), `2`=high (op + shapes + time). Controlled via `compute.set_trace(level)` and `--trace 0|1|2` CLI flag.
- **Trace file logging**: All trace output (Rust `[compute]` and Python `[generate]`) goes to a log file only (not console), keeping the terminal clean. Each generated token is also logged with `[generated_token]` prefix. Enabled via `--trace-file path` or `compute.set_trace_file(path)`.
- **tok/s display**: When tracing is enabled, tokens/sec is printed to the console after each response.

## Project Structure

```
llm/
├── plan.md                     # This file
├── compute/                    # Rust crate (PyO3 + maturin)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       ├── lib.rs              # PyO3 module entry + all ops exposed
│       └── tensor.rs           # Tensor struct + all math ops (flat file)
├── tokenizer.py                # BPE encode/decode (single file)
├── model.py                    # GPT-2 transformer (single file)
├── loader.py                   # Safetensors parser + config loader
├── generate.py                 # Greedy generation loop
├── trace.py                    # Debug trace (simple print-based)
├── chat.py                     # CLI entry point
├── tests/
│   ├── test_tensor.py          # vs numpy
│   ├── test_tokenizer.py       # vs tiktoken
│   ├── test_loader.py          # vs safetensors lib
│   ├── test_model.py           # vs HuggingFace transformers
│   └── test_generate.py        # vs HuggingFace generate()
├── requirements.txt            # Runtime: (empty — only the built compute crate)
└── requirements-test.txt       # Test-only: numpy, tiktoken, transformers, torch, safetensors
```

**Why flat files?** Each `.py` file maps 1:1 to a concept. No packages, no `__init__.py`, no import gymnastics. A reader can understand the full system by reading 5 Python files + 2 Rust files.

## Golden References

| Component | Our Code | Golden Reference | Test Method |
|---|---|---|---|
| Tensor math | `compute/` (Rust) | **numpy** | `np.allclose(ours, numpy, atol=1e-5)` |
| Tokenizer | `tokenizer.py` | **tiktoken** | Identical token IDs for same input |
| Model loader | `loader.py` | **safetensors** (Python lib) | Byte-identical tensor values |
| GPT-2 forward pass | `model.py` | **HuggingFace transformers** `GPT2LMHeadModel` | `torch.allclose(our_logits, hf_logits, atol=1e-4)` |
| Text generation | `generate.py` | **HuggingFace** `model.generate(do_sample=False)` | Identical greedy output |

## Core Phases (MVP)

### Phase 1: Rust Tensor Library

Build the minimal math engine. No SIMD, no parallelism, no fancy abstractions — just correct f32 tensor operations.

1. Initialize maturin project in `llm/compute/` with `maturin init --bindings pyo3`. Cargo.toml: edition 2021, depends only on `pyo3`. Target `x86_64-unknown-linux-gnu` (PyO3 needs glibc).
2. Implement `Tensor` struct in `tensor.rs`:
   - Storage: `Vec<f32>`, always contiguous (no strides — `transpose` returns a new copied tensor).
   - Shape: `Vec<usize>`.
   - Constructors: `zeros(shape)`, `from_data(data, shape)`.
   - Accessors: `shape()`, `to_vec()` (return flat f32 data to Python).
   - Reshape: `reshape(new_shape)` (just changes shape metadata, panics if sizes don't match).
3. Implement ops as methods/free functions on Tensor — all naive loops:
   - `matmul(a, b)` → triple nested loop.
   - `add(a, b)`, `mul(a, b)`, `mul_scalar(a, f32)` → elementwise.
   - `gelu(a)` → `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` per element.
   - `softmax(a, axis)` → subtract max, exp, divide by sum.
   - `layer_norm(x, gamma, beta, eps)` → `(x - mean) / sqrt(var + eps) * gamma + beta`.
   - `embedding_lookup(table, indices)` → gather rows by index.
   - `transpose_2d(a)` → swap last two dims, return new tensor.
   - `slice_row(a, idx)` → extract a single row (for getting last token logits).
   - `cat(tensors, axis)` → concatenate along axis.
   - `tri_mask(size)` → upper-triangular mask filled with `-1e9`, zeros on/below diagonal.
4. Expose via `#[pyclass]` / `#[pymethods]` in `lib.rs`. Python: `from compute import Tensor`.
5. **Test**: `tests/test_tensor.py` — compare every op against numpy with `np.allclose(..., atol=1e-6)`.

**Tracing**: add a module-level `set_trace(bool)` function. When enabled, each op prints: op name, input shapes, output shape, time taken. Example: `[compute] matmul [1,768] × [768,3072] → [1,3072] (2.1ms)`.

### Phase 2: Tokenizer

BPE encode/decode in a single Python file. No training — use GPT-2's pre-trained vocabulary.

6. Download GPT-2 vocab files: `vocab.json` (token↔id mapping, 50257 entries) and `merges.txt` (BPE merge rules, ~50k lines).
7. Implement `tokenizer.py` — single class `Tokenizer`:
   - `__init__(vocab_path, merges_path)`: load vocab dict and merge list.
   - `encode(text) → list[int]`: pre-tokenize with GPT-2's regex pattern (`'s|'t|'re|...|\s|\S`), convert each word to byte sequence, iteratively apply BPE merges in priority order.
   - `decode(tokens) → str`: look up each token ID → byte string, concatenate, decode UTF-8.
8. **Test**: `tests/test_tokenizer.py` — compare against `tiktoken.get_encoding("gpt2")` for a suite of strings.

**Tracing**: when enabled, print: input text, byte sequences after pre-tokenization, merge steps applied, final token IDs, decoded verification.

### Phase 3: Model Loader

Parse safetensors binary format and GPT-2 config. Single Python file.

9. Implement `loader.py`:
   - `load_config(path) → dict`: parse GPT-2 `config.json`, extract: `n_layer`, `n_head`, `n_embd`, `vocab_size`, `n_positions`.
   - `load_weights(path) → dict[str, Tensor]`: parse safetensors format (8-byte LE header length → JSON header → raw tensor data). Convert each tensor's raw bytes to our Rust `Tensor`. Handle GPT-2's Conv1D weight layout (transpose `[in, out]` → `[out, in]`).
10. **Test**: `tests/test_loader.py` — load same model file with our loader and `safetensors.torch.load_file()`, compare all tensor values.

**Tracing**: when enabled, print: each tensor name, dtype, shape, size in bytes as it's loaded.

### Phase 4: GPT-2 Transformer

The entire GPT-2 model in a single Python file. No KV cache — recompute the full forward pass every token. This is slower but the code is dramatically simpler (no mutable state, no cache management).

11. Implement `model.py` — single class `GPT2`:
    - `__init__(config, weights)`: store config dict and weights dict.
    - `forward(token_ids: list[int]) → Tensor`: full forward pass returning logits for every position.
      - **Embedding**: `wte[token_ids] + wpe[0..seq_len]` (token + position embeddings).
      - **Transformer blocks** (×12): for each block:
        - `x = x + attention(layer_norm(x))` (pre-norm + residual)
        - `x = x + ffn(layer_norm(x))` (pre-norm + residual)
      - **Attention** (as an inline function, not a class):
        - Linear projection: `x @ W_qkv + b_qkv` → split into Q, K, V.
        - Reshape to `[n_heads, seq_len, head_dim]`.
        - Scores: `Q @ K.T / sqrt(head_dim)`.
        - Causal mask: add upper-triangular `-1e9` mask.
        - `softmax(scores) @ V`.
        - Reshape back, output projection.
      - **FFN** (as an inline function):
        - `gelu(x @ W1 + b1) @ W2 + b2` (768 → 3072 → 768).
      - **Final**: `layer_norm(x) @ wte.T` → logits (weight tying).
12. **Test**: `tests/test_model.py` — load GPT-2 weights, run same `input_ids`, assert `torch.allclose(our_logits, hf_logits, atol=1e-4)`.

**Tracing**: when enabled, print per-layer: input shape, attention score stats (min/max/mean), top-5 attention positions, FFN activation stats, output shape, layer time.

### Phase 5: Generation + CLI

Greedy text generation and a minimal chat interface. Two files.

13. Implement `generate.py`:
    - `generate(model, tokenizer, prompt, max_tokens, trace) → str`:
      - Encode prompt → `token_ids`.
      - Loop `max_tokens` times:
        - `logits = model.forward(token_ids)` (full recompute each step — no KV cache).
        - `next_token = argmax(logits[-1])` (greedy — just pick the highest probability token).
        - Append `next_token` to `token_ids`.
        - Print decoded token immediately (streaming effect).
      - Return full decoded text.
14. Implement `chat.py`:
    - Parse args: `--model-dir`, `--max-tokens`, `--trace`.
    - Load model and tokenizer.
    - Loop: `input("You: ")` → `generate()` → print response.
    - Commands: `/quit`, `/trace on|off`.
    - Single-turn only (no conversation history — each prompt is independent).
15. **Test**: `tests/test_generate.py` — compare greedy output with `transformers.GPT2LMHeadModel.generate(do_sample=False)`.

**Tracing**: when enabled, print per-token: token ID, decoded text, top-5 candidates with probabilities, tokens/sec.

## Enhancement Phases (Deferred)

These are ordered by educational value, not difficulty. Each is independent — implement in any order after the MVP works.

### Phase 6: Sampling (temperature, top-k, top-p)

Add stochastic generation. Currently greedy (argmax) always picks the same token — output is deterministic but repetitive.

- Add `temperature` scaling: `logits / temperature` before softmax.
- Add `top_k`: zero out all but the top-k logits.
- Add `top_p` (nucleus): sort by probability, keep smallest set summing to ≥ p.
- Sample from the filtered distribution instead of argmax.
- **Golden ref**: compare distribution shape against HuggingFace `LogitsProcessor`.

### Phase 7: KV Cache

Avoid recomputing attention for all previous tokens on every step. Currently the model runs the full forward pass for the entire sequence every time a new token is generated.

- Cache K and V tensors from each layer.
- On subsequent tokens, only compute Q/K/V for the new token, concatenate with cached K/V.
- **Speedup**: O(n) per step instead of O(n²). For 100-token generation, ~50× faster.
- **Golden ref**: output must remain identical before/after adding cache.

### Phase 8: Multi-Turn Chat

Add conversation history so the model sees prior turns.

- Maintain `token_ids` across turns (concatenate user + assistant tokens).
- Add a system prompt prefix.
- Track context window usage (GPT-2 limit: 1024 tokens).
- Add `/clear` command to reset history.

### Phase 9: SIMD Acceleration (AVX2) ✅

Pure performance optimization. Speeds up matmul ~8–20× on this hardware.

- ✅ `#[target_feature(enable = "avx2,fma")]` GEMM using `_mm256_fmadd_ps`.
- ✅ K-tiled (128) loop with 8-wide column processing for cache efficiency.
- ✅ AVX2 paths for elementwise ops (GELU, add, mul, softmax, layer_norm).
- ✅ Runtime toggle: `compute.set_avx2(True/False)`, `--no-avx2` CLI flag.
- ✅ Padé tanh approximation and 6th-order Taylor exp for GELU/softmax.
- ✅ All 130 existing tests still pass — drop-in replacement, no API change.
- Measured improvement: ~9× faster (full test suite: 12min → 80s).

### Phase 10: Parallelism (rayon)

Multi-threaded matmul and elementwise ops via rayon.

- Parallelize outer loop of matmul across rows.
- Parallelize elementwise ops across chunks.
- Add `rayon` dependency to Cargo.toml.
- Complements SIMD — orthogonal optimization.

### Phase 11: GPU Backend (Intel Arc via Level Zero)

Offload large matmuls to GPU.

- Intel Arc GPU support via Level Zero API (`ze_*` FFI bindings).
- Implement GPU GEMM kernel in SPIR-V.
- Add `--device cpu|gpu|auto` flag.
- Requires `level-zero` system library in WSL.

### Phase 12: NPU Backend (Intel AI Boost)

Offload inference to the NPU.

- Requires `intel_vpu` kernel driver (may not work in WSL2).
- Likely needs bare-metal Linux.

### Phase 13: Llama Architecture

Support modern instruction-tuned models that can actually chat.

- RMSNorm (instead of LayerNorm), RoPE (instead of absolute positional embeddings), SwiGLU (instead of GELU), Grouped-Query Attention (instead of MHA).
- Add `rms_norm` and `silu` ops to compute crate.
- Target: `SmolLM-135M-Instruct` or `TinyLlama-1.1B-Chat`.
- Implement chat template formatting (system/user/assistant turns).

### Phase 14: BPE Training

Implement the BPE merge-learning algorithm to train a tokenizer from a corpus.

- Count byte-pair frequencies, merge most frequent, repeat.
- Validate by training on a small corpus and comparing with `tokenizers` library.

## Verification

- **Per-phase unit tests**: each phase includes tests comparing against the golden reference. Run with `pytest tests/`.
- **End-to-end smoke test** (after Phase 5): `python chat.py --model-dir ./models/gpt2 --prompt "The capital of France is" --max-tokens 20` and verify output matches HuggingFace GPT-2 greedy output exactly.
- **Trace walkthrough**: enable `--trace`, run a short prompt, manually verify shapes match expected GPT-2 dimensions (e.g., attention shape `[12, seq_len, seq_len]`, FFN hidden dim 3072).

## Key Decisions

- **Naive ops first, optimize later**: single-threaded scalar loops for all math. SIMD/parallelism/GPU are deferred enhancements. This keeps the Rust code ~200 lines instead of ~1500.
- **Always-contiguous tensors**: no stride tracking. `transpose` copies data. Simpler at the cost of memory.
- **No KV cache in MVP**: recompute full forward pass per token. Slower but eliminates mutable state and cache management complexity.
- **Greedy-only in MVP**: `argmax` is one line. Sampling is a separate enhancement.
- **Flat file layout**: one file per concept (tokenizer.py, model.py, etc). No Python packages.
- **F32 only**: no F16/I8 quantization. One dtype keeps the Rust code simple.
- **Safetensors over PyTorch .bin**: simpler format (no pickle), direct byte parsing.
- **GPT-2 Conv1D convention**: GPT-2 stores linear weights as `[in, out]` instead of `[out, in]`. The loader transposes them.
- **glibc target for Rust**: PyO3 requires glibc. Build target is `x86_64-unknown-linux-gnu`.
- **No external ML runtime deps**: the only runtime dependency is the self-built compute crate. numpy/torch/tiktoken/transformers are test-only.

## What Was Deferred and Why

| Deferred Item | Why | Educational Cost |
|---|---|---|
| ~~SIMD (AVX2)~~ | ✅ Implemented | — |
| Parallelism (rayon) | Pure performance. | None |
| KV cache | Eliminates complex mutable state. Full recompute is clearer. | Low — can explain concept without implementing |
| Sampling (temp/top-k/top-p) | Greedy works. Sampling is an independent concept. | Low |
| Multi-turn chat | Single-turn is simpler. History is just token concatenation. | Low |
| GPU/NPU backends | Requires driver setup, FFI bindings, kernel code. | None for LLM understanding |
| Llama architecture | GPT-2 teaches the same transformer fundamentals. | Low — RoPE/GQA are refinements |
| BPE training | Encode/decode teaches BPE. Training is a separate algorithm. | Low |
| F16/I8/quantization | F32-only keeps one code path. | None for basics |
| Backend trait | No abstraction needed with one backend. | None |
| Strides/non-contiguous | Always copying is simpler. | None |
