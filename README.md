# Local LLM Inference Engine

A from-scratch implementation of GPT-2 inference, built for learning. Rust handles tensor math (via PyO3), Python handles everything else: tokenizer, transformer, generation, and CLI.

## Important Limitations

**This is an educational project, not a production system.** It is designed to teach how transformer-based language models work at every layer of the stack — from byte-pair encoding to matrix multiplication to autoregressive decoding.

Key limitations:

- **Extremely slow.** Even with AVX2 SIMD acceleration, generation runs at ~1–2 tokens/sec for GPT-2-124M. Without AVX2, it's ~0.05–0.1 tok/s. There is no KV cache — the full forward pass is recomputed for the entire sequence on every token.
- **No randomness.** Generation is greedy (argmax only) — it always picks the single most probable next token. This means output is deterministic but highly repetitive. There is no temperature, top-k, or top-p sampling.
- **No stop condition.** GPT-2 is a base model (not instruction-tuned), so it almost never predicts the end-of-text token during greedy decoding. Generation always runs until `--max-tokens` is reached.
- **Single-turn only.** Each prompt is independent — there is no conversation history or chat template.
- **F32 only.** No quantization (F16/I8). The full GPT-2-124M model uses ~500MB of memory.
- **CPU only.** No GPU or NPU offloading.

The objective is **clarity and correctness** — every component is tested against a golden-reference library (numpy, tiktoken, HuggingFace transformers) to prove it produces identical results.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  chat.py  — CLI interface (single-turn)         │
├─────────────────────────────────────────────────┤
│  generate.py — autoregressive loop (greedy)     │
├─────────────────────────────────────────────────┤
│  model.py — GPT-2 transformer (12 layers)       │
├──────────────────┬──────────────────────────────┤
│  tokenizer.py    │  loader.py                   │
│  BPE encode/     │  safetensors parser +        │
│  decode          │  config loader               │
├──────────────────┴──────────────────────────────┤
│  compute (Rust/PyO3) — tensor math              │
│  AVX2+FMA SIMD with naive scalar fallback       │
└─────────────────────────────────────────────────┘
```

| File | Purpose | Lines |
|------|---------|-------|
| `compute/src/tensor.rs` | Tensor struct, all math ops, SIMD kernels | ~1000 |
| `compute/src/lib.rs` | PyO3 module, configuration API | ~55 |
| `tokenizer.py` | GPT-2 BPE encode/decode | ~170 |
| `loader.py` | Safetensors binary parser + config | ~100 |
| `model.py` | Full GPT-2 forward pass | ~230 |
| `generate.py` | Greedy decoding loop | ~165 |
| `chat.py` | Interactive CLI | ~165 |

## Inference Flow

The following sequence diagram shows the complete path from user prompt to generated response. Derived from actual `trace.log` output.

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam sequenceMessageAlign center
skinparam responseMessageBelowArrow true

actor User
participant "chat.py" as Chat
participant "generate.py" as Gen
participant "tokenizer.py" as Tok
participant "model.py" as Model
participant "compute\n(Rust/PyO3)" as Compute

User -> Chat : input prompt
Chat -> Gen : generate(model, tokenizer, prompt, max_tokens)

== Encode Prompt ==

Gen -> Tok : encode(prompt)
note right: BPE: regex split → byte encoding\n→ iterative merge by priority
Tok --> Gen : token_ids (e.g. 5 tokens)

loop **max_tokens times** (e.g. 40)\n//Repeats until max_tokens reached or EOS;//\n//each iteration generates one token//

    == Forward Pass (full recompute) ==

    Gen -> Model : forward(token_ids)

    == Embeddings ==
    Model -> Compute : embedding_lookup(wte, token_ids)
    Compute --> Model : tok_emb [seq_len, 768]
    Model -> Compute : embedding_lookup(wpe, [0..seq_len])
    Compute --> Model : pos_emb [seq_len, 768]
    Model -> Compute : add(tok_emb, pos_emb)
    Compute --> Model : x [seq_len, 768]

    loop **12 transformer layers**\n//GPT-2 has 12 identical blocks;\n//each contains attention + FFN//

        == Pre-Norm + Attention + Residual ==
        Model -> Compute : layer_norm(x)
        Compute --> Model : ln1
        Model -> Compute : matmul(ln1, W_qkv) + bias
        note right: QKV projection\n[seq, 768] × [768, 2304]
        Compute --> Model : qkv [seq_len, 2304]
        Model -> Compute : slice → Q, K, V
        Model -> Compute : transpose → [12 heads, seq, 64]
        Model -> Compute : matmul(Q, K.T)
        note right: Attention scores\n[12, seq, seq]
        Model -> Compute : mul_scalar(1/√64)
        Model -> Compute : add(causal mask)
        Model -> Compute : softmax(-1)
        Model -> Compute : matmul(scores, V)
        note right: Weighted values\n[12, seq, 64]
        Model -> Compute : transpose + reshape → [seq, 768]
        Model -> Compute : matmul(out, W_proj) + bias
        note right: Output projection\n[seq, 768] × [768, 768]
        Model -> Compute : add(x, attn_out)
        note right: Residual connection
        Compute --> Model : x

        == Pre-Norm + FFN + Residual ==
        Model -> Compute : layer_norm(x)
        Compute --> Model : ln2
        Model -> Compute : matmul(ln2, W_fc) + bias
        note right: Up-project\n[seq, 768] × [768, 3072]
        Model -> Compute : gelu()
        Model -> Compute : matmul(h, W_proj) + bias
        note right: Down-project\n[seq, 3072] × [3072, 768]
        Model -> Compute : add(x, ffn_out)
        note right: Residual connection
        Compute --> Model : x

    end

    == LM Head ==
    Model -> Compute : layer_norm(x)
    note right: Final layer norm
    Model -> Compute : transpose(wte)
    note right: Weight tying:\nreuse token embeddings\n[50257, 768] → [768, 50257]
    Model -> Compute : matmul(x, wte.T)
    note right: Logits\n[seq, 768] × [768, 50257]
    Compute --> Model : logits [seq_len, 50257]
    Model --> Gen : logits

    == Token Selection ==
    Gen -> Compute : slice_row(-1)
    note right: Last position only\n→ [50257]
    Gen -> Compute : argmax()
    note right: Greedy: pick highest\nprobability token
    Compute --> Gen : next_token_id
    Gen -> Tok : decode([next_token_id])
    Tok --> Gen : token_text
    Gen -> User : print token (streaming)
    note right: Token appears on screen\nimmediately as generated

end

Gen --> Chat : (generated_text, tok_per_sec)
Chat -> User : print tok/s (if tracing on)

@enduml
```

## Setup

```bash
# Prerequisites: Python 3.12+, Rust 1.70+, maturin
cd llm
python -m venv .venv
source .venv/bin/activate

# Build the Rust tensor library
cd compute
maturin develop --release
cd ..

# Download GPT-2 124M model files (~550MB)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai-community/gpt2', local_dir='models/gpt2')
"
```

The download places `config.json`, `vocab.json`, `merges.txt`, `tokenizer.json`, and `model.safetensors` into `models/gpt2/`. The safetensors file (523MB) is excluded from git — each clone needs to download it.

## Usage

```bash
# Basic chat
python chat.py

# With tracing (log to file, show tok/s on console)
python chat.py --trace 1 --trace-file trace.log

# High verbosity trace (per-op shapes and timing)
python chat.py --trace 2 --trace-file trace.log

# Force naive scalar ops (disable SIMD, for benchmarking/debugging)
python chat.py --no-avx2

# All options
python chat.py --model-dir models/gpt2 --max-tokens 40 --trace 2 --trace-file trace.log --no-avx2
```

### Interactive Commands

| Command | Effect |
|---------|--------|
| `/quit` | Exit |
| `/trace 0\|1\|2` | Set trace verbosity (0=off, 1=low, 2=high) |
| `/avx2 on\|off` | Toggle SIMD acceleration at runtime |

### Trace Levels

- **Level 0**: No trace output.
- **Level 1 (low)**: Rust ops log `[compute] op_name (Xms)` to trace file. Python logs generation summary. Console shows tok/s after each response.
- **Level 2 (high)**: Rust ops additionally log input/output shapes. Python logs per-token top-5 candidates with probabilities.

Trace output goes to the log file only (not the console). Generated tokens are logged with `[generated_token]` prefix.

## Testing

```bash
pip install -r requirements-test.txt
python -m pytest tests/ -v
```

All 130 tests compare our implementation against golden-reference libraries:

| Test Suite | Golden Reference | Tests |
|-----------|-----------------|-------|
| `test_tensor.py` | numpy | 49 |
| `test_tokenizer.py` | tiktoken | 32 |
| `test_loader.py` | safetensors (Python lib) | 27 |
| `test_model.py` | HuggingFace transformers | 13 |
| `test_generate.py` | HuggingFace `model.generate()` | 9 |

## Project Goals

1. **Understand every layer** — from BPE merges to attention masks to SIMD intrinsics
2. **Correctness first** — every op matches its golden reference exactly
3. **Minimal dependencies** — runtime needs only the self-built `compute` crate; all ML libraries are test-only
4. **Read the code** — flat file layout, one concept per file, no frameworks or abstractions

See [plan.md](plan.md) for the full development plan and enhancement roadmap.
