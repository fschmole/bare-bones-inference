"""
chat.py — Minimal interactive CLI for GPT-2 text generation.

Single-turn: each prompt is independent (no conversation history).
Commands:
    /quit        — exit
    /trace 0|1|2 — set trace verbosity (0=off, 1=low, 2=high)
    /avx2 on|off — toggle AVX2 SIMD acceleration

Usage:
    python chat.py --model-dir models/gpt2 --max-tokens 40 --trace 1
    python chat.py --no-avx2 --trace 2 --trace-file trace.log
"""

import argparse
import sys
import time

import compute
from generate import generate, set_trace as set_generate_trace, set_trace_file as set_generate_trace_file
from loader import load_config, load_weights
from model import GPT2
from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="GPT-2 chat (single-turn)")
    parser.add_argument(
        "--model-dir",
        default="models/gpt2",
        help="Directory containing model files (default: models/gpt2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=40,
        help="Maximum number of tokens to generate per response (default: 40)",
    )
    parser.add_argument(
        "--trace",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Trace verbosity: 0=off, 1=low (summary), 2=high (per-token detail)",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Also write trace output to this file (append mode)",
    )
    parser.add_argument(
        "--no-avx2",
        action="store_true",
        help="Disable AVX2 SIMD acceleration (force naive scalar ops)",
    )
    args = parser.parse_args()

    # --- AVX2 setup ---
    if args.no_avx2:
        compute.set_avx2(False)

    # --- Load model ---
    print(f"Loading model from {args.model_dir}...", file=sys.stderr)
    t0 = time.time()

    config = load_config(f"{args.model_dir}/config.json")
    weights = load_weights(f"{args.model_dir}/model.safetensors")
    tokenizer = Tokenizer(
        f"{args.model_dir}/vocab.json",
        f"{args.model_dir}/merges.txt",
    )
    model = GPT2(config, weights)

    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s", file=sys.stderr)
    print(
        f"  {config['n_layer']} layers, {config['n_head']} heads, "
        f"d_model={config['n_embd']}, vocab={config['vocab_size']}",
        file=sys.stderr,
    )
    print(f"  max_tokens={args.max_tokens}, avx2={'ON' if compute.get_avx2() else 'OFF'}", file=sys.stderr)

    # --- Trace setup ---
    if args.trace > 0:
        set_generate_trace(args.trace)
        compute.set_trace(args.trace)
        print(f"Tracing: level {args.trace} ({'low' if args.trace == 1 else 'high'})", file=sys.stderr)
    else:
        print("Tracing: OFF (use /trace 1 or /trace 2 to enable)", file=sys.stderr)

    if args.trace_file:
        set_generate_trace_file(args.trace_file)
        compute.set_trace_file(args.trace_file)
        print(f"Trace file: {args.trace_file}", file=sys.stderr)

    print()

    # --- Chat loop ---
    while True:
        try:
            prompt = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        prompt = prompt.strip()
        if not prompt:
            continue

        # --- Commands ---
        if prompt == "/quit":
            print("Bye!")
            break

        if prompt.startswith("/trace "):
            try:
                level = int(prompt.split()[1])
                if level not in (0, 1, 2):
                    raise ValueError
                set_generate_trace(level)
                compute.set_trace(level)
                labels = {0: "OFF", 1: "low", 2: "high"}
                print(f"Tracing: {labels[level]}", file=sys.stderr)
            except (ValueError, IndexError):
                print("Usage: /trace 0|1|2", file=sys.stderr)
            continue

        if prompt.startswith("/avx2 "):
            arg = prompt.split()[1] if len(prompt.split()) > 1 else ""
            if arg == "on":
                compute.set_avx2(True)
                print("AVX2: ON", file=sys.stderr)
            elif arg == "off":
                compute.set_avx2(False)
                print("AVX2: OFF (using naive ops)", file=sys.stderr)
            else:
                print("Usage: /avx2 on|off", file=sys.stderr)
            continue

        if prompt.startswith("/"):
            print(
                f"Unknown command: {prompt}. "
                "Available: /quit, /trace 0|1|2, /avx2 on|off",
                file=sys.stderr,
            )
            continue

        # --- Generate ---
        print("GPT-2: ", end="", flush=True)
        _, tok_per_sec = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            stream=True,
        )

        # Print tokens/sec if tracing is enabled
        if compute.get_trace() > 0:
            print(f"  [{tok_per_sec:.2f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
