"""Extract teacher logits from local GGUF model.

Uses Qwen3-Code-Reasoning 6B on Vulkan GPU to generate teacher logits
for MoQE distillation. No cloud needed.

Reads text from .txt or .jsonl, tokenizes, runs forward pass,
saves (input_tokens, logits) in CubeMind's .npz format.

Usage:
    python -u -m scripts.extract_local_logits --max-seqs 2000
    python -u -m scripts.extract_local_logits --input data.txt --max-seqs 500
    python -u -m scripts.extract_local_logits --model path/to/model.gguf
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def extract(
    model_path: str,
    input_paths: list[str],
    output_dir: str = "data/teacher/qwen3_6b",
    max_seqs: int = 2000,
    max_seq_len: int = 512,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    skip_existing: bool = True,
):
    from llama_cpp import Llama

    os.makedirs(output_dir, exist_ok=True)

    # Count existing
    existing = len(list(Path(output_dir).glob("sequence_*.npz")))
    if skip_existing and existing >= max_seqs:
        print(f"Already have {existing} sequences, skipping")
        return

    print(f"Loading model: {model_path}")
    print(f"n_ctx={n_ctx}, max_seq_len={max_seq_len}, gpu_layers={n_gpu_layers}")
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=True,
        verbose=False,
    )
    vocab_size = model.n_vocab()
    print(f"Vocab: {vocab_size}, ctx: {n_ctx}")

    # Collect texts from all input files
    def iter_texts():
        for path in input_paths:
            p = Path(path)
            if p.suffix == ".txt":
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 20:
                            yield line
            elif p.suffix == ".jsonl":
                import json
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", obj.get("content", ""))
                            if len(text) > 20:
                                yield text
                        except Exception:
                            continue

    t0 = time.time()
    count = existing
    total_tokens = 0

    for text in iter_texts():
        if count >= max_seqs:
            break

        seq_path = Path(output_dir) / f"sequence_{count:06d}.npz"
        if skip_existing and seq_path.exists():
            count += 1
            continue

        # Tokenize
        tokens = model.tokenize(text.encode("utf-8", errors="ignore"))
        if len(tokens) < 10:
            continue
        tokens = tokens[:max_seq_len]

        # Forward pass with logits
        try:
            model.reset()
            model.eval(tokens)

            # Extract logits for each position
            # llama-cpp stores scores after eval
            n_tokens = len(tokens)
            logits = np.zeros((n_tokens, vocab_size), dtype=np.float16)

            for i in range(n_tokens):
                try:
                    scores = model.scores[i]
                    logits[i] = np.array(scores, dtype=np.float16)
                except (IndexError, AttributeError):
                    break

            if np.all(logits == 0):
                # Fallback: try _scores
                try:
                    all_scores = np.array(model._scores, dtype=np.float16)
                    if all_scores.ndim == 2:
                        logits[:min(n_tokens, len(all_scores))] = all_scores[:n_tokens]
                except Exception:
                    continue

        except Exception as e:
            print(f"  Error on seq {count}: {e}")
            continue

        # Save
        np.savez_compressed(
            seq_path,
            input_tokens=np.array(tokens, dtype=np.int32),
            logits=logits,
            identity_len=np.array([n_tokens], dtype=np.int32),
        )

        count += 1
        total_tokens += n_tokens

        if count % 10 == 0:
            elapsed = time.time() - t0
            rate = (count - existing) / max(elapsed, 1)
            print(f"  [{count}/{max_seqs}] {total_tokens:,} tokens, "
                  f"{elapsed:.0f}s, {rate:.1f} seq/s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {count} sequences, {total_tokens:,} tokens in {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/external_llms/Qwen3-Code-Reasoning-Instruct-6B-Brainstorm20x.Q4_K_M.gguf")
    parser.add_argument("--input", nargs="+", default=[
        "D:/grillcheese_training_data/unified/microsoft_orca_agentinstruct_1m_v1.full.jsonl",
        "D:/grillcheese_training_data/unified/nemotron_cc_math_v1_4plus_mind.500m_tokens.jsonl",
        "D:/grillcheese_training_data/unified/allenai_c4_multilingual.500m_tokens.txt",
        "D:/grillcheese_training_data/unified/allenai_c4_realnewslike.500m_tokens.jsonl",
    ])
    parser.add_argument("--output", default="data/teacher/qwen3_6b")
    parser.add_argument("--max-seqs", type=int, default=2000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    args = parser.parse_args()

    extract(
        model_path=args.model,
        input_paths=args.input,
        output_dir=args.output,
        max_seqs=args.max_seqs,
        max_seq_len=args.max_seq_len,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )
