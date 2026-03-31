"""H200 multi-teacher logit extraction + Harrier embedding eval.

One-shot script for an H200 (141GB VRAM) session:
  1. Extract teacher logits from 3 models sequentially
  2. Evaluate Harrier-OSS-v1 embeddings
  3. Package everything for download

Teachers (all MIT/Apache, permissive):
  - Qwen3-Coder-Next 80B (Apache 2.0)
  - GLM-4.7 (MIT)
  - DeepSeek R1 distill (MIT)

Usage:
    python scripts/h200_extract_logits.py --corpus data/training_texts.jsonl
    python scripts/h200_extract_logits.py --corpus data/training_texts.jsonl --skip-harrier
    python scripts/h200_extract_logits.py --teacher qwen --max-seqs 500

Estimated time: ~6-8 hours total on H200
Estimated cost: ~$25 at $3.59/hr
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# ── Teacher configs ──────────────────────────────────────────────────────────

TEACHERS = {
    "qwen": {
        "name": "Qwen/Qwen3-Coder-Next-80B",
        "short": "qwen80b",
        "dtype": torch.float16,
        "trust_remote_code": True,
    },
    "glm": {
        "name": "THUDM/glm-4-9b-chat",  # GLM-4 9B as proxy (fits easily)
        "short": "glm4",
        "dtype": torch.float16,
        "trust_remote_code": True,
    },
    "deepseek": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "short": "dsr1_32b",
        "dtype": torch.float16,
        "trust_remote_code": True,
    },
}

# Harrier embedding model
HARRIER_MODEL = "microsoft/harrier-oss-v1-0.6b"


def free_gpu():
    """Aggressively free GPU memory between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def extract_logits_for_teacher(
    teacher_key: str,
    corpus_path: str,
    output_dir: str,
    max_seqs: int = 2000,
    max_seq_len: int = 512,
    batch_size: int = 1,
):
    """Extract teacher logits from a single model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = TEACHERS[teacher_key]
    out_dir = Path(output_dir) / cfg["short"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many already done
    existing = len(list(out_dir.glob("sequence_*.npz")))
    if existing >= max_seqs:
        print(f"  {cfg['short']}: {existing} sequences already extracted, skipping")
        return existing

    print(f"\n{'='*60}")
    print(f"  Teacher: {cfg['name']}")
    print(f"  Output:  {out_dir}")
    print(f"  Target:  {max_seqs} sequences (max_len={max_seq_len})")
    print(f"{'='*60}")

    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["name"], trust_remote_code=cfg.get("trust_remote_code", False))

    print(f"  Loading model ({cfg['dtype']})...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        torch_dtype=cfg["dtype"],
        device_map="auto",
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    model.eval()

    # Get vocab size
    vocab_size = model.config.vocab_size
    print(f"  Vocab: {vocab_size}, Device: {next(model.parameters()).device}")

    # Load corpus
    texts = []
    corpus = Path(corpus_path)
    if corpus.suffix == ".jsonl":
        with open(corpus) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", obj.get("prompt", "")))
                if text:
                    texts.append(text)
    elif corpus.suffix == ".txt":
        with open(corpus) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported corpus format: {corpus.suffix}")

    print(f"  Corpus: {len(texts)} texts loaded")

    t0 = time.perf_counter()
    n_extracted = existing
    total_tokens = 0

    for i, text in enumerate(texts):
        if n_extracted >= max_seqs:
            break

        seq_path = out_dir / f"sequence_{n_extracted:06d}.npz"
        if seq_path.exists():
            n_extracted += 1
            continue

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=True,
                                  max_length=max_seq_len, truncation=True)
        if len(tokens) < 10:
            continue

        input_ids = torch.tensor([tokens], dtype=torch.long).cuda()

        # Forward pass — get logits
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0].cpu().half().numpy()  # (seq_len, vocab)

        input_tokens = np.array(tokens, dtype=np.int32)

        # Save
        np.savez_compressed(
            seq_path,
            input_tokens=input_tokens,
            logits=logits,
            identity_len=np.array([len(tokens)], dtype=np.int32),
        )

        n_extracted += 1
        total_tokens += len(tokens)

        if n_extracted % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = n_extracted / elapsed * 3600
            print(f"  [{n_extracted}/{max_seqs}] {total_tokens:,} tokens, "
                  f"{elapsed:.0f}s, ~{rate:.0f} seq/hr")

    elapsed = time.perf_counter() - t0
    print(f"\n  Done: {n_extracted} sequences, {total_tokens:,} tokens in {elapsed:.0f}s")

    # Free model
    del model
    del tokenizer
    free_gpu()

    return n_extracted


def eval_harrier(corpus_path: str, output_dir: str, max_texts: int = 500):
    """Evaluate Harrier-OSS-v1 embeddings on the training corpus."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"  Evaluating Harrier-OSS-v1 Embeddings")
    print(f"{'='*60}")

    print(f"  Loading {HARRIER_MODEL}...")
    model = SentenceTransformer(HARRIER_MODEL, trust_remote_code=True)

    # Load texts
    texts = []
    corpus = Path(corpus_path)
    if corpus.suffix == ".jsonl":
        with open(corpus) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", obj.get("content", obj.get("prompt", "")))
                if text:
                    texts.append(text[:512])  # truncate for embedding
    elif corpus.suffix == ".txt":
        with open(corpus) as f:
            texts = [line.strip()[:512] for line in f if line.strip()]

    texts = texts[:max_texts]
    print(f"  Encoding {len(texts)} texts...")

    t0 = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    elapsed = time.perf_counter() - t0

    # Stats
    emb_dim = embeddings.shape[1]
    norms = np.linalg.norm(embeddings, axis=1)

    # Self-similarity matrix (sample)
    n_sample = min(100, len(embeddings))
    sample = embeddings[:n_sample]
    sim_matrix = (sample @ sample.T) / (
        np.linalg.norm(sample, axis=1, keepdims=True) @
        np.linalg.norm(sample, axis=1, keepdims=True).T + 1e-8
    )
    # Off-diagonal mean similarity
    mask = ~np.eye(n_sample, dtype=bool)
    avg_sim = sim_matrix[mask].mean()

    print(f"\n  Harrier-OSS-v1 Results:")
    print(f"  Dimension:       {emb_dim}")
    print(f"  Texts encoded:   {len(texts)}")
    print(f"  Time:            {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/s)")
    print(f"  Norm (mean±std): {norms.mean():.3f} ± {norms.std():.3f}")
    print(f"  Avg cosine sim:  {avg_sim:.4f} (off-diagonal)")
    print(f"  Min/Max sim:     {sim_matrix[mask].min():.4f} / {sim_matrix[mask].max():.4f}")

    # Save embeddings
    out_path = Path(output_dir) / "harrier_embeddings.npz"
    np.savez_compressed(out_path, embeddings=embeddings,
                        dim=emb_dim, model=HARRIER_MODEL)
    print(f"  Saved to {out_path}")

    del model
    free_gpu()


def main():
    parser = argparse.ArgumentParser(description="H200 multi-teacher logit extraction")
    parser.add_argument("--corpus", required=True, help="Training corpus (.jsonl or .txt)")
    parser.add_argument("--output", default="data/h200_logits", help="Output directory")
    parser.add_argument("--max-seqs", type=int, default=2000, help="Max sequences per teacher")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--teacher", type=str, default=None,
                        help="Run single teacher (qwen/glm/deepseek)")
    parser.add_argument("--skip-harrier", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  H200 Multi-Teacher Logit Extraction")
    print("=" * 60)
    print(f"  Corpus:    {args.corpus}")
    print(f"  Output:    {args.output}")
    print(f"  Max seqs:  {args.max_seqs}")
    print(f"  Teachers:  {list(TEACHERS.keys()) if not args.teacher else [args.teacher]}")
    print(f"  Harrier:   {'skip' if args.skip_harrier else HARRIER_MODEL}")
    print(f"  GPU:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_mem/1e9:.0f} GB"
          if torch.cuda.is_available() else "")

    t_start = time.perf_counter()

    # Extract logits from each teacher
    teachers_to_run = [args.teacher] if args.teacher else ["qwen", "glm", "deepseek"]
    for teacher_key in teachers_to_run:
        extract_logits_for_teacher(
            teacher_key=teacher_key,
            corpus_path=args.corpus,
            output_dir=args.output,
            max_seqs=args.max_seqs,
            max_seq_len=args.max_seq_len,
        )

    # Evaluate Harrier embeddings
    if not args.skip_harrier:
        eval_harrier(args.corpus, args.output)

    total_time = time.perf_counter() - t_start
    cost = total_time / 3600 * 3.59

    print(f"\n{'='*60}")
    print(f"  All done!")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Est. cost:  ${cost:.2f}")
    print(f"  Output:     {args.output}/")

    # List outputs
    for d in Path(args.output).iterdir():
        if d.is_dir():
            n = len(list(d.glob("sequence_*.npz")))
            print(f"    {d.name}/: {n} sequences")
        elif d.suffix == ".npz":
            print(f"    {d.name}: {d.stat().st_size / 1e6:.1f} MB")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
