"""Colab H100 cell: Download 4 teacher logit datasets + preprocess to CubeMind format.

Run each section as a separate Colab cell.
Total: ~30 min on H100 (mostly download + parquet decode)
Output: data/teachers/{qwen235b,glm358b,deepseek_r1,qwen30b_moe}/*.npz

All datasets use DistillKit compressed format:
  - compressed_logprobs: bfloat16, 128 top-k values per token
  - bytepacked_indices: 18-bit big-endian packed, 128 indices per token
"""

# ============================================================================
# CELL 1: Setup
# ============================================================================

# !pip install -q datasets numpy tqdm torch

import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

BASE = Path("/content/cubemind/data/teachers")
MAX_SEQ_LEN = 512  # Match our training config
MAX_SEQS_PER_TEACHER = 2000  # Adjust as needed
TOP_K = 128  # DistillKit exact_k


def decode_bf16_logprobs(raw_uint8):
    """Decode bfloat16 logprobs from uint8 array."""
    raw = np.array(raw_uint8, dtype=np.uint8)
    t = torch.frombuffer(raw.tobytes(), dtype=torch.bfloat16)
    return t.float().numpy()


def unpack_18bit_be(raw_uint8, n_indices=128):
    """Unpack 128 x 18-bit big-endian packed indices from 288 bytes."""
    raw = np.array(raw_uint8, dtype=np.uint8)
    bits = np.unpackbits(raw)
    indices = np.zeros(n_indices, dtype=np.int32)
    for i in range(min(n_indices, len(bits) // 18)):
        start = i * 18
        val = 0
        for b in range(18):
            val |= int(bits[start + b]) << (17 - b)
        indices[i] = val
    return indices


def decode_distillkit_row(row, max_seq_len=512):
    """Decode a single DistillKit row into (input_tokens, top_k_indices, top_k_logprobs).

    Auto-detects k from the actual byte sizes:
      logprobs bytes / 2 = k (bfloat16)
      indices bytes * 8 / 18 = k (18-bit packed)

    Returns arrays truncated to max_seq_len.
    """
    input_ids = np.array(row['input_ids'], dtype=np.int32)
    n_tokens = min(len(input_ids) - 1, max_seq_len)

    compressed_lp = row['compressed_logprobs']
    bytepacked_idx = row['bytepacked_indices']

    if n_tokens < 1:
        return input_ids[:1], np.zeros((0, 1), dtype=np.int32), np.zeros((0, 1), dtype=np.float32)

    # Auto-detect k from first token's byte sizes
    lp_bytes = len(np.array(compressed_lp[0], dtype=np.uint8))
    idx_bytes = len(np.array(bytepacked_idx[0], dtype=np.uint8))
    k = lp_bytes // 2  # bfloat16 = 2 bytes each

    all_indices = np.zeros((n_tokens, k), dtype=np.int32)
    all_logprobs = np.zeros((n_tokens, k), dtype=np.float32)

    for t in range(n_tokens):
        lp_raw = np.array(compressed_lp[t], dtype=np.uint8)
        idx_raw = np.array(bytepacked_idx[t], dtype=np.uint8)

        lp = decode_bf16_logprobs(lp_raw)
        idx = unpack_18bit_be(idx_raw, n_indices=k)

        all_logprobs[t, :len(lp)] = lp[:k]
        all_indices[t, :len(idx)] = idx[:k]

    return input_ids[:n_tokens + 1], all_indices, all_logprobs


def save_sequence_topk(out_dir, idx, input_tokens, top_k_indices, top_k_logprobs):
    """Save in CubeMind top-k format. Handles variable k across teachers."""
    path = out_dir / f"sequence_{idx:06d}.npz"
    if path.exists():
        return False
    np.savez_compressed(
        path,
        input_tokens=input_tokens.astype(np.int32),
        top_k_indices=top_k_indices.astype(np.int32),
        top_k_logprobs=top_k_logprobs.astype(np.float16),
        k=np.array([top_k_indices.shape[1]], dtype=np.int32),
        identity_len=np.array([len(input_tokens)], dtype=np.int32),
    )
    return True


def save_sequence_full(out_dir, idx, input_tokens, logits):
    """Save in CubeMind full-logits format (for datasets that have them)."""
    path = out_dir / f"sequence_{idx:06d}.npz"
    if path.exists():
        return False
    np.savez_compressed(
        path,
        input_tokens=np.array(input_tokens[:MAX_SEQ_LEN], dtype=np.int32),
        logits=np.array(logits[:MAX_SEQ_LEN], dtype=np.float16),
        identity_len=np.array([len(input_tokens)], dtype=np.int32),
    )
    return True


print("Setup done. MAX_SEQ_LEN={}, MAX_SEQS={}, TOP_K={}".format(
    MAX_SEQ_LEN, MAX_SEQS_PER_TEACHER, TOP_K))

# ============================================================================
# CELL 2: Qwen3-235B logits (DistillKit compressed, <1K sequences)
# ============================================================================

from datasets import load_dataset

out_dir = BASE / "qwen235b"
out_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Qwen3-235B logits...")
ds = load_dataset("AdrienB134/Qwen3-235B-Logits-Packed-8192-subsample", split="train")

count = 0
for row in tqdm(ds, desc="Qwen3-235B"):
    if count >= MAX_SEQS_PER_TEACHER:
        break
    if count == 0:
        print(f"  Columns: {list(row.keys())}")

    if 'compressed_logprobs' in row and 'bytepacked_indices' in row:
        input_ids, indices, logprobs = decode_distillkit_row(row, MAX_SEQ_LEN)
        # Split packed sequence into MAX_SEQ_LEN chunks
        n = len(indices)
        for start in range(0, n - 10, MAX_SEQ_LEN):
            if count >= MAX_SEQS_PER_TEACHER:
                break
            end = min(start + MAX_SEQ_LEN, n)
            if end - start < 10:
                continue
            save_sequence_topk(out_dir, count,
                               input_ids[start:end + 1],
                               indices[start:end],
                               logprobs[start:end])
            count += 1
    else:
        print(f"  Unknown format: {list(row.keys())}")
        break

print(f"Qwen3-235B: {count} sequences saved to {out_dir}")

# ============================================================================
# CELL 3: GLM-4.7-358B logits (DistillKit compressed, 100K+, MIT)
# ============================================================================

from datasets import load_dataset

out_dir = BASE / "glm358b"
out_dir.mkdir(parents=True, exist_ok=True)

print("Streaming GLM-4.7-358B logits...")
ds = load_dataset("JackBinary/GLM-4.7-358B-logits", split="train", streaming=True)

count = 0
for row in tqdm(ds, desc="GLM-358B", total=MAX_SEQS_PER_TEACHER):
    if count >= MAX_SEQS_PER_TEACHER:
        break
    if count == 0:
        print(f"  Columns: {list(row.keys())}")

    if 'compressed_logprobs' in row and 'bytepacked_indices' in row:
        input_ids, indices, logprobs = decode_distillkit_row(row, MAX_SEQ_LEN)
        n = len(indices)
        for start in range(0, n - 10, MAX_SEQ_LEN):
            if count >= MAX_SEQS_PER_TEACHER:
                break
            end = min(start + MAX_SEQ_LEN, n)
            if end - start < 10:
                continue
            save_sequence_topk(out_dir, count,
                               input_ids[start:end + 1],
                               indices[start:end],
                               logprobs[start:end])
            count += 1
    else:
        print(f"  Unknown format: {list(row.keys())}")
        break

print(f"GLM-358B: {count} sequences saved to {out_dir}")

# ============================================================================
# CELL 4: DeepSeek R1 logits (different compression: k=32, vocab=129280)
# ============================================================================

from datasets import load_dataset

out_dir = BASE / "deepseek_r1"
out_dir.mkdir(parents=True, exist_ok=True)

# DeepSeek uses legacy_logit_compression with k=32, vocab=129280
# Different format from GLM/Qwen — need to probe columns first
print("Streaming DeepSeek R1 logits...")
ds = load_dataset("arcee-ai/DeepSeek-MixedModeReasoning-Logits-Packed-16384",
                   split="train", streaming=True)

count = 0
ds_format = None
for row in tqdm(ds, desc="DeepSeek-R1", total=MAX_SEQS_PER_TEACHER):
    if count >= MAX_SEQS_PER_TEACHER:
        break

    if count == 0:
        print(f"  Columns: {list(row.keys())}")
        # Detect format
        if 'compressed_logprobs' in row:
            ds_format = "distillkit"
            # DeepSeek config: exact_k=32, vocab=129280
            # 32 indices need ceil(32 * 17 / 8) = 68 bytes (17 bits for 129K vocab)
            # Or maybe same 18-bit format
            bp_shape = np.array(row['bytepacked_indices'][0]).shape
            lp_shape = np.array(row['compressed_logprobs'][0]).shape
            print(f"  bytepacked shape per token: {bp_shape}")
            print(f"  logprobs shape per token: {lp_shape}")
            # k=32: logprobs = 32 * 2 bytes(bf16) = 64 bytes
            # indices: 32 * 18 bits = 576 bits = 72 bytes
            ds_k = lp_shape[0] // 2  # bf16 = 2 bytes each
            print(f"  Detected k={ds_k}")
        else:
            print("  Non-DistillKit format — saving raw")
            ds_format = "raw"

    if ds_format == "distillkit":
        input_ids = np.array(row['input_ids'], dtype=np.int32)
        compressed_lp = row['compressed_logprobs']
        bytepacked_idx = row['bytepacked_indices']

        n_tokens = min(len(compressed_lp), MAX_SEQ_LEN)
        if n_tokens < 10:
            continue

        # Decode with DeepSeek's k (auto-detected from shape)
        lp_bytes = np.array(compressed_lp[0], dtype=np.uint8)
        ds_k = len(lp_bytes) // 2  # bf16 = 2 bytes each
        idx_bytes = np.array(bytepacked_idx[0], dtype=np.uint8)
        bits_per_idx = (len(idx_bytes) * 8) // ds_k

        all_indices = np.zeros((n_tokens, ds_k), dtype=np.int32)
        all_logprobs = np.zeros((n_tokens, ds_k), dtype=np.float32)

        for t in range(n_tokens):
            lp_raw = np.array(compressed_lp[t], dtype=np.uint8)
            idx_raw = np.array(bytepacked_idx[t], dtype=np.uint8)
            all_logprobs[t] = decode_bf16_logprobs(lp_raw)[:ds_k]
            all_indices[t] = unpack_18bit_be(idx_raw, n_indices=ds_k)

        # Split into chunks
        for start in range(0, n_tokens - 10, MAX_SEQ_LEN):
            if count >= MAX_SEQS_PER_TEACHER:
                break
            end = min(start + MAX_SEQ_LEN, n_tokens)
            save_sequence_topk(out_dir, count,
                               input_ids[start:end + 1],
                               all_indices[start:end],
                               all_logprobs[start:end])
            count += 1
    else:
        # Raw format fallback
        tokens = np.array(row.get('input_ids', []), dtype=np.int32)
        if len(tokens) < 10:
            continue
        path = out_dir / f"sequence_{count:06d}.npz"
        if not path.exists():
            np.savez_compressed(path, input_tokens=tokens[:MAX_SEQ_LEN])
        count += 1

print(f"DeepSeek-R1: {count} sequences saved to {out_dir}")

# ============================================================================
# CELL 5: Qwen3-30B MoE patterns + logits (math domain)
# ============================================================================

from datasets import load_dataset

out_dir = BASE / "qwen30b_moe"
out_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Qwen3-30B MoE logits...")
ds = load_dataset("chaik3/aime2024_Qwen3-30B-A3B_moe_patterns_logits", split="train")

count = 0
for row in tqdm(ds, desc="Qwen3-30B-MoE"):
    if count >= MAX_SEQS_PER_TEACHER:
        break
    if count == 0:
        print(f"  Columns: {list(row.keys())}")

    # Qwen3-30B MoE format: decode_ids + decode_pattern_logits + decode_pattern
    tokens = row.get('decode_ids', row.get('input_ids', None))
    logits = row.get('decode_pattern_logits', row.get('logits', None))
    moe_patterns = row.get('decode_pattern', row.get('moe_patterns', None))

    if tokens is None:
        if count == 0:
            print(f"  No tokens found: {list(row.keys())}")
        break

    tokens = np.array(tokens, dtype=np.int32)
    total_len = len(tokens)
    if total_len < 10:
        continue

    for start in range(0, total_len - 10, MAX_SEQ_LEN):
        if count >= MAX_SEQS_PER_TEACHER:
            break
        end = min(start + MAX_SEQ_LEN, total_len)
        if end - start < 10:
            continue

        save_dict = {
            "input_tokens": tokens[start:end].astype(np.int32),
        }
        if logits is not None:
            save_dict["logits"] = np.array(logits)[start:end].astype(np.float16)
        if moe_patterns is not None:
            save_dict["moe_patterns"] = np.array(moe_patterns)[start:end]

        path = out_dir / f"sequence_{count:06d}.npz"
        if not path.exists():
            np.savez_compressed(path, **save_dict)
        count += 1

print(f"Qwen3-30B-MoE: {count} sequences saved to {out_dir}")

# ============================================================================
# CELL 6: Summary + package for download
# ============================================================================


print("\n" + "=" * 60)
print("  Teacher Logits Summary")
print("=" * 60)

total_seqs = 0
total_bytes = 0
for teacher_dir in sorted(BASE.iterdir()):
    if not teacher_dir.is_dir():
        continue
    seqs = list(teacher_dir.glob("sequence_*.npz"))
    n = len(seqs)
    size = sum(f.stat().st_size for f in seqs)
    total_seqs += n
    total_bytes += size
    print(f"  {teacher_dir.name:20s}: {n:5d} seqs, {size/1e9:.2f} GB")

print(f"  {'TOTAL':20s}: {total_seqs:5d} seqs, {total_bytes/1e9:.2f} GB")
print("=" * 60)

# Optional: tar for download
# !tar czf /content/teacher_logits.tar.gz -C /content/cubemind/data teachers/
print("\nTo download: !tar czf /content/teachers.tar.gz -C /content/cubemind/data teachers/")
