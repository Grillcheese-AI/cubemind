"""Decode DistillKit compressed logit format.

Config from dataset:
  exact_k: 128, exact_dtype: bfloat16, k: 128
  vocab: 151366, delta_encoding: false
  288 bytes = 128 indices × 18 bits packed
  256 bytes = 128 bfloat16 logprobs

Run on Colab:
    python scripts/decode_distillkit.py
"""

import numpy as np
import torch
from datasets import load_dataset


def unpack_18bit(packed_bytes, n_indices=128):
    """Unpack 128 × 18-bit indices from 288 bytes."""
    raw = np.frombuffer(
        packed_bytes if isinstance(packed_bytes, bytes) else bytes(packed_bytes),
        dtype=np.uint8,
    )
    bits = np.unpackbits(raw)
    indices = []
    for i in range(n_indices):
        start = i * 18
        if start + 18 > len(bits):
            break
        val = 0
        for b in range(18):
            val |= int(bits[start + b]) << (17 - b)
        indices.append(val)
    return np.array(indices, dtype=np.int32)


def unpack_18bit_le(packed_bytes, n_indices=128):
    """Unpack 128 × 18-bit LE indices from 288 bytes."""
    raw = np.frombuffer(
        packed_bytes if isinstance(packed_bytes, bytes) else bytes(packed_bytes),
        dtype=np.uint8,
    )
    bits = np.unpackbits(raw, bitorder='little')
    indices = []
    for i in range(n_indices):
        start = i * 18
        if start + 18 > len(bits):
            break
        val = 0
        for b in range(18):
            val |= int(bits[start + b]) << b
        indices.append(val)
    return np.array(indices, dtype=np.int32)


def unpack_9bytes_4idx(packed_bytes, n_indices=128):
    """Decode 4 indices from every 9 bytes (4 × 18 = 72 = 9 × 8)."""
    result = []
    data = bytes(packed_bytes) if not isinstance(packed_bytes, bytes) else packed_bytes
    for chunk_start in range(0, len(data), 9):
        chunk = data[chunk_start:chunk_start + 9]
        if len(chunk) < 9:
            break
        val = int.from_bytes(chunk, 'little')
        for j in range(4):
            idx = (val >> (j * 18)) & 0x3FFFF
            result.append(idx)
            if len(result) >= n_indices:
                break
        if len(result) >= n_indices:
            break
    return np.array(result[:n_indices], dtype=np.int32)


def decode_bf16_logprobs(raw_bytes):
    """Decode 256 bytes as 128 bfloat16 values via torch."""
    raw = np.frombuffer(
        raw_bytes if isinstance(raw_bytes, bytes) else bytes(raw_bytes),
        dtype=np.uint8,
    )
    # bfloat16 is 2 bytes: convert via torch
    t = torch.frombuffer(raw.tobytes(), dtype=torch.bfloat16)
    return t.float().numpy()


def main():
    ds = load_dataset("JackBinary/GLM-4.7-358B-logits", split="train", streaming=True)
    row = next(iter(ds))

    input_ids = np.array(row['input_ids'])
    cl = np.array(row['compressed_logprobs'][0], dtype=np.uint8)
    bp = np.array(row['bytepacked_indices'][0], dtype=np.uint8)

    next_token = input_ids[1]
    print(f"input_ids: shape={input_ids.shape} max={input_ids.max()}")
    print(f"next_token (ground truth): {next_token}")
    print()

    # ── Decode logprobs as bfloat16 ──────────────────────────────────────────
    logprobs = decode_bf16_logprobs(cl.tobytes())
    print(f"bfloat16 logprobs ({len(logprobs)} values):")
    print(f"  first 10: {logprobs[:10]}")
    print(f"  min={logprobs.min():.4f} max={logprobs.max():.4f}")
    probs = np.exp(logprobs)
    print(f"  exp(logprobs) sum = {probs.sum():.4f}")
    print(f"  exp(logprobs[:32]) sum = {probs[:32].sum():.4f}")
    print()

    # ── Also try as float16 for comparison ───────────────────────────────────
    lp_f16 = np.frombuffer(cl.tobytes(), dtype=np.float16)
    print(f"float16 logprobs for comparison: {lp_f16[:5]}")
    print()

    # ── Decode indices: try all 3 methods ────────────────────────────────────
    for name, func in [
        ("18-bit BE", unpack_18bit),
        ("18-bit LE", unpack_18bit_le),
        ("9-byte chunks", unpack_9bytes_4idx),
    ]:
        idx = func(bp.tobytes())
        valid = np.sum((idx >= 0) & (idx < 160000))
        has_next = next_token in idx
        print(f"{name}: valid={valid}/{len(idx)} max={idx.max()} "
              f"next_token_found={has_next}")
        if has_next:
            pos = np.where(idx == next_token)[0][0]
            print(f"  *** MATCH at position {pos}, logprob={logprobs[pos]:.4f}")
        print(f"  first 10: {idx[:10]}")
        print()

    # ── Check token 2 as well ────────────────────────────────────────────────
    print("--- Token 2 ---")
    cl2 = np.array(row['compressed_logprobs'][1], dtype=np.uint8)
    bp2 = np.array(row['bytepacked_indices'][1], dtype=np.uint8)
    next_token2 = input_ids[2]
    lp2 = decode_bf16_logprobs(cl2.tobytes())

    for name, func in [
        ("18-bit BE", unpack_18bit),
        ("18-bit LE", unpack_18bit_le),
        ("9-byte chunks", unpack_9bytes_4idx),
    ]:
        idx = func(bp2.tobytes())
        has_next = next_token2 in idx
        print(f"{name}: next_token={next_token2} found={has_next} max={idx.max()}")
        if has_next:
            pos = np.where(idx == next_token2)[0][0]
            print(f"  *** MATCH at position {pos}, logprob={lp2[pos]:.4f}")


if __name__ == "__main__":
    main()
