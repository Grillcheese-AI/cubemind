"""Pre-process teacher logits for fast training.

Reads full-length .npz files from source dir, truncates to max_seq_len,
saves as compact .npz to local SSD. Reduces I/O from 310MB → ~4MB per file.

Usage:
    python scripts/preprocess_logits.py --max-seq-len 512
    python scripts/preprocess_logits.py --max-seq-len 64 --out data/logits_64
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def _process_one(args: tuple) -> tuple[int, int, int]:
    """Process a single file. Runs in worker process.

    Returns (n_tokens, bytes_in, bytes_out) or (0, 0, 0) on skip/error.
    """
    path, out_path, max_seq_len = args

    if os.path.exists(out_path):
        return (0, 0, 0)

    try:
        data = np.load(path)
        tokens = data["input_tokens"][:max_seq_len].astype(np.int32)
        logits = data["logits"][:max_seq_len]

        if len(tokens) < 2:
            return (0, 0, 0)

        save_dict = {"input_tokens": tokens, "logits": logits}
        if "identity_len" in data:
            save_dict["identity_len"] = data["identity_len"]

        np.savez_compressed(out_path, **save_dict)

        return (len(tokens), os.path.getsize(path), os.path.getsize(out_path))

    except Exception as e:
        print(f"  ERROR {os.path.basename(path)}: {e}", flush=True)
        return (0, 0, 0)


def preprocess(src_dir: str, dst_dir: str, max_seq_len: int, workers: int = 8) -> None:
    files = sorted(glob.glob(os.path.join(src_dir, "sequence_*.npz")))
    if not files:
        print(f"No sequence_*.npz files in {src_dir}")
        sys.exit(1)

    os.makedirs(dst_dir, exist_ok=True)
    print(f"Preprocessing {len(files)} files: {src_dir} -> {dst_dir}")
    print(f"  max_seq_len={max_seq_len}, workers={workers}")

    tasks = [
        (path, os.path.join(dst_dir, os.path.basename(path)), max_seq_len)
        for path in files
    ]

    total_tokens = 0
    total_bytes_in = 0
    total_bytes_out = 0
    done = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, t): t[0] for t in tasks}

        for future in as_completed(futures):
            n_tok, b_in, b_out = future.result()
            total_tokens += n_tok
            total_bytes_in += b_in
            total_bytes_out += b_out
            done += 1

            if done % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed
                eta = (len(files) - done) / max(rate, 0.01)
                print(f"  [{done:>4}/{len(files)}] {rate:.1f} files/s "
                      f"| {total_tokens/1e3:.0f}K tok "
                      f"| {total_bytes_out/1e6:.0f}MB out "
                      f"| ETA {eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {done} files in {elapsed:.1f}s ({done/elapsed:.1f} files/s)")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Input:  {total_bytes_in/1e9:.1f} GB")
    print(f"  Output: {total_bytes_out/1e9:.2f} GB")
    if total_bytes_out > 0:
        print(f"  Ratio:  {total_bytes_in/total_bytes_out:.0f}x smaller")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process teacher logits")
    parser.add_argument("--src", default="G:/MYDRIVE", help="Source directory")
    parser.add_argument("--out", default="data/logits", help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    preprocess(args.src, args.out, args.max_seq_len, args.workers)
