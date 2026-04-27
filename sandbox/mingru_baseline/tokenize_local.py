"""Parallel local tokenizer for CubeMind-LM training corpora.

Produces the same four-file layout as ``prepare_data_from_file()`` in
``train_torch.py`` (lines 336-400):

    data/train_<tag>_<tok_tag>.bin       raw uint16 token IDs
    data/val_<tag>_<tok_tag>.bin         raw uint16 token IDs
    data/meta_<tag>_<tok_tag>.json       {"source","tag","vocab","train_tokens","val_tokens"}
    data/tokenizer_<tag>_<tok_tag>.json  {"spm_path","vocab_size"}

Tag derivation matches train_torch.py exactly:
    tag     = src.stem.replace('.', '_')[:40]
    tok_tag = f"spm_{tokenizer_path.stem}"

With these four files present in the pod's
``sandbox/mingru_baseline/data/`` directory, the ``needs_build`` check
in ``prepare_data_from_file`` is False and tokenization is skipped.

Uses a multiprocessing pool so it saturates every core instead of
single-threading like the H200 path does. Typical 16-core workstation
tokenizes 3-4 GB of text in a few minutes.

Example:

    python sandbox/mingru_baseline/tokenize_local.py \\
        --data-path D:\\grillcheese_training_data\\temporal_corpus_v4.txt \\
        --tokenizer-path D:\\grillcheese_training_data\\tokenizer\\grillcheese_spm32k_v2.model
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

_SPM = None  # per-worker singleton, populated by _worker_init


def _worker_init(spm_path: str) -> None:
    global _SPM
    import sentencepiece as spm
    _SPM = spm.SentencePieceProcessor()
    _SPM.Load(spm_path)


def _encode_chunk(chunk: str) -> np.ndarray:
    assert _SPM is not None
    return np.asarray(_SPM.EncodeAsIds(chunk), dtype=np.uint16)


def _chunk_stream(src: Path, chunk_bytes: int):
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            c = f.read(chunk_bytes)
            if not c:
                break
            yield c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, type=Path,
                    help="UTF-8 text corpus to tokenize")
    ap.add_argument("--tokenizer-path", required=True, type=Path,
                    help="SentencePiece .model file")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "data",
                    help="Output directory (default: sandbox/mingru_baseline/data)")
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--workers", type=int,
                    default=max(1, (mp.cpu_count() or 2) - 2),
                    help="Pool workers (default: cpu_count - 2)")
    ap.add_argument("--chunk-mb", type=int, default=4,
                    help="Text chunk size per task (MB)")
    args = ap.parse_args()

    src: Path = args.data_path.resolve()
    pre: Path = args.tokenizer_path.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not src.exists():
        raise SystemExit(f"FATAL: data path not found: {src}")
    if not pre.exists():
        raise SystemExit(f"FATAL: tokenizer not found: {pre}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = src.stem.replace(".", "_")[:40]
    tok_tag = f"spm_{pre.stem}"
    train_bin = out_dir / f"train_{tag}_{tok_tag}.bin"
    val_bin   = out_dir / f"val_{tag}_{tok_tag}.bin"
    meta_path = out_dir / f"meta_{tag}_{tok_tag}.json"
    tok_path  = out_dir / f"tokenizer_{tag}_{tok_tag}.json"

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(pre))
    vocab_size = int(sp.GetPieceSize())
    del sp

    tok_path.write_text(json.dumps(
        {"spm_path": str(pre), "vocab_size": vocab_size}
    ))

    file_gb = src.stat().st_size / 1e9
    chunk_bytes = args.chunk_mb * 1_000_000

    print(f"  source:    {src.name}  ({file_gb:.2f} GB)")
    print(f"  tokenizer: {pre.name}  (vocab={vocab_size})")
    print(f"  out_dir:   {out_dir}")
    print(f"  workers:   {args.workers}  |  chunk: {args.chunk_mb} MB")
    print(f"  targets:")
    for p in (train_bin, val_bin, meta_path, tok_path):
        print(f"    {p.name}")
    print()

    t0 = time.time()
    parts: list[np.ndarray] = []
    total_tok = 0
    total_mb = 0.0

    with mp.Pool(args.workers, initializer=_worker_init,
                 initargs=(str(pre),)) as pool:
        for i, arr in enumerate(
            pool.imap(_encode_chunk, _chunk_stream(src, chunk_bytes),
                      chunksize=1)
        ):
            parts.append(arr)
            total_tok += int(arr.shape[0])
            total_mb += args.chunk_mb
            if i % 25 == 0:
                elapsed = time.time() - t0
                rate = total_tok / max(elapsed, 1e-6)
                print(
                    f"    {total_tok:>12,} tokens  |  "
                    f"{total_mb:>6.0f} MB read  |  "
                    f"{rate/1e6:>5.2f} M tok/s  |  "
                    f"{elapsed:>5.1f}s",
                    end="\r",
                )

    elapsed = time.time() - t0
    print(f"    {total_tok:>12,} tokens  |  done in {elapsed:.1f}s"
          f" ({total_tok/max(elapsed,1e-6)/1e6:.2f} M tok/s){' '*20}")

    all_ids = np.concatenate(parts) if len(parts) > 1 else parts[0]
    del parts
    gc.collect()

    split_idx = int(len(all_ids) * (1.0 - args.val_split))
    train_ids = all_ids[:split_idx]
    val_ids   = all_ids[split_idx:]

    print(f"  writing train_bin ({len(train_ids):,} tokens, "
          f"{train_ids.nbytes/1e9:.2f} GB)")
    train_ids.tofile(str(train_bin))
    print(f"  writing val_bin   ({len(val_ids):,} tokens, "
          f"{val_ids.nbytes/1e9:.2f} GB)")
    val_ids.tofile(str(val_bin))

    meta_path.write_text(json.dumps({
        "source": str(src),
        "tag": tag,
        "vocab": vocab_size,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
    }, indent=2))

    print(f"  train: {len(train_ids):>12,} tokens")
    print(f"  val:   {len(val_ids):>12,} tokens")
    print()
    print("  upload these four files to:")
    print("    /workspace/cubemind/sandbox/mingru_baseline/data/")


if __name__ == "__main__":
    main()
