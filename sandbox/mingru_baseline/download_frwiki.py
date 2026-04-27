"""Stream French Wikipedia from Hugging Face → flat JSONL.

Pulls ``wikimedia/wikipedia`` (config ``20231101.fr``) in streaming mode
so we never materialize the full ~5 GB parquet locally. Each output line
is a single JSON object the build_pretrain_corpus.py pipeline accepts
without further transformation.

Output schema (one record per line):
    {"text": "<article body>", "title": "<title>", "id": "<wiki id>"}

Sizing note: French Wikipedia averages ~750 SPM tokens per article with
the grillcheese_spm32k_v2 tokenizer (slightly higher than English wiki
because of accents and longer words). For ~15% of a 2 B-token Stage-1
budget = ~300 M French tokens → cap at ~400 k articles.

Example:

    python sandbox/mingru_baseline/download_frwiki.py \\
        --output D:\\grillcheese_training_data\\jsonl\\frwiki_400k.jsonl \\
        --max-records 400000 \\
        --min-chars 500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path,
                    help="Output JSONL path")
    ap.add_argument("--config", default="20231101.fr",
                    help="HF wikimedia/wikipedia config (default: 20231101.fr)")
    ap.add_argument("--max-records", type=int, default=400_000,
                    help="Stop after writing this many articles "
                         "(default: 400 000 → ~300 M SPM tokens)")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="Skip stub articles shorter than N characters "
                         "(default: 500). Wikipedia stubs add noise, not signal.")
    args = ap.parse_args()

    # Lazy import — datasets is heavy and not needed for the rest of the
    # pretrain pipeline; only this downloader pulls it.
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "FATAL: install `datasets` first: pip install datasets\n"
            f"  underlying error: {e}"
        ) from e

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"  config:      wikimedia/wikipedia / {args.config}")
    print(f"  output:      {args.output}")
    print(f"  max records: {args.max_records:,}")
    print(f"  min chars:   {args.min_chars}")
    print()

    ds = load_dataset("wikimedia/wikipedia", args.config,
                      split="train", streaming=True)

    n_in = n_out = n_skip = chars_out = 0
    t0 = time.time()
    with args.output.open("w", encoding="utf-8") as fout:
        for row in ds:
            n_in += 1
            txt = (row.get("text") or "").strip()
            if len(txt) < args.min_chars:
                n_skip += 1
                continue
            rec = {
                "text": txt,
                "title": row.get("title", ""),
                "id": str(row.get("id", "")),
            }
            fout.write(json.dumps(rec, ensure_ascii=False))
            fout.write("\n")
            n_out += 1
            chars_out += len(txt)

            if n_out % 5_000 == 0:
                elapsed = time.time() - t0
                rate = n_in / max(elapsed, 1e-6)
                print(f"    in={n_in:>8,}  out={n_out:>8,}  "
                      f"skip={n_skip:>6,}  {chars_out/1e9:>5.2f} GB  "
                      f"{rate:>5.0f} rec/s  {elapsed:>5.0f}s",
                      end="\r")

            if n_out >= args.max_records:
                break

    print()
    elapsed = time.time() - t0
    print(f"  DONE in={n_in:,}  out={n_out:,}  skip={n_skip:,}  "
          f"{chars_out/1e9:.2f} GB  {elapsed/60:.1f}m")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
