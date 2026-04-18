#!/usr/bin/env python3
"""Extract OpenThoughts-114k JSONL to chat-formatted plain text for
stage-1 LM pretraining.

OpenThoughts-114k is 114K reasoning samples in JSONL with
``system``/``user``/``assistant`` fields. The assistant field already
contains the ``<|begin_of_thought|> ... <|end_of_thought|> ...
<|begin_of_solution|> ... <|end_of_solution|>`` structure that the
model should learn to emit.

We format each row as:

    <|system|>
    {system text}
    <|user|>
    {user text}
    <|assistant|>
    {assistant text — already contains the thought/solution markers}

Then a blank line between rows so SentencePiece doesn't blur them.

The grillcheese_spm32k_v2 tokenizer has the four chat tags
(<|system|>, <|user|>, <|assistant|>, <|tool|>) as single forced
tokens, so the chat structure compresses to 3 tokens of overhead per
row instead of 30+.

Run::

    python sandbox/mingru_baseline/extract_openthoughts.py \\
        --input  D:/grillcheese_training_data/jsonl/OpenThoughts-114k.jsonl \\
        --output D:/grillcheese_training_data/openthoughts_chat.txt

Then either:
  - Use as a standalone stage-1 corpus: --data-path openthoughts_chat.txt
  - Concat with c4 for a richer mix:
      cat allenai_c4_realnewslike.500m_tokens.txt openthoughts_chat.txt \\
          > stage1_lm_combined.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract(input_path: Path, output_path: Path,
            limit: int | None = None) -> dict:
    """Walk the JSONL, format each row, write to output. Skips rows
    missing any of the three role fields."""
    n_in = 0
    n_out = 0
    n_skip = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", errors="replace") as f, \
         output_path.open("w", encoding="utf-8") as out:
        for line in f:
            n_in += 1
            if limit is not None and n_out >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_skip += 1
                continue

            # OpenThoughts uses ShareGPT format: top-level ``system`` (str)
            # + ``conversations`` (list of {from, value} dicts where
            # ``from`` ∈ {"user", "human", "assistant", "gpt"}).
            system = (row.get("system") or "").strip()
            convs = row.get("conversations") or []
            if not isinstance(convs, list):
                n_skip += 1
                continue

            parts: list[str] = []
            if system:
                parts.append(f"<|system|>\n{system}")
            has_assistant = False
            for turn in convs:
                if not isinstance(turn, dict):
                    continue
                src = (turn.get("from") or "").lower()
                val = (turn.get("value") or "").strip()
                if not val:
                    continue
                if src in ("human", "user"):
                    parts.append(f"<|user|>\n{val}")
                elif src in ("assistant", "gpt", "model"):
                    parts.append(f"<|assistant|>\n{val}")
                    has_assistant = True
                elif src in ("system",):
                    # Some datasets put system in conversations instead of
                    # at the top level — handle both.
                    if not system:
                        parts.insert(0, f"<|system|>\n{val}")
                elif src in ("tool", "function"):
                    parts.append(f"<|tool|>\n{val}")
                # else: silently drop unknown roles

            if not has_assistant:
                n_skip += 1
                continue

            out.write("\n".join(parts))
            out.write("\n\n")  # blank line between rows so SPM doesn't blur
            n_out += 1
            if n_out % 10_000 == 0:
                print(f"  {n_out:>7,} rows written", file=sys.stderr)

    return {
        "rows_in":      n_in,
        "rows_written": n_out,
        "rows_skipped": n_skip,
        "output_size_mb": output_path.stat().st_size / 1e6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N rows (for smoke-testing the format)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    print(f"  input:  {args.input} ({args.input.stat().st_size/1e9:.2f} GB)")
    print(f"  output: {args.output}")

    stats = extract(args.input, args.output, limit=args.limit)
    print(f"\n  rows in:      {stats['rows_in']:,}")
    print(f"  rows written: {stats['rows_written']:,}")
    print(f"  rows skipped: {stats['rows_skipped']:,}")
    print(f"  output size:  {stats['output_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
