#!/usr/bin/env python3
"""Scrub a combined multitask JSONL so all label IDs fit the trainer's
head sizes.

The Colab two-stage validation surfaced 4 issues with the raw
``multitask_combined.jsonl`` (Gemini cleaned + SVC with Gemini schemas):

  1. ``opcode_id`` out of range — SVC mapper occasionally emits ids
     >= 55 (the trainer's default opcode head size).
  2. ``schema_id`` out of range — Gemini classified into 21 schemas
     but the head is sized for 16.
  3. ``rule_id`` wildly out of range — 3,500+ distinct rules vs a
     32-class head. Most rules appear once; bucketing all but the top
     31 into "other" is the right move.
  4. ``intent_id`` missing on SVC rows — the field is required by the
     intent head but only the Gemini factory emits it.

Without this scrub, stage 2 dies on the first batch with
``Assertion 't >= 0 && t < n_classes' failed`` from
``nll_loss_forward_reduce_cuda_kernel_2d``.

This script does a 2-pass cleanup:

  * Pass 1 tallies ``schema_name`` / ``rule_name`` frequencies.
  * Pass 2 rewrites each row:
        - clamp ``opcode_id`` to ``[0, top_opcodes)`` (extras → 0)
        - remap schema/rule by name into top-K + "other" bucket
        - inject ``intent_id`` default (0) where missing

Run::

    python sandbox/mingru_baseline/scrub_multitask.py \\
        --input  /workspace/data/multitask_combined.jsonl \\
        --output /workspace/data/multitask_combined_clean.jsonl

The defaults match the trainer's head sizes in ``train_torch.py``
(``num_opcode_classes=55``, ``num_schema_classes=16``,
``num_rule_classes=32``, ``num_intent_classes=6``).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path


def _iter_rows(path: Path):
    with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tally(path: Path) -> tuple[Counter, Counter, int]:
    schemas: Counter = Counter()
    rules: Counter = Counter()
    n = 0
    for row in _iter_rows(path):
        n += 1
        schemas[row.get("schema_name", "")] += 1
        rules[row.get("rule_name", "")] += 1
    return schemas, rules, n


def keep_map(counter: Counter, top_k: int,
             other_name: str = "__OTHER__") -> dict[str, int]:
    """Top-(K-1) names get ids 0..K-2, the catch-all gets id K-1."""
    keep = {name: i for i, (name, _) in enumerate(counter.most_common(top_k - 1))}
    keep[other_name] = top_k - 1
    return keep


def scrub(input_path: Path, output_path: Path,
          top_opcodes: int, top_schemas: int, top_rules: int,
          default_intent_id: int) -> dict:
    schemas, rules, n_in = tally(input_path)
    schema_map = keep_map(schemas, top_schemas)
    rule_map   = keep_map(rules,   top_rules)

    n_out = clamped_op = bucketed_sch = bucketed_rule = added_intent = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(str(input_path), "r", encoding="utf-8", errors="replace") as fin, \
         io.open(str(output_path), "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            op = int(row.get("opcode_id", 0))
            if op >= top_opcodes or op < 0:
                row["opcode_id"] = 0
                clamped_op += 1

            s_name = row.get("schema_name", "__OTHER__")
            if s_name in schema_map:
                row["schema_id"] = schema_map[s_name]
            else:
                row["schema_id"] = schema_map["__OTHER__"]
                row["schema_name"] = "other"
                bucketed_sch += 1

            r_name = row.get("rule_name", "__OTHER__")
            if r_name in rule_map:
                row["rule_id"] = rule_map[r_name]
            else:
                row["rule_id"] = rule_map["__OTHER__"]
                row["rule_name"] = "other"
                bucketed_rule += 1

            if "intent_id" not in row:
                row["intent_id"] = default_intent_id
                added_intent += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    return {
        "rows_in":            n_in,
        "rows_out":           n_out,
        "schemas_distinct":   len(schemas),
        "rules_distinct":     len(rules),
        "opcode_clamped":     clamped_op,
        "schema_bucketed":    bucketed_sch,
        "rule_bucketed":      bucketed_rule,
        "intent_id_added":    added_intent,
        "output_size_mb":     output_path.stat().st_size / 1e6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input",  type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-opcodes",       type=int, default=55,
                    help="opcode_id >= this is clamped to 0 (NOOP).")
    ap.add_argument("--top-schemas",       type=int, default=16,
                    help="Top-(K-1) schema names kept; rest -> 'other'.")
    ap.add_argument("--top-rules",         type=int, default=32,
                    help="Top-(K-1) rule names kept; rest -> 'other'.")
    ap.add_argument("--default-intent-id", type=int, default=0,
                    help="Used when a row has no intent_id field.")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    print(f"  input:  {args.input} ({args.input.stat().st_size/1e9:.2f} GB)")
    print(f"  output: {args.output}")
    stats = scrub(
        args.input, args.output,
        top_opcodes=args.top_opcodes,
        top_schemas=args.top_schemas,
        top_rules=args.top_rules,
        default_intent_id=args.default_intent_id,
    )
    print(f"\n  rows in:           {stats['rows_in']:,}")
    print(f"  rows out:          {stats['rows_out']:,}")
    print(f"  distinct schemas:  {stats['schemas_distinct']:,}  -> top-{args.top_schemas-1} + other")
    print(f"  distinct rules:    {stats['rules_distinct']:,}  -> top-{args.top_rules-1} + other")
    print(f"  opcode clamped:    {stats['opcode_clamped']:,}")
    print(f"  schema bucketed:   {stats['schema_bucketed']:,}")
    print(f"  rule bucketed:     {stats['rule_bucketed']:,}")
    print(f"  intent_id added:   {stats['intent_id_added']:,}")
    print(f"  output size:       {stats['output_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
