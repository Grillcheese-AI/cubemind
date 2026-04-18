#!/usr/bin/env python3
"""Post-process the raw Gemini factory output for training.

The factory emits one row per real instruction with Gemini-chosen
schema/rule names. On 4.4K rows we see ~3100 distinct schema names
(most appear once) which gives the multitask schema head near-zero
per-class learning signal.

This pass:

  1. Tallies schema + rule name frequencies in the input.
  2. Keeps the **top-(K-1) schemas** and **top-(M-1) rules** by
     frequency; remaps the rest to a single catch-all ``"other"``
     bucket at id ``K-1`` / ``M-1``.
  3. Rewrites ``schema_id`` / ``rule_id`` on each row accordingly.
     Leaves ``schema_name`` / ``rule_name`` text verbatim when it's
     in the top set; rewrites them to ``"other"`` when bucketed, so
     the label strings match the IDs.
  4. Leaves ``text`` and ``cubelang_program`` **unchanged** — those
     still carry Gemini's original name inside the tag block, which
     is useful rich signal for the LM head even when the schema head
     sees the bucketed label.
  5. Writes ``_clean.jsonl`` + a sidecar JSON metadata file listing
     kept classes, their counts, and the collision rate.

Run::

    python sandbox/mingru_baseline/postprocess_multitask.py \\
        --input  sandbox/mingru_baseline/data/multitask_gemini_v1.jsonl \\
        --output sandbox/mingru_baseline/data/multitask_gemini_v1_clean.jsonl \\
        --top-schemas 16 --top-rules 32

The default top-K matches the multitask config's ``num_schema_classes``
and ``num_rule_classes``. Re-run when the raw file grows (it's cheap).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tally_frequencies(path: Path) -> tuple[Counter, Counter, int]:
    """Count schema_name + rule_name occurrences."""
    schemas: Counter = Counter()
    rules: Counter = Counter()
    n = 0
    for row in _iter_rows(path):
        n += 1
        schemas[row.get("schema_name", "")] += 1
        rules[row.get("rule_name", "")] += 1
    return schemas, rules, n


def build_keep_map(freq: Counter, top_k: int, other_name: str = "other") -> tuple[dict[str, int], int]:
    """Assign IDs 0..top_k-2 to the top-(K-1) names; reserve K-1 for
    the catch-all ``other_name``. Returns the map + the other-id."""
    keep = {name: i for i, (name, _) in enumerate(freq.most_common(top_k - 1))}
    other_id = top_k - 1
    keep[other_name] = other_id
    return keep, other_id


def postprocess(
    input_path: Path,
    output_path: Path,
    top_schemas: int = 16,
    top_rules: int = 32,
    dedup: bool = True,
) -> dict:
    schemas, rules, total = tally_frequencies(input_path)
    schema_map, schema_other = build_keep_map(schemas, top_schemas)
    rule_map, rule_other = build_keep_map(rules, top_rules)

    seen_texts: set[str] = set() if dedup else set()
    dup_count = 0

    stats = {
        "total_in": total,
        "kept": 0,
        "duplicates_dropped": 0,
        "schema_bucketed_to_other": 0,
        "rule_bucketed_to_other": 0,
        "schema_kept_classes": {k: v for k, v in schemas.most_common(top_schemas - 1)},
        "rule_kept_classes":   {k: v for k, v in rules.most_common(top_rules - 1)},
        "top_schemas": top_schemas,
        "top_rules":   top_rules,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in _iter_rows(input_path):
            if dedup:
                key = row.get("text", "")
                if key in seen_texts:
                    dup_count += 1
                    continue
                seen_texts.add(key)

            sname = row.get("schema_name", "")
            if sname in schema_map:
                row["schema_id"] = schema_map[sname]
            else:
                row["schema_id"]   = schema_other
                row["schema_name"] = "other"
                stats["schema_bucketed_to_other"] += 1

            rname = row.get("rule_name", "")
            if rname in rule_map:
                row["rule_id"] = rule_map[rname]
            else:
                row["rule_id"]   = rule_other
                row["rule_name"] = "other"
                stats["rule_bucketed_to_other"] += 1

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    stats["duplicates_dropped"] = dup_count
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-schemas", type=int, default=16,
                    help="Keep the top-(K-1) schemas by frequency; others "
                         "go to id K-1 / name 'other'. Match "
                         "TrainConfig.num_schema_classes.")
    ap.add_argument("--top-rules", type=int, default=32,
                    help="Same for rules. Match num_rule_classes.")
    ap.add_argument("--no-dedup", action="store_true",
                    help="Skip text-level deduplication")
    args = ap.parse_args()

    stats = postprocess(
        args.input, args.output,
        top_schemas=args.top_schemas,
        top_rules=args.top_rules,
        dedup=not args.no_dedup,
    )
    print(f"  input:  {args.input} ({stats['total_in']:,} rows)")
    print(f"  output: {args.output} ({stats['kept']:,} rows kept)")
    print(f"  duplicates dropped: {stats['duplicates_dropped']:,}")
    print(f"  schema bucketed to 'other': {stats['schema_bucketed_to_other']:,} "
          f"({stats['schema_bucketed_to_other']/max(stats['kept'],1)*100:.1f}%)")
    print(f"  rule   bucketed to 'other': {stats['rule_bucketed_to_other']:,} "
          f"({stats['rule_bucketed_to_other']/max(stats['kept'],1)*100:.1f}%)")
    print(f"  top schemas kept: "
          f"{list(stats['schema_kept_classes'].items())[:5]}")
    print(f"  meta: {args.output.with_suffix(args.output.suffix + '.meta.json')}")


if __name__ == "__main__":
    main()
