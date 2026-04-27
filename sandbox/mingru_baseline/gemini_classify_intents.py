#!/usr/bin/env python3
"""Use Gemini to classify English root verbs into the 6 intent buckets
the multitask intent head trains on.

The Rust SVC emitter (``emit_multitask_jsonl.rs``) ships with a
hand-rolled ``verb_to_intent`` map covering ~80 verbs — everything
else falls back to ``inform`` (id 0). On the 329K SVC corpus that
means 83% of rows land in the fallback bucket, leaving the intent
head with very little class-balance signal.

This script fixes that by sending Gemini the **distinct verbs** (not
rows) and asking it to assign each to one of:

    0 inform     — declarative, explanation, definition, statement
    1 ask        — interrogative, query for information
    2 produce    — make / create / generate / write something new
    3 modify     — transform / convert / arithmetic / in-place change
    4 evaluate   — compare, analyze, classify, judge, rate
    5 recall     — memory access (read or write)

Output is a ``verb_intent_map.json`` ({verb: [name, id]}) which the
``apply_intent_map.py`` companion script (or a one-line invocation)
applies to every row of an existing multitask JSONL.

Cost model: top-1000 verbs covers ~98% of rows. With ~10 verbs per
batch and gemini-2.5-flash pricing this is well under $1.

Usage::

    python sandbox/mingru_baseline/gemini_classify_intents.py \\
        --input  sandbox/mingru_baseline/data/multitask_svc_v2.jsonl \\
        --output sandbox/mingru_baseline/data/verb_intent_map.json \\
        --top-verbs 1000 --model gemini-2.5-flash --batch-size 50
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

from google import genai  # noqa: E402

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("GEMINI_API_KEY not set — add it to cubemind/.env")
GENAI_CLIENT = genai.Client(api_key=API_KEY)


INTENT_DOCS = """
0 inform    — declarative, explanation, definition, statement (be, have, mean, describe, explain)
1 ask       — interrogative, query for information (find, search, lookup, ask, identify)
2 produce   — make / create / generate / write something new (create, build, write, design)
3 modify    — transform / convert / arithmetic / in-place change (transform, convert, add, scale)
4 evaluate  — compare, analyze, classify, judge, rate (compare, analyze, classify, predict)
5 recall    — memory access, read or write (store, save, recall, retrieve, remember)
"""

NAME_TO_ID = {
    "inform":   0,
    "ask":      1,
    "produce":  2,
    "modify":   3,
    "evaluate": 4,
    "recall":   5,
}


PROMPT_TEMPLATE = """You are labeling English root verbs with their high-level user-intent class.

The 6 intent classes (with examples):
{docs}

For each verb in the list below, pick exactly ONE intent class name.

Rules:
- Pick the most COMMON usage if a verb is ambiguous.
- "be", "have", "do" → "inform" (declarative copulas / generic verbs).
- Aux/modal verbs ("can", "will", "should") → "inform".
- Movement verbs that change state ("transfer", "move", "send") → "modify".
- Movement verbs that don't change state ("come", "go") → "inform".

Return ONE JSON object: {{"verb1": "name1", "verb2": "name2", ...}}
Only use these names: inform, ask, produce, modify, evaluate, recall.

VERBS:
{verbs}
"""


def collect_top_verbs(input_path: Path, top_k: int) -> list[tuple[str, int]]:
    """Return [(verb, count), ...] sorted by frequency, top-K."""
    counts: Counter = Counter()
    with io.open(str(input_path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = row.get("rule_name", "").strip().lower()
            if v:
                counts[v] += 1
    return counts.most_common(top_k)


def classify_batch(verbs: list[str], model: str) -> dict[str, str]:
    """One Gemini call for a batch of verbs. Returns {verb: name}.
    Verbs that can't be parsed are silently dropped."""
    prompt = PROMPT_TEMPLATE.format(
        docs=INTENT_DOCS.strip(),
        verbs="\n".join(f"- {v}" for v in verbs),
    )
    resp = GENAI_CLIENT.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json",
                "temperature": 0.0,
                "max_output_tokens": 4096},
    )
    raw = (resp.text or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to recover — Gemini sometimes wraps in code fences
        m = raw.find("{"); n = raw.rfind("}")
        if m >= 0 and n > m:
            parsed = json.loads(raw[m:n+1])
        else:
            print(f"  WARN: unparseable response, skipping batch: {raw[:120]!r}", file=sys.stderr)
            return {}
    out = {}
    for v in verbs:
        name = parsed.get(v) or parsed.get(v.lower())
        if isinstance(name, str) and name.lower() in NAME_TO_ID:
            out[v] = name.lower()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input",  type=Path, required=True,
                    help="Multitask JSONL with rule_name (= root verb) per row.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output verb_intent_map.json")
    ap.add_argument("--top-verbs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    print(f"  input: {args.input}")
    print(f"  collecting top-{args.top_verbs} verbs...")
    top = collect_top_verbs(args.input, args.top_verbs)
    if not top:
        sys.exit("no verbs found — check the rule_name field")
    coverage_total = sum(c for _, c in top)
    # Get total row count to compute coverage
    with io.open(str(args.input), "r", encoding="utf-8", errors="replace") as f:
        total_rows = sum(1 for _ in f)
    print(f"  top-{len(top)} verbs cover {coverage_total/total_rows*100:.1f}% of {total_rows:,} rows")

    verb_list = [v for v, _ in top]

    # Resume support — load existing map if present
    out_map: dict[str, list] = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            out_map = existing
            print(f"  resuming: {len(out_map):,} verbs already classified")
        except Exception:
            pass

    # Classify in batches, skipping verbs we already have
    pending = [v for v in verb_list if v not in out_map]
    print(f"  pending: {len(pending):,} verbs")
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for bi, start in enumerate(range(0, len(pending), args.batch_size)):
        batch = pending[start:start + args.batch_size]
        try:
            mapped = classify_batch(batch, args.model)
        except Exception as e:
            print(f"  batch {bi+1}/{n_batches} FAILED: {e}", file=sys.stderr)
            continue
        for v, name in mapped.items():
            out_map[v] = [name, NAME_TO_ID[name]]
        # Persist every batch so we can resume
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out_map, indent=2, sort_keys=True),
                               encoding="utf-8")
        elapsed = time.time() - t0
        print(f"  batch {bi+1}/{n_batches}: +{len(mapped)} mapped "
              f"(total {len(out_map):,}, {elapsed:.1f}s)")

    print(f"\n  done: {len(out_map):,} verbs in {args.output}")
    # Distribution summary
    dist: Counter = Counter()
    for v, (name, _) in out_map.items():
        dist[name] += sum(c for vv, c in top if vv == v)
    rows_classified = sum(dist.values())
    print(f"  rows covered by map: {rows_classified:,}")
    for name in ["inform", "ask", "produce", "modify", "evaluate", "recall"]:
        c = dist.get(name, 0)
        print(f"    {name:<10} {c:>8,}  {c/max(rows_classified,1)*100:>5.1f}%")


if __name__ == "__main__":
    main()
