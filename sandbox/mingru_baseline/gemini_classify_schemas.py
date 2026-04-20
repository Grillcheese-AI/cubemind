#!/usr/bin/env python3
"""Re-label SVC multitask rows with Gemini-assigned schemas.

The SVC emitter (opcode-vsa-rs) produces 330K multitask rows with
verb-derived schemas — too coarse. Rather than hand-crafting a better
schema derivation (or writing a local text classifier), this script
lets Gemini do the classification in bulk: each call sends a batch of
100 instruction texts plus the canonical schema list and asks Gemini
to return one schema name per instruction.

That's much cheaper than the full annotation mode — each row costs
~15 input tokens + ~5 output tokens, so 330K rows fits in a few
dollars of Gemini credits with gemini-3-flash-preview.

Pipeline:

  1. Derive the canonical schema list from the Gemini factory's
     existing output (top-K by frequency). Pinned so schema_id is
     stable across runs.
  2. Stream the SVC input JSONL.
  3. Batch N rows per call; send (schema_list, instruction_texts) to
     Gemini; expect a JSON array of schema_name strings back.
  4. Rewrite each row's schema_name + schema_id with the Gemini pick
     (fall back to "other" if Gemini returns a name not in the list).
  5. Append to output JSONL, flush each batch. Resumable.

Usage::

    python sandbox/mingru_baseline/gemini_classify_schemas.py \\
        --gemini-factory sandbox/mingru_baseline/data/multitask_gemini_v1.jsonl \\
        --input  sandbox/mingru_baseline/data/multitask_svc_v1.jsonl \\
        --output sandbox/mingru_baseline/data/multitask_svc_v1_gemini_schemas.jsonl \\
        --model  gemini-3-flash-preview \\
        --workers 6 --rows-per-call 100 --top-schemas 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

from google import genai  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("GEMINI_API_KEY not set — add it to cubemind/.env")
GENAI_CLIENT = genai.Client(api_key=API_KEY)


INSTR_RE = re.compile(r"<INSTR>(.*?)</INSTR>", re.DOTALL)


def _extract_instruction(row: dict) -> str:
    """Pull the natural-language instruction out of a row's text field.
    Falls back to the whole text minus tag markup."""
    text = row.get("text", "")
    m = INSTR_RE.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return re.sub(r"<[^>]+>", " ", text).strip()


# ── Schema vocabulary derivation ────────────────────────────────────────

def derive_schema_vocab(gemini_path: Path, top_k: int) -> list[str]:
    """Return the top-K schemas from the Gemini factory output.

    ``top_k - 1`` real names + ``"other"`` catch-all at index ``top_k - 1``.
    The position in the list IS the schema_id, so this list must be
    persisted alongside the dataset (we write it to the meta file).
    """
    counts: Counter = Counter()
    with gemini_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = r.get("schema_name", "")
            if name and name != "other":
                counts[name] += 1
    top = [n for n, _ in counts.most_common(top_k - 1)]
    top.append("other")  # reserved catch-all at the end
    return top


def build_prompt(schemas: list[str], instructions: list[str]) -> str:
    """Per-call prompt: canonical schemas + numbered instructions."""
    schema_bullets = "\n".join(f"  - {s}" for s in schemas)
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(instructions))
    return f"""\
Classify each instruction into exactly one of these schemas. Return a
JSON array with ONE schema name per instruction, in the same order,
same length as the input. Use "other" when nothing fits well.

Canonical schema list (use these exact names, case-sensitive):
{schema_bullets}

Instructions:
{numbered}

Output a JSON array of {len(instructions)} strings. No commentary, no
markdown fences.
"""


# ── Async worker pool ───────────────────────────────────────────────────

async def _classify_batch(model_name: str, schemas: list[str],
                          instructions: list[str],
                          retries: int = 4) -> list[str] | None:
    """One Gemini call → list of schema names (length == len(instructions))."""
    prompt = build_prompt(schemas, instructions)
    cfg = genai_types.GenerateContentConfig(
        temperature=0.0,   # classification — deterministic
        top_p=1.0,
        max_output_tokens=4096,
        response_mime_type="application/json",
    )
    backoff = 2.0
    last_err = None
    for attempt in range(retries):
        try:
            resp = await asyncio.to_thread(
                GENAI_CLIENT.models.generate_content,
                model=model_name,
                contents=prompt,
                config=cfg,
            )
            text = (resp.text or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError(f"expected array, got {type(data).__name__}")
            if len(data) != len(instructions):
                raise ValueError(
                    f"returned {len(data)} labels for {len(instructions)} inputs")
            return [str(x) for x in data]
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2.0
    print(f"  WARN batch failed after {retries}: {last_err}", file=sys.stderr)
    return None


def _iter_input_rows(path: Path, skip: int):
    """Yield rows past the ``skip``-th line (already-written on resume)."""
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if n < skip:
                n += 1
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


async def _producer(
    input_path: Path,
    output_path: Path,
    schemas: list[str],
    schema_to_id: dict[str, int],
    model_name: str,
    workers: int,
    rows_per_call: int,
):
    already = _count_existing(output_path)
    if already:
        print(f"  resume: {already:,} rows already in {output_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = output_path.open("a", encoding="utf-8")

    sem = asyncio.Semaphore(workers)
    other_id = schema_to_id["other"]
    hist: Counter = Counter()
    t0 = time.time()
    written = already
    row_iter = _iter_input_rows(input_path, skip=already)
    exhausted = False

    async def _one_batch(batch_rows: list[dict]):
        nonlocal written
        async with sem:
            texts = [_extract_instruction(r) for r in batch_rows]
            labels = await _classify_batch(model_name, schemas, texts)
        if labels is None:
            return 0
        for row, lab in zip(batch_rows, labels):
            if lab in schema_to_id:
                row["schema_name"] = lab
                row["schema_id"] = schema_to_id[lab]
            else:
                row["schema_name"] = "other"
                row["schema_id"] = other_id
            hist[row["schema_name"]] += 1
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        written += len(batch_rows)
        return len(batch_rows)

    in_flight: list[asyncio.Task] = []
    while True:
        # Refill up to 2× workers of outstanding batches.
        while len(in_flight) < workers * 2 and not exhausted:
            batch: list[dict] = []
            for _ in range(rows_per_call):
                try:
                    batch.append(next(row_iter))
                except StopIteration:
                    exhausted = True
                    break
            if not batch:
                break
            in_flight.append(asyncio.create_task(_one_batch(batch)))
        if not in_flight:
            break
        done, pending = await asyncio.wait(
            in_flight, return_when=asyncio.FIRST_COMPLETED)
        in_flight = list(pending)
        elapsed = time.time() - t0
        rate = (written - already) / max(elapsed, 1.0)
        top3 = hist.most_common(3)
        print(f"  [{written:>7,}]  {rate:.1f} rows/s  "
              f"elapsed {elapsed/60:.1f}m  top3: {top3}")

    out_f.close()
    return written, hist


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gemini-factory", type=Path, required=True,
                    help="Gemini factory JSONL — source of the canonical schema list")
    ap.add_argument("--input", type=Path, required=True,
                    help="Multitask JSONL to re-label (e.g. SVC emitter output)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="gemini-2.5-flash is the workhorse here — "
                         "gemini-3-flash-preview reliably truncates the "
                         "classification response at rows_per_call>=50.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rows-per-call", type=int, default=100)
    ap.add_argument("--top-schemas", type=int, default=16,
                    help="Keep top-(K-1) schemas from Gemini factory "
                         "+ 'other' catch-all at id K-1")
    args = ap.parse_args()

    print(f"  deriving schema vocab from {args.gemini_factory}")
    schemas = derive_schema_vocab(args.gemini_factory, args.top_schemas)
    schema_to_id = {name: i for i, name in enumerate(schemas)}
    print(f"  {len(schemas)} canonical schemas (last one is 'other'):")
    for i, s in enumerate(schemas):
        print(f"    {i:>2}: {s}")
    print(f"\n  model: {args.model}  workers: {args.workers}  "
          f"rows/call: {args.rows_per_call}\n")

    # Persist the pinned schema list alongside the output so downstream
    # trainers know the exact schema_id → name mapping.
    meta = {
        "schemas": schemas,
        "schema_to_id": schema_to_id,
        "source_gemini_file": str(args.gemini_factory),
        "source_svc_file": str(args.input),
        "model": args.model,
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".schema_map.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    written, hist = asyncio.run(_producer(
        input_path=args.input,
        output_path=args.output,
        schemas=schemas,
        schema_to_id=schema_to_id,
        model_name=args.model,
        workers=args.workers,
        rows_per_call=args.rows_per_call,
    ))
    print(f"\n  total rows: {written:,}")
    print("  final distribution:")
    total = sum(hist.values())
    for name, ct in hist.most_common():
        pct = ct / max(total, 1) * 100
        print(f"    {name:30s} {ct:>7}  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
