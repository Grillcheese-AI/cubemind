#!/usr/bin/env python3
"""Classify the *subject era* of each book via Gemini.

Companion to ``build_temporal_corpus.py``. The temporal corpus uses
two distinct date tags:

  * ``[PUB:YYYY]``  — when the book was written (controls register / lens)
  * ``[SUBJ:YYYY]`` — what historical period the book *discusses*

This script generates the SUBJ side of that split. For each book in
``books-dir``, it sends the title + first ~1.5 KB of body text to
Gemini and asks "what historical period is this book primarily
about?". Output is a flat JSON map ``{filename: subject_era_string}``
that ``build_temporal_corpus.py --subj-map`` then applies during the
corpus build.

The format of ``subject_era_string`` is what Gemini returns under the
contract:
  * ``"YYYY"``           — single year focus
  * ``"YYYY-YYYY"``      — range
  * ``"-NNN"``            — BCE year (negative)
  * ``"multi"``           — encyclopedic / multi-period reference
  * ``"unknown"``         — Gemini couldn't determine (skipped from map)

Cost: ~243 books × ~600 input tokens × ~30 output tokens with
``gemini-3.1-flash-lite-preview`` runs at well under $1.

Run::

    python sandbox/mingru_baseline/classify_book_subjects.py \\
        --books-dir D:/grillcheese_training_data/knowledgetxt \\
        --output    D:/grillcheese_training_data/book_subj_map.json \\
        --batch-size 20 --model gemini-3.1-flash-lite-preview

Resumable: existing entries in ``--output`` are loaded and skipped.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

from google import genai  # noqa: E402

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("GEMINI_API_KEY not set — add it to cubemind/.env")
GENAI_CLIENT = genai.Client(api_key=API_KEY)


PROMPT = """You are labeling history/non-fiction books with the historical period each one PRIMARILY discusses.

For each book, return ONE of:
  * A single year as "YYYY" (e.g. "1066")
  * A year range as "YYYY-YYYY" (e.g. "1861-1865")
  * "multi" if the book is encyclopedic / spans many disconnected eras
  * "unknown" if the title and excerpt give no hint

Rules:
  * The PUBLICATION date of the book is irrelevant. We want the SUBJECT period.
  * For biographies, use the subject's lifespan (or main career period).
  * For period histories ("Civil War", "WWII"), use the standard date range.
  * For ancient/BCE periods use a negative year prefix (e.g. "-44" for Caesar's death).
  * Never invent dates — if uncertain, return "unknown".

Return ONE JSON object: {"filename1": "1861-1865", "filename2": "multi", ...}

BOOKS:
{items}"""


_TITLE_FROM_FILENAME = re.compile(r"\.txt$")


def _book_summary(text: str, max_chars: int = 1500) -> str:
    """First ~max_chars of book body, whitespace-collapsed."""
    snip = re.sub(r"\s+", " ", text[: max_chars * 2]).strip()
    return snip[:max_chars]


def classify_batch(items: list[tuple[str, str]], model: str) -> dict[str, str]:
    """One Gemini call for a batch. ``items`` is [(filename, prompt_chunk), ...]."""
    listing = "\n".join(
        f"--- {fname} ---\n{chunk}" for fname, chunk in items
    )
    prompt = PROMPT.replace("{items}", listing)
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
        m = raw.find("{"); n = raw.rfind("}")
        if m >= 0 and n > m:
            try: parsed = json.loads(raw[m:n+1])
            except json.JSONDecodeError: parsed = {}
        else:
            parsed = {}
    out: dict[str, str] = {}
    for fname, _ in items:
        v = parsed.get(fname)
        if not isinstance(v, str): continue
        v = v.strip()
        # Validate format: year, year-range, "multi", "unknown" (skipped)
        if v in ("multi",): out[fname] = v
        elif v == "unknown": continue
        elif re.fullmatch(r"-?\d{1,4}(-\-?\d{1,4})?", v): out[fname] = v
        # else: silently drop malformed entries
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--books-dir", type=Path, required=True)
    ap.add_argument("--output",    type=Path, required=True,
                    help="JSON map {filename: subject_era}")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-chars-per-book", type=int, default=1500)
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    args = ap.parse_args()

    if not args.books_dir.exists():
        sys.exit(f"books-dir not found: {args.books_dir}")

    files = sorted(args.books_dir.glob("*.txt"))
    print(f"  books-dir:   {args.books_dir}  ({len(files):,} files)")

    out_map: dict[str, str] = {}
    if args.output.exists():
        try:
            out_map = json.loads(args.output.read_text(encoding="utf-8"))
            print(f"  resuming:    {len(out_map):,} books already classified")
        except Exception:
            out_map = {}

    pending: list[tuple[str, str]] = []
    for path in files:
        if path.name in out_map:
            continue
        try:
            with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
                head = f.read(args.max_chars_per_book * 2)
        except Exception as e:
            print(f"  skip {path.name}: {e}", file=sys.stderr)
            continue
        title = _TITLE_FROM_FILENAME.sub("", path.name)
        snip = _book_summary(head, max_chars=args.max_chars_per_book)
        chunk = f"TITLE: {title}\nEXCERPT: {snip}"
        pending.append((path.name, chunk))

    print(f"  pending:     {len(pending):,} new classifications")
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for bi, start in enumerate(range(0, len(pending), args.batch_size)):
        batch = pending[start : start + args.batch_size]
        try:
            mapped = classify_batch(batch, args.model)
        except Exception as e:
            print(f"  batch {bi+1}/{n_batches} FAILED: {e}", file=sys.stderr)
            continue
        out_map.update(mapped)
        # Persist after every batch so the run is resumable.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out_map, indent=2, sort_keys=True),
                               encoding="utf-8")
        elapsed = time.time() - t0
        print(f"  batch {bi+1}/{n_batches}: +{len(mapped)} mapped "
              f"(total {len(out_map):,}, {elapsed:.0f}s)")

    print(f"\n  done: {len(out_map):,} books in {args.output}")
    # Era distribution
    from collections import Counter
    buckets = Counter()
    for v in out_map.values():
        if v == "multi": buckets["multi"] += 1
        elif "-" in v.lstrip("-"):
            try:
                start = int(v.split("-")[0] if not v.startswith("-")
                            else v.split("-", 2)[1] if v.count("-") >= 2 else v)
                century = (start // 100) * 100
                buckets[f"{century}s"] += 1
            except ValueError:
                buckets["unparsed"] += 1
        else:
            try:
                y = int(v); century = (y // 100) * 100
                buckets[f"{century}s"] += 1
            except ValueError:
                buckets["unparsed"] += 1
    for k, c in sorted(buckets.items()):
        print(f"    {k:<10} {c:>5}")


if __name__ == "__main__":
    main()
