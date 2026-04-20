#!/usr/bin/env python3
"""Build a temporally-tagged training corpus from multiple sources.

Aggregates three classes of historical text into a single ``.txt`` file
where every record is prefixed by a date or era marker so the model can
learn to associate calendar time with vocabulary, topics, and writing
style. The format is plain text (no special tokenizer changes required) —
the date tokens render as natural digits and word fragments that the
existing ``grillcheese_spm32k_v2`` tokenizer handles cleanly.

**Sources (configurable):**

1. **NYT archive JSONs** (``YYYY_M.json``) — NYT Article Search API
   format. Each file is a list of articles for one year/month. We use
   ``pub_date`` for the date tag and concatenate
   ``headline.main + section_name + lead_paragraph + snippet + abstract``
   as the body. Many pre-1990 articles have body-only-headline
   metadata; that's still useful (the headline language is era-distinct).

2. **Historical events JSONL** — pre-curated date-tagged event lists
   (e.g. ``historical_events_1800-1850.jsonl``). Each line is a single
   event dict with an ``event``/``description`` field and a year/date.

3. **Knowledge text books** — ``.txt`` chapters. No per-row dates, but
   we add an optional **era hint** parsed from the filename when one
   matches a year pattern (``1066 -``, ``1491_``, ``1492_``, etc.).
   Books without a parseable date stay untagged but still feed the
   corpus.

**Output format per record (one line, with explicit ``\\n\\n`` separator
between records so SentencePiece doesn't blur them):**

::

    [DATE:1852-01-15] [SEC:News] HEADLINE: Crystal Palace Lectures
    BODY: A series of lectures was given at the Crystal Palace, ...

    [DATE:1492] BOOK: 1493 - Uncovering the New World Columbus Cr
    Chapter 3: ... [book chapter text] ...

    [DATE:1066] HISTORICAL EVENT: Battle of Hastings, William the
    Conqueror defeats King Harold ...

The ``[DATE:YYYY...]`` token is plain text — no tokenizer regen needed.
The model learns the convention by repetition. ``YYYY`` for books and
events with year-only resolution; ``YYYY-MM-DD`` for NYT articles.

Run::

    python sandbox/mingru_baseline/build_temporal_corpus.py \\
        --nyt-dir       D:/grillcheese_training_data/temporal/nyt_data \\
        --events-glob   "D:/grillcheese_training_data/temporal/historical/historical_events_*.jsonl" \\
        --books-dir     D:/grillcheese_training_data/knowledgetxt \\
        --output        D:/grillcheese_training_data/temporal_corpus.txt \\
        --max-mb        4096

The ``--max-mb`` cap lets you sample a fixed budget across sources
proportional to their declared weights (defaults to 50% NYT / 10% events
/ 40% books).
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import re
import sys
from pathlib import Path

# Filename → year extractor for book hints. Matches:
#   "1492_ The Year ... .txt" → 1492
#   "1066 - Andrew Bridgeford.txt" → 1066
#   "10 Methods of Warfare.txt" → None (not a year)
_BOOK_YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-2][0-9])[\s_\-]")


def _extract_book_year(filename: str) -> int | None:
    """Pull a 4-digit year (1000-2029) from the leading filename token.
    Returns None if the filename doesn't start with one."""
    m = _BOOK_YEAR_RE.match(filename)
    if not m:
        return None
    y = int(m.group(1))
    return y if 1000 <= y <= 2029 else None


def _normalize_text(s: str, max_chars: int = 4000) -> str:
    """Collapse whitespace, strip, optionally truncate. Long passages
    truncated to ``max_chars`` to keep individual records manageable
    for the trainer's seq_len chunking."""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0]
    return s


# ==NYT articles ────────────────────────────────────────────────────────────

def emit_nyt(nyt_dir: Path, out, byte_budget: int) -> dict:
    """Walk every ``*.json`` (year/month bucket) under ``nyt_dir``, emit
    one record per article. Returns {n_files, n_articles_in, n_records,
    bytes_written}."""
    files = sorted(nyt_dir.glob("*.json"))
    n_files = n_in = n_out = bytes_w = 0
    for path in files:
        n_files += 1
        try:
            with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
                rows = json.load(f)
        except Exception as e:
            print(f"  skip NYT {path.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(rows, list):
            continue
        for art in rows:
            n_in += 1
            if not isinstance(art, dict):
                continue
            pub_date = (art.get("pub_date") or "")[:10]   # YYYY-MM-DD
            if not pub_date:
                continue
            head = (art.get("headline") or {}).get("main", "") if isinstance(art.get("headline"), dict) else ""
            head = _normalize_text(head, 200)
            sec  = _normalize_text(art.get("section_name", ""), 60)
            lead = _normalize_text(art.get("lead_paragraph", ""), 1500)
            snip = _normalize_text(art.get("snippet", ""), 500)
            abst = _normalize_text(art.get("abstract", ""), 500)
            # Skip empty-headline + empty-body articles (lots of these
            # in pre-1900 data — pure index entries).
            if not head and not lead and not snip and not abst:
                continue
            body_parts: list[str] = []
            if lead: body_parts.append(lead)
            if snip and snip not in (lead or ""): body_parts.append(snip)
            if abst and abst not in (lead or "") and abst not in (snip or ""):
                body_parts.append(abst)
            body = " ".join(body_parts)

            sec_tag = f" [SEC:{sec}]" if sec else ""
            head_tag = f" HEADLINE: {head}" if head else ""
            body_tag = f" BODY: {body}" if body else ""
            record = f"[DATE:{pub_date}]{sec_tag}{head_tag}{body_tag}\n\n"

            chunk = record.encode("utf-8")
            if bytes_w + len(chunk) > byte_budget:
                return {"n_files": n_files, "n_articles_in": n_in,
                        "n_records": n_out, "bytes_written": bytes_w,
                        "stopped_at": str(path.name)}
            out.write(chunk)
            bytes_w += len(chunk)
            n_out += 1
        if n_files % 50 == 0:
            print(f"    NYT {n_files} files, {n_out:,} articles, {bytes_w/1e6:.1f} MB",
                  file=sys.stderr)
    return {"n_files": n_files, "n_articles_in": n_in,
            "n_records": n_out, "bytes_written": bytes_w}


# ==Historical events JSONL ────────────────────────────────────────────────

def emit_events(globs: list[str], out, byte_budget: int) -> dict:
    """Each line is a dict with at least one date-ish field and one
    text field. We accept several common schemas and try them in
    order — robust to whichever generator wrote the file."""
    paths: list[Path] = []
    for g in globs:
        paths.extend(Path(p) for p in glob.glob(g))
    n_files = n_in = n_out = bytes_w = 0
    for path in paths:
        n_files += 1
        try:
            with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    n_in += 1
                    line = line.strip()
                    if not line: continue
                    try: row = json.loads(line)
                    except json.JSONDecodeError: continue
                    # Date: try several keys, take the first that yields a year.
                    year = None
                    for k in ("year", "date", "pub_date", "event_date", "timestamp"):
                        v = row.get(k)
                        if isinstance(v, int) and 1000 <= v <= 2029:
                            year = v; break
                        if isinstance(v, str):
                            m = re.search(r"(1[0-9]{3}|20[0-2][0-9])", v)
                            if m:
                                year = int(m.group(1)); break
                    text = ""
                    for k in ("event", "description", "text", "summary", "title"):
                        v = row.get(k)
                        if isinstance(v, str) and v.strip():
                            text = _normalize_text(v, 1500); break
                    if not text or year is None:
                        continue
                    record = f"[DATE:{year}] HISTORICAL EVENT: {text}\n\n"
                    chunk = record.encode("utf-8")
                    if bytes_w + len(chunk) > byte_budget:
                        return {"n_files": n_files, "n_lines_in": n_in,
                                "n_records": n_out, "bytes_written": bytes_w,
                                "stopped_at": str(path.name)}
                    out.write(chunk)
                    bytes_w += len(chunk)
                    n_out += 1
        except Exception as e:
            print(f"  skip events {path.name}: {e}", file=sys.stderr)
    return {"n_files": n_files, "n_lines_in": n_in,
            "n_records": n_out, "bytes_written": bytes_w}


# ==Knowledge text books ───────────────────────────────────────────────────

def emit_books(books_dir: Path, out, byte_budget: int,
               chunk_chars: int = 3000) -> dict:
    """Walk every ``*.txt`` under ``books_dir``, split each book into
    overlap-free ``chunk_chars``-sized chunks, emit each as a separate
    record. Books whose filename starts with a 4-digit year get a
    ``[DATE:YYYY]`` tag (e.g. ``1066 - Andrew Bridgeford.txt``); others
    are emitted as ``[BOOK]`` with no date — still useful as undated
    historical context."""
    files = sorted(books_dir.glob("*.txt"))
    n_files = n_chunks = n_dated = bytes_w = 0
    for path in files:
        n_files += 1
        year = _extract_book_year(path.name)
        title = _normalize_text(path.stem, 200)
        try:
            with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            print(f"  skip book {path.name}: {e}", file=sys.stderr)
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        date_tag = f"[DATE:{year}] " if year else "[BOOK] "
        if year:
            n_dated += 1
        for i in range(0, len(text), chunk_chars):
            piece = text[i : i + chunk_chars]
            if i + chunk_chars < len(text):
                # Avoid mid-word split
                piece = piece.rsplit(" ", 1)[0]
            record = f"{date_tag}TITLE: {title}\nBODY: {piece}\n\n"
            chunk = record.encode("utf-8")
            if bytes_w + len(chunk) > byte_budget:
                return {"n_files": n_files, "n_dated": n_dated,
                        "n_records": n_chunks, "bytes_written": bytes_w,
                        "stopped_at": path.name}
            out.write(chunk)
            bytes_w += len(chunk)
            n_chunks += 1
        if n_files % 100 == 0:
            print(f"    books {n_files} files, {n_chunks:,} chunks, "
                  f"{bytes_w/1e6:.1f} MB ({n_dated} dated)",
                  file=sys.stderr)
    return {"n_files": n_files, "n_dated": n_dated,
            "n_records": n_chunks, "bytes_written": bytes_w}


# ==Driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nyt-dir",     type=Path, required=True)
    ap.add_argument("--events-glob", action="append", default=[],
                    help="JSONL glob; can be passed multiple times")
    ap.add_argument("--books-dir",   type=Path, required=True)
    ap.add_argument("--output",      type=Path, required=True)
    ap.add_argument("--max-mb",      type=float, default=4096.0,
                    help="Total output size cap in MB (default 4 GB)")
    ap.add_argument("--nyt-frac",    type=float, default=0.50)
    ap.add_argument("--events-frac", type=float, default=0.10)
    ap.add_argument("--books-frac",  type=float, default=0.40)
    ap.add_argument("--book-chunk-chars", type=int, default=3000)
    args = ap.parse_args()

    s = args.nyt_frac + args.events_frac + args.books_frac
    if abs(s - 1.0) > 0.01:
        print(f"  WARN: source fractions sum to {s:.2f}, not 1.0", file=sys.stderr)
    total_bytes = int(args.max_mb * 1024 * 1024)
    nyt_budget    = int(total_bytes * args.nyt_frac)
    events_budget = int(total_bytes * args.events_frac)
    books_budget  = int(total_bytes * args.books_frac)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"  output:   {args.output}")
    print(f"  budget:   {args.max_mb:.0f} MB total")
    print(f"    NYT:    {nyt_budget/1e6:.0f} MB ({args.nyt_frac*100:.0f}%)")
    print(f"    events: {events_budget/1e6:.0f} MB ({args.events_frac*100:.0f}%)")
    print(f"    books:  {books_budget/1e6:.0f} MB ({args.books_frac*100:.0f}%)")
    print()

    summary: dict = {"sources": {}}
    with io.open(str(args.output), "wb") as out:
        if args.nyt_dir.exists():
            print("  == ingesting NYT archive ──")
            summary["sources"]["nyt"] = emit_nyt(args.nyt_dir, out, nyt_budget)
            print(f"    done: {summary['sources']['nyt']}")
        if args.events_glob:
            print("  == ingesting historical events ──")
            summary["sources"]["events"] = emit_events(args.events_glob, out, events_budget)
            print(f"    done: {summary['sources']['events']}")
        if args.books_dir.exists():
            print("  == ingesting knowledge books ──")
            summary["sources"]["books"] = emit_books(
                args.books_dir, out, books_budget,
                chunk_chars=args.book_chunk_chars)
            print(f"    done: {summary['sources']['books']}")

    summary["output"] = str(args.output)
    summary["total_bytes"] = args.output.stat().st_size

    sidecar = args.output.with_suffix(args.output.suffix + ".meta.json")
    sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.output} ({summary['total_bytes']/1e6:.1f} MB)")
    print(f"  meta  {sidecar}")


if __name__ == "__main__":
    main()
