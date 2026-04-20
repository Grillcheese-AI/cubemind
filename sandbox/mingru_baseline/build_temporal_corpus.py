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


# Content-based publication-year fallback. Patterns are intentionally
# restricted to *explicit publication markers* (copyright lines, "First
# published in YYYY", etc.) so encyclopedias and history books — which
# mention hundreds of historical years like "in 1875, the Civil War..."
# — don't get falsely tagged with their *content* dates. We only match
# years that appear inside one of these phrases.
#
# We pick the MOST RECENT year matched: revisions and reprints push
# the publication date forward, so taking max() across hits captures
# the latest edition's date.
_CONTENT_YEAR_PATTERNS = [
    re.compile(r"(?:©|\(c\)|Copyright)\s+(?:\d{4}\s*[-,]\s*)?(\d{4})", re.IGNORECASE),
    re.compile(r"First\s+published\s+(?:in\s+)?(\d{4})", re.IGNORECASE),
    re.compile(r"First\s+edition\s+(?:in\s+)?(\d{4})", re.IGNORECASE),
    re.compile(r"Originally\s+published\s+(?:in\s+)?(\d{4})", re.IGNORECASE),
    re.compile(r"Published\s+(?:in\s+)?(\d{4})", re.IGNORECASE),
    re.compile(r"Printed\s+in\s+(?:\w+\s+)?(\d{4})", re.IGNORECASE),
    re.compile(r"This\s+edition\s+(?:was\s+)?(?:first\s+)?published\s+(?:in\s+)?(\d{4})", re.IGNORECASE),
    re.compile(r"All\s+rights\s+reserved\.?\s*[\(\)\s\-,c©]*(\d{4})", re.IGNORECASE),
    re.compile(r"ISBN[\s:-]*[\d\-X]+\s*[\(\)\s,]*(\d{4})", re.IGNORECASE),
]


def _extract_content_year(text: str,
                          head_chars: int = 16000,
                          tail_chars: int = 6000) -> int | None:
    """Scan first ``head_chars`` + last ``tail_chars`` of book text for
    explicit publication markers. Front matter usually carries the
    copyright line; back matter often repeats it (publisher info,
    'About the Author', reprint history). Returns the most recent year
    found across all patterns, range-limited to 1500-2029.
    """
    if not text:
        return None
    head = text[:head_chars]
    tail = text[-tail_chars:] if len(text) > head_chars else ""
    haystack = head + "\n" + tail
    candidates: list[int] = []
    for pat in _CONTENT_YEAR_PATTERNS:
        for m in pat.finditer(haystack):
            try:
                y = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if 1500 <= y <= 2029:
                candidates.append(y)
    if not candidates:
        return None
    return max(candidates)


# Subject-era extraction (B). Books frequently encode the period they
# discuss in the title — these tags let the model distinguish between
# the publication date (when written = register/style/lens) and the
# subject date (what's being discussed = topics, named entities).
#
# Format: subj is a single year or a "YYYY-YYYY" range. Picked via
# (1) explicit 4-digit year tokens in the title, then (2) era keyword
# match against the table below. Title text comes from filename stem
# minus author suffix (everything after " - " is treated as the byline).
SUBJ_ERA_KEYWORDS: dict[str, str] = {
    # Antiquity
    "ancient egypt":     "-3000-30",
    "pharaoh":           "-3000-30",
    "ancient greece":    "-800--146",
    "athens":            "-800--146",
    "sparta":            "-800--146",
    "ancient rome":      "-753-476",
    "roman empire":      "-27-476",
    "roman republic":    "-509--27",
    "byzantine":         "330-1453",
    "constantinople":    "330-1453",

    # Medieval / early modern
    "vikings":           "793-1066",
    "viking":            "793-1066",
    "norman":            "1066-1154",
    "crusade":           "1095-1291",
    "medieval":          "500-1500",
    "middle ages":       "500-1500",
    "renaissance":       "1400-1600",
    "tudor":             "1485-1603",
    "elizabethan":       "1558-1603",
    "reformation":       "1517-1648",
    "thirty years":      "1618-1648",  # War
    "enlightenment":     "1685-1815",

    # Conflicts (most-mentioned in this corpus)
    "american revolution":  "1775-1783",
    "revolutionary war":    "1775-1783",
    "founding":             "1775-1789",   # Founding Brothers, etc.
    "war of 1812":          "1812-1815",
    "napoleon":             "1799-1815",
    "civil war":            "1861-1865",   # American — most common context
    "antebellum":           "1820-1861",
    "reconstruction":       "1865-1877",
    "wild west":            "1865-1895",
    "gilded age":           "1870-1900",
    "victorian":            "1837-1901",
    "world war i":          "1914-1918",
    "world war 1":          "1914-1918",
    "world war one":        "1914-1918",
    "great war":            "1914-1918",
    "wwi":                  "1914-1918",
    "russian revolution":   "1917-1923",
    "interwar":             "1919-1939",
    "weimar":               "1919-1933",
    "great depression":     "1929-1939",
    "world war ii":         "1939-1945",
    "world war 2":          "1939-1945",
    "world war two":        "1939-1945",
    "wwii":                 "1939-1945",
    "third reich":          "1933-1945",
    "nazi":                 "1933-1945",
    "holocaust":            "1933-1945",
    "battle of britain":    "1940",
    "cold war":             "1947-1991",
    "korean war":           "1950-1953",
    "vietnam":              "1955-1975",
    "civil rights":         "1954-1968",
    "gulf war":             "1990-1991",
    "9/11":                 "2001",
    "iraq war":             "2003-2011",
    "afghanistan war":      "2001-2021",

    # Periods / civilizations / discoveries
    "columbus":             "1492-1506",
    "conquistador":         "1492-1600",
    "new world":            "1492-1607",
    "colonial america":     "1607-1776",
    "early america":        "1776-1815",
    "industrial revolution": "1760-1840",
    "atomic age":           "1945-1991",
    "space race":           "1957-1975",
    "internet":             "1990-2025",
    "21st century":         "2001-2025",
    "20th century":         "1901-2000",
    "19th century":         "1801-1900",
    "18th century":         "1701-1800",
    "17th century":         "1601-1700",
}


# 4-digit year tokens in the title. Same range as the filename matcher
# (1000-2029), but we accept anywhere in the title rather than only the
# leading position. Returns ALL matches so the caller can pick (e.g.
# "1491 / 1492 / 1493" trilogy → range).
_TITLE_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def _extract_title_subj(title_or_filename: str) -> str | None:
    """Return a SUBJ tag value (single year or 'YYYY-YYYY' range) from
    a book title. Tries (1) explicit 4-digit year(s), then (2) era
    keyword match. None if nothing recognized."""
    if not title_or_filename:
        return None
    # Strip author suffix (everything after " - ") — that's the byline,
    # not the subject. Same for ".txt" extension.
    title = title_or_filename
    if " - " in title:
        title = title.split(" - ", 1)[0]
    title = title.replace(".txt", "").replace("_", " ").strip()

    # 1. Year tokens — pick min/max if multiple, else single year.
    years = [int(y) for y in _TITLE_YEAR_RE.findall(title)
             if 1000 <= int(y) <= 2029]
    if years:
        if len(years) == 1:
            return str(years[0])
        return f"{min(years)}-{max(years)}"

    # 2. Era keywords — match longest keyword first to avoid false
    # positives like "WWII in Europe" matching the WWI pattern (which
    # is a substring of WWII). Lowercase the title once.
    title_lc = title.lower()
    for kw in sorted(SUBJ_ERA_KEYWORDS, key=len, reverse=True):
        if kw in title_lc:
            return SUBJ_ERA_KEYWORDS[kw]
    return None


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
            # NYT articles are primary sources — publication date IS the
            # subject date. Emit both tags so the model learns the
            # equivalence and downstream prompts work with either.
            year_only = pub_date[:4]
            record = (f"[PUB:{pub_date}] [SUBJ:{year_only}]"
                      f"{sec_tag}{head_tag}{body_tag}\n\n")

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
                    # Includes the schema used by the historical_events_*.jsonl
                    # files in temporal/historical/ which key dates as
                    # earliest_date_year + latest_date_year.
                    year = None
                    for k in ("year", "date", "pub_date", "event_date", "timestamp",
                              "earliest_date_year", "latest_date_year",
                              "earliestDateYear"):
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
                    # Historical events: the date IS the subject date
                    # (when the event happened). The publication date of
                    # the JSONL itself isn't meaningful — these are
                    # post-hoc curated. So emit only [SUBJ:].
                    record = f"[SUBJ:{year}] HISTORICAL EVENT: {text}\n\n"
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
               chunk_chars: int = 3000,
               subj_map: dict[str, str] | None = None) -> dict:
    """Walk every ``*.txt`` under ``books_dir``, split each book into
    overlap-free ``chunk_chars``-sized chunks, emit each as a separate
    record.

    Tagging scheme (post pub/subj split):
      * ``[PUB:YYYY]`` from filename-leading year OR content scan
        (copyright/publication markers). Modern revisions push this
        forward — represents the writing era.
      * ``[SUBJ:YYYY]`` or ``[SUBJ:YYYY-YYYY]`` from:
            (1) ``subj_map`` lookup by filename (Gemini-classified),
            (2) title-hint extraction (4-digit years or era keywords),
        in that priority. None if neither yields a result.

    Books with neither tag emit as bare ``[BOOK]`` — still feeds the
    corpus as undated historical context.
    """
    files = sorted(books_dir.glob("*.txt"))
    n_files = n_chunks = n_pub = n_pub_fn = n_pub_ct = 0
    n_subj = n_subj_map = n_subj_title = bytes_w = 0
    for path in files:
        n_files += 1
        # Publication date — filename leading year first, then content scan.
        pub_year = _extract_book_year(path.name)
        pub_source = "filename" if pub_year else None
        try:
            with io.open(str(path), "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            print(f"  skip book {path.name}: {e}", file=sys.stderr)
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if pub_year is None:
            pub_year = _extract_content_year(text)
            if pub_year is not None:
                pub_source = "content"

        # Subject era — priority is Gemini map (--subj-map) over title hints.
        subj_era: str | None = None
        subj_source: str | None = None
        if subj_map is not None and path.name in subj_map:
            subj_era = subj_map[path.name]
            if subj_era:
                subj_source = "map"
        if subj_era is None:
            subj_era = _extract_title_subj(path.name)
            if subj_era is not None:
                subj_source = "title"

        title = _normalize_text(path.stem, 200)

        # Build the tag prefix — PUB and SUBJ are independent.
        tag_parts: list[str] = []
        if pub_year:
            tag_parts.append(f"[PUB:{pub_year}]")
            n_pub += 1
            if pub_source == "filename": n_pub_fn += 1
            else: n_pub_ct += 1
        if subj_era:
            tag_parts.append(f"[SUBJ:{subj_era}]")
            n_subj += 1
            if subj_source == "map":   n_subj_map += 1
            else:                       n_subj_title += 1
        if not tag_parts:
            tag_parts.append("[BOOK]")
        date_tag = " ".join(tag_parts) + " "
        for i in range(0, len(text), chunk_chars):
            piece = text[i : i + chunk_chars]
            if i + chunk_chars < len(text):
                # Avoid mid-word split
                piece = piece.rsplit(" ", 1)[0]
            record = f"{date_tag}TITLE: {title}\nBODY: {piece}\n\n"
            chunk = record.encode("utf-8")
            if bytes_w + len(chunk) > byte_budget:
                return {"n_files": n_files,
                        "n_pub": n_pub, "n_pub_filename": n_pub_fn,
                        "n_pub_content": n_pub_ct,
                        "n_subj": n_subj, "n_subj_map": n_subj_map,
                        "n_subj_title": n_subj_title,
                        "n_records": n_chunks, "bytes_written": bytes_w,
                        "stopped_at": path.name}
            out.write(chunk)
            bytes_w += len(chunk)
            n_chunks += 1
        if n_files % 100 == 0:
            print(f"    books {n_files} files, {n_chunks:,} chunks, "
                  f"{bytes_w/1e6:.1f} MB (pub {n_pub}: {n_pub_fn} fn / "
                  f"{n_pub_ct} ct, subj {n_subj}: {n_subj_map} map / "
                  f"{n_subj_title} title)",
                  file=sys.stderr)
    return {"n_files": n_files,
            "n_pub": n_pub, "n_pub_filename": n_pub_fn,
            "n_pub_content": n_pub_ct,
            "n_subj": n_subj, "n_subj_map": n_subj_map,
            "n_subj_title": n_subj_title,
            "n_records": n_chunks, "bytes_written": bytes_w}


# ==Project Gutenberg (factual/) ───────────────────────────────────────────

# Subject field date patterns from Library of Congress headings used in
# Gutenberg metadata. Examples seen in the corpus:
#   "World War, 1914-1918"
#   "United States -- History -- Civil War, 1861-1865"
#   "France -- History -- Revolution, 1789-1799"
#   "Verplanck, Gulian C. (Gulian Crommelin), 1786-1870"  (biographical — author dates)
_SUBJECT_RANGE_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\s*[-‐–—]\s*(1[0-9]{3}|20[0-2][0-9])\b")
_SUBJECT_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def _extract_subj_from_subjects(subjects: list) -> str | None:
    """Subjects field is LoC-style — often contains explicit date ranges
    or single years (e.g. "Civil War, 1861-1865"). Returns the first
    range or the min..max of single years across all subjects."""
    if not isinstance(subjects, list):
        return None
    for s in subjects:
        if not isinstance(s, str):
            continue
        m = _SUBJECT_RANGE_RE.search(s)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    # Fall back: single years across all subject strings
    years: list[int] = []
    for s in subjects:
        if not isinstance(s, str):
            continue
        for y_str in _SUBJECT_YEAR_RE.findall(s):
            y = int(y_str)
            if 1500 <= y <= 2029:
                years.append(y)
    if not years:
        return None
    if len(set(years)) == 1:
        return str(years[0])
    return f"{min(years)}-{max(years)}"


def _extract_pub_from_authors(authors: list) -> int | None:
    """For Gutenberg public-domain books, the author's death year is a
    reasonable upper bound on publication date. Returns the latest
    death_year across all authors, or birth_year + 30 as fallback."""
    if not isinstance(authors, list):
        return None
    deaths: list[int] = []
    births: list[int] = []
    for a in authors:
        if not isinstance(a, dict):
            continue
        d = a.get("death_year")
        if isinstance(d, int) and 1500 <= d <= 2029:
            deaths.append(d)
        b = a.get("birth_year")
        if isinstance(b, int) and 1500 <= b <= 2029:
            births.append(b)
    if deaths:
        return max(deaths)
    if births:
        # Mid-career estimate when death is unknown (still-living
        # author's birth + 30y rule of thumb).
        return max(births) + 30
    return None


def emit_factual(factual_dir: Path, out, byte_budget: int,
                 chunk_chars: int = 3000) -> dict:
    """Walk every ``*_metadata.json`` under ``factual_dir``, pair with
    its sibling ``<id>.txt``, extract PUB (author dates) + SUBJ
    (subjects field LoC dates), chunk the text, and emit records.

    Returns counts with the same shape as ``emit_books`` so the summary
    JSON has consistent fields across both book sources.
    """
    meta_files = sorted(factual_dir.glob("*_metadata.json"))
    n_files = n_chunks = n_pub = n_subj = bytes_w = 0
    for meta_path in meta_files:
        n_files += 1
        text_path = meta_path.with_name(meta_path.name.replace("_metadata.json", ".txt"))
        if not text_path.exists():
            continue
        try:
            with io.open(str(meta_path), "r", encoding="utf-8", errors="replace") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"  skip factual meta {meta_path.name}: {e}", file=sys.stderr)
            continue
        try:
            with io.open(str(text_path), "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            print(f"  skip factual text {text_path.name}: {e}", file=sys.stderr)
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        title = _normalize_text(meta.get("title", ""), 200)
        pub_year = _extract_pub_from_authors(meta.get("authors", []))
        subj_era = _extract_subj_from_subjects(meta.get("subjects", []))
        # If no SUBJ from LoC subjects, try title-hint extraction
        # (catches era keywords like "Civil War" in the title itself).
        if subj_era is None:
            subj_era = _extract_title_subj(title)

        tag_parts: list[str] = []
        if pub_year:
            tag_parts.append(f"[PUB:{pub_year}]")
            n_pub += 1
        if subj_era:
            tag_parts.append(f"[SUBJ:{subj_era}]")
            n_subj += 1
        if not tag_parts:
            tag_parts.append("[BOOK]")
        date_tag = " ".join(tag_parts) + " "

        for i in range(0, len(text), chunk_chars):
            piece = text[i : i + chunk_chars]
            if i + chunk_chars < len(text):
                piece = piece.rsplit(" ", 1)[0]
            record = f"{date_tag}TITLE: {title}\nBODY: {piece}\n\n"
            chunk = record.encode("utf-8")
            if bytes_w + len(chunk) > byte_budget:
                return {"n_files": n_files, "n_pub": n_pub,
                        "n_subj": n_subj, "n_records": n_chunks,
                        "bytes_written": bytes_w,
                        "stopped_at": meta_path.name}
            out.write(chunk)
            bytes_w += len(chunk)
            n_chunks += 1
        if n_files % 200 == 0:
            print(f"    factual {n_files} files, {n_chunks:,} chunks, "
                  f"{bytes_w/1e6:.1f} MB ({n_pub} pub / {n_subj} subj)",
                  file=sys.stderr)
    return {"n_files": n_files, "n_pub": n_pub, "n_subj": n_subj,
            "n_records": n_chunks, "bytes_written": bytes_w}


# ==Driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nyt-dir",     type=Path, required=True)
    ap.add_argument("--events-glob", action="append", default=[],
                    help="JSONL glob; can be passed multiple times")
    ap.add_argument("--books-dir",   type=Path, required=True)
    ap.add_argument("--factual-dir", type=Path, default=None,
                    help="Project Gutenberg-style factual/ dir with paired "
                         "<id>.txt + <id>_metadata.json files")
    ap.add_argument("--subj-map",    type=Path, default=None,
                    help="Optional JSON map {filename: subject_era_string} "
                         "from a Gemini classifier (see "
                         "classify_book_subjects.py). Takes priority over "
                         "title-hint extraction for books in the map.")
    ap.add_argument("--output",      type=Path, required=True)
    ap.add_argument("--max-mb",      type=float, default=6144.0,
                    help="Total output size cap in MB (default 6 GB to "
                         "fit all four sources)")
    ap.add_argument("--nyt-frac",     type=float, default=0.40)
    ap.add_argument("--events-frac",  type=float, default=0.05)
    ap.add_argument("--books-frac",   type=float, default=0.30)
    ap.add_argument("--factual-frac", type=float, default=0.25)
    ap.add_argument("--book-chunk-chars", type=int, default=3000)
    args = ap.parse_args()

    s = args.nyt_frac + args.events_frac + args.books_frac + args.factual_frac
    if abs(s - 1.0) > 0.01:
        print(f"  WARN: source fractions sum to {s:.2f}, not 1.0", file=sys.stderr)
    total_bytes = int(args.max_mb * 1024 * 1024)
    nyt_budget     = int(total_bytes * args.nyt_frac)
    events_budget  = int(total_bytes * args.events_frac)
    books_budget   = int(total_bytes * args.books_frac)
    factual_budget = int(total_bytes * args.factual_frac)

    # Optional Gemini subj_map for books
    subj_map: dict[str, str] | None = None
    if args.subj_map and args.subj_map.exists():
        subj_map = json.loads(args.subj_map.read_text(encoding="utf-8"))
        print(f"  loaded subj_map: {len(subj_map):,} entries from {args.subj_map}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"  output:   {args.output}")
    print(f"  budget:   {args.max_mb:.0f} MB total")
    print(f"    NYT:     {nyt_budget/1e6:.0f} MB ({args.nyt_frac*100:.0f}%)")
    print(f"    events:  {events_budget/1e6:.0f} MB ({args.events_frac*100:.0f}%)")
    print(f"    books:   {books_budget/1e6:.0f} MB ({args.books_frac*100:.0f}%)")
    print(f"    factual: {factual_budget/1e6:.0f} MB ({args.factual_frac*100:.0f}%)")
    print()

    summary: dict = {"sources": {}}
    with io.open(str(args.output), "wb") as out:
        if args.nyt_dir.exists():
            print("  == ingesting NYT archive ==")
            summary["sources"]["nyt"] = emit_nyt(args.nyt_dir, out, nyt_budget)
            print(f"    done: {summary['sources']['nyt']}")
        if args.events_glob:
            print("  == ingesting historical events ==")
            summary["sources"]["events"] = emit_events(args.events_glob, out, events_budget)
            print(f"    done: {summary['sources']['events']}")
        if args.books_dir.exists():
            print("  == ingesting knowledge books ==")
            summary["sources"]["books"] = emit_books(
                args.books_dir, out, books_budget,
                chunk_chars=args.book_chunk_chars,
                subj_map=subj_map)
            print(f"    done: {summary['sources']['books']}")
        if args.factual_dir is not None and args.factual_dir.exists():
            print("  == ingesting Project Gutenberg (factual/) ==")
            summary["sources"]["factual"] = emit_factual(
                args.factual_dir, out, factual_budget,
                chunk_chars=args.book_chunk_chars)
            print(f"    done: {summary['sources']['factual']}")

    summary["output"] = str(args.output)
    summary["total_bytes"] = args.output.stat().st_size

    sidecar = args.output.with_suffix(args.output.suffix + ".meta.json")
    sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.output} ({summary['total_bytes']/1e6:.1f} MB)")
    print(f"  meta  {sidecar}")


if __name__ == "__main__":
    main()
