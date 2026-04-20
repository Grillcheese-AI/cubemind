#!/usr/bin/env python3
"""Sample N text lines uniformly across heterogeneous sources.

Walks a configured set of root directories, dispatches per-file by
extension to the right text extractor, reservoir-samples N total
lines, writes the result to a single scratch ``.txt`` corpus that
SentencePiece can ingest directly.

Supported extensions:

    .txt   raw text, one logical line per record
    .jsonl one JSON record per line; extracts ``text`` / ``content`` /
           ``body`` / ``message``
    .json  either a single object (recursive walk for any string field
           that looks like prose) OR an array of records (NYT-style;
           extracts ``headline.main`` + ``abstract`` + ``lead_paragraph``)
    .pdf   extracted via PyMuPDF (fitz) — page-by-page text

Skipped: .epub, .opf, .csv, .jpg, .tsx, .ts, .R, anything else.

Reservoir sampling guarantees uniform N-line draw across however many
billion candidate lines exist, in one pass, with O(N) memory. We sample
across ALL files at once (not per-file), so big files contribute
proportionally without dominating.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator


# ── Default source list ────────────────────────────────────────────────
#
# Add new roots here. Each entry is (label, root_path, recursive).
# The aggregator walks each root, picks up every file matching one of
# the supported extensions, and dispatches to the right extractor.
#
# Order matters only for the progress log — sampling is uniform.

DEFAULT_SOURCES: list[tuple[str, str, bool]] = [
    # D: drive — grillcheese pretraining corpus
    ("unified",              "D:/grillcheese_training_data/unified",          False),
    ("factual",              "D:/grillcheese_training_data/factual",          True),
    ("gbooks",               "D:/grillcheese_training_data/gbooks",           True),
    ("jsonl",                "D:/grillcheese_training_data/jsonl",            False),
    ("knowledgetxt",         "D:/grillcheese_training_data/knowledgetxt",     True),
    ("temporal_historical",  "D:/grillcheese_training_data/temporal/historical", False),
    ("temporal_nyt",         "D:/grillcheese_training_data/temporal/nyt_data",   False),
    ("pre",                  "D:/grillcheese_training_data/pre",              True),
    ("svc_root",             "D:/grillcheese_training_data",                  False),  # root-level *.jsonl

    # E: drive — RAW_TEXTS
    ("realms",        "E:/RAW_TEXTS/realms",        True),
    ("to_validate",   "E:/RAW_TEXTS/to_validate",   True),
    ("wsj_q1",        "E:/RAW_TEXTS/WSJ 1st Qtr 2025", True),
    ("wsj_q2",        "E:/RAW_TEXTS/WSJ 2nd Qtr 2025", True),
    ("wsj_q3",        "E:/RAW_TEXTS/WSJ 3rd Qtr 2025", True),
    ("wsj_q4",        "E:/RAW_TEXTS/WSJ 4th Qtr 2025", True),
    ("e_jsonl",       "E:/jsonl",                   False),

    # E: drive — Reader's Digest PDFs
    ("readers_digest", "E:/Reader's Digest USA 2025 Complete", False),
]

EXCLUDE_FILE_PREFIXES = ("hf_",)        # skip duplicated HF-mirror files
SUPPORTED_EXT = {".txt", ".jsonl", ".json", ".pdf"}


# ── File walk ──────────────────────────────────────────────────────────

def walk_sources(sources: list[tuple[str, str, bool]]) -> list[tuple[str, Path]]:
    """Return [(label, path), …] for every supported file under each root."""
    out: list[tuple[str, Path]] = []
    for label, root_str, recursive in sources:
        root = Path(root_str)
        if not root.exists():
            print(f"  [skip] {label}: {root} does not exist")
            continue
        if recursive:
            paths = root.rglob("*")
        else:
            paths = root.glob("*")
        for p in paths:
            if not p.is_file():
                continue
            if p.suffix.lower() not in SUPPORTED_EXT:
                continue
            if any(p.name.startswith(pref) for pref in EXCLUDE_FILE_PREFIXES):
                continue
            out.append((label, p))
    return out


# ── Per-format extractors ──────────────────────────────────────────────

def extract_txt(path: Path) -> Iterator[str]:
    """Yield logical lines from a plain .txt file (encoding-tolerant)."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                yield line


_JSONL_TEXT_KEYS = ("text", "content", "body", "message", "completion", "response")


def extract_jsonl(path: Path) -> Iterator[str]:
    """Yield ``text`` / ``content`` / ``body`` / etc. fields from a JSONL."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            for key in _JSONL_TEXT_KEYS:
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    for sub in v.split("\n"):
                        if sub.strip():
                            yield sub
                    break  # one text field per record


def extract_json(path: Path) -> Iterator[str]:
    """Yield strings from a .json file. Handles arrays of records (NYT
    format with headline/abstract/lead_paragraph) and single nested
    objects (recursive walk for any string >= 20 chars)."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            parts: list[str] = []
            # NYT-style fields
            hl = rec.get("headline")
            if isinstance(hl, dict) and isinstance(hl.get("main"), str):
                parts.append(hl["main"])
            for k in ("abstract", "lead_paragraph", "snippet", "text", "content"):
                v = rec.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v)
            joined = " ".join(parts).strip()
            if joined:
                yield joined
    elif isinstance(data, dict):
        # Single-object: walk recursively, yield any prose-looking string.
        stack: list = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, str):
                cur = cur.strip()
                if len(cur) >= 20 and " " in cur:  # heuristic: prose
                    yield cur


def extract_pdf(path: Path) -> Iterator[str]:
    """Page-by-page text via PyMuPDF. Yields one line per page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return
    try:
        doc = fitz.open(path)
    except Exception:
        return
    for page in doc:
        try:
            text = page.get_text("text")
        except Exception:
            continue
        for line in text.split("\n"):
            line = line.strip()
            if line:
                yield line
    doc.close()


EXTRACTORS = {
    ".txt":   extract_txt,
    ".jsonl": extract_jsonl,
    ".json":  extract_json,
    ".pdf":   extract_pdf,
}


# ── Reservoir sampling ─────────────────────────────────────────────────

def reservoir_sample(files: list[tuple[str, Path]], n_target: int,
                     seed: int = 0) -> list[str]:
    """Single-pass reservoir sampling across all files. Returns N lines."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    label_counts: dict[str, int] = {}
    t0 = time.time()
    last_print = t0

    for label, path in files:
        ext = path.suffix.lower()
        extractor = EXTRACTORS.get(ext)
        if extractor is None:
            continue
        try:
            for line in extractor(path):
                if not line:
                    continue
                seen += 1
                if len(reservoir) < n_target:
                    reservoir.append(line)
                else:
                    # Replace existing element with probability n/seen
                    j = rng.randint(0, seen - 1)
                    if j < n_target:
                        reservoir[j] = line
                label_counts[label] = label_counts.get(label, 0) + 1
                # Progress every ~10s
                now = time.time()
                if now - last_print > 10:
                    rate = seen / max(now - t0, 1)
                    print(f"  scanned {seen:>11,} lines  "
                          f"reservoir {len(reservoir):,}/{n_target:,}  "
                          f"{rate:>10,.0f} lines/s  "
                          f"current: {label}")
                    last_print = now
        except Exception as e:
            print(f"  [warn] {label}/{path.name}: {e}", file=sys.stderr)
            continue

    elapsed = time.time() - t0
    print(f"\n  total scanned: {seen:,} lines in {elapsed/60:.1f}m")
    print(f"  per-source line counts seen:")
    for lab, ct in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {lab:25s} {ct:>12,}  ({ct/max(seen,1)*100:>5.1f}%)")
    return reservoir


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to write the sampled scratch corpus.")
    ap.add_argument("--sample-lines", type=int, default=2_000_000,
                    help="Total lines to sample. 2M is plenty for a 32K SPM.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"  walking {len(DEFAULT_SOURCES)} source roots...")
    files = walk_sources(DEFAULT_SOURCES)
    print(f"  {len(files):,} candidate files found")
    by_label: dict[str, int] = {}
    for label, _ in files:
        by_label[label] = by_label.get(label, 0) + 1
    for lab, ct in sorted(by_label.items(), key=lambda x: -x[1]):
        print(f"    {lab:25s} {ct:>5} files")

    print(f"\n  reservoir-sampling {args.sample_lines:,} lines (seed={args.seed})...")
    sample = reservoir_sample(files, args.sample_lines, seed=args.seed)

    print(f"\n  writing {len(sample):,} lines to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for line in sample:
            f.write(line + "\n")
    sz = args.output.stat().st_size / 1e6
    print(f"  done — {sz:.1f} MB scratch corpus ready for SPM training")


if __name__ == "__main__":
    main()
