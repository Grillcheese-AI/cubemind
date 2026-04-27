"""Flatten one or more JSONL and/or .txt-dir pretrain corpora into a single .txt.

Two input modes (can be mixed in one invocation):
    --input <file.jsonl>    one JSON object per line, configurable text field
    --txt-dir <dir>         every ``*.txt`` under the directory = one record

Records in the output are separated by a blank line, matching the
convention already used by ``allenai_c4_realnewslike.500m_tokens.txt``.

Example (Nemotron + Wikibooks + Gutenberg books + factual books):

    python sandbox/mingru_baseline/build_pretrain_corpus.py \\
        --input   D:\\grillcheese_training_data\\unified\\nemotron_cc_v2_high_quality.1b_tokens.jsonl \\
        --input   D:\\grillcheese_training_data\\jsonl\\wikibooks_corpus.jsonl \\
        --txt-dir E:\\RAW_TEXTS\\realms\\gutemberg_books_unclassified \\
        --txt-dir E:\\RAW_TEXTS\\realms\\factual_books \\
        --output  D:\\grillcheese_training_data\\pretrain_ext_v1.txt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _flatten_txt_dir(
    src_dir: Path,
    fout,
    min_chars: int,
    max_records: int | None,
) -> tuple[int, int, int, int, int, int]:
    """Flatten every ``*.txt`` under ``src_dir`` (recursive). One file = one record.

    Return (n_in, n_out, n_skip_short, n_skip_bad, n_skip_missing, chars_out)
    with the same fields as ``_flatten_one`` so aggregation in ``main`` works
    uniformly. ``n_skip_bad`` counts files that failed to read; the
    ``n_skip_missing`` slot is unused for txt-dirs and always 0.
    """
    n_in = n_out = n_skip_short = n_skip_bad = chars_out = 0
    t0 = time.time()

    txt_files = sorted(src_dir.rglob("*.txt"))
    total_files = len(txt_files)
    print(f"    {total_files:,} .txt files under {src_dir.name}")

    for path in txt_files:
        n_in += 1
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            n_skip_bad += 1
            continue

        txt = txt.strip()
        if len(txt) < min_chars:
            n_skip_short += 1
            continue

        fout.write(txt)
        fout.write("\n\n")
        n_out += 1
        chars_out += len(txt)

        if n_out % 500 == 0:
            elapsed = time.time() - t0
            rate = n_in / max(elapsed, 1e-6)
            print(
                f"    in={n_in:>8,}/{total_files:,}  out={n_out:>8,}  "
                f"skip={n_skip_bad + n_skip_short:>6,}  "
                f"{chars_out/1e9:>5.2f} GB  "
                f"{rate:>5.0f} file/s  {elapsed:>5.0f}s",
                end="\r",
            )

        if max_records and n_out >= max_records:
            break

    print()
    return n_in, n_out, n_skip_short, n_skip_bad, 0, chars_out


def _flatten_one(
    src: Path,
    fout,
    field: str,
    min_chars: int,
    max_records: int | None,
) -> tuple[int, int, int, int, int, int]:
    """Return (n_in, n_out, n_skip_short, n_skip_bad, n_skip_missing, chars_out)."""
    n_in = n_out = n_skip_short = n_skip_bad = n_skip_missing = chars_out = 0
    t0 = time.time()

    with src.open("r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            n_in += 1
            try:
                row = json.loads(line)
            except Exception:
                n_skip_bad += 1
                continue

            txt = row.get(field)
            if not isinstance(txt, str):
                n_skip_missing += 1
                continue

            txt = txt.strip()
            if len(txt) < min_chars:
                n_skip_short += 1
                continue

            fout.write(txt)
            fout.write("\n\n")
            n_out += 1
            chars_out += len(txt)

            if n_out % 50_000 == 0:
                elapsed = time.time() - t0
                rate = n_in / max(elapsed, 1e-6)
                print(
                    f"    in={n_in:>10,}  out={n_out:>10,}  "
                    f"skip={n_skip_bad + n_skip_missing + n_skip_short:>8,}  "
                    f"{chars_out/1e9:>5.2f} GB  "
                    f"{rate:>7.0f} rec/s  {elapsed:>5.0f}s",
                    end="\r",
                )

            if max_records and n_out >= max_records:
                break

    print()
    return n_in, n_out, n_skip_short, n_skip_bad, n_skip_missing, chars_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, action="append", default=[],
                    help="JSONL file (repeat for multiple sources)")
    ap.add_argument("--input-cap", action="append", default=[],
                    help="Per-source record cap, paired positionally with "
                         "--input. Use 0 for no cap. Example for 85/15 "
                         "bilingual mix:\n"
                         "  --input nemotron.jsonl --input-cap 0 \\\n"
                         "  --input frwiki.jsonl   --input-cap 400000")
    ap.add_argument("--txt-dir", type=Path, action="append", default=[],
                    help="Directory of .txt files — one file = one record "
                         "(recursive). Repeat for multiple directories.")
    ap.add_argument("--txt-dir-cap", action="append", default=[],
                    help="Per-source record cap, paired positionally with "
                         "--txt-dir. Use 0 for no cap.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output .txt file")
    ap.add_argument("--field", default="text",
                    help="JSON field to extract from --input (default: text)")
    ap.add_argument("--min-chars", type=int, default=100,
                    help="Skip records with fewer than N characters (default: 100)")
    ap.add_argument("--max-records-per-source", type=int, default=None,
                    help="Global per-source fallback cap. Used for any "
                         "source without a specific --input-cap / "
                         "--txt-dir-cap entry.")
    args = ap.parse_args()

    if not args.input and not args.txt_dir:
        raise SystemExit("FATAL: pass at least one --input or --txt-dir")

    inputs = [p.resolve() for p in args.input]
    txt_dirs = [p.resolve() for p in args.txt_dir]

    # Resolve per-source caps. Each --input-cap / --txt-dir-cap entry
    # pairs positionally with the corresponding --input / --txt-dir. A
    # missing or 0 entry falls back to --max-records-per-source. Returns
    # ``None`` (no cap) if both the per-source slot and the global
    # fallback are absent.
    def _resolve_caps(srcs, raw_caps, fallback):
        out = []
        for i in range(len(srcs)):
            cap = 0
            if i < len(raw_caps):
                try:
                    cap = int(raw_caps[i])
                except (ValueError, TypeError):
                    cap = 0
            out.append(cap if cap > 0 else fallback)
        return out

    input_caps = _resolve_caps(inputs, args.input_cap,
                               args.max_records_per_source)
    txt_dir_caps = _resolve_caps(txt_dirs, args.txt_dir_cap,
                                 args.max_records_per_source)
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"FATAL: input not found: {p}")
    for d in txt_dirs:
        if not d.is_dir():
            raise SystemExit(f"FATAL: txt-dir not a directory: {d}")

    dst: Path = args.output.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    jsonl_size_gb = sum(p.stat().st_size for p in inputs) / 1e9
    n_total = len(inputs) + len(txt_dirs)
    print(f"  sources ({n_total}):")
    for p in inputs:
        print(f"    JSONL   {p}  ({p.stat().st_size/1e9:.2f} GB)")
    for d in txt_dirs:
        n_txt = sum(1 for _ in d.rglob("*.txt"))
        print(f"    TXT-DIR {d}  ({n_txt:,} .txt files)")
    print(f"  total JSONL bytes: {jsonl_size_gb:.2f} GB")
    print(f"  output: {dst}")
    print(f"  field:  {args.field}")
    print(f"  min_chars: {args.min_chars}")
    if args.max_records_per_source:
        print(f"  max_records_per_source: {args.max_records_per_source:,}")
    print()

    agg = {"n_in": 0, "n_out": 0, "n_skip_short": 0,
           "n_skip_bad": 0, "n_skip_missing": 0, "chars_out": 0}

    step = 0
    with dst.open("w", encoding="utf-8") as fout:
        for i, src in enumerate(inputs):
            step += 1
            cap = input_caps[i]
            cap_str = f"  cap={cap:,}" if cap else ""
            print(f"  [{step}/{n_total}] JSONL {src.name}{cap_str}")
            n_in, n_out, ns, nb, nm, co = _flatten_one(
                src, fout, args.field, args.min_chars, cap,
            )
            agg["n_in"] += n_in
            agg["n_out"] += n_out
            agg["n_skip_short"] += ns
            agg["n_skip_bad"] += nb
            agg["n_skip_missing"] += nm
            agg["chars_out"] += co
            print(f"    -> {n_out:,} records ({co/1e9:.2f} GB)")

        for i, d in enumerate(txt_dirs):
            step += 1
            cap = txt_dir_caps[i]
            cap_str = f"  cap={cap:,}" if cap else ""
            print(f"  [{step}/{n_total}] TXT-DIR {d.name}{cap_str}")
            n_in, n_out, ns, nb, nm, co = _flatten_txt_dir(
                d, fout, args.min_chars, cap,
            )
            agg["n_in"] += n_in
            agg["n_out"] += n_out
            agg["n_skip_short"] += ns
            agg["n_skip_bad"] += nb
            agg["n_skip_missing"] += nm
            agg["chars_out"] += co
            print(f"    -> {n_out:,} records ({co/1e9:.2f} GB)")

    print()
    print(f"  TOTAL")
    print(f"    in:    {agg['n_in']:,}")
    print(f"    out:   {agg['n_out']:,} records "
          f"({agg['chars_out']/1e9:.2f} GB)")
    print(f"    skip:  short={agg['n_skip_short']:,}  "
          f"bad_json={agg['n_skip_bad']:,}  "
          f"missing_field={agg['n_skip_missing']:,}")
    print()
    print("  next step (tokenize locally):")
    print(f"    python sandbox/mingru_baseline/tokenize_local.py \\")
    print(f"        --data-path {dst} \\")
    print(f"        --tokenizer-path D:\\grillcheese_training_data\\tokenizer\\grillcheese_spm32k_v2.model")


if __name__ == "__main__":
    main()
