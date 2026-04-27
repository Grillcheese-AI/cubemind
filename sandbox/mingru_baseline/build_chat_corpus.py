"""Flatten Cubby chat-pair JSONL into SPM chat-token .txt for Stage 1.6.

Consumes the output of ``generate_cubby_corpus.py`` (rows shaped
``{"user": ..., "cubby": ..., "category": ..., "has_aside": ...}``) and
emits plain UTF-8 text with SPM forced chat tokens
(``<|user|>`` / ``<|assistant|>``) framing each turn. Pairs are
separated by a blank line, matching the convention used by
``allenai_c4_realnewslike.500m_tokens.txt`` and
``build_pretrain_corpus.py`` so ``tokenize_local.py`` can consume it
unchanged.

Output format per pair:

    <|user|>
    {user text}
    <|assistant|>
    {cubby text}
    [blank line]

The inner newlines between role token and content are intentional:
SPM tokenizes ``<|user|>`` as a single forced token, then ``\\n`` as
the next token, then the content. This gives the model clear
segmentation points to learn turn boundaries.

Multi-input: pass ``--input`` multiple times to concatenate several
JSONL sources into one ``.txt``. Useful if we later fold in additional
chat sources (pre/conversation.jsonl, batch_chat_templates_*.jsonl,
etc.) alongside the Cubby corpus.

Optional filters: ``--category`` (repeat to include multiple categories
only; default: all), ``--min-user-chars`` / ``--min-assistant-chars``
(drop pairs where either side is shorter than N).

Example:

    python sandbox/mingru_baseline/build_chat_corpus.py \\
        --input  sandbox/mingru_baseline/data/cubby_chat_v1.jsonl \\
        --output sandbox/mingru_baseline/data/cubby_chat_v1.txt

    # Tokenize with existing SPM tool:
    python sandbox/mingru_baseline/tokenize_local.py \\
        --data-path sandbox/mingru_baseline/data/cubby_chat_v1.txt \\
        --tokenizer-path D:\\grillcheese_training_data\\tokenizer\\grillcheese_spm32k_v2.model

Known schema support:

- ``cubby``: ``{"user": str, "cubby": str, ...}``  (primary, this file)
- ``assistant``: ``{"user": str, "assistant": str}`` (alias; some
  future generators may use the generic field name)
- ``turns``: ``{"turns": [{"role": "user"|"assistant", "content": str}, ...]}``
  (multi-turn sources like pre/conversation.jsonl)

Auto-detected from the first valid row.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"


def _detect_schema(first_row: dict) -> str:
    if "cubby" in first_row and "user" in first_row:
        return "cubby"
    if "assistant" in first_row and "user" in first_row:
        return "assistant"
    if "turns" in first_row and isinstance(first_row["turns"], list):
        return "turns"
    raise ValueError(
        f"Unrecognized row schema. Expected 'cubby'+'user', "
        f"'assistant'+'user', or 'turns'. Got keys: {list(first_row.keys())}"
    )


def _format_pair(user_text: str, assistant_text: str) -> str:
    """One user/assistant turn in SPM chat-token form."""
    return (
        f"{USER_TAG}\n{user_text}\n"
        f"{ASSISTANT_TAG}\n{assistant_text}\n"
    )


def _format_turns(turns: list[dict]) -> str | None:
    """Multi-turn conversation. Returns None if the turns list is
    malformed or contains roles we do not emit."""
    parts = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str):
            return None
        content = content.strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"{USER_TAG}\n{content}\n")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_TAG}\n{content}\n")
        else:
            return None  # system/tool/etc -- skip for v1
    if not parts:
        return None
    return "".join(parts)


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _flatten_source(
    src: Path,
    fout,
    schema: str,
    categories_filter: set[str] | None,
    min_user_chars: int,
    min_assistant_chars: int,
) -> dict:
    """Process one JSONL source. Returns stats dict."""
    stats = {
        "n_in": 0, "n_out": 0, "n_skip_short": 0,
        "n_skip_category": 0, "n_skip_malformed": 0,
        "chars_out": 0,
    }
    t0 = time.time()
    for row in _iter_rows(src):
        stats["n_in"] += 1
        try:
            if schema == "cubby":
                if categories_filter is not None:
                    cat = row.get("category")
                    if cat not in categories_filter:
                        stats["n_skip_category"] += 1
                        continue
                user_text = (row.get("user") or "").strip()
                asst_text = (row.get("cubby") or "").strip()
                if (len(user_text) < min_user_chars
                        or len(asst_text) < min_assistant_chars):
                    stats["n_skip_short"] += 1
                    continue
                block = _format_pair(user_text, asst_text)

            elif schema == "assistant":
                user_text = (row.get("user") or "").strip()
                asst_text = (row.get("assistant") or "").strip()
                if (len(user_text) < min_user_chars
                        or len(asst_text) < min_assistant_chars):
                    stats["n_skip_short"] += 1
                    continue
                block = _format_pair(user_text, asst_text)

            elif schema == "turns":
                block = _format_turns(row.get("turns", []))
                if block is None:
                    stats["n_skip_malformed"] += 1
                    continue
                if len(block) < min_user_chars + min_assistant_chars:
                    stats["n_skip_short"] += 1
                    continue

            else:
                stats["n_skip_malformed"] += 1
                continue

        except Exception:
            stats["n_skip_malformed"] += 1
            continue

        fout.write(block)
        fout.write("\n")  # blank line between pairs/conversations
        stats["n_out"] += 1
        stats["chars_out"] += len(block)

        if stats["n_out"] % 1000 == 0:
            elapsed = time.time() - t0
            rate = stats["n_in"] / max(elapsed, 1e-6)
            print(
                f"    in={stats['n_in']:>7,}  out={stats['n_out']:>7,}  "
                f"{stats['chars_out']/1e6:>6.1f} MB  "
                f"{rate:>6.0f} row/s  {elapsed:>5.0f}s",
                end="\r",
            )

    if stats["n_out"] >= 1000:
        print()  # newline after progress line
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, action="append", required=True,
                    help="JSONL source (repeat for multiple inputs)")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output .txt file")
    ap.add_argument("--category", action="append", default=None,
                    help="Only include these categories (repeat). Default: all")
    ap.add_argument("--min-user-chars", type=int, default=5)
    ap.add_argument("--min-assistant-chars", type=int, default=20)
    args = ap.parse_args()

    inputs = [p.resolve() for p in args.input]
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")

    dst: Path = args.output.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    categories_filter = set(args.category) if args.category else None

    print(f"  sources ({len(inputs)}):")
    for p in inputs:
        print(f"    {p}  ({p.stat().st_size/1e6:.1f} MB)")
    print(f"  output: {dst}")
    if categories_filter:
        print(f"  category filter: {sorted(categories_filter)}")
    print(f"  min_chars: user={args.min_user_chars} "
          f"assistant={args.min_assistant_chars}")
    print()

    agg = {
        "n_in": 0, "n_out": 0, "n_skip_short": 0,
        "n_skip_category": 0, "n_skip_malformed": 0,
        "chars_out": 0,
    }

    with dst.open("w", encoding="utf-8") as fout:
        for i, src in enumerate(inputs, 1):
            # Detect schema from first valid row
            schema = None
            for row in _iter_rows(src):
                try:
                    schema = _detect_schema(row)
                    break
                except ValueError:
                    continue
            if schema is None:
                print(f"  [{i}/{len(inputs)}] {src.name}  "
                      f"SKIP (no recognizable rows)")
                continue

            print(f"  [{i}/{len(inputs)}] {src.name}  (schema: {schema})")
            stats = _flatten_source(
                src, fout, schema, categories_filter,
                args.min_user_chars, args.min_assistant_chars,
            )
            for k in agg:
                agg[k] += stats.get(k, 0)
            print(
                f"    -> {stats['n_out']:,} pairs "
                f"({stats['chars_out']/1e6:.1f} MB)  "
                f"skip_short={stats['n_skip_short']:,}  "
                f"skip_category={stats['n_skip_category']:,}  "
                f"skip_malformed={stats['n_skip_malformed']:,}"
            )

    print()
    print(f"  TOTAL")
    print(f"    rows read:   {agg['n_in']:,}")
    print(f"    pairs out:   {agg['n_out']:,} ({agg['chars_out']/1e6:.2f} MB)")
    print(f"    dropped:     short={agg['n_skip_short']:,}  "
          f"category={agg['n_skip_category']:,}  "
          f"malformed={agg['n_skip_malformed']:,}")
    print()
    print("  next step (tokenize locally):")
    print(f"    python sandbox/mingru_baseline/tokenize_local.py \\")
    print(f"        --data-path {dst} \\")
    print(f"        --tokenizer-path D:\\grillcheese_training_data\\tokenizer\\grillcheese_spm32k_v2.model")


if __name__ == "__main__":
    main()
