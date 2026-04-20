#!/usr/bin/env python3
"""Regenerate grillcheese_spm32k with multitask tags + VM opcodes as
forced tokens.

The existing tokenizer at ``D:/grillcheese_training_data/tokenizer/
grillcheese_spm32k.model`` was trained from the full unified corpus
(562M lines, 204 GB characters, 42 files — verified via its
report.json). It covers chat/instruction/thinking/memory special
tokens but splits our multitask structural tags into 3-5 subword
pieces each, and fragments the 58 VM opcode mnemonics similarly —
wasting ~30% of effective sequence length on structural markers.

This regen:

  1. Reuses the same unified corpus as input (the original 194 GB
     corpus.txt), sampled via SentencePiece's native line-sampling
     (``input_sentence_size``) so we don't duplicate the sample.
  2. Adds the multitask tags, the 58 canonical opcodes, and the 8
     universal VSA-VM role names as ``user_defined_symbols``. These
     get guaranteed single-token IDs and NEVER fragment.
  3. Keeps all the original chat/instruction/thinking/memory tokens
     so existing code that relies on them keeps working.
  4. Saves as ``grillcheese_spm32k_v2.model`` — old model stays in
     place so we can A/B the two tokenizers on the same data.

Run::

    python sandbox/mingru_baseline/regen_tokenizer.py \\
        --input-dir D:/grillcheese_training_data/unified \\
        --output-dir D:/grillcheese_training_data/tokenizer \\
        --sample-sentences 2000000
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vm_opcodes import CANONICAL_OPCODES  # noqa: E402


# ── Reserved tokens ─────────────────────────────────────────────────────
#
# ORDER MATTERS: SentencePiece assigns sequential IDs starting after the
# control tokens (<pad>=0, <unk>=1, <s>=2, </s>=3). Appending here means
# append-only at the end of user_defined_symbols — never reorder, never
# insert in the middle, because checkpoints reference tokens by ID.

# Existing chat / agent tokens (kept verbatim from the original tokenizer
# report.json so downstream code that references them keeps working).
LEGACY_CHAT_TOKENS = [
    "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>",
    "<|image|>", "<|audio|>",
    "[MY_STATE]",    "[/MY_STATE]",
    "[INSTRUCTION]", "[/INSTRUCTION]",
    "[THINKING]",    "[/THINKING]",
    "[MEMORY]",      "[/MEMORY]",
    "[SPECIALIST]",  "[/SPECIALIST]",
]

# Multitask structural tags (from gemini_factory / train_torch).
# Paired open/close stay adjacent so the vocab is scan-friendly.
MULTITASK_TAGS = [
    "<TASK:SCHEMA2RULE>", "<TASK:REPAIR>",
    "<SCHEMA>",    "</SCHEMA>",
    "<OPCODE>",    "</OPCODE>",
    "<RULE>",      "</RULE>",
    "<VALID>",     "</VALID>",
    "<VALID>yes</VALID>",  "<VALID>no</VALID>",
    "<TRACE>",     "</TRACE>",
    "<ROLES>",     "</ROLES>",
    "<INSTR>",     "</INSTR>",
    "<INTENT>",    "</INTENT>",
    "<ACT>",       "</ACT>",
    "<FIX_RULE>",   "</FIX_RULE>",
    "<FIX_OPCODE>", "</FIX_OPCODE>",
    "<ERROR>",     "</ERROR>",
]

# Universal VSA-VM semantic roles (match `cubemind/reasoning/vm.py` +
# `opcode-vsa-rs/src/ir.rs::CubeMindRole`).
VSA_ROLES = [
    "AGENT", "ACTION", "OBJECT", "QUANTITY",
    "SOURCE", "DESTINATION", "CONTEXT", "STATE",
]


def build_user_symbols() -> list[str]:
    """Build the full ``user_defined_symbols`` list. Deduplicated but
    order-preserving (insertion order) because IDs follow positional
    order once SentencePiece assigns them."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in (LEGACY_CHAT_TOKENS
                + MULTITASK_TAGS
                + list(CANONICAL_OPCODES)
                + VSA_ROLES):
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


# ── Training driver ─────────────────────────────────────────────────────

def train(
    input_paths: list[Path],
    output_prefix: Path,
    vocab_size: int,
    sample_sentences: int,
    character_coverage: float,
) -> dict:
    """Call SentencePiece with the canonical training settings."""
    import sentencepiece as spm

    user_symbols = build_user_symbols()
    print(f"  {len(user_symbols)} user_defined_symbols will be forced into vocab")
    print(f"  {len(CANONICAL_OPCODES)} opcodes + {len(VSA_ROLES)} roles + "
          f"{len(MULTITASK_TAGS)} multitask tags + "
          f"{len(LEGACY_CHAT_TOKENS)} legacy chat tokens")

    # SentencePiece wants a comma-separated list of input paths OR a
    # single file. For multiple files we write a tiny manifest.
    if len(input_paths) == 1:
        input_arg = str(input_paths[0])
    else:
        input_arg = ",".join(str(p) for p in input_paths)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    spm.SentencePieceTrainer.Train(
        input=input_arg,
        model_prefix=str(output_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        input_sentence_size=sample_sentences,
        shuffle_input_sentence=True,
        user_defined_symbols=user_symbols,
        # Reserve standard control IDs at the front.
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<s>", eos_piece="</s>",
        num_threads=16,
        # Split digits so the model treats "2019" as "2 0 1 9" under
        # the subword merges — important for the number-magnitude
        # binding we want the VSA head to learn.
        split_digits=True,
        # Allow byte-fallback so no OOV panics on rare unicode/emoji.
        byte_fallback=True,
        # Normalisation: keep whitespace as-is so legal-document and
        # news formatting is preserved (matches the original model).
        normalization_rule_name="identity",
    )
    elapsed = time.time() - t0
    print(f"  SentencePiece training done in {elapsed/60:.1f}m")
    return {
        "elapsed_minutes": elapsed / 60,
        "user_symbols_count": len(user_symbols),
        "vocab_size": vocab_size,
    }


# ── Validation ─────────────────────────────────────────────────────────

def validate(model_path: Path) -> dict:
    """Tokenize a battery of multitask strings and verify every forced
    symbol IS single-token in the new model."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    actual_vocab = sp.GetPieceSize()

    failures: list[tuple[str, int]] = []
    user_symbols = build_user_symbols()
    for tok in user_symbols:
        pieces = sp.EncodeAsPieces(tok)
        # SPM prefaces standalone words with the word-boundary marker
        # U+2581 (▁) — that's a legitimate "this piece starts a new
        # word" signal, not fragmentation. Strip it before counting;
        # the tag itself is single-token if exactly one non-▁ piece
        # contains the full symbol.
        non_boundary = [p for p in pieces if p != "\u2581"]
        if len(non_boundary) != 1 or tok not in non_boundary[0]:
            failures.append((tok, len(pieces)))

    # Fragmentation check on a representative multitask row.
    sample = (
        "<TASK:SCHEMA2RULE> <SCHEMA>resource_transfer</SCHEMA> "
        "<ROLES>SOURCE DESTINATION OBJECT QUANTITY</ROLES> "
        "<TRACE>CREATE ASSIGN BIND_ROLE</TRACE> "
        "<RULE>bind_source_role</RULE> <OPCODE>BIND_ROLE</OPCODE> "
        "<VALID>yes</VALID>"
    )
    row_ids = sp.EncodeAsIds(sample)
    # Rough structural-markup count — the tags + opcode mnemonics
    # should all be single tokens; only content words fragment.
    return {
        "actual_vocab_size": actual_vocab,
        "forced_symbols_that_fragmented": failures,
        "sample_row": sample,
        "sample_row_token_count": len(row_ids),
    }


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input-dir", action="append", type=Path,
                    default=None,
                    help="Repeatable: directory to scan recursively for "
                         ".txt and .jsonl files. Default: the canonical set "
                         "of grillcheese_training_data/{unified, factual, "
                         "gbooks, jsonl, knowledgetxt, temporal} + the root.")
    ap.add_argument("--exclude-prefix", action="append", default=["hf_"],
                    help="Skip files whose basename starts with any listed "
                         "prefix. Defaults to skipping hf_* (HuggingFace "
                         "datasets that duplicate content available elsewhere).")
    ap.add_argument("--scratch-corpus", type=Path,
                    default=Path("D:/grillcheese_training_data/tokenizer/_scratch_v2_sample.txt"),
                    help="Where to write the sampled corpus that SPM "
                         "trains on. ~600MB-2GB depending on sample-sentences.")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("D:/grillcheese_training_data/tokenizer"),
                    help="Where to write grillcheese_spm32k_v2.*")
    ap.add_argument("--prefix", default="grillcheese_spm32k_v2",
                    help="Model prefix (output files: <prefix>.model/.vocab)")
    ap.add_argument("--vocab-size", type=int, default=32128,
                    help="Target vocab size. 32128 = 32000 + 128 headroom "
                         "so the ~94 forced tokens don't crowd out learned subwords.")
    ap.add_argument("--sample-sentences", type=int, default=2_000_000,
                    help="Lines to sample from input for SPM training. "
                         "2M is enough for a 32K unigram model; 5-10M for "
                         "belt-and-suspenders.")
    ap.add_argument("--character-coverage", type=float, default=0.9995,
                    help="Matches the original tokenizer's 0.9995.")
    args = ap.parse_args()

    all_files = sorted(Path(args.input_dir).glob(args.input_glob))
    excludes = tuple(args.exclude_prefix or [])
    files = [p for p in all_files if not p.name.startswith(excludes)]
    if not files:
        sys.exit(f"no files matching {args.input_glob} in {args.input_dir} "
                 f"(after excluding prefix {excludes})")
    sizes_gb = sum(f.stat().st_size for f in files) / 1e9
    skipped = len(all_files) - len(files)
    print(f"  input: {len(files)} files, {sizes_gb:.1f} GB total "
          f"({skipped} excluded by prefix {excludes}) — sampling "
          f"{args.sample_sentences:,} lines across all")
    for p in files:
        print(f"    {p.stat().st_size / 1e9:>6.2f} GB  {p.name}")

    output_prefix = args.output_dir / args.prefix
    report = train(
        input_paths=files,
        output_prefix=output_prefix,
        vocab_size=args.vocab_size,
        sample_sentences=args.sample_sentences,
        character_coverage=args.character_coverage,
    )
    model_path = output_prefix.with_suffix(".model")
    print(f"\n  wrote {model_path}")

    print("\n  validating...")
    val = validate(model_path)
    print(f"  actual vocab size: {val['actual_vocab_size']}")
    if val["forced_symbols_that_fragmented"]:
        print(f"  [!] FAILED forced symbols: "
              f"{val['forced_symbols_that_fragmented']}")
    else:
        print(f"  all {len(build_user_symbols())} forced symbols are single-token")
    print(f"  sample row: {val['sample_row_token_count']} tokens")
    print(f"    text: {val['sample_row']}")

    # Write report.json alongside the model.
    report_path = output_prefix.with_suffix(".report.json")
    report_path.write_text(json.dumps({
        **report,
        **val,
        "input_files": [str(p) for p in files],
        "input_glob": args.input_glob,
    }, indent=2, ensure_ascii=False))
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
