"""Pre-tokenize TinyStories with SentencePiece unigram for VSA-LM.

SentencePiece unigram is better than BPE for VSA architectures:
- More consistent token boundaries → cleaner block-code bindings
- Better compression at small vocab sizes
- Morphologically coherent subwords

Usage:
    python -m cubemind.training.prepare_data_sp
    python -m cubemind.training.prepare_data_sp --vocab-size 8192 --out-dir data/sp_tokens
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger


def prepare(
    vocab_size: int = 8192,
    out_dir: str = "sandbox/vsa_lm/data_sp",
    text_source: str = "data/tinystories_50k.json",
    max_stories: int = 0,
) -> None:
    import sentencepiece as spm

    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "sp_model")

    # Load text
    if os.path.exists(text_source):
        with open(text_source, encoding="utf-8") as f:
            texts = json.load(f)
        logger.info("Loaded {} texts from {}", len(texts), text_source)
    else:
        logger.error("Text source not found: {}", text_source)
        return

    if max_stories > 0:
        texts = texts[:max_stories]

    # Write temp file for SentencePiece training
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for t in texts:
            f.write(t.strip() + "\n")
        tmp_path = f.name

    # Train SentencePiece model
    logger.info("Training SentencePiece unigram model (vocab={})", vocab_size)
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        normalization_rule_name="identity",
        num_threads=os.cpu_count() or 4,
    )
    os.unlink(tmp_path)
    logger.info("Model saved: {}.model", model_prefix)

    # Load trained model and tokenize all texts
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    actual_vocab = sp.get_piece_size()
    logger.info("Actual vocab: {}", actual_vocab)

    all_tokens = []
    for i, text in enumerate(texts):
        ids = sp.encode(text, out_type=int)
        all_tokens.extend(ids)
        if (i + 1) % 10000 == 0:
            logger.info("Tokenized {}/{} stories ({} tokens so far)",
                        i + 1, len(texts), len(all_tokens))

    tokens = np.array(all_tokens, dtype=np.int32)
    np.save(os.path.join(out_dir, "tokens.npy"), tokens)
    np.save(os.path.join(out_dir, "vocab_size.npy"), np.array([actual_vocab]))

    # Stats
    avg_len = len(tokens) / len(texts)
    logger.info("Done: {} tokens from {} stories (avg {:.1f} tokens/story)",
                len(tokens), len(texts), avg_len)
    logger.info("Saved to {}/", out_dir)

    # Compare with BPE if exists
    bpe_path = Path("sandbox/vsa_lm/data/tokens.npy")
    if bpe_path.exists():
        bpe_tokens = np.load(bpe_path)
        bpe_per_story = len(bpe_tokens) / len(texts) if len(texts) > 0 else 0
        logger.info("Comparison: BPE={:.1f} tok/story, SP={:.1f} tok/story ({:.0f}% of BPE)",
                    bpe_per_story, avg_len, 100 * avg_len / max(bpe_per_story, 1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--out-dir", default="sandbox/vsa_lm/data_sp")
    parser.add_argument("--text-source", default="data/tinystories_50k.json")
    parser.add_argument("--max-stories", type=int, default=0)
    args = parser.parse_args()
    prepare(args.vocab_size, args.out_dir, args.text_source, args.max_stories)
