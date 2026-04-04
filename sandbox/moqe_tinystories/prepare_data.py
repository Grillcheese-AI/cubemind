"""Prepare TinyStories as .npz token files for MoQE CE-only training.

No teacher logits — just tokenized sequences. The distillation pipeline
detects None teacher_logits and switches to CE-only mode automatically.

Run: python -u sandbox/moqe_tinystories/prepare_data.py
"""

import os
import tempfile

import numpy as np
from loguru import logger


def main(max_stories: int = 300000, vocab_size: int = 1000, max_seq_len: int = 512,
         output_dir: str = "sandbox/moqe_tinystories/data"):
    from datasets import load_dataset
    import sentencepiece as spm

    logger.info("Loading TinyStories ({} stories)...", max_stories)
    ds = load_dataset("roneneldan/TinyStories", split="train")

    texts = []
    for i, item in enumerate(ds):
        if i >= max_stories:
            break
        texts.append(item["text"])
    logger.info("Loaded {} stories", len(texts))

    # Train SentencePiece
    sp_model_prefix = os.path.join(tempfile.gettempdir(), "moqe_tinystories_sp")
    corpus_file = sp_model_prefix + "_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    logger.info("Training SentencePiece BPE (vocab={})", vocab_size)
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=sp_model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=3,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_prefix + ".model")
    actual_vocab = sp.get_piece_size()
    logger.info("Vocab: {}", actual_vocab)

    # Save tokenizer model alongside data
    os.makedirs(output_dir, exist_ok=True)
    import shutil
    shutil.copy2(sp_model_prefix + ".model", os.path.join(output_dir, "tokenizer.model"))

    # Tokenize and save as .npz (same format as teacher logit files, but no logits)
    count = 0
    total_tokens = 0
    for i, text in enumerate(texts):
        tokens = sp.encode(text, out_type=int)
        if len(tokens) < 4:
            continue
        tokens = np.array(tokens[:max_seq_len], dtype=np.int32)
        np.savez_compressed(
            os.path.join(output_dir, f"sequence_{count:06d}.npz"),
            input_tokens=tokens,
        )
        total_tokens += len(tokens)
        count += 1
        if (count % 10000) == 0:
            logger.info("  {} sequences, {} tokens", count, total_tokens)

    logger.info("Done: {} sequences, {} tokens → {}", count, total_tokens, output_dir)

    # Save vocab size for training script
    np.save(os.path.join(output_dir, "vocab_size.npy"), np.array([actual_vocab]))


if __name__ == "__main__":
    main()
