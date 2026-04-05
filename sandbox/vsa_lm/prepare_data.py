"""Pre-tokenize TinyStories for VSA-LM.

Uses ByteLevel BPE (same as FlashLM v5) with vocab=8192 for fair PPL comparison.
Saves all tokens as one .npy file for instant loading (<1s).

Run once: python -u sandbox/vsa_lm/prepare_data.py
"""

import os
import numpy as np
from loguru import logger


def main(max_stories=0, vocab_size=8192, output_dir="sandbox/vsa_lm/data"):
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    logger.info("Loading TinyStories (max={})...", max_stories or "all")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    texts = []
    for i, item in enumerate(ds):
        if max_stories > 0 and i >= max_stories:
            break
        texts.append(item["text"])
    logger.info("Loaded {} stories", len(texts))

    os.makedirs(output_dir, exist_ok=True)

    # Train ByteLevel BPE — same algorithm as FlashLM v5
    logger.info("Training ByteLevel BPE tokenizer (vocab={})...", vocab_size)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    actual_vocab = tokenizer.get_vocab_size()
    logger.info("Tokenizer trained: {} tokens, saved to {}/tokenizer.json", actual_vocab, output_dir)

    # Tokenize all text into one flat array
    logger.info("Tokenizing {} stories...", len(texts))
    all_tokens = []
    for i, t in enumerate(texts):
        ids = tokenizer.encode(t).ids
        all_tokens.extend(ids)
        if (i + 1) % 50000 == 0:
            logger.info("  {}/{} stories, {} tokens", i + 1, len(texts), len(all_tokens))

    tokens = np.array(all_tokens, dtype=np.int32)
    logger.info("Total: {} tokens ({:.0f}MB)", len(tokens), tokens.nbytes / 1e6)

    # Save
    np.save(os.path.join(output_dir, "tokens.npy"), tokens)
    np.save(os.path.join(output_dir, "vocab_size.npy"), np.array([actual_vocab]))

    logger.info("Done: {}/tokens.npy + tokenizer.json", output_dir)


if __name__ == "__main__":
    main(max_stories=50000)  # 50K stories (~25M tokens)
