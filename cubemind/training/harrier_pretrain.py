"""Harrier 0.6B MindForge pre-training — overnight run.

Extracts teacher logits from Harrier and trains MindForge to forge LoRA
adapters that approximate the teacher's output distribution.

Training loop:
  1. Sample text from TinyStories
  2. Extract Harrier logits (teacher)
  3. Encode context as block-code
  4. MindForge forges adapter from context
  5. VSA-LM forward with forged adapter → student logits
  6. KL divergence loss: student vs teacher
  7. Error-driven update on MindForge weights

Usage:
    python -m cubemind.training.harrier_pretrain
    python -m cubemind.training.harrier_pretrain --steps 50000 --seq-len 64
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from loguru import logger

from cubemind.ops.block_codes import BlockCodes
from cubemind.execution.mindforge import MindForge
from cubemind.functional.math import softmax


# ── Config ───────────────────────────────────────────────────────────────


TEACHER_PRESETS = {
    "harrier": {
        "path": "data/external_llms/harrier-oss-v1-0.6b.Q8_0.gguf",
        "d_model": 1024, "n_layers": 24, "vocab": 151936, "gpu": False,
    },
    "ministral": {
        "path": "data/external_llms/Ministral-3-3B-Reasoning-2512-Q8_0.gguf",
        "d_model": 3072, "n_layers": 32, "vocab": 131072, "gpu": True,
    },
    "llama": {
        "path": "data/external_llms/Llama3.3-8b-instruct-reasoning.gguf",
        "d_model": 4096, "n_layers": 32, "vocab": 128256, "gpu": True,
    },
}


@dataclass
class HarrierPretrainConfig:
    # Teacher selection
    teacher_name: str = "llama"
    teacher_path: str = ""
    data_dir: str = "sandbox/vsa_lm/data"

    # Teacher dims (set from preset)
    teacher_d_model: int = 4096
    teacher_n_layers: int = 32
    teacher_vocab: int = 128256
    teacher_gpu: bool = True

    # MindForge
    k: int = 16
    l: int = 16
    forge_rank: int = 8
    forge_basis: int = 16
    forge_d_hidden: int = 128

    def __post_init__(self):
        if not self.teacher_path and self.teacher_name in TEACHER_PRESETS:
            p = TEACHER_PRESETS[self.teacher_name]
            self.teacher_path = p["path"]
            self.teacher_d_model = p["d_model"]
            self.teacher_n_layers = p["n_layers"]
            self.teacher_vocab = p["vocab"]
            self.teacher_gpu = p["gpu"]

    # Training
    seq_len: int = 128
    n_ctx: int = 1024
    lr: float = 4e-3
    lr_min: float = 3e-4
    train_steps: int = 50000
    log_every: int = 100
    save_every: int = 1000
    warmup_steps: int = 500

    # Distillation
    temperature: float = 2.0  # softmax temperature for KL div
    top_k_logits: int = 256   # only distill top-K logits (memory saving)
    max_stories: int = 0      # 0 = all stories

    seed: int = 42


# ── Teacher Logit Extraction ─────────────────────────────────────────────


class TeacherExtractor:
    """Extract logits from Harrier GGUF via llama-cpp-python."""

    def __init__(self, config: HarrierPretrainConfig):
        from llama_cpp import Llama

        logger.info("Loading teacher: {}", config.teacher_path)
        self.model = Llama(
            model_path=config.teacher_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=-1,
            verbose=False,
            logits_all=True,
        )
        self.vocab_size = config.teacher_vocab
        self.top_k = config.top_k_logits
        logger.info("Teacher loaded: vocab={}, d_model={}", self.vocab_size, config.teacher_d_model)

    def extract_from_text(self, text: str, max_tokens: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize text with teacher's tokenizer and extract logits.

        Args:
            text: Raw text string.
            max_tokens: Maximum tokens to process.

        Returns:
            (token_ids, logits) where token_ids is (seq,) int32 and
            logits is (seq, top_k) float32.
        """
        tokens = self.model.tokenize(text.encode("utf-8"))[:max_tokens]
        if len(tokens) < 2:
            tokens = self.model.tokenize(b"The cat sat on the mat.")[:max_tokens]

        self.model.reset()
        self.model.eval(tokens)
        full_logits = np.array(self.model.scores[:len(tokens)], dtype=np.float32)

        token_ids = np.array(tokens, dtype=np.int32)

        if self.top_k < full_logits.shape[1]:
            top_indices = np.argpartition(full_logits, -self.top_k, axis=1)[:, -self.top_k:]
            top_logits = np.take_along_axis(full_logits, top_indices, axis=1)
            return token_ids, top_logits
        return token_ids, full_logits

    def __del__(self):
        if hasattr(self, "model"):
            del self.model


# ── Data Loading ─────────────────────────────────────────────────────────


def load_text_data(config: HarrierPretrainConfig) -> list[str]:
    """Load TinyStories as raw text for teacher tokenization.

    Priority: local JSON > local .txt > HuggingFace > synthetic
    """
    import json

    # 1. Local JSON (fastest — pre-downloaded)
    json_path = Path("data/tinystories_50k.json")
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            texts = json.load(f)
        logger.info("Loaded {} stories from {}", len(texts), json_path)
        return texts[:config.max_stories] if config.max_stories > 0 else texts

    # 2. Local .txt files
    text_files = sorted(Path(config.data_dir).glob("*.txt"))
    if text_files:
        texts = []
        for f in text_files:
            texts.extend(f.read_text(encoding="utf-8", errors="ignore").split("\n\n"))
        texts = [t.strip() for t in texts if len(t.strip()) > 50]
        if texts:
            logger.info("Loaded {} text chunks from {}", len(texts), config.data_dir)
            return texts

    # 3. HuggingFace (slow, may have import conflicts)
    try:
        import sys
        grilly_paths = [p for p in sys.path if "grilly" in p.lower()]
        for p in grilly_paths:
            sys.path.remove(p)
        try:
            import datasets
            logger.info("Loading TinyStories from HuggingFace...")
            ds = datasets.load_dataset("roneneldan/TinyStories", split="train")
            texts = [row["text"] for row in ds if len(row.get("text", "")) > 50]
            logger.info("Loaded {} stories from HF", len(texts))

            # Cache locally for next time
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(texts[:50000], f)
            logger.info("Cached to {}", json_path)

            return texts[:config.max_stories] if config.max_stories > 0 else texts
        finally:
            for p in grilly_paths:
                if p not in sys.path:
                    sys.path.append(p)
    except Exception as e:
        logger.warning("HF load failed: {}", e)

    # 4. Synthetic fallback
    logger.warning("No text data found. Using synthetic training sentences.")
    return [
        f"Once upon a time there was a {animal} who liked to {action}."
        for animal in ["cat", "dog", "bird", "fish", "bear", "fox", "rabbit"]
        for action in ["run", "jump", "play", "eat", "sleep", "sing", "dance", "swim"]
    ] * 100


# ── Training Loop ────────────────────────────────────────────────────────


def train(config: HarrierPretrainConfig | None = None):
    if config is None:
        config = HarrierPretrainConfig()

    rng = np.random.default_rng(config.seed)

    # Load data as raw text (teacher will tokenize with its own vocab)
    texts = load_text_data(config)
    n_texts = len(texts)

    # Init teacher
    teacher = TeacherExtractor(config)

    # Init MindForge
    bc = BlockCodes(k=config.k, l=config.l)
    forge = MindForge(
        k=config.k, l=config.l,
        n_layers=config.teacher_n_layers,
        d_target=config.teacher_d_model,
        rank=config.forge_rank,
        n_basis=config.forge_basis,
        d_hidden=config.forge_d_hidden,
        seed=config.seed,
    )

    n_params = (forge.A_basis.size + forge.B_basis.size
                + forge.W_proj.size + forge.W_h.size + forge.W_coeff.size
                + forge.layer_embeddings.size)
    logger.info("MindForge: {}K params, rank={}, basis={}, d_hidden={}",
                n_params // 1000, config.forge_rank, config.forge_basis, config.forge_d_hidden)

    # Training state
    best_loss = float("inf")
    loss_ema = 0.0
    t0 = time.time()

    # Init VM for structured context encoding
    from cubemind.reasoning.vm import VSAVM
    vm = VSAVM(bc=bc, seed=config.seed)

    # Build a persistent codebook of text patterns the VM discovers
    # Each unique pattern type gets a stable context block-code

    logger.info("Starting MindForge pre-training: {} steps, {} texts, lr={}",
                config.train_steps, n_texts, config.lr)

    for step in range(config.train_steps):
        # ── 1. Sample two consecutive texts (before → after) ─────────────
        text_idx = int(rng.integers(0, n_texts - 1))
        text_a = texts[text_idx]
        text_b = texts[text_idx + 1]

        # ── 2. Teacher tokenizes and produces logits ─────────────────────
        ids_a, logits_a = teacher.extract_from_text(text_a, max_tokens=config.seq_len)
        ids_b, logits_b = teacher.extract_from_text(text_b, max_tokens=config.seq_len)

        # ── 3. VM builds structured context via DISCOVER ─────────────────
        # Encode each text as a block-code using positional binding (SEQ)
        vecs_a = [bc.random_discrete(seed=int(t) + 1) for t in ids_a[:8]]
        vecs_b = [bc.random_discrete(seed=int(t) + 1) for t in ids_b[:8]]
        bc_a = vm.execute("SEQ", vecs_a) if len(vecs_a) > 0 else bc.random_discrete(seed=0)
        bc_b = vm.execute("SEQ", vecs_b) if len(vecs_b) > 0 else bc.random_discrete(seed=1)

        # DISCOVER the transition rule between the two texts
        discovery = vm.execute("DISCOVER", bc_a, bc_b)

        # The context for MindForge is the STRUCTURED delta, not a random hash
        if discovery["rule_type"] == "bind" and discovery["delta"] is not None:
            context = discovery["delta"]  # structured transformation vector
        else:
            context = bc.bind(bc_a, bc_b)  # fallback: bind the two encodings

        # Store in cleanup memory for SDLS purification
        pattern_key = f"step_{step % 1000}"
        vm.cleanup_mem.store(pattern_key, context)

        # ── 4. Teacher soft targets ──────────────────────────────────────
        teacher_probs_b = np.asarray(
            softmax(logits_b / config.temperature), dtype=np.float32,
        )
        teacher_probs_b = np.clip(teacher_probs_b, 1e-7, 1.0)

        # ── 5. Forge adapter from structured context ─────────────────────
        layer_id = int(rng.integers(0, config.teacher_n_layers))
        A, B = forge.forge(context, layer_id)

        # ── 6. LR schedule ───────────────────────────────────────────────
        lr = config.lr
        if step < config.warmup_steps:
            lr = config.lr * max(step, 1) / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / max(
                config.train_steps - config.warmup_steps, 1,
            )
            lr = config.lr_min + 0.5 * (config.lr - config.lr_min) * (
                1 + math.cos(math.pi * progress)
            )

        # ── 7. CE + KL loss ──────────────────────────────────────────────
        # Use teacher logits_a as the "hidden state" input to the adapter
        # (text_a is what the model saw, adapter should help predict text_b)
        d = config.teacher_d_model
        tk = min(config.top_k_logits, logits_a.shape[1])
        teacher_mean_a = logits_a.mean(axis=0).astype(np.float32)[:tk]  # (tk,)

        h_input = np.zeros(d, dtype=np.float32)
        h_input[:tk] = teacher_mean_a

        # Apply adapter
        lora_out = (h_input @ A.T @ B.T).astype(np.float32)
        student_out = h_input + forge.scale * lora_out

        # Student probabilities
        s_logits = student_out[:tk]
        s_scaled = s_logits / config.temperature
        s_shifted = s_scaled - np.max(s_scaled)
        s_exp = np.exp(np.clip(s_shifted, -20, 20))
        s_probs = (s_exp / (np.sum(s_exp) + 1e-8)).astype(np.float32)
        s_probs = np.clip(s_probs, 1e-7, 1.0)

        # Teacher target (text_b distribution)
        t_probs = teacher_probs_b.mean(axis=0).astype(np.float32)[:tk]
        t_probs = np.clip(t_probs, 1e-7, 1.0)

        # KL divergence
        kl = float(np.sum(t_probs * (np.log(t_probs) - np.log(s_probs)))) * (config.temperature ** 2)
        kl = min(kl, 100.0)  # cap to prevent explosion

        # CE
        labels = ids_b[1:] if len(ids_b) > 1 else ids_b
        ce = 0.0
        for t in range(min(len(labels), config.seq_len)):
            tok = int(labels[t]) % tk
            ce -= math.log(max(float(s_probs[tok]), 1e-8))
        ce /= max(len(labels), 1)

        loss = 0.3 * ce + 0.6 * kl

        # ── 8. Error-driven update ───────────────────────────────────────
        grad_logits = (s_probs - t_probs).astype(np.float32)

        # Backprop through LoRA
        grad_lora = np.zeros(d, dtype=np.float32)
        grad_lora[:tk] = grad_logits / max(config.temperature, 0.1)
        grad_A = np.outer(grad_lora[:forge.rank], h_input)
        grad_B = np.outer(h_input, grad_lora[:forge.rank])

        # Get mixing coefficients
        ctx_flat = bc.to_flat(context).astype(np.float32)
        ctx_proj = (ctx_flat @ forge.W_proj.T).astype(np.float32)
        layer_emb = forge.layer_embeddings[layer_id]
        combined = np.concatenate([ctx_proj, layer_emb])
        from cubemind.execution.mindforge import gelu
        h_h = np.asarray(gelu(combined @ forge.W_h.T + forge.b_h), dtype=np.float32)
        c_raw = h_h @ forge.W_coeff.T + forge.b_coeff
        c_exp = np.exp(c_raw - np.max(c_raw))
        c_soft = (c_exp / np.sum(c_exp)).astype(np.float32)

        # Update basis adapters (proportional to coefficient)
        for i in range(forge.n_basis):
            w = float(c_soft[i])
            if w < 0.01:
                continue
            forge.A_basis[i] -= lr * w * np.clip(grad_A, -0.05, 0.05).astype(np.float32)
            forge.B_basis[i] -= lr * w * np.clip(grad_B, -0.05, 0.05).astype(np.float32)

        # Update W_proj (slow)
        gp = np.outer(grad_lora[:forge.W_proj.shape[0]], ctx_flat[:forge.W_proj.shape[1]])
        if gp.shape == forge.W_proj.shape:
            forge.W_proj -= lr * 0.01 * np.clip(gp, -0.01, 0.01).astype(np.float32)

        # Update W_coeff via coefficient error (push toward uniform when loss high)
        if loss > 1.0:
            uniform = np.ones(forge.n_basis, dtype=np.float32) / forge.n_basis
            coeff_error = c_soft - uniform
            grad_wc = np.outer(coeff_error, h_h[:forge.W_coeff.shape[1]])
            if grad_wc.shape == forge.W_coeff.shape:
                forge.W_coeff -= lr * 0.1 * np.clip(grad_wc, -0.01, 0.01).astype(np.float32)

        # Track loss
        loss_ema = 0.99 * loss_ema + 0.01 * loss if step > 0 else loss

        if step % config.log_every == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "step={:6d} | loss={:.4f} | ema={:.4f} | lr={:.6f} | {:.1f} stp/s",
                step, loss, loss_ema, lr, sps,
            )

        if step % config.save_every == 0 and step > 0:
            save_path = f"data/checkpoints/harrier_forge_step{step}.npz"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(
                save_path,
                W_proj=forge.W_proj,
                W_h=forge.W_h,
                b_h=forge.b_h,
                W_coeff=forge.W_coeff,
                b_coeff=forge.b_coeff,
                A_basis=forge.A_basis,
                B_basis=forge.B_basis,
                layer_embeddings=forge.layer_embeddings,
                step=step,
                loss_ema=loss_ema,
            )
            logger.info("Saved checkpoint: {}", save_path)

            if loss_ema < best_loss:
                best_loss = loss_ema

    elapsed = time.time() - t0
    logger.info("Training done: {} steps in {:.0f}s, best_loss={:.4f}", config.train_steps, elapsed, best_loss)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Harrier 0.6B MindForge pre-training")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    config = HarrierPretrainConfig(
        train_steps=args.steps,
        seq_len=args.seq_len,
        lr=args.lr,
        save_every=args.save_every,
        log_every=args.log_every,
        k=16,
        l=32
    )
    train(config)


if __name__ == "__main__":
    main()
