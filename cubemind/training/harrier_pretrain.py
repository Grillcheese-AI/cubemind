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
    lr: float = 2e-3
    lr_min: float = 3e-4
    train_steps: int = 50000
    log_every: int = 100
    save_every: int = 5000
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
        n_gpu = -1 if config.teacher_gpu else 0
        self.model = Llama(
            model_path=config.teacher_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=n_gpu,
            verbose=False,
            logits_all=True,
        )
        self.vocab_size = config.teacher_vocab
        self.top_k = config.top_k_logits
        logger.info("Teacher loaded: vocab={}, d_model={}", self.vocab_size, config.teacher_d_model)

    def extract_from_text(self, text: str, max_tokens: int = 128) -> tuple[np.ndarray, np.ndarray]:
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

    logger.info("Starting MindForge pre-training: {} steps, {} texts, lr={}",
                config.train_steps, n_texts, config.lr)

    for step in range(config.train_steps):
        # Sample a random text and tokenize with teacher's tokenizer
        text_idx = int(rng.integers(0, n_texts))
        text = texts[text_idx]

        # Extract logits using teacher's own tokenizer
        input_ids, teacher_logits = teacher.extract_from_text(text, max_tokens=config.seq_len)
        labels = input_ids[1:]  # shifted labels

        # Create context block-code from token sequence
        context = bc.random_discrete(seed=0)
        for t_idx in range(min(8, len(input_ids))):
            tok_vec = bc.random_discrete(seed=int(input_ids[t_idx]) + 1)
            pos_shift = np.roll(tok_vec, t_idx, axis=-1)
            context = bc.discretize(context.astype(np.float32) + pos_shift.astype(np.float32))

        # Teacher soft targets: (seq, top_k) → probabilities
        teacher_probs = np.asarray(
            softmax(teacher_logits / config.temperature), dtype=np.float32,
        )
        teacher_probs = np.clip(teacher_probs, 1e-7, 1.0)

        # Forge adapter for one random layer
        layer_id = int(rng.integers(0, config.teacher_n_layers))
        A, B = forge.forge(context, layer_id)  # A: (rank, d), B: (d, rank)

        # LR schedule: cosine with warmup
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

        # ── CE + KL loss with direct error-driven update ──────────────────
        # Same approach as vsa_lm.py: compute error, update weights directly.
        # No random hidden states — use teacher logits as both input and target.

        # Teacher probs as target distribution: (seq, top_k)
        # Student: forge adapter, apply to teacher's mean hidden state estimate
        # The adapter should learn to project the input toward teacher's output dist.

        # Use teacher logits mean as a proxy for the hidden state at this layer
        teacher_mean = teacher_logits.mean(axis=0).astype(np.float32)  # (top_k,)

        # Pad/truncate to d_target for adapter application
        d = config.teacher_d_model
        tk = len(teacher_mean)
        if tk >= d:
            h_input = teacher_mean[:d]
        else:
            h_input = np.zeros(d, dtype=np.float32)
            h_input[:tk] = teacher_mean

        # Apply LoRA adapter: output = h + scale * h @ A.T @ B.T
        lora_out = (h_input @ A.T @ B.T).astype(np.float32)  # (d,)
        student_out = h_input + forge.scale * lora_out

        # Project student output to logit-sized space
        s_logits = student_out[:tk]
        s_scaled = s_logits / config.temperature
        s_shifted = s_scaled - np.max(s_scaled)
        s_exp = np.exp(s_shifted)
        s_probs = (s_exp / (np.sum(s_exp) + 1e-8)).astype(np.float32)
        s_probs = np.clip(s_probs, 1e-7, 1.0)

        # Teacher target (average across sequence positions)
        t_probs = teacher_probs.mean(axis=0).astype(np.float32)
        t_probs = np.clip(t_probs, 1e-7, 1.0)

        # KL divergence: teacher || student
        kl = float(np.sum(t_probs * (np.log(t_probs) - np.log(s_probs)))) * (config.temperature ** 2)

        # CE on hard labels
        ce = 0.0
        for t in range(min(len(labels), config.seq_len)):
            tok = int(labels[t]) % tk
            ce -= math.log(max(float(s_probs[tok]), 1e-8))
        ce /= max(config.seq_len, 1)

        loss = 0.3 * ce + 0.6 * kl

        # ── Direct error-driven update (like vsa_lm.py) ──────────────────
        # Error: gradient of CE+KL w.r.t. student logits = s_probs - t_probs
        grad_logits = (s_probs - t_probs).astype(np.float32)  # (top_k,)

        # Backprop through output projection: grad w.r.t. lora_out
        grad_lora = np.zeros(d, dtype=np.float32)
        grad_lora[:tk] = grad_logits / config.temperature

        # Backprop through LoRA: lora_out = h @ A.T @ B.T
        # grad_A ≈ outer(B @ grad_lora, h) → (rank, d)
        # grad_B ≈ outer(h, A @ grad_lora) → (d, rank)
        grad_A_direct = np.outer(grad_lora[:forge.rank], h_input)  # (rank, d)
        grad_B_direct = np.outer(h_input, grad_lora[:forge.rank])  # (d, rank)

        # Get softmax coefficients for this context+layer
        ctx_flat = bc.to_flat(context).astype(np.float32)
        ctx_proj = (ctx_flat @ forge.W_proj.T).astype(np.float32)
        layer_emb = forge.layer_embeddings[layer_id]
        combined = np.concatenate([ctx_proj, layer_emb])
        from cubemind.execution.mindforge import gelu
        h_hidden = np.asarray(gelu(combined @ forge.W_h.T + forge.b_h), dtype=np.float32)
        coeffs_raw = h_hidden @ forge.W_coeff.T + forge.b_coeff
        coeffs_exp = np.exp(coeffs_raw - np.max(coeffs_raw))
        coeffs_soft = (coeffs_exp / np.sum(coeffs_exp)).astype(np.float32)

        # Update each basis adapter proportional to its coefficient
        for i in range(forge.n_basis):
            w = float(coeffs_soft[i])
            if w < 0.01:
                continue
            forge.A_basis[i] -= lr * w * np.clip(grad_A_direct, -0.1, 0.1).astype(np.float32)
            forge.B_basis[i] -= lr * w * np.clip(grad_B_direct, -0.1, 0.1).astype(np.float32)

        # Update output projection (W_proj) — scatter-add style
        grad_proj = np.outer(
            (grad_lora[:forge.W_proj.shape[0]]),
            ctx_flat[:forge.W_proj.shape[1]],
        )
        if grad_proj.shape == forge.W_proj.shape:
            forge.W_proj -= lr * 0.01 * np.clip(grad_proj, -0.01, 0.01).astype(np.float32)

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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    config = HarrierPretrainConfig(
        train_steps=args.steps,
        seq_len=args.seq_len,
        lr=args.lr,
        save_every=args.save_every,
        log_every=args.log_every,
    )
    train(config)


if __name__ == "__main__":
    main()
