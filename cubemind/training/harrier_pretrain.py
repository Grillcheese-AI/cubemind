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
    seq_len: int = 64
    n_ctx: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    train_steps: int = 50000
    log_every: int = 100
    save_every: int = 5000
    warmup_steps: int = 500

    # Distillation
    temperature: float = 2.0  # softmax temperature for KL div
    top_k_logits: int = 256   # only distill top-K logits (memory saving)

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

    def extract(self, token_ids: np.ndarray) -> np.ndarray:
        """Extract teacher logits for a token sequence.

        Args:
            token_ids: (seq_len,) int32 array of token IDs.

        Returns:
            (seq_len, top_k) float32 logits (top-K only for memory).
        """
        tokens = token_ids.tolist()
        self.model.reset()  # clear KV cache between extractions
        self.model.eval(tokens)
        full_logits = np.array(self.model.scores[:len(tokens)], dtype=np.float32)

        # Keep only top-K logits per position (saves memory)
        if self.top_k < full_logits.shape[1]:
            top_indices = np.argpartition(full_logits, -self.top_k, axis=1)[:, -self.top_k:]
            top_logits = np.take_along_axis(full_logits, top_indices, axis=1)
            return top_logits
        return full_logits

    def __del__(self):
        if hasattr(self, "model"):
            del self.model


# ── Data Loading ─────────────────────────────────────────────────────────


def load_tokens(config: HarrierPretrainConfig) -> np.ndarray:
    """Load pre-tokenized TinyStories."""
    tokens_path = os.path.join(config.data_dir, "tokens.npy")
    if os.path.exists(tokens_path):
        tokens = np.load(tokens_path)
        logger.info("Loaded {} tokens from {}", len(tokens), tokens_path)
        return tokens
    else:
        logger.error("No tokens found at {}. Run sandbox/vsa_lm/prepare_data.py first.", tokens_path)
        raise FileNotFoundError(tokens_path)


# ── Training Loop ────────────────────────────────────────────────────────


def train(config: HarrierPretrainConfig | None = None):
    if config is None:
        config = HarrierPretrainConfig()

    rng = np.random.default_rng(config.seed)

    # Load data
    tokens = load_tokens(config)
    n_tokens = len(tokens)

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

    logger.info("Starting Harrier pre-training: {} steps, seq_len={}, lr={}",
                config.train_steps, config.seq_len, config.lr)

    for step in range(config.train_steps):
        # Sample a random sequence
        start = rng.integers(0, n_tokens - config.seq_len - 1)
        input_ids = tokens[start:start + config.seq_len].astype(np.int32)

        # Get teacher logits
        teacher_logits = teacher.extract(input_ids)  # (seq, top_k)

        # Create context block-code from token sequence
        context = bc.random_discrete(seed=0)
        for t_idx in range(min(8, len(input_ids))):
            tok_vec = bc.random_discrete(seed=int(input_ids[t_idx]) + 1)
            pos_shift = np.roll(tok_vec, t_idx, axis=-1)
            context = bc.discretize(context.astype(np.float32) + pos_shift.astype(np.float32))

        # Teacher target: project logits to d_model-sized target vector
        # Use top-K logit values as a compact supervision signal
        teacher_probs = np.asarray(
            softmax(teacher_logits / config.temperature), dtype=np.float32,
        )
        # Mean over sequence → (top_k,) target distribution
        target_dist = teacher_probs.mean(axis=0)
        # Project to d_model size via tiling/truncation
        d = config.teacher_d_model
        if len(target_dist) >= d:
            target_vec = target_dist[:d].astype(np.float32)
        else:
            repeats = d // len(target_dist) + 1
            target_vec = np.tile(target_dist, repeats)[:d].astype(np.float32)
        # Normalize to unit vector
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)

        # Forge adapter for one random layer
        layer_id = int(rng.integers(0, config.teacher_n_layers))
        A, B = forge.forge(context, layer_id)

        # Apply adapter to target_vec itself (self-reconstruction)
        # The adapter should learn to preserve/enhance the teacher's signal
        adapted = (target_vec @ A.T @ B.T).astype(np.float32)
        adapted_norm = np.linalg.norm(adapted)
        if adapted_norm > 1e-8:
            adapted = adapted / adapted_norm

        # Loss: negative cosine similarity (want adapted ≈ target direction)
        cosine_sim = float(np.dot(adapted, target_vec))
        loss = 1.0 - cosine_sim  # range [0, 2], 0 = perfect alignment

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

        # ── Gradient update on basis adapters (the actual weights) ────────
        # The forged adapter is: A = sum(coeffs[i] * A_basis[i])
        # To minimize loss, update each basis adapter toward producing
        # adapters that align target_vec with itself.
        #
        # d(loss)/d(A_basis[i]) ≈ coeffs[i] * (target @ B.T).T ⊗ (adapted - target)
        # Simplified: push A_basis toward target alignment, scaled by coefficient

        # Get the softmax coefficients for this context+layer
        ctx_flat = bc.to_flat(context).astype(np.float32)
        ctx_proj = (ctx_flat @ forge.W_proj.T).astype(np.float32)
        layer_emb = forge.layer_embeddings[layer_id]
        combined = np.concatenate([ctx_proj, layer_emb])
        from cubemind.execution.mindforge import gelu
        h = np.asarray(gelu(combined @ forge.W_h.T + forge.b_h), dtype=np.float32)
        coeffs = h @ forge.W_coeff.T + forge.b_coeff
        coeffs_exp = np.exp(coeffs - np.max(coeffs))
        coeffs_soft = (coeffs_exp / np.sum(coeffs_exp)).astype(np.float32)

        # Error direction in d_model space
        error_dir = (target_vec - adapted).astype(np.float32)  # (d_model,)

        # Update each basis adapter proportional to its coefficient
        for i in range(forge.n_basis):
            w = float(coeffs_soft[i])
            if w < 0.01:
                continue  # skip negligible basis

            # Gradient for A_basis[i]: push adapter to better reconstruct target
            grad_A = w * np.outer(
                error_dir[:forge.rank],
                target_vec[:forge.d_target],
            )
            forge.A_basis[i] += lr * np.clip(grad_A, -0.01, 0.01).astype(np.float32)

            # Gradient for B_basis[i]: similar
            grad_B = w * np.outer(
                target_vec[:forge.d_target],
                error_dir[:forge.rank],
            )
            forge.B_basis[i] += lr * np.clip(grad_B, -0.01, 0.01).astype(np.float32)

        # Also update W_proj (slower, smaller LR)
        grad_proj = np.outer(
            (loss * ctx_proj)[:forge.W_proj.shape[0]],
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
