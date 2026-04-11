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

        # ── EGGROLL: rank-1 ES on basis adapters (backprop-free) ──────────
        # Generate N perturbations, evaluate each, keep top-k, update.
        n_workers = 16
        top_k_keep = 4

        basis_idx = int(rng.integers(0, forge.n_basis))
        orig_A = forge.A_basis[basis_idx].copy()
        orig_B = forge.B_basis[basis_idx].copy()

        perturbations = []
        losses_p = []

        for _ in range(n_workers):
            # Rank-1 perturbation
            dA = (rng.standard_normal((forge.rank, 1)) @ rng.standard_normal((1, forge.d_target))
                  ).astype(np.float32) * 0.02
            dB = (rng.standard_normal((forge.d_target, 1)) @ rng.standard_normal((1, forge.rank))
                  ).astype(np.float32) * 0.02

            forge.A_basis[basis_idx] = orig_A + dA
            forge.B_basis[basis_idx] = orig_B + dB

            A_p, B_p = forge.forge(context, layer_id)
            out_p = (target_vec @ A_p.T @ B_p.T).astype(np.float32)
            n_p = np.linalg.norm(out_p)
            if n_p > 1e-8:
                out_p /= n_p
            loss_p = 1.0 - float(np.dot(out_p, target_vec))

            perturbations.append((dA, dB))
            losses_p.append(loss_p)

        # Restore
        forge.A_basis[basis_idx] = orig_A
        forge.B_basis[basis_idx] = orig_B

        # Select top-k lowest loss, merit-weighted update
        ranked = np.argsort(losses_p)[:top_k_keep]
        avg_dA = np.zeros_like(orig_A)
        avg_dB = np.zeros_like(orig_B)
        total_w = 0.0
        for idx in ranked:
            merit = max(0.0, loss - losses_p[idx]) * 10.0 + 1.0
            avg_dA += merit * perturbations[idx][0]
            avg_dB += merit * perturbations[idx][1]
            total_w += merit

        if total_w > 0:
            forge.A_basis[basis_idx] += lr * (avg_dA / total_w)
            forge.B_basis[basis_idx] += lr * (avg_dB / total_w)

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
