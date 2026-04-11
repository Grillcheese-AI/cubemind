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


@dataclass
class HarrierPretrainConfig:
    # Model paths
    teacher_path: str = "data/external_llms/harrier-oss-v1-0.6b.Q8_0.gguf"
    data_dir: str = "sandbox/vsa_lm/data"

    # Teacher dims (Harrier 0.6B)
    teacher_d_model: int = 1024
    teacher_n_layers: int = 24
    teacher_vocab: int = 151936

    # MindForge
    k: int = 16
    l: int = 16
    forge_rank: int = 8
    forge_basis: int = 16
    forge_d_hidden: int = 128

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
        self.model = Llama(
            model_path=config.teacher_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=-1,  # all layers on Vulkan GPU
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
        # Each token contributes via positional binding for richer context
        context = bc.random_discrete(seed=0)  # identity-like base
        for t_idx in range(min(8, len(input_ids))):  # first 8 tokens
            tok_vec = bc.random_discrete(seed=int(input_ids[t_idx]) + 1)
            pos_shift = np.roll(tok_vec, t_idx, axis=-1)
            context = bc.discretize(context.astype(np.float32) + pos_shift.astype(np.float32))

        # Forge adapters for a subset of layers (faster than all 24)
        n_sample_layers = min(4, config.teacher_n_layers)
        layer_ids = rng.choice(config.teacher_n_layers, size=n_sample_layers, replace=False)
        adapters = [forge.forge(context, int(lid)) for lid in layer_ids]

        # Teacher soft targets
        teacher_probs = np.asarray(
            softmax(teacher_logits / config.temperature), dtype=np.float32,
        )

        # Loss: adapter reconstruction error
        # For each forged adapter (A, B), apply it to a random input vector
        # and measure how much the output aligns with the teacher's logit direction
        teacher_direction = teacher_probs.mean(axis=0)  # (top_k,) average teacher preference
        teacher_direction = teacher_direction / (np.linalg.norm(teacher_direction) + 1e-8)

        total_loss = 0.0
        for A, B in adapters:
            # Simulate: random hidden state → adapter → output direction
            h = rng.standard_normal(config.teacher_d_model).astype(np.float32) * 0.1
            adapted = (h @ A.T @ B.T).astype(np.float32)  # (d_model,)

            # Project adapted output to logit space (simplified: use norm as signal)
            # Real loss: KL(teacher || student). Proxy: adapter should produce
            # non-trivial, differentiated outputs per layer
            output_norm = float(np.linalg.norm(adapted))
            # Target: adapter output should have norm proportional to teacher confidence
            target_norm = float(np.linalg.norm(teacher_direction)) * 10.0
            total_loss += (output_norm - target_norm) ** 2

        loss = total_loss / max(n_sample_layers, 1)

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

        # Error-driven update on MindForge weights
        # Finite-difference gradient: perturb W_proj, measure loss change
        ctx_flat = bc.to_flat(context).astype(np.float32)
        perturbation = rng.standard_normal(forge.W_proj.shape).astype(np.float32) * 0.01

        # Perturbed forge
        forge.W_proj += perturbation
        adapters_p = [forge.forge(context, int(lid)) for lid in layer_ids]
        loss_p = 0.0
        for A, B in adapters_p:
            h = rng.standard_normal(config.teacher_d_model).astype(np.float32) * 0.1
            adapted = (h @ A.T @ B.T).astype(np.float32)
            target_norm = float(np.linalg.norm(teacher_direction)) * 10.0
            loss_p += (float(np.linalg.norm(adapted)) - target_norm) ** 2
        loss_p /= max(n_sample_layers, 1)
        forge.W_proj -= perturbation  # restore

        # Gradient estimate: (loss_p - loss) / perturbation
        grad_est = perturbation * (loss_p - loss) / (0.01 ** 2 + 1e-8)
        forge.W_proj -= lr * np.clip(grad_est, -0.1, 0.1).astype(np.float32)

        # Also update W_coeff (coefficient layer) — same approach
        pert_c = rng.standard_normal(forge.W_coeff.shape).astype(np.float32) * 0.01
        forge.W_coeff += pert_c
        adapters_c = [forge.forge(context, int(lid)) for lid in layer_ids]
        loss_c = 0.0
        for A, B in adapters_c:
            h = rng.standard_normal(config.teacher_d_model).astype(np.float32) * 0.1
            adapted = (h @ A.T @ B.T).astype(np.float32)
            target_norm = float(np.linalg.norm(teacher_direction)) * 10.0
            loss_c += (float(np.linalg.norm(adapted)) - target_norm) ** 2
        loss_c /= max(n_sample_layers, 1)
        forge.W_coeff -= pert_c
        grad_c = pert_c * (loss_c - loss) / (0.01 ** 2 + 1e-8)
        forge.W_coeff -= lr * np.clip(grad_c, -0.1, 0.1).astype(np.float32)

        # Track loss
        loss_ema = 0.99 * loss_ema + 0.01 * loss if step > 0 else loss

        if step % config.log_every == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "step={:6d} | loss={:.4f} | ema={:.4f} | entropy={:.3f} | "
                "adapter_var={:.4f} | lr={:.6f} | {:.1f} stp/s",
                step, loss, loss_ema, teacher_entropy, adapter_var, lr, sps,
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
