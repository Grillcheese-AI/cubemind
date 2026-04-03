"""HE-MoE on TinyStories — pure grilly, no torch.

Uses grilly.nn for layers, grilly.backend for GPU dispatch.
No torch dependency except for dataset loading (HF datasets).

Run: python -u sandbox/he_moe_tinystories/experiment_grilly.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

# grilly imports
from grilly import nn as gnn
from grilly.optim import AdamW

from cubemind.brain.addition_linear import AdditionLinear, SignActivation
from cubemind.functional import F


@dataclass
class Config:
    seq_len: int = 64
    hidden_dim: int = 128
    n_experts: int = 8
    top_k: int = 2
    fast_lr: float = 0.01
    slow_lr: float = 0.005
    sigma: float = 2.0
    fast_ratio: float = 0.7
    train_steps: int = 5000
    warmup_steps: int = 500
    val_every: int = 50
    batch_size: int = 32
    max_stories: int = 3000
    use_addition_only: bool = False


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(cfg: Config):
    """Load TinyStories with SentencePiece BPE tokenizer."""
    from datasets import load_dataset
    import sentencepiece as spm
    import tempfile
    import os

    logger.info("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Collect raw text
    texts = []
    for i, item in enumerate(ds):
        if i >= cfg.max_stories:
            break
        texts.append(item["text"])
    logger.info("Loaded {} stories", len(texts))

    # Train SentencePiece BPE model on the data
    sp_model_prefix = os.path.join(tempfile.gettempdir(), "tinystories_sp")
    corpus_file = sp_model_prefix + "_corpus.txt"

    with open(corpus_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    vocab_target = min(4000, max(1000, len(texts) // 2))
    logger.info("Training SentencePiece BPE (vocab={})", vocab_target)
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=sp_model_prefix,
        vocab_size=vocab_target,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=3,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_prefix + ".model")
    vocab_size = sp.get_piece_size()
    logger.info("SentencePiece vocab: {}", vocab_size)

    # Tokenize all text
    all_tokens = []
    for t in texts:
        all_tokens.extend(sp.encode(t, out_type=int))
    mapped = np.array(all_tokens, dtype=np.int32)
    logger.info("Total tokens: {}", len(mapped))

    # Split 80/10/10 + make sequences
    n = len(mapped)
    train_tok = mapped[:int(0.8 * n)]
    val_tok = mapped[int(0.8 * n):int(0.9 * n)]

    def make_seqs(tok):
        x, y = [], []
        for i in range(0, len(tok) - cfg.seq_len - 1, cfg.seq_len // 2):
            x.append(tok[i:i + cfg.seq_len])
            y.append(tok[i + 1:i + cfg.seq_len + 1])
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    train_x, train_y = make_seqs(train_tok)
    val_x, val_y = make_seqs(val_tok)
    logger.info("Train: {} seqs, Val: {} seqs", len(train_x), len(val_x))

    return train_x, train_y, val_x, val_y, vocab_size, sp


# ── Model (pure grilly/numpy) ───────────────────────────────────────────────

class HEMoEGrilly:
    """HE-MoE using grilly nn layers. No torch."""

    def __init__(self, vocab_size: int, cfg: Config, seed: int = 42):
        self.cfg = cfg
        d = cfg.hidden_dim
        rng = np.random.default_rng(seed)

        # Embedding (lookup table)
        self.embed = rng.normal(0, 0.02, (vocab_size, d)).astype(np.float32)

        # Positional encoding (sinusoidal, precomputed)
        pe = np.zeros((cfg.seq_len + 16, d), dtype=np.float32)
        pos = np.arange(cfg.seq_len + 16).reshape(-1, 1).astype(np.float32)
        div = np.exp(np.arange(0, d, 2).astype(np.float32) * -(math.log(10000) / d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d // 2])
        self.pe = pe

        # Fast path: grilly Linear → ReLU → Linear
        self.fast_w1 = rng.normal(0, 0.02, (d, d)).astype(np.float32)
        self.fast_w2 = rng.normal(0, 0.01, (vocab_size, d)).astype(np.float32)

        # Experts
        self.expert_mu = rng.normal(0, 0.5, (cfg.n_experts, d)).astype(np.float32)
        self.expert_w = rng.normal(0, 0.02, (cfg.n_experts, d, d)).astype(np.float32)
        self.expert_charge = np.ones(cfg.n_experts, dtype=np.float32)
        self.expert_charge[1::2] = -1.0

        # Gated recurrence
        self.expert_h = np.zeros((cfg.n_experts, d), dtype=np.float32)
        self.expert_gate = np.ones(cfg.n_experts, dtype=np.float32) * 0.5

        # Traces
        self.a_trace = np.zeros(cfg.n_experts, dtype=np.float32)
        self.e_trace = np.zeros((cfg.n_experts, d), dtype=np.float32)

        self._step = 0

    def _softmax(self, x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

    def forward(self, x_ids: np.ndarray) -> np.ndarray:
        """x_ids: (batch, seq) → logits: (batch*seq, vocab)."""
        B, S = x_ids.shape
        d = self.cfg.hidden_dim

        # Embed + position
        h = self.embed[x_ids].reshape(B * S, d)  # (B*S, d)
        h = h + self.pe[:S].reshape(1, S, d).repeat(B, axis=0).reshape(B * S, d)

        # Fast path: Linear → ReLU → Linear (grilly _bridge if available)
        h_fast = np.maximum(h @ self.fast_w1.T, 0)  # ReLU
        logits = h_fast @ self.fast_w2.T  # (B*S, vocab)

        # Slow path: electrostatic routing
        # Kernel scores
        scores = np.zeros((B * S, self.cfg.n_experts), dtype=np.float32)
        for i in range(self.cfg.n_experts):
            diff = h - self.expert_mu[i]
            dist_sq = np.sum(diff * diff, axis=-1)
            kernel = np.exp(-dist_sq / (2 * self.cfg.sigma ** 2))
            scores[:, i] = self.expert_charge[i] * kernel

        # Top-k (global)
        mean_scores = scores.mean(axis=0)
        top_idx = np.argsort(mean_scores)[-self.cfg.top_k:][::-1]

        slow_out = np.zeros((B * S, d), dtype=np.float32)
        for idx in top_idx:
            w = scores[:, idx:idx + 1]  # (B*S, 1)
            expert_out = h @ self.expert_w[idx].T

            # Gated recurrence
            gate = 1.0 / (1.0 + np.exp(-self.expert_gate[idx]))
            self.expert_h[idx] = gate * self.expert_h[idx] + (1 - gate) * expert_out.mean(axis=0)
            slow_out += w * (expert_out + self.expert_h[idx])

        # Residual merge
        h_merged = h_fast + (1.0 - self.cfg.fast_ratio) * slow_out
        logits = h_merged @ self.fast_w2.T

        return logits

    def train_step(self, x_ids: np.ndarray, y_ids: np.ndarray, lr_scale: float = 1.0) -> float:
        """One step: forward + Hebbian update. Returns loss."""
        self._step += 1
        B, S = x_ids.shape
        d = self.cfg.hidden_dim
        lr = self.cfg.fast_lr * lr_scale

        # Forward (inline for access to intermediates)
        h = self.embed[x_ids].reshape(B * S, d)
        h = h + self.pe[:S].reshape(1, S, d).repeat(B, axis=0).reshape(B * S, d)

        h_fast = np.maximum(h @ self.fast_w1.T, 0)
        logits = h_fast @ self.fast_w2.T

        # Loss
        y_flat = y_ids.ravel()
        probs = self._softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(len(y_flat)), y_flat] + 1e-8))

        # Error
        target = np.zeros_like(probs)
        target[np.arange(len(y_flat)), y_flat] = 1.0
        error = target - probs

        # Fast W2 update (Hebbian delta)
        grad_w2 = error.T @ h_fast / (B * S)
        self.fast_w2 += np.clip(lr * grad_w2, -0.1, 0.1)

        # Fast W1 update
        error_h = (error @ self.fast_w2) * (h_fast > 0).astype(np.float32)
        grad_w1 = error_h.T @ h / (B * S)
        self.fast_w1 += np.clip(lr * 0.5 * grad_w1, -0.1, 0.1)

        # Expert updates (error-driven + Coulomb)
        error_hidden = (error.mean(axis=0) @ self.fast_w2).astype(np.float32)
        h_mean = h.mean(axis=0)

        # Kernel scores for routing
        scores = np.zeros(self.cfg.n_experts, dtype=np.float32)
        for i in range(self.cfg.n_experts):
            diff = h_mean - self.expert_mu[i]
            scores[i] = self.expert_charge[i] * np.exp(-np.dot(diff, diff) / (2 * self.cfg.sigma ** 2))
        top_idx = np.argsort(scores)[-self.cfg.top_k:][::-1]

        for idx in top_idx:
            # Coulomb force
            force = self.expert_charge[idx] * (h_mean - self.expert_mu[idx])
            self.expert_mu[idx] += self.cfg.slow_lr * np.clip(force, -0.5, 0.5)

            # Error-driven expert weight update
            grad_ew = np.outer(error_hidden, h_mean)
            self.expert_w[idx] += np.clip(self.cfg.slow_lr * grad_ew, -0.1, 0.1)

            # Traces
            self.a_trace[idx] = 0.9 * self.a_trace[idx] + abs(scores[idx])
            self.e_trace[idx] = 0.9 * self.e_trace[idx] + error_hidden

        return float(loss)


def compute_ppl(model: HEMoEGrilly, x_data, y_data, batch_size=32, max_batches=50):
    """Compute perplexity."""
    total_loss, total_tokens = 0.0, 0
    for i in range(0, min(len(x_data), max_batches * batch_size), batch_size):
        x_batch = x_data[i:i + batch_size]
        y_batch = y_data[i:i + batch_size]
        if len(x_batch) == 0:
            break
        logits = model.forward(x_batch)
        y_flat = y_batch.ravel()
        probs = model._softmax(logits)
        log_probs = np.log(probs[np.arange(len(y_flat)), y_flat] + 1e-8)
        total_loss -= log_probs.sum()
        total_tokens += len(y_flat)
    avg = total_loss / max(total_tokens, 1)
    return float(np.exp(min(avg, 20))), float(avg)


def lr_schedule(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def main():
    cfg = Config(
        train_steps=5000,
        max_stories=3000,
        batch_size=32,
        n_experts=8,
        top_k=2,
    )

    train_x, train_y, val_x, val_y, vocab_size, enc = load_data(cfg)
    model = HEMoEGrilly(vocab_size, cfg)

    logger.info("HE-MoE grilly: vocab={}, d={}, experts={}", vocab_size, cfg.hidden_dim, cfg.n_experts)

    t0 = time.time()
    best_ppl = float("inf")
    n_batches = len(train_x) // cfg.batch_size
    step = 0

    while step < cfg.train_steps:
        # Shuffle each epoch
        perm = np.random.permutation(len(train_x))
        train_x, train_y = train_x[perm], train_y[perm]

        for i in range(0, len(train_x) - cfg.batch_size, cfg.batch_size):
            if step >= cfg.train_steps:
                break

            x_batch = train_x[i:i + cfg.batch_size]
            y_batch = train_y[i:i + cfg.batch_size]

            lr_s = lr_schedule(step, cfg.warmup_steps, cfg.train_steps)
            loss = model.train_step(x_batch, y_batch, lr_scale=lr_s)

            if step % cfg.val_every == 0:
                val_ppl, val_loss = compute_ppl(model, val_x, val_y)
                logger.info("step={:5d} lr={:.4f} | loss={:.3f} | val_ppl={:.2f}",
                            step, cfg.fast_lr * lr_s, loss, val_ppl)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl

            step += 1

    elapsed = time.time() - t0
    val_ppl, _ = compute_ppl(model, val_x, val_y)

    print(f"\n{'='*50}")
    print(f"  HE-MoE (grilly, no torch) Results")
    print(f"{'='*50}")
    print(f"  Final val PPL:  {val_ppl:.2f}")
    print(f"  Best val PPL:   {best_ppl:.2f}")
    print(f"  Steps:          {step}")
    print(f"  Time:           {elapsed:.0f}s")
    print(f"  Steps/sec:      {step / elapsed:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
