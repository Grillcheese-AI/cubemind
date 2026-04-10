"""HE-MoE v2 — Hybrid: Coulomb routing + NLMS experts + LiquidStateCell.

Merges the best of HE-MoE (electrostatic expert positioning) with
LiquidMoE (NLMS O(D) updates, UCB exploration, novelty augmentation).

Key insight: LiquidMoE never builds (N, vocab) error matrices. Each expert
is a linear regressor updated with NLMS on its own prediction error. This
makes it O(K*D) per step instead of O(N*V).

Architecture:
  1. Embedding + position → h (N, d)
  2. LiquidStateCell: h += beta * novelty(h, ema_state)
  3. Fast path: h @ W1 → ReLU → h_fast (N, d)
  4. Routing: Coulomb kernel + UCB exploration → top-k experts
  5. Expert forward: h @ expert_w[i] → expert_out (N, d)
  6. Output: (h_fast + slow_out) @ W2 → logits (N, vocab)
  7. Update: NLMS per expert + Coulomb position drift + bandit reward

Run: python -u sandbox/he_moe_tinystories/experiment_hybrid.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class Config:
    seq_len: int = 64
    hidden_dim: int = 256
    n_experts: int = 32
    top_k: int = 4
    fast_lr: float = 0.032231
    expert_mu: float = 0.25       # NLMS step size (from LiquidMoE)
    expert_l2: float = 0.001    # NLMS weight decay
    slow_lr: float = 0.0032231     # Coulomb position update rate
    sigma: float = 2.0           # RBF kernel bandwidth
    fast_ratio: float = 0.8
    explore_c: float = 1.25      # UCB exploration coefficient
    beta_cos: float = 0.80       # Cosine vs reward blend
    alpha_liquid: float = 0.001   # LiquidStateCell EMA decay
    beta_liquid: float = 0.30    # Novelty signal strength
    train_steps: int = 5000
    warmup_steps: int = 1000
    val_every: int = 50
    batch_size: int = 32
    max_stories: int = 30000


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(cfg: Config):
    """Load TinyStories with SentencePiece BPE tokenizer."""
    from datasets import load_dataset
    import sentencepiece as spm
    import tempfile
    import os

    logger.info("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    texts = []
    for i, item in enumerate(ds):
        if i >= cfg.max_stories:
            break
        texts.append(item["text"])
    logger.info("Loaded {} stories", len(texts))

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

    all_tokens = []
    for t in texts:
        all_tokens.extend(sp.encode(t, out_type=int))
    mapped = np.array(all_tokens, dtype=np.int32)
    logger.info("Total tokens: {}", len(mapped))

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


# ── LiquidStateCell ──────────────────────────────────────────────────────────

class LiquidStateCell:
    """EMA state + novelty augmentation from LiquidMoE."""

    def __init__(self, dim: int, alpha: float = 0.08, beta: float = 0.20):
        self.alpha = alpha
        self.beta = beta
        self.state = np.zeros(dim, dtype=np.float32)

    def step(self, h_mean: np.ndarray) -> np.ndarray:
        """Update state, return novelty-augmented representation."""
        self.state = (1.0 - self.alpha) * self.state + self.alpha * h_mean
        deviation = h_mean - self.state
        norm = np.linalg.norm(deviation) + 1e-8
        novelty = deviation / norm
        return h_mean + self.beta * novelty


# ── Model ────────────────────────────────────────────────────────────────────

class HEMoEHybrid:
    """HE-MoE v2: Coulomb routing + NLMS experts + LiquidStateCell."""

    def __init__(self, vocab_size: int, cfg: Config, seed: int = 42):
        self.cfg = cfg
        d = cfg.hidden_dim
        rng = np.random.default_rng(seed)

        # Embedding
        self.embed = rng.normal(0, 0.02, (vocab_size, d)).astype(np.float32)

        # Positional encoding
        pe = np.zeros((cfg.seq_len + 16, d), dtype=np.float32)
        pos = np.arange(cfg.seq_len + 16).reshape(-1, 1).astype(np.float32)
        div = np.exp(np.arange(0, d, 2).astype(np.float32) * -(math.log(10000) / d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d // 2])
        self.pe = pe

        # Fast path: h → W1 → ReLU → W2 → logits
        self.fast_w1 = rng.normal(0, 0.02, (d, d)).astype(np.float32)
        self.fast_w2 = rng.normal(0, 0.01, (vocab_size, d)).astype(np.float32)

        # Expert weights: each expert is a linear (d → d) regressor
        self.expert_w = rng.normal(0, 0.02, (cfg.n_experts, d, d)).astype(np.float32)

        # Coulomb: expert positions + charges in hidden space
        self.expert_mu = rng.normal(0, 0.5, (cfg.n_experts, d)).astype(np.float32)
        self.expert_charge = np.ones(cfg.n_experts, dtype=np.float32)
        self.expert_charge[1::2] = -1.0

        # UCB bandit state (from LiquidMoE)
        self.expert_counts = np.ones(cfg.n_experts, dtype=np.float32)
        self.expert_reward = np.zeros(cfg.n_experts, dtype=np.float32)

        # Gated recurrence
        self.expert_h = np.zeros((cfg.n_experts, d), dtype=np.float32)
        self.expert_gate = np.ones(cfg.n_experts, dtype=np.float32) * 0.5

        # Liquid state cell
        self.cell = LiquidStateCell(d, cfg.alpha_liquid, cfg.beta_liquid)

        self._step = 0
        self._total_t = 1  # for UCB log(t)

    def _softmax(self, x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

    def _route(self, h_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hybrid routing: Coulomb kernel + UCB exploration + reward prior."""
        cfg = self.cfg

        # Coulomb kernel scores (cosine-like via RBF)
        diffs = h_mean - self.expert_mu  # (E, d)
        dist_sq = np.sum(diffs * diffs, axis=-1)  # (E,)
        kernel = self.expert_charge * np.exp(-dist_sq / (2 * cfg.sigma ** 2))

        # UCB exploration bonus
        ucb = cfg.explore_c * np.sqrt(np.log(self._total_t + 1) / (self.expert_counts + 1))

        # Blend: kernel routing + reward prior + UCB
        scores = cfg.beta_cos * kernel + (1.0 - cfg.beta_cos) * self.expert_reward + ucb

        # Top-k
        top_idx = np.argsort(scores)[-cfg.top_k:][::-1]
        weights = np.exp(scores[top_idx])
        weights /= weights.sum() + 1e-8

        return top_idx, weights

    def forward(self, x_ids: np.ndarray) -> np.ndarray:
        """x_ids: (batch, seq) → logits: (batch*seq, vocab)."""
        B, S = x_ids.shape
        d = self.cfg.hidden_dim
        N = B * S

        h = self.embed[x_ids].reshape(N, d)
        h += np.tile(self.pe[:S], (B, 1))

        # Fast path
        h_fast = np.maximum(h @ self.fast_w1.T, 0)

        # Liquid state augmentation on mean
        h_mean = self.cell.step(h.mean(axis=0))

        # Route
        top_idx, weights = self._route(h_mean)

        # Slow path: experts
        slow_out = np.zeros((N, d), dtype=np.float32)
        for j, idx in enumerate(top_idx):
            expert_out = h @ self.expert_w[idx].T
            gate = 1.0 / (1.0 + np.exp(-self.expert_gate[idx]))
            self.expert_h[idx] = gate * self.expert_h[idx] + (1 - gate) * expert_out.mean(axis=0)
            slow_out += weights[j] * (expert_out + self.expert_h[idx])

        # Merge + project
        h_merged = h_fast + (1.0 - self.cfg.fast_ratio) * slow_out
        return h_merged @ self.fast_w2.T

    def train_step(self, x_ids: np.ndarray, y_ids: np.ndarray, lr_scale: float = 1.0) -> float:
        """One step: forward + NLMS expert update + Coulomb drift."""
        self._step += 1
        self._total_t += 1
        B, S = x_ids.shape
        d = self.cfg.hidden_dim
        N = B * S
        lr = self.cfg.fast_lr * lr_scale

        # ── Forward ──
        h = self.embed[x_ids].reshape(N, d)
        h += np.tile(self.pe[:S], (B, 1))

        h_fast = np.maximum(h @ self.fast_w1.T, 0)
        logits = h_fast @ self.fast_w2.T

        # Loss
        y_flat = y_ids.ravel()
        logits -= logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-8)
        loss = -np.mean(np.log(probs[np.arange(N), y_flat] + 1e-8))

        # ── Fast path gradient (same as before, unavoidable (N,V) matmuls) ──
        grad = probs
        grad[np.arange(N), y_flat] -= 1.0

        grad_w2 = grad.T @ h_fast / N
        np.clip(lr * grad_w2, -0.05, 0.05, out=grad_w2)
        self.fast_w2 -= grad_w2

        relu_mask = (h_fast > 0)
        grad_h = (grad @ self.fast_w2) * relu_mask
        grad_w1 = grad_h.T @ h / N
        np.clip(lr * 0.5 * grad_w1, -0.05, 0.05, out=grad_w1)
        self.fast_w1 -= grad_w1

        # ── Liquid state + routing ──
        h_mean = self.cell.step(h.mean(axis=0))
        top_idx, weights = self._route(h_mean)

        # ── NLMS expert updates (O(K*D) — no (N,V) matrix!) ──
        # Use mean hidden as the "input" for NLMS
        # Target: error_hidden from grad backprop
        error_hidden = grad_h.mean(axis=0)  # (d,) — direction of error in hidden space
        denom = np.dot(h_mean, h_mean) + 1e-8  # NLMS normalization

        for j, idx in enumerate(top_idx):
            # Expert prediction on mean
            expert_pred = self.expert_w[idx] @ h_mean  # (d,)
            expert_error = error_hidden - expert_pred   # prediction error

            # NLMS update: w += mu * gate * (h / ||h||^2) * error
            nlms_grad = np.outer(expert_error, h_mean) / denom
            step_size = self.cfg.expert_mu * weights[j] * lr_scale
            self.expert_w[idx] = (1.0 - self.cfg.expert_l2) * self.expert_w[idx]
            self.expert_w[idx] += np.clip(step_size * nlms_grad, -0.05, 0.05)

            # Coulomb position drift
            force = self.expert_charge[idx] * (h_mean - self.expert_mu[idx])
            self.expert_mu[idx] += self.cfg.slow_lr * np.clip(force, -0.5, 0.5)

            # Update bandit counts
            self.expert_counts[idx] += 1

        # Bandit reward: inverse loss
        reward = 1.0 / (1.0 + loss)
        for idx in top_idx:
            self.expert_reward[idx] = 0.9 * self.expert_reward[idx] + 0.1 * reward

        return float(loss)


def compute_ppl(model: HEMoEHybrid, x_data, y_data, batch_size=32, max_batches=50):
    """Compute perplexity."""
    total_loss, total_tokens = 0.0, 0
    for i in range(0, min(len(x_data), max_batches * batch_size), batch_size):
        xb = x_data[i:i + batch_size]
        yb = y_data[i:i + batch_size]
        if len(xb) == 0:
            break
        logits = model.forward(xb)
        y_flat = yb.ravel()
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
        train_steps=8000,
        max_stories=3000,
        batch_size=32,
        n_experts=16,
        top_k=2,
    )

    train_x, train_y, val_x, val_y, vocab_size, enc = load_data(cfg)
    model = HEMoEHybrid(vocab_size, cfg)

    logger.info("HE-MoE Hybrid: vocab={}, d={}, experts={}, batch={}, top_k={}",
                vocab_size, cfg.hidden_dim, cfg.n_experts, cfg.batch_size, cfg.top_k)
    logger.info("  NLMS mu={}, UCB c={}, liquid alpha={}, beta={}",
                cfg.expert_mu, cfg.explore_c, cfg.alpha_liquid, cfg.beta_liquid)

    t0 = time.time()
    best_ppl = float("inf")
    step = 0

    while step < cfg.train_steps:
        perm = np.random.permutation(len(train_x))
        train_x, train_y = train_x[perm], train_y[perm]

        for i in range(0, len(train_x) - cfg.batch_size, cfg.batch_size):
            if step >= cfg.train_steps:
                break

            xb = train_x[i:i + cfg.batch_size]
            yb = train_y[i:i + cfg.batch_size]

            lr_s = lr_schedule(step, cfg.warmup_steps, cfg.train_steps)
            loss = model.train_step(xb, yb, lr_scale=lr_s)

            if step % cfg.val_every == 0:
                val_ppl, val_loss = compute_ppl(model, val_x, val_y)
                elapsed = time.time() - t0
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                logger.info("step={:5d} lr={:.4f} | loss={:.3f} | val_ppl={:.2f} | {:.1f} stp/s",
                            step, cfg.fast_lr * lr_s, loss, val_ppl, sps)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl

            step += 1

    elapsed = time.time() - t0
    val_ppl, _ = compute_ppl(model, val_x, val_y)

    print(f"\n{'='*55}")
    print("  HE-MoE Hybrid (Coulomb + NLMS + Liquid) Results")
    print(f"{'='*55}")
    print(f"  Final val PPL:  {val_ppl:.2f}")
    print(f"  Best val PPL:   {best_ppl:.2f}")
    print(f"  Steps:          {step}")
    print(f"  Time:           {elapsed:.0f}s")
    print(f"  Steps/sec:      {step / elapsed:.1f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
