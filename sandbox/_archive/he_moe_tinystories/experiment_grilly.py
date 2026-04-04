"""HE-MoE on TinyStories — numpy BLAS for matmul, grilly for future scale.

At d=256, V=4000, batch=64: numpy BLAS beats Vulkan compute due to dispatch
overhead. Uses optimized numpy path now. When scaling to d=1024+ / V=32K+,
swap to _bridge.linear for GPU acceleration.

Run: python -u sandbox/he_moe_tinystories/experiment_grilly.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class Config:
    seq_len: int = 128
    hidden_dim: int = 128
    n_experts: int = 32
    top_k: int = 2
    fast_lr: float = 0.03
    slow_lr: float = 0.001
    sigma: float = 2.0
    fast_ratio: float = 0.7
    train_steps: int = 5000
    warmup_steps: int = 500
    val_every: int = 50
    batch_size: int = 64
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


# ── Model ────────────────────────────────────────────────────────────────────

class HEMoEGrilly:
    """HE-MoE with numpy BLAS. 32 experts, Coulomb routing, Hebbian update."""

    def __init__(self, vocab_size: int, cfg: Config, seed: int = 42):
        self.cfg = cfg
        d = cfg.hidden_dim
        rng = np.random.default_rng(seed)

        self.embed = rng.normal(0, 0.02, (vocab_size, d)).astype(np.float32)

        pe = np.zeros((cfg.seq_len + 16, d), dtype=np.float32)
        pos = np.arange(cfg.seq_len + 16).reshape(-1, 1).astype(np.float32)
        div = np.exp(np.arange(0, d, 2).astype(np.float32) * -(math.log(10000) / d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d // 2])
        self.pe = pe

        self.fast_w1 = rng.normal(0, 0.02, (d, d)).astype(np.float32)
        self.fast_w2 = rng.normal(0, 0.01, (vocab_size, d)).astype(np.float32)

        self.expert_mu = rng.normal(0, 0.5, (cfg.n_experts, d)).astype(np.float32)
        self.expert_w = rng.normal(0, 0.02, (cfg.n_experts, d, d)).astype(np.float32)
        self.expert_charge = np.ones(cfg.n_experts, dtype=np.float32)
        self.expert_charge[1::2] = -1.0

        self.expert_h = np.zeros((cfg.n_experts, d), dtype=np.float32)
        self.expert_gate = np.ones(cfg.n_experts, dtype=np.float32) * 0.5

        self.a_trace = np.zeros(cfg.n_experts, dtype=np.float32)
        self.e_trace = np.zeros((cfg.n_experts, d), dtype=np.float32)

        self._step = 0

    def _softmax(self, x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

    def forward(self, x_ids: np.ndarray) -> np.ndarray:
        """x_ids: (batch, seq) -> logits: (batch*seq, vocab)."""
        B, S = x_ids.shape
        d = self.cfg.hidden_dim
        N = B * S

        # Embed + position
        h = self.embed[x_ids].reshape(N, d)
        h = h + np.tile(self.pe[:S], (B, 1))

        # Fast path: linear -> relu -> linear
        h_fast = np.maximum(h @ self.fast_w1.T, 0)

        # Slow path: electrostatic routing on mean(h)
        h_mean = h.mean(axis=0)
        diffs = h_mean - self.expert_mu
        dist_sq = np.sum(diffs * diffs, axis=-1)
        scores = self.expert_charge * np.exp(-dist_sq / (2 * self.cfg.sigma ** 2))
        top_idx = np.argsort(scores)[-self.cfg.top_k:][::-1]

        slow_out = np.zeros((N, d), dtype=np.float32)
        for idx in top_idx:
            expert_out = h @ self.expert_w[idx].T
            gate = 1.0 / (1.0 + np.exp(-self.expert_gate[idx]))
            self.expert_h[idx] = gate * self.expert_h[idx] + (1 - gate) * expert_out.mean(axis=0)
            slow_out += scores[idx] * (expert_out + self.expert_h[idx])

        # Merge + project
        h_merged = h_fast + (1.0 - self.cfg.fast_ratio) * slow_out
        return h_merged @ self.fast_w2.T

    def train_step(self, x_ids: np.ndarray, y_ids: np.ndarray, lr_scale: float = 1.0) -> float:
        """One step: forward + Hebbian update. Optimized to avoid redundant (N,V) matmuls."""
        self._step += 1
        B, S = x_ids.shape
        d = self.cfg.hidden_dim
        N = B * S
        lr = self.cfg.fast_lr * lr_scale

        # Forward
        h = self.embed[x_ids].reshape(N, d)
        h += np.tile(self.pe[:S], (B, 1))  # in-place

        h_fast = np.maximum(h @ self.fast_w1.T, 0)  # (N, d) — 4ms
        logits = h_fast @ self.fast_w2.T              # (N, V) — 45ms

        # Loss (stable log-softmax, no separate softmax alloc)
        y_flat = y_ids.ravel()
        logits -= logits.max(axis=-1, keepdims=True)  # in-place shift
        exp_logits = np.exp(logits)                    # (N, V)
        sum_exp = exp_logits.sum(axis=-1, keepdims=True)
        probs = exp_logits                             # reuse buffer
        probs /= sum_exp                               # in-place normalize
        loss = -np.mean(np.log(probs[np.arange(N), y_flat] + 1e-8))

        # Gradient: probs - onehot (in-place, reuse probs buffer as grad)
        grad = probs  # grad = probs (reuse, no alloc)
        grad[np.arange(N), y_flat] -= 1.0  # grad = probs - onehot

        # W2 update: grad.T @ h_fast — (V, N) @ (N, d) = (V, d) — 45ms
        grad_w2 = grad.T @ h_fast
        grad_w2 /= N
        np.clip(lr * grad_w2, -0.05, 0.05, out=grad_w2)
        self.fast_w2 -= grad_w2

        # W1 update: exploit sparsity of error signal
        # grad_h_fast = grad @ W2 - but grad is (N,V), W2 is (V,d) — 45ms
        # Optimization: W2[y_flat] is the onehot part, rest is -probs @ W2
        # grad_h_fast = probs @ W2 - W2[y_flat]  (but probs was modified above)
        # Just do the matmul — it's the bottleneck but unavoidable for now
        relu_mask = (h_fast > 0)
        grad_h = (grad @ self.fast_w2)  # (N, d) — 45ms
        grad_h *= relu_mask  # in-place mask

        grad_w1 = grad_h.T @ h  # (d, N) @ (N, d) = (d, d) — 4ms
        grad_w1 /= N
        np.clip(lr * 0.5 * grad_w1, -0.05, 0.05, out=grad_w1)
        self.fast_w1 -= grad_w1

        # Expert updates (cheap: only operates on mean vectors)
        h_mean = h.mean(axis=0)  # (d,)
        error_hidden = grad_h.mean(axis=0)  # (d,) — already computed, free

        diffs = h_mean - self.expert_mu
        dist_sq = np.sum(diffs * diffs, axis=-1)
        scores = self.expert_charge * np.exp(-dist_sq / (2 * self.cfg.sigma ** 2))
        top_idx = np.argsort(scores)[-self.cfg.top_k:][::-1]

        for idx in top_idx:
            force = self.expert_charge[idx] * diffs[idx]
            self.expert_mu[idx] += self.cfg.slow_lr * np.clip(force, -0.5, 0.5)

            grad_ew = np.outer(error_hidden, h_mean)
            self.expert_w[idx] -= np.clip(self.cfg.slow_lr * grad_ew, -0.05, 0.05)

            self.a_trace[idx] = 0.9 * self.a_trace[idx] + abs(scores[idx])
            self.e_trace[idx] = 0.9 * self.e_trace[idx] + error_hidden

        return float(loss)


def compute_ppl(model: HEMoEGrilly, x_data, y_data, batch_size=32, max_batches=50):
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
        train_steps=15000,
        max_stories=30000,
        batch_size=64,
        n_experts=32,
        top_k=2,
    )

    train_x, train_y, val_x, val_y, vocab_size, enc = load_data(cfg)
    model = HEMoEGrilly(vocab_size, cfg)

    logger.info("HE-MoE: vocab={}, d={}, experts={}, batch={}, seq={}",
                vocab_size, cfg.hidden_dim, cfg.n_experts, cfg.batch_size, cfg.seq_len)

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

    print(f"\n{'='*50}")
    print(f"  HE-MoE Results (32 experts, BPE)")
    print(f"{'='*50}")
    print(f"  Final val PPL:  {val_ppl:.2f}")
    print(f"  Best val PPL:   {best_ppl:.2f}")
    print(f"  Steps:          {step}")
    print(f"  Time:           {elapsed:.0f}s")
    print(f"  Steps/sec:      {step / elapsed:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
