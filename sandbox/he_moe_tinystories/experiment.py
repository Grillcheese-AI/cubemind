"""HE-MoE on TinyStories — real text next-token prediction benchmark.

No backprop through MoE. Fast path (Hebbian) + slow electrostatic path.
Compares against MLP and Softmax MoE baselines.

Run: python sandbox/he_moe_tinystories/experiment.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    seq_len: int = 64
    hidden_dim: int = 128
    n_experts: int = 8
    top_k: int = 2
    fast_lr: float = 0.05
    slow_lr: float = 0.005
    consol_lr: float = 0.002
    repulsion_weight: float = 0.005
    sigma: float = 2.0
    temp: float = 1.0
    fast_ratio: float = 0.7
    train_steps: int = 5000
    val_every: int = 500
    batch_size: int = 32
    max_text_chars: int = 2_000_000  # Limit for quick testing
    device: str = "cpu"


# ── Data ─────────────────────────────────────────────────────────────────────

def load_tinystories(cfg: Config):
    """Load TinyStories, build char-level vocab, create sequences."""
    from datasets import load_dataset

    logger.info("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Collect text (limited for speed)
    texts = []
    total_chars = 0
    for item in ds:
        texts.append(item["text"])
        total_chars += len(item["text"])
        if total_chars >= cfg.max_text_chars:
            break
    all_text = "\n".join(texts)
    logger.info("Loaded {} chars from {} stories", len(all_text), len(texts))

    # Char vocab
    chars = sorted(set(all_text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    logger.info("Vocab: {} chars", vocab_size)

    # Encode
    encoded = [c2i[c] for c in all_text]

    # Split 80/10/10
    n = len(encoded)
    train_enc = encoded[:int(0.8 * n)]
    val_enc = encoded[int(0.8 * n):int(0.9 * n)]
    test_enc = encoded[int(0.9 * n):]

    def make_dataset(enc):
        x, y = [], []
        for i in range(0, len(enc) - cfg.seq_len - 1, cfg.seq_len):
            x.append(enc[i:i + cfg.seq_len])
            y.append(enc[i + 1:i + cfg.seq_len + 1])
        return TensorDataset(
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

    train_ds = make_dataset(train_enc)
    val_ds = make_dataset(val_enc)
    test_ds = make_dataset(test_enc)

    logger.info("Train: {} seqs, Val: {} seqs, Test: {} seqs",
                len(train_ds), len(val_ds), len(test_ds))

    return train_ds, val_ds, test_ds, vocab_size, i2c


# ── Models ───────────────────────────────────────────────────────────────────

class HEMoETorch(nn.Module):
    """HE-MoE in PyTorch. No backprop through experts — Hebbian + Coulomb."""

    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        d = cfg.hidden_dim
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, d)

        # Fast path: 2-layer MLP (Hebbian updated)
        self.fast_w1 = nn.Parameter(torch.randn(d, d) * 0.05)
        self.fast_w2 = nn.Parameter(torch.randn(vocab_size, d) * 0.01)

        # Slow path: charged experts
        self.expert_mu = nn.Parameter(torch.randn(cfg.n_experts, d) * 0.5)
        self.expert_w = nn.Parameter(torch.randn(cfg.n_experts, d, d) * 0.05)
        self.expert_charge = torch.ones(cfg.n_experts)
        self.expert_charge[1::2] = -1.0  # alternating

        # Traces
        self.a_trace = torch.zeros(cfg.n_experts)
        self.e_trace = torch.zeros(cfg.n_experts, d)

    def kernel(self, h, mu):
        """RBF kernel: (batch, d) vs (n_experts, d) → (batch, n_experts)."""
        return torch.exp(-torch.cdist(h, mu).pow(2) / (2 * self.cfg.sigma ** 2))

    def forward(self, x):
        """x: (batch, seq_len) → logits: (batch*seq_len, vocab_size)."""
        B, S = x.shape
        h = self.embed(x).view(B * S, -1)  # (B*S, d)

        # Fast path
        h_fast = torch.relu(h @ self.fast_w1.T)  # (B*S, d)
        fast_logits = h_fast @ self.fast_w2.T  # (B*S, vocab)

        # Slow path: electrostatic routing
        kernels = self.kernel(h, self.expert_mu)  # (B*S, n_exp)
        scores = kernels * self.expert_charge.unsqueeze(0)  # charge-weighted
        _, top_idx = scores.mean(dim=0).topk(self.cfg.top_k)  # global top-k

        slow_out = torch.zeros_like(h)
        for idx in top_idx:
            w = scores[:, idx].unsqueeze(-1)  # (B*S, 1)
            expert_out = h @ self.expert_w[idx].T  # (B*S, d)
            slow_out += w * expert_out

        # Residual merge
        h_merged = h_fast + (1.0 - self.cfg.fast_ratio) * slow_out
        logits = h_merged @ self.fast_w2.T

        return logits

    @torch.no_grad()
    def hebbian_update(self, x, y):
        """No-backprop update: Hebbian on fast path + Coulomb on slow."""
        B, S = x.shape
        h = self.embed(x).view(B * S, -1)
        y_flat = y.view(-1)

        # Fast path forward
        h_fast = torch.relu(h @ self.fast_w1.T)
        logits = h_fast @ self.fast_w2.T
        probs = TF.softmax(logits / self.cfg.temp, dim=-1)

        # Error signal
        target = TF.one_hot(y_flat, logits.shape[-1]).float()
        error = target - probs  # (B*S, vocab)

        # Update fast_w2 (Hebbian delta)
        grad_w2 = error.T @ h_fast / (B * S)
        self.fast_w2.data += torch.clamp(self.cfg.fast_lr * grad_w2, -0.1, 0.1)

        # Update fast_w1 (propagated Hebbian)
        error_h = (error @ self.fast_w2) * (h_fast > 0).float()
        grad_w1 = error_h.T @ h / (B * S)
        self.fast_w1.data += torch.clamp(self.cfg.fast_lr * 0.5 * grad_w1, -0.1, 0.1)

        # Slow path: expert position + weight updates
        kernels = self.kernel(h, self.expert_mu)
        scores = kernels * self.expert_charge.unsqueeze(0)
        _, top_idx = scores.mean(dim=0).topk(self.cfg.top_k)

        h_mean = h.mean(dim=0)
        error_mean = error.mean(dim=0)

        for idx in top_idx:
            # Coulomb force: move expert toward input centroid
            force = self.expert_charge[idx] * (h_mean - self.expert_mu[idx])
            self.expert_mu.data[idx] += self.cfg.slow_lr * torch.clamp(force, -0.5, 0.5)

            # Expert weight update (error-driven)
            e_out = h @ self.expert_w[idx].T
            e_error = error_mean[:self.cfg.hidden_dim] if error_mean.shape[0] > self.cfg.hidden_dim else error_mean
            grad_ew = torch.outer(e_error[:self.cfg.hidden_dim], h_mean)
            self.expert_w.data[idx] += torch.clamp(self.cfg.slow_lr * grad_ew, -0.1, 0.1)

            # Traces
            self.a_trace[idx] = 0.9 * self.a_trace[idx] + kernels[:, idx].mean()
            self.e_trace[idx] = 0.9 * self.e_trace[idx] + e_error[:self.cfg.hidden_dim]

        # Consolidate inactive
        for idx in range(self.cfg.n_experts):
            if idx not in top_idx and self.a_trace[idx] > 0.01:
                e_norm = self.e_trace[idx].norm()
                if e_norm > 1e-6:
                    direction = self.e_trace[idx] / e_norm
                    self.expert_w.data[idx] += self.cfg.consol_lr * self.a_trace[idx] * torch.outer(direction, h_mean)

        # Expert repulsion
        for i in range(self.cfg.n_experts):
            for j in range(i + 1, self.cfg.n_experts):
                diff = self.expert_mu[i] - self.expert_mu[j]
                dist = diff.norm() + 0.1
                push = self.cfg.repulsion_weight * diff / (dist ** 2)
                self.expert_mu.data[i] += push
                self.expert_mu.data[j] -= push

        return float(TF.cross_entropy(logits, y_flat).item())


class MLPBaseline(nn.Module):
    """Standard MLP with backprop (Adam)."""

    def __init__(self, vocab_size: int, d: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, vocab_size)

    def forward(self, x):
        h = torch.relu(self.fc1(self.embed(x).view(-1, self.embed.embedding_dim)))
        return self.fc2(h)


# ── Training + Eval ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, loader, device, max_batches=50):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = TF.cross_entropy(logits, y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    avg = total_loss / max(total_tokens, 1)
    return float(np.exp(min(avg, 20))), avg


def train_he_moe(cfg: Config):
    """Full training + eval pipeline."""
    train_ds, val_ds, test_ds, vocab_size, i2c = load_tinystories(cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = HEMoETorch(vocab_size, cfg).to(cfg.device)
    model.train()

    logger.info("Training HE-MoE ({} experts, top-{}, fast_ratio={})",
                cfg.n_experts, cfg.top_k, cfg.fast_ratio)

    t0 = time.time()
    best_val_ppl = float("inf")

    for step, (x, y) in enumerate(train_loader):
        if step >= cfg.train_steps:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        loss = model.hebbian_update(x, y)

        if step % cfg.val_every == 0:
            val_ppl, val_loss = compute_ppl(model, val_loader, cfg.device)
            logger.info("step={:5d} | train_loss={:.3f} | val_ppl={:.2f} | val_loss={:.3f}",
                        step, loss, val_ppl, val_loss)
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl

    elapsed = time.time() - t0
    test_ppl, test_loss = compute_ppl(model, test_loader, cfg.device)
    logger.success("HE-MoE: test_ppl={:.2f} | best_val_ppl={:.2f} | {:.0f}s",
                    test_ppl, best_val_ppl, elapsed)

    # MLP baseline (with backprop for comparison)
    logger.info("Training MLP baseline (Adam)...")
    mlp = MLPBaseline(vocab_size, cfg.hidden_dim).to(cfg.device)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
    for step, (x, y) in enumerate(train_loader):
        if step >= min(cfg.train_steps, 3000):
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        opt.zero_grad()
        loss = TF.cross_entropy(mlp(x), y.view(-1))
        loss.backward()
        opt.step()

    mlp_ppl, _ = compute_ppl(mlp, test_loader, cfg.device)
    logger.success("MLP (Adam): test_ppl={:.2f}", mlp_ppl)

    # Results
    print("\n" + "=" * 50)
    print("  TinyStories Perplexity Results")
    print("=" * 50)
    print(f"  HE-MoE (no backprop): {test_ppl:.2f}")
    print(f"  MLP (Adam backprop):  {mlp_ppl:.2f}")
    print(f"  Ratio:                {test_ppl / mlp_ppl:.1f}x")
    print("=" * 50)

    return test_ppl, mlp_ppl


if __name__ == "__main__":
    cfg = Config(
        train_steps=3000,
        max_text_chars=500_000,  # ~500K chars for quick test
        batch_size=32,
    )
    train_he_moe(cfg)
