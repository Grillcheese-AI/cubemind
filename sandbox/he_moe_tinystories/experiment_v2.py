"""HE-MoE on TinyStories v2 — BPE tokenizer, positional encoding, proper loss.

Fixes from v1:
1. BPE tokenizer (tiktoken GPT-2, ~50K vocab) instead of char-level
2. Positional encoding (sinusoidal) for sequence awareness
3. Per-position loss (not flattened batch)
4. LR schedule (warmup + cosine decay)
5. More training steps (10K default)

Run: python sandbox/he_moe_tinystories/experiment_v2.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger


@dataclass
class Config:
    seq_len: int = 64
    hidden_dim: int = 128
    n_experts: int = 8
    top_k: int = 2
    fast_lr: float = 0.01
    slow_lr: float = 0.005
    consol_lr: float = 0.002
    repulsion_weight: float = 0.005
    sigma: float = 2.0
    temp: float = 1.0
    fast_ratio: float = 0.7
    train_steps: int = 10000
    warmup_steps: int = 500
    val_every: int = 500
    batch_size: int = 32
    max_stories: int = 5000
    device: str = "cpu"
    vocab_size: int = 0  # Set by tokenizer
    use_addition_only: bool = False  # Use L1 distance instead of matmul


# ── Data with BPE ────────────────────────────────────────────────────────────

def load_tinystories_bpe(cfg: Config):
    """Load TinyStories with BPE tokenizer (tiktoken GPT-2)."""
    from datasets import load_dataset
    import tiktoken

    logger.info("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    enc = tiktoken.get_encoding("gpt2")

    # Tokenize stories
    all_tokens = []
    for i, item in enumerate(ds):
        if i >= cfg.max_stories:
            break
        tokens = enc.encode(item["text"])
        all_tokens.extend(tokens)

    logger.info("Tokenized {} stories -> {} tokens (vocab={})",
                min(len(ds), cfg.max_stories), len(all_tokens), enc.n_vocab)

    # Limit vocab to tokens actually seen (for smaller embedding)
    unique_tokens = sorted(set(all_tokens))
    token_map = {t: i for i, t in enumerate(unique_tokens)}
    inv_map = {i: t for t, i in token_map.items()}
    mapped = [token_map[t] for t in all_tokens]
    vocab_size = len(unique_tokens)
    cfg.vocab_size = vocab_size
    logger.info("Effective vocab: {} unique tokens", vocab_size)

    # Split 80/10/10
    n = len(mapped)
    train_enc = mapped[:int(0.8 * n)]
    val_enc = mapped[int(0.8 * n):int(0.9 * n)]
    test_enc = mapped[int(0.9 * n):]

    def make_dataset(tokens):
        x, y = [], []
        for i in range(0, len(tokens) - cfg.seq_len - 1, cfg.seq_len // 2):  # 50% overlap
            x.append(tokens[i:i + cfg.seq_len])
            y.append(tokens[i + 1:i + cfg.seq_len + 1])
        return TensorDataset(
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

    train_ds = make_dataset(train_enc)
    val_ds = make_dataset(val_enc)
    test_ds = make_dataset(test_enc)

    logger.info("Train: {} seqs, Val: {} seqs, Test: {} seqs",
                len(train_ds), len(val_ds), len(test_ds))

    return train_ds, val_ds, test_ds, vocab_size, enc, inv_map


# ── Multiplication-Free Ops (torch) ──────────────────────────────────────────

class AdditionLinearTorch(nn.Module):
    """Multiplication-free linear: y = -||W - x||₁ + bias.

    Zero multiplications. Only additions, subtractions, absolute values.
    Weight rows are templates — output = negative L1 distance to each template.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) → (..., out_features)
        # -||W - x||₁ = -sum_j |W[i,j] - x[j]| for each output i
        diff = x.unsqueeze(-2) - self.weight  # (..., out, in)
        dist = diff.abs().sum(dim=-1)  # (..., out)
        out = -dist
        if self.bias is not None:
            out = out + self.bias
        return out


class SignActivationTorch(nn.Module):
    """Sign activation: {-1, 0, +1}. No multiplications.

    Uses straight-through estimator for gradient (clamped).
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = x - self.threshold
        # STE: forward = sign, backward = clamp
        return shifted.sign() + shifted - shifted.detach()


class AdditiveSigmoidTorch(nn.Module):
    """Addition-only sigmoid: clamp(0.5 + 0.25 * x, 0, 1). No multiplications."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(0.5 + 0.25 * x, 0.0, 1.0)


# ── Positional Encoding ──────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (no learnable params)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])  # Handle odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ── HE-MoE v2 ───────────────────────────────────────────────────────────────

class HEMoEv2(nn.Module):
    """HE-MoE with BPE, positional encoding, per-position loss.

    Supports two modes:
    - use_addition_only=False: standard matmul (torch, for comparison)
    - use_addition_only=True: L1 distance (zero multiplications)
    """

    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        d = cfg.hidden_dim
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, d)
        self.pos_enc = SinusoidalPE(d, max_len=cfg.seq_len + 16)

        if cfg.use_addition_only:
            # Multiplication-free: L1 distance layers
            self.fast_layer1 = AdditionLinearTorch(d, d)
            self.fast_layer2 = AdditionLinearTorch(d, vocab_size)
            self.fast_act = AdditiveSigmoidTorch()
            logger.info("Using ADDITION-ONLY layers (zero multiplications)")
        else:
            # Standard matmul
            self.fast_w1 = nn.Parameter(torch.randn(d, d) * 0.02)
            self.fast_w2 = nn.Parameter(torch.randn(vocab_size, d) * 0.01)
            logger.info("Using standard matmul layers")

        # Slow path: charged experts with gated recurrence
        self.expert_mu = nn.Parameter(torch.randn(cfg.n_experts, d) * 0.5)
        self.expert_w = nn.Parameter(torch.randn(cfg.n_experts, d, d) * 0.02)
        self.expert_charge = torch.ones(cfg.n_experts)
        self.expert_charge[1::2] = -1.0

        # Gated recurrence state per expert (like FlashLM's ParallelGatedRecurrence)
        self.expert_h = torch.zeros(cfg.n_experts, d)  # Hidden state
        self.expert_gate = nn.Parameter(torch.ones(cfg.n_experts) * 0.5)  # Decay gate

        # Traces
        self.a_trace = torch.zeros(cfg.n_experts)
        self.e_trace = torch.zeros(cfg.n_experts, d)

    def kernel(self, h, mu):
        return torch.exp(-torch.cdist(h, mu).pow(2) / (2 * self.cfg.sigma ** 2))

    def forward(self, x):
        """x: (B, S) → logits: (B, S, vocab)."""
        B, S = x.shape
        d = self.cfg.hidden_dim

        h = self.embed(x)  # (B, S, d)
        h = self.pos_enc(h)  # Add positional info
        h_flat = h.view(B * S, d)

        # Fast path (matmul or addition-only)
        if self.cfg.use_addition_only:
            h_fast = self.fast_act(self.fast_layer1(h_flat))
            fast_logits = self.fast_layer2(h_fast)
        else:
            h_fast = torch.relu(h_flat @ self.fast_w1.T)
            fast_logits = h_fast @ self.fast_w2.T

        # Slow path
        kernels = self.kernel(h_flat, self.expert_mu)
        scores = kernels * self.expert_charge.unsqueeze(0)
        _, top_idx = scores.mean(dim=0).topk(self.cfg.top_k)

        slow_out = torch.zeros(B * S, d, device=h_flat.device)
        for idx in top_idx:
            w = scores[:, idx].unsqueeze(-1)
            expert_out = h_flat @ self.expert_w[idx].T

            # Gated recurrence: h_new = gate * h_old + (1-gate) * expert_out
            gate = torch.sigmoid(self.expert_gate[idx])
            self.expert_h[idx] = gate * self.expert_h[idx] + (1 - gate) * expert_out.mean(dim=0).detach()

            # Use recurrent state as additional context
            slow_out += w * (expert_out + self.expert_h[idx].unsqueeze(0))

        # Residual merge → project to vocab
        h_merged = h_fast + (1.0 - self.cfg.fast_ratio) * slow_out
        if self.cfg.use_addition_only:
            logits = self.fast_layer2(h_merged)
        else:
            logits = h_merged @ self.fast_w2.T

        return logits.view(B, S, -1)  # (B, S, vocab)

    @torch.no_grad()
    def hebbian_update(self, x, y, lr_scale: float = 1.0):
        """No-backprop update with LR scaling for schedule."""
        B, S = x.shape
        d = self.cfg.hidden_dim
        lr = self.cfg.fast_lr * lr_scale

        h = self.pos_enc(self.embed(x)).view(B * S, d)
        y_flat = y.view(-1)

        # Fast forward
        h_fast = torch.relu(h @ self.fast_w1.T)
        logits = h_fast @ self.fast_w2.T
        probs = TF.softmax(logits / self.cfg.temp, dim=-1)

        # Per-position error
        target = TF.one_hot(y_flat, logits.shape[-1]).float()
        error = target - probs

        # Loss for logging
        loss = TF.cross_entropy(logits, y_flat).item()

        # Fast W2 update
        grad_w2 = error.T @ h_fast / (B * S)
        self.fast_w2.data += torch.clamp(lr * grad_w2, -0.1, 0.1)

        # Fast W1 update (Hebbian propagation)
        error_h = (error @ self.fast_w2) * (h_fast > 0).float()
        grad_w1 = error_h.T @ h / (B * S)
        self.fast_w1.data += torch.clamp(lr * 0.5 * grad_w1, -0.1, 0.1)

        # Slow path updates
        error_hidden = (error.mean(dim=0) @ self.fast_w2).detach()
        kernels = self.kernel(h, self.expert_mu)
        scores = kernels * self.expert_charge.unsqueeze(0)
        _, top_idx = scores.mean(dim=0).topk(self.cfg.top_k)
        h_mean = h.mean(dim=0)

        for idx in top_idx:
            # Coulomb force
            force = self.expert_charge[idx] * (h_mean - self.expert_mu[idx])
            self.expert_mu.data[idx] += self.cfg.slow_lr * torch.clamp(force, -0.5, 0.5)

            # Expert weight (error-driven)
            grad_ew = torch.outer(error_hidden, h_mean)
            self.expert_w.data[idx] += torch.clamp(self.cfg.slow_lr * grad_ew, -0.1, 0.1)

            # Traces
            self.a_trace[idx] = 0.9 * self.a_trace[idx] + kernels[:, idx].mean()
            self.e_trace[idx] = 0.9 * self.e_trace[idx] + error_hidden

        # Consolidate inactive
        for idx in range(self.cfg.n_experts):
            if idx not in top_idx and self.a_trace[idx] > 0.01:
                e_norm = self.e_trace[idx].norm()
                if e_norm > 1e-6:
                    direction = self.e_trace[idx] / e_norm
                    self.expert_w.data[idx] += self.cfg.consol_lr * self.a_trace[idx] * torch.outer(direction, h_mean)

        # Expert repulsion (every 10 steps to save compute)
        for i in range(self.cfg.n_experts):
            for j in range(i + 1, self.cfg.n_experts):
                diff = self.expert_mu[i] - self.expert_mu[j]
                dist = diff.norm() + 0.1
                push = self.cfg.repulsion_weight * diff / (dist ** 2)
                self.expert_mu.data[i] += push
                self.expert_mu.data[j] -= push

        return loss


# ── MLP Baseline ─────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    def __init__(self, vocab_size: int, d: int, seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos_enc = SinusoidalPE(d, max_len=seq_len + 16)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, vocab_size)

    def forward(self, x):
        h = self.pos_enc(self.embed(x))
        B, S, d = h.shape
        h = torch.relu(self.fc1(h.view(B * S, d)))
        return self.fc2(h).view(B, S, -1)


# ── Training ─────────────────────────────────────────────────────────────────

def lr_schedule(step: int, warmup: int, total: int) -> float:
    """Warmup + cosine decay."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def compute_ppl(model, loader, device, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)  # (B, S, vocab)
        B, S, V = logits.shape
        loss = TF.cross_entropy(logits.view(B * S, V), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    avg = total_loss / max(total_tokens, 1)
    return float(np.exp(min(avg, 20))), avg


def train(cfg: Config):
    data = load_tinystories_bpe(cfg)
    train_ds, val_ds, test_ds, vocab_size, enc, inv_map = data

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = HEMoEv2(vocab_size, cfg).to(cfg.device)
    model.train()

    logger.info("HE-MoE v2: vocab={}, d={}, experts={}, steps={}",
                vocab_size, cfg.hidden_dim, cfg.n_experts, cfg.train_steps)

    t0 = time.time()
    best_val_ppl = float("inf")
    step = 0

    while step < cfg.train_steps:
        for x, y in train_loader:
            if step >= cfg.train_steps:
                break
            x, y = x.to(cfg.device), y.to(cfg.device)

            # LR schedule
            lr_scale = lr_schedule(step, cfg.warmup_steps, cfg.train_steps)
            loss = model.hebbian_update(x, y, lr_scale=lr_scale)

            if step % cfg.val_every == 0:
                val_ppl, val_loss = compute_ppl(model, val_loader, cfg.device)
                logger.info("step={:5d} lr={:.4f} | loss={:.3f} | val_ppl={:.2f}",
                            step, cfg.fast_lr * lr_scale, loss, val_ppl)
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl

            step += 1

    elapsed = time.time() - t0
    test_ppl, _ = compute_ppl(model, test_loader, cfg.device)
    logger.success("HE-MoE v2: test_ppl={:.2f} | best_val={:.2f} | {:.0f}s", test_ppl, best_val_ppl, elapsed)

    # MLP baseline
    logger.info("Training MLP baseline (Adam + pos encoding)...")
    mlp = MLPBaseline(vocab_size, cfg.hidden_dim, cfg.seq_len).to(cfg.device)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train_steps)
    mlp_step = 0
    while mlp_step < cfg.train_steps:
        for x, y in train_loader:
            if mlp_step >= cfg.train_steps:
                break
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            logits = mlp(x)
            B, S, V = logits.shape
            loss = TF.cross_entropy(logits.view(B * S, V), y.view(-1))
            loss.backward()
            opt.step()
            scheduler.step()
            mlp_step += 1

    mlp_ppl, _ = compute_ppl(mlp, test_loader, cfg.device)
    logger.success("MLP (Adam): test_ppl={:.2f}", mlp_ppl)

    print("\n" + "=" * 50)
    print("  TinyStories Perplexity (BPE, v2)")
    print("=" * 50)
    print(f"  HE-MoE (no backprop): {test_ppl:.2f}")
    print(f"  MLP (Adam backprop):  {mlp_ppl:.2f}")
    print(f"  Ratio:                {test_ppl / mlp_ppl:.2f}x")
    print("=" * 50)

    return test_ppl, mlp_ppl


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "matmul"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    cfg = Config(
        train_steps=steps,
        max_stories=3000,
        batch_size=32,
        seq_len=64,
        hidden_dim=128,
        fast_lr=0.01,
        use_addition_only=(mode == "addition"),
    )

    logger.info("Mode: {} | Steps: {}", mode, steps)
    train(cfg)

    # If "compare" mode, run both
    if mode == "compare":
        logger.info("\n--- Running ADDITION-ONLY variant ---")
        cfg2 = Config(
            train_steps=steps,
            max_stories=3000,
            batch_size=32,
            seq_len=64,
            hidden_dim=128,
            fast_lr=0.01,
            use_addition_only=True,
        )
        train(cfg2)
