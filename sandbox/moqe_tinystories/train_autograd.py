"""MoQE training via grilly autograd — no manual backward.

grilly.nn.autograd builds computation graph during forward,
then .backward() computes all gradients automatically.
GPU-accelerated backward via Vulkan compute shaders.

Step 1: python -u sandbox/moqe_tinystories/prepare_data.py
Step 2: python -u sandbox/moqe_tinystories/train_autograd.py
"""

import math
import os
import time

import numpy as np
from loguru import logger

from grilly.nn.autograd import (
    Variable, matmul, relu, softmax, cross_entropy, no_grad, mean,
)


# ── Config ───────────────────────────────────────────────────────────────────

D_MODEL = 1024
N_LAYERS = 12
N_EXPERTS = 4
SEQ_LEN = 768
LR = 3e-4
LR_MIN = 1e-5
WARMUP = 1000
TRAIN_STEPS = 100000
VAL_EVERY = 10
SAVE_EVERY = 5000
MAX_GRAD_NORM = 1.0
SEED = 42


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(data_dir: str, seq_len: int):
    """Load pre-tokenized .npz sequences into train/val arrays."""
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, "sequence_*.npz")))
    logger.info("Loading {} sequences from {}", len(files), data_dir)

    all_tokens = []
    for f in files:
        tokens = np.load(f)["input_tokens"]
        if len(tokens) >= 4:
            all_tokens.extend(tokens.tolist())

    mapped = np.array(all_tokens, dtype=np.int32)
    logger.info("Total tokens: {}", len(mapped))

    n = len(mapped)
    train_tok = mapped[:int(0.8 * n)]
    val_tok = mapped[int(0.8 * n):int(0.9 * n)]

    def make_seqs(tok):
        x, y = [], []
        for i in range(0, len(tok) - seq_len - 1, seq_len // 2):
            x.append(tok[i:i + seq_len])
            y.append(tok[i + 1:i + seq_len + 1])
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    train_x, train_y = make_seqs(train_tok)
    val_x, val_y = make_seqs(val_tok)
    logger.info("Train: {} seqs, Val: {} seqs", len(train_x), len(val_x))
    return train_x, train_y, val_x, val_y


# ── Model ────────────────────────────────────────────────────────────────────

class MoQEAutograd:
    """MoQE with grilly autograd. Forward builds graph, backward() does the rest."""

    def __init__(self, vocab_size: int, d: int, n_layers: int, n_experts: int,
                 seed: int = 42):
        self.vocab_size = vocab_size
        self.d = d
        self.n_layers = n_layers
        self.n_experts = n_experts

        rng = np.random.default_rng(seed)
        std = 1.0 / math.sqrt(d)

        # Parameters (all autograd Variables)
        self.embed = Variable(rng.normal(0, 0.02, (vocab_size, d)).astype(np.float32),
                              requires_grad=True)
        self.pos = Variable(rng.normal(0, 0.02, (SEQ_LEN + 16, d)).astype(np.float32),
                            requires_grad=True)
        self.out_w = Variable(rng.normal(0, 0.02, (vocab_size, d)).astype(np.float32),
                              requires_grad=True)

        # Per-layer: expert weights + router
        self.experts = []   # [layer][expert] = Variable (d, d)
        self.routers_W = [] # [layer] = Variable (n_experts, d)
        self.routers_b = [] # [layer] = Variable (n_experts,)

        for _ in range(n_layers):
            layer_exp = [Variable(rng.normal(0, std, (d, d)).astype(np.float32),
                                  requires_grad=True) for _ in range(n_experts)]
            self.experts.append(layer_exp)
            self.routers_W.append(Variable(
                (rng.standard_normal((n_experts, d)) * 0.01).astype(np.float32),
                requires_grad=True))
            self.routers_b.append(Variable(
                np.zeros(n_experts, dtype=np.float32), requires_grad=True))

    def params(self) -> list[Variable]:
        """All trainable parameters."""
        p = [self.embed, self.pos, self.out_w]
        for l in range(self.n_layers):
            p.extend(self.experts[l])
            p.append(self.routers_W[l])
            p.append(self.routers_b[l])
        return p

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def forward(self, input_ids: np.ndarray) -> Variable:
        """Forward pass. Builds autograd graph automatically."""
        from grilly.nn.autograd import add, mul as ag_mul, index, transpose

        S = len(input_ids)

        # Embed lookup through autograd (index op tracks grad to embed table)
        x = index(self.embed, input_ids)        # (S, d) — grad flows to embed
        pe = index(self.pos, np.arange(S))       # (S, d) — grad flows to pos
        x = add(x, pe)

        for l in range(self.n_layers):
            # Router: softmax(W @ mean(x) + b)
            x_mean = mean(x, dim=0)
            logits = add(matmul(self.routers_W[l], x_mean), self.routers_b[l])
            weights = softmax(logits, dim=-1)  # (n_experts,)

            # Expert forward: soft-weighted sum through autograd
            expert_outs = [matmul(x, self.experts[l][e]) for e in range(self.n_experts)]

            # Weighted combination (all through autograd graph)
            out = ag_mul(expert_outs[0], Variable(
                np.full((S, 1), weights.data[0], dtype=np.float32)))
            for e in range(1, self.n_experts):
                scaled = ag_mul(expert_outs[e], Variable(
                    np.full((S, 1), weights.data[e], dtype=np.float32)))
                out = add(out, scaled)

            x = add(x, out)  # residual

        # Output: x @ out_w.T — transpose through autograd
        out_w_T = transpose(self.out_w)  # grad flows to out_w
        logits = matmul(x, out_w_T)
        return logits


# ── Eval ─────────────────────────────────────────────────────────────────────

def compute_ppl(model, x_data, y_data, max_samples=200):
    total_loss, n = 0.0, 0
    with no_grad():
        for i in range(min(len(x_data), max_samples)):
            logits = model.forward(x_data[i])
            p = logits.data - logits.data.max(axis=-1, keepdims=True)
            p = np.exp(p)
            p /= p.sum(axis=-1, keepdims=True) + 1e-8
            for t in range(len(y_data[i])):
                tok = int(y_data[i][t])
                if 0 <= tok < p.shape[1]:
                    total_loss -= math.log(max(float(p[t, tok]), 1e-8))
                    n += 1
    return float(math.exp(min(total_loss / max(n, 1), 20)))


def lr_schedule(step):
    if step < WARMUP:
        return LR_MIN + (LR - LR_MIN) * step / max(WARMUP, 1)
    t = (step - WARMUP) / max(TRAIN_STEPS - WARMUP, 1)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * t))


def save_checkpoint(model, step, best_ppl, path="sandbox/moqe_tinystories/checkpoints"):
    os.makedirs(path, exist_ok=True)
    d = {"step": np.array([step]), "best_ppl": np.array([best_ppl]),
         "embed": model.embed.data, "pos": model.pos.data, "out_w": model.out_w.data}
    for l in range(model.n_layers):
        for e in range(model.n_experts):
            d[f"L{l}_E{e}"] = model.experts[l][e].data
        d[f"L{l}_rW"] = model.routers_W[l].data
        d[f"L{l}_rb"] = model.routers_b[l].data
    f = os.path.join(path, f"moqe_ag_step{step}.npz")
    np.savez_compressed(f, **d)
    logger.info("Saved: {} ({:.1f}MB)", f, os.path.getsize(f) / 1e6)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    vocab_size = int(np.load("sandbox/moqe_tinystories/data/vocab_size.npy")[0])
    logger.info("Vocab: {}", vocab_size)

    train_x, train_y, val_x, val_y = load_data("sandbox/moqe_tinystories/data", SEQ_LEN)

    model = MoQEAutograd(vocab_size, D_MODEL, N_LAYERS, N_EXPERTS, SEED)
    n_params = sum(p.data.size for p in model.params())
    logger.info("MoQE autograd: d={}, layers={}, experts={}, params={:.1f}M",
                D_MODEL, N_LAYERS, N_EXPERTS, n_params / 1e6)

    # AdamW optimizer state
    adam_m = {id(p): np.zeros_like(p.data) for p in model.params()}
    adam_v = {id(p): np.zeros_like(p.data) for p in model.params()}
    b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01

    t0 = time.time()
    best_ppl = float("inf")
    step = 0

    while step < TRAIN_STEPS:
        perm = np.random.permutation(len(train_x))
        train_x, train_y = train_x[perm], train_y[perm]

        for i in range(len(train_x)):
            if step >= TRAIN_STEPS:
                break

            lr = lr_schedule(step)

            # ── Forward + loss (graph built automatically) ──
            model.zero_grad()
            logits = model.forward(train_x[i])
            loss = cross_entropy(logits, train_y[i].ravel().astype(np.int64))

            # ── Backward (one call — grilly autograd) ──
            loss.backward(use_gpu=True)

            # ── Gradient clipping ──
            gnorm_sq = sum(float(np.sum(p.grad ** 2)) for p in model.params() if p.grad is not None)
            gnorm = math.sqrt(gnorm_sq)
            if MAX_GRAD_NORM > 0 and gnorm > MAX_GRAD_NORM:
                clip = MAX_GRAD_NORM / (gnorm + 1e-8)
                for p in model.params():
                    if p.grad is not None:
                        p.grad *= clip

            # ── AdamW step ──
            for p in model.params():
                if p.grad is None:
                    continue
                pid = id(p)
                adam_m[pid] = b1 * adam_m[pid] + (1 - b1) * p.grad
                adam_v[pid] = b2 * adam_v[pid] + (1 - b2) * (p.grad ** 2)
                mh = adam_m[pid] / (1 - b1 ** (step + 1))
                vh = adam_v[pid] / (1 - b2 ** (step + 1))
                p.data -= lr * (mh / (np.sqrt(vh) + eps) + wd * p.data)

            # ── Log ──
            if step % VAL_EVERY == 0:
                ppl = compute_ppl(model, val_x, val_y)
                el = time.time() - t0
                sps = (step + 1) / el if el > 0 else 0
                logger.info("step={:6d} lr={:.5f} | loss={:.3f} gnorm={:.1f} | PPL={:.1f} | {:.1f} stp/s",
                            step, lr, float(loss.data), gnorm, ppl, sps)
                if ppl < best_ppl:
                    best_ppl = ppl
                    save_checkpoint(model, step, best_ppl)

            if step > 0 and step % SAVE_EVERY == 0:
                save_checkpoint(model, step, best_ppl)

            step += 1

    save_checkpoint(model, step, best_ppl)
    ppl = compute_ppl(model, val_x, val_y)
    print(f"\n{'='*50}")
    print(f"  MoQE (grilly autograd) — Final PPL: {ppl:.1f}")
    print(f"  Best PPL: {best_ppl:.1f}  |  Steps: {step}")
    print(f"  Time: {time.time() - t0:.0f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
