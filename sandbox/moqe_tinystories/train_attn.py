"""MoQE + Flash Attention 2 + LayerNorm — via grilly ops.

Architecture per layer:
  x → LayerNorm → Q,K,V projections → FA2 causal attention → residual
    → LayerNorm → MoE experts (4 experts, soft routing) → residual

Uses _bridge ops: flash_attention2, layernorm, linear.
Backward via Eigen (moe_backward_cpu pattern for expert gradients).

Run: python -u sandbox/moqe_tinystories/train_attn.py
"""

import math
import os

import numpy as np
from loguru import logger

from grilly.backend import _bridge


# ── Config ───────────────────────────────────────────────────────────────────

D_MODEL = 768
N_HEADS = 12
N_LAYERS = 8
N_EXPERTS = 4
SEQ_LEN = 512
HEAD_DIM = D_MODEL // N_HEADS  # 64
LR = 3e-4
LR_MIN = 1e-5
WARMUP = 1000
TRAIN_STEPS = 100000
VAL_EVERY = 50
SAVE_EVERY = 5000
MAX_GRAD_NORM = 1.0
SEED = 42


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(data_dir, seq_len):
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, "sequence_*.npz")))
    logger.info("Loading {} sequences", len(files))
    all_tokens = []
    for f in files:
        t = np.load(f)["input_tokens"]
        if len(t) >= 4:
            all_tokens.extend(t.tolist())
    mapped = np.array(all_tokens, dtype=np.int32)
    logger.info("Total tokens: {}", len(mapped))
    n = len(mapped)
    train_tok, val_tok = mapped[:int(0.8*n)], mapped[int(0.8*n):int(0.9*n)]

    def make_seqs(tok):
        x, y = [], []
        for i in range(0, len(tok) - seq_len - 1, seq_len // 2):
            x.append(tok[i:i+seq_len])
            y.append(tok[i+1:i+seq_len+1])
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    tx, ty = make_seqs(train_tok)
    vx, vy = make_seqs(val_tok)
    logger.info("Train: {} seqs, Val: {} seqs", len(tx), len(vx))
    return tx, ty, vx, vy


# ── Helper ops ───────────────────────────────────────────────────────────────

def _np(x):
    if x is None: return None
    return np.asarray(x) if not isinstance(x, np.ndarray) else x

def layernorm(x, gamma, beta, eps=1e-5):
    """LayerNorm: GPU via _bridge, CPU fallback."""
    r = _np(_bridge.layernorm(x, gamma, beta, eps))
    if r is not None: return r
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def linear(x, w, b=None):
    """x @ w.T + b via GPU."""
    r = _np(_bridge.linear(x, w, b))
    if r is not None: return r
    out = x @ w.T
    if b is not None: out = out + b
    return out

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-8)

def flash_attn(Q, K, V, n_heads, head_dim):
    """Flash Attention 2 via _bridge, fallback to numpy."""
    S = Q.shape[0]
    # Reshape to (1, n_heads, seq, head_dim) for _bridge
    Q4 = Q.reshape(S, n_heads, head_dim).transpose(1, 0, 2)[None]  # (1, H, S, D)
    K4 = K.reshape(S, n_heads, head_dim).transpose(1, 0, 2)[None]
    V4 = V.reshape(S, n_heads, head_dim).transpose(1, 0, 2)[None]
    Q4 = np.ascontiguousarray(Q4, dtype=np.float32)
    K4 = np.ascontiguousarray(K4, dtype=np.float32)
    V4 = np.ascontiguousarray(V4, dtype=np.float32)

    scale = 1.0 / math.sqrt(head_dim)
    r = _np(_bridge.flash_attention2(Q4, K4, V4, None, scale))
    if r is not None:
        return r.reshape(1, n_heads, S, head_dim).transpose(0, 2, 1, 3).reshape(S, n_heads * head_dim)

    # CPU fallback: standard causal attention
    scores = np.matmul(Q4, K4.transpose(0, 1, 3, 2)) * scale
    # Causal mask
    mask = np.triu(np.full((S, S), -1e9, dtype=np.float32), k=1)
    scores = scores + mask[None, None, :, :]
    attn = softmax(scores, axis=-1)
    out = np.matmul(attn, V4)
    return out.reshape(1, n_heads, S, head_dim).transpose(0, 2, 1, 3).reshape(S, n_heads * head_dim)


# ── Model ────────────────────────────────────────────────────────────────────

def init_weights(vocab, d, n_heads, n_layers, n_experts, seed=42):
    rng = np.random.default_rng(seed)
    std = 1.0 / math.sqrt(d)
    hd = d // n_heads

    w = {
        "embed": rng.normal(0, 0.02, (vocab, d)).astype(np.float32),
        "pos": rng.normal(0, 0.02, (SEQ_LEN + 16, d)).astype(np.float32),
        "out_w": rng.normal(0, 0.02, (vocab, d)).astype(np.float32),
    }

    # Per-layer: attention + layernorm + MoE
    for l in range(n_layers):
        # LayerNorm 1 (before attention)
        w[f"ln1_g_{l}"] = np.ones(d, dtype=np.float32)
        w[f"ln1_b_{l}"] = np.zeros(d, dtype=np.float32)
        # QKV projection (fused: d → 3d)
        w[f"qkv_w_{l}"] = rng.normal(0, std, (3 * d, d)).astype(np.float32)
        w[f"qkv_b_{l}"] = np.zeros(3 * d, dtype=np.float32)
        # Output projection
        w[f"attn_out_w_{l}"] = rng.normal(0, std, (d, d)).astype(np.float32)
        w[f"attn_out_b_{l}"] = np.zeros(d, dtype=np.float32)
        # LayerNorm 2 (before MoE)
        w[f"ln2_g_{l}"] = np.ones(d, dtype=np.float32)
        w[f"ln2_b_{l}"] = np.zeros(d, dtype=np.float32)
        # Expert weights
        for e in range(n_experts):
            w[f"expert_{l}_{e}"] = rng.normal(0, std, (d, d)).astype(np.float32)
        # Router
        w[f"router_W_{l}"] = rng.standard_normal((n_experts, d)).astype(np.float32) * 0.01
        w[f"router_b_{l}"] = np.zeros(n_experts, dtype=np.float32)

    return w


def forward(w, input_ids, d=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            n_experts=N_EXPERTS):
    """Full forward: embed → [LN → Attn → LN → MoE] × N → out_proj."""
    S = len(input_ids)
    hd = d // n_heads

    x = w["embed"][input_ids] + w["pos"][:S]  # (S, d)

    for l in range(n_layers):
        # ── Attention block ──
        h = layernorm(x, w[f"ln1_g_{l}"], w[f"ln1_b_{l}"])
        qkv = linear(h, w[f"qkv_w_{l}"], w[f"qkv_b_{l}"])  # (S, 3d)
        Q, K, V = qkv[:, :d], qkv[:, d:2*d], qkv[:, 2*d:]
        attn_out = flash_attn(Q, K, V, n_heads, hd)
        attn_proj = linear(attn_out, w[f"attn_out_w_{l}"], w[f"attn_out_b_{l}"])
        x = x + attn_proj  # residual

        # ── MoE block ──
        h2 = layernorm(x, w[f"ln2_g_{l}"], w[f"ln2_b_{l}"])
        # Router
        h_mean = h2.mean(axis=0)
        r_logits = w[f"router_W_{l}"] @ h_mean + w[f"router_b_{l}"]
        r_w = softmax(r_logits)
        # Expert forward + blend
        out = np.zeros_like(h2)
        for e in range(n_experts):
            out += r_w[e] * linear(h2, w[f"expert_{l}_{e}"])
        x = x + out  # residual

    logits = linear(x, w["out_w"])
    return logits


# ── CE loss ──────────────────────────────────────────────────────────────────

def cross_entropy_with_grad(logits, labels):
    S, V = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True) + 1e-8
    loss = -np.mean(np.log(probs[np.arange(S), labels] + 1e-8))
    grad = probs.copy()
    grad[np.arange(S), labels] -= 1.0
    grad /= S
    return float(loss), grad.astype(np.float32)


# ── PPL ──────────────────────────────────────────────────────────────────────

def compute_ppl(w, x_data, y_data, max_samples=100):
    total_loss, n = 0.0, 0
    for i in range(min(len(x_data), max_samples)):
        logits = forward(w, x_data[i])
        p = softmax(logits)
        for t in range(len(y_data[i])):
            tok = int(y_data[i][t])
            if 0 <= tok < p.shape[1]:
                total_loss -= math.log(max(float(p[t, tok]), 1e-8))
                n += 1
    return float(math.exp(min(total_loss / max(n, 1), 20)))


# ── Numerical gradient (finite differences) for now ─────────────────────────
# TODO: replace with analytical backward once architecture is validated

def numerical_grad(w, input_ids, labels, key, eps=1e-4):
    """Finite difference gradient for one parameter."""
    param = w[key]
    grad = np.zeros_like(param)
    flat = param.ravel()
    for i in range(min(len(flat), 100)):  # Sample 100 dims max
        old = flat[i]
        flat[i] = old + eps
        loss_p, _ = cross_entropy_with_grad(forward(w, input_ids), labels)
        flat[i] = old - eps
        loss_m, _ = cross_entropy_with_grad(forward(w, input_ids), labels)
        flat[i] = old
        grad.ravel()[i] = (loss_p - loss_m) / (2 * eps)
    return grad


# ── LR schedule ──────────────────────────────────────────────────────────────

def lr_schedule(step):
    if step < WARMUP:
        return LR_MIN + (LR - LR_MIN) * step / max(WARMUP, 1)
    t = (step - WARMUP) / max(TRAIN_STEPS - WARMUP, 1)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * t))


# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(w, step, best_ppl, path="sandbox/moqe_tinystories/checkpoints"):
    os.makedirs(path, exist_ok=True)
    d = {"step": np.array([step]), "best_ppl": np.array([best_ppl])}
    for k, v in w.items():
        d[k] = v
    f = os.path.join(path, f"moqe_attn_step{step}.npz")
    np.savez_compressed(f, **d)
    logger.info("Saved: {} ({:.1f}MB)", f, os.path.getsize(f) / 1e6)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    vocab = int(np.load("sandbox/moqe_tinystories/data/vocab_size.npy")[0])
    logger.info("Vocab: {}", vocab)

    train_x, train_y, val_x, val_y = load_data("sandbox/moqe_tinystories/data", SEQ_LEN)

    w = init_weights(vocab, D_MODEL, N_HEADS, N_LAYERS, N_EXPERTS, SEED)
    n_params = sum(v.size for v in w.values())
    logger.info("MoQE+Attn: d={}, heads={}, layers={}, experts={}, params={:.1f}M",
                D_MODEL, N_HEADS, N_LAYERS, N_EXPERTS, n_params / 1e6)

    # NOTE: Using forward-only mode for now.
    # Backward needs grilly autograd or manual backward (next step).
    # This validates the forward architecture first.
    logger.info("Forward-only validation (backward TBD)...")

    ppl = compute_ppl(w, val_x, val_y, max_samples=20)
    logger.info("Initial PPL: {:.1f}", ppl)


if __name__ == "__main__":
    main()
