"""MoQE distillation from Qwen 80B teacher logits.

Uses grilly_core.moe_forward/backward with Qwen vocabulary (151936).
Teacher logits at data/teacher/qwen80b/ provide soft targets for KD.

Loss = 0.3 * CE(hard) + 0.6 * KL(teacher) + 0.1 * router_balance

Run: python -u sandbox/moqe_tinystories/train_distill.py
"""

import math
import os
import time

import numpy as np
from loguru import logger

import grilly_core as gc


# ── Config ───────────────────────────────────────────────────────────────────

VOCAB = 151936       # Qwen vocab
D_MODEL = 768
N_LAYERS = 8
N_EXPERTS = 4
SEQ_LEN = 512
LR = 3e-4
LR_MIN = 1e-5
WARMUP = 500
TRAIN_STEPS = 50000
VAL_EVERY = 50
SAVE_EVERY = 2000
MAX_GRAD_NORM = 1.0
KD_TEMP = 2.0
SEED = 42


# ── Data ─────────────────────────────────────────────────────────────────────

def load_teacher_data(data_dir, max_seq=SEQ_LEN):
    """Load teacher logit .npz files. Returns list of (input_ids, labels, teacher_logits)."""
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, "sequence_*.npz")))
    logger.info("Loading {} teacher sequences from {}", len(files), data_dir)

    TOP_K = 64  # Only keep top-64 teacher logits (saves 99.96% RAM)
    data = []
    skipped = 0
    for f in files:
        try:
            d = np.load(f)
            tokens = d["input_tokens"][:max_seq].astype(np.int32)
            if len(tokens) < 4:
                continue
            # Process row-by-row to avoid (S, 151K) float32 allocation
            S = min(len(tokens) - 1, max_seq - 1)
            top_idx = np.zeros((S, TOP_K), dtype=np.int32)
            top_lp = np.zeros((S, TOP_K), dtype=np.float16)
            for row in range(S):
                row_logits = d["logits"][row].astype(np.float32)  # (V,) — one row
                idx = np.argpartition(-row_logits, TOP_K)[:TOP_K]
                vals = row_logits[idx]
                order = np.argsort(-vals)
                top_idx[row] = idx[order]
                top_lp[row] = vals[order].astype(np.float16)
            teacher = {"top_k_indices": top_idx, "top_k_logprobs": top_lp}
            data.append((tokens[:-1], tokens[1:], teacher))
            if len(data) % 100 == 0:
                logger.info("  Loaded {}/{} sequences", len(data), len(files))
        except Exception:
            skipped += 1
    if skipped:
        logger.warning("Skipped {} corrupt files", skipped)

    # Split 90/10
    n = len(data)
    split = int(0.9 * n)
    train, val = data[:split], data[split:]
    logger.info("Train: {} seqs, Val: {} seqs", len(train), len(val))
    return train, val


# ── Loss functions ───────────────────────────────────────────────────────────

def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-8)


def ce_with_grad(logits, labels):
    """Cross-entropy loss + gradient."""
    S, V = logits.shape
    probs = softmax_np(logits)
    valid = (labels >= 0) & (labels < V)
    loss = 0.0
    count = 0
    for i in range(S):
        if valid[i]:
            loss -= math.log(max(float(probs[i, labels[i]]), 1e-8))
            count += 1
    loss /= max(count, 1)
    grad = probs.copy()
    for i in range(S):
        if valid[i]:
            grad[i, labels[i]] -= 1.0
    grad /= max(count, 1)
    return loss, grad.astype(np.float32)


def sparse_kl_with_grad(student_logits, top_k_indices, top_k_logprobs, temperature):
    """Sparse KL: only compute on teacher's top-k tokens. Saves 99%+ RAM."""
    S, V = student_logits.shape
    k = top_k_indices.shape[1]

    # Teacher distribution over top-k (softmax of top-k logprobs)
    t_lp = top_k_logprobs.astype(np.float32) / temperature
    t_lp -= t_lp.max(axis=-1, keepdims=True)
    t_probs = np.exp(t_lp)
    t_probs /= t_probs.sum(axis=-1, keepdims=True) + 1e-8

    # Student log-softmax at temperature
    s_scaled = student_logits / temperature
    s_max = s_scaled.max(axis=-1, keepdims=True)
    s_exp = np.exp(s_scaled - s_max)
    s_sum = s_exp.sum(axis=-1, keepdims=True)
    s_log_softmax = s_scaled - s_max - np.log(s_sum + 1e-8)

    # Gather student log-probs at teacher's top-k positions
    row_idx = np.arange(S)[:, None]
    s_log_at_topk = s_log_softmax[row_idx, top_k_indices]  # (S, k)

    # KL per token
    t_log = np.log(np.clip(t_probs, 1e-7, 1.0))
    kl = np.sum(t_probs * (t_log - s_log_at_topk), axis=-1)
    kl = np.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
    loss = float(np.mean(kl)) * (temperature ** 2)

    # Gradient: sparse — only non-zero at top-k positions + softmax baseline
    # Instead of full (S, V) grad, compute directly as:
    # grad = softmax(s/T) * T^3 / S  (baseline for all tokens)
    # grad[i, top_k[i]] -= teacher_probs[i] * T^3 / S  (correction at top-k)
    # This is mathematically equivalent but we return full grad for moe_backward
    scale = temperature * (temperature ** 2) / S
    s_softmax = s_exp / (s_sum + 1e-8)  # (S, V) — unavoidable for full grad
    grad = s_softmax * scale
    for i in range(S):
        grad[i, top_k_indices[i]] -= t_probs[i] * scale

    return loss, grad.astype(np.float32)


# ── Weight init ──────────────────────────────────────────────────────────────

def init_weights(vocab, d, n_layers, n_experts, seed=42):
    rng = np.random.default_rng(seed)
    std = 1.0 / math.sqrt(d)
    return {
        "embed": rng.normal(0, 0.02, (vocab, d)).astype(np.float32),
        "pos": rng.normal(0, 0.02, (SEQ_LEN + 16, d)).astype(np.float32),
        "out_w": rng.normal(0, 0.02, (vocab, d)).astype(np.float32),
        "experts": [rng.normal(0, std, (d, d)).astype(np.float32)
                     for _ in range(n_layers * n_experts)],
        "routers_W": [rng.standard_normal((n_experts, d)).astype(np.float32) * 0.01
                       for _ in range(n_layers)],
        "routers_b": [np.zeros(n_experts, dtype=np.float32)
                       for _ in range(n_layers)],
    }


# ── LR + checkpoint ─────────────────────────────────────────────────────────

def lr_schedule(step):
    if step < WARMUP:
        return LR_MIN + (LR - LR_MIN) * step / max(WARMUP, 1)
    t = (step - WARMUP) / max(TRAIN_STEPS - WARMUP, 1)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * t))


def save_checkpoint(w, step, best_ppl, path="sandbox/moqe_tinystories/checkpoints"):
    os.makedirs(path, exist_ok=True)
    d = {"step": np.array([step]), "best_ppl": np.array([best_ppl]),
         "embed": w["embed"], "pos": w["pos"], "out_w": w["out_w"]}
    for i, e in enumerate(w["experts"]):
        d[f"expert_{i}"] = e
    for i, r in enumerate(w["routers_W"]):
        d[f"router_W_{i}"] = r
    for i, b in enumerate(w["routers_b"]):
        d[f"router_b_{i}"] = b
    f = os.path.join(path, f"moqe_distill_step{step}.npz")
    np.savez_compressed(f, **d)
    logger.info("Saved: {} ({:.1f}MB)", f, os.path.getsize(f) / 1e6)


# ── PPL ──────────────────────────────────────────────────────────────────────

def compute_ppl(device, handle, val_data, max_samples=50):
    total_loss, n = 0.0, 0
    for i in range(min(len(val_data), max_samples)):
        ids, labels, _ = val_data[i]
        logits = gc.moe_forward(device, handle, ids)
        p = softmax_np(logits)
        for t in range(len(labels)):
            tok = int(labels[t])
            if 0 <= tok < p.shape[1]:
                total_loss -= math.log(max(float(p[t, tok]), 1e-8))
                n += 1
    return float(math.exp(min(total_loss / max(n, 1), 20)))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    train_data, val_data = load_teacher_data("data/logits_512")

    w = init_weights(VOCAB, D_MODEL, N_LAYERS, N_EXPERTS, SEED)
    n_params = sum(v.size if isinstance(v, np.ndarray) else sum(x.size for x in v) for v in w.values())
    logger.info("MoQE distill: vocab={}, d={}, layers={}, experts={}, params={:.1f}M",
                VOCAB, D_MODEL, N_LAYERS, N_EXPERTS, n_params / 1e6)

    # Upload to GPU
    device = gc.Device()
    device.load_shaders('C:/Users/grill/Documents/GitHub/grilly/shaders/spv')

    # Upload as float32 (GPU needs f32, but we store f16 on CPU to save RAM)
    handle = gc.moe_upload(device,
                            w["embed"].astype(np.float32),
                            w["pos"],
                            w["experts"],
                            w["routers_W"], w["routers_b"],
                            w["out_w"].astype(np.float32),
                            N_LAYERS, N_EXPERTS)
    logger.success("GPU MoE uploaded: handle={}", handle)

    # Freeze embed + out_w (too large for adam state at 151K vocab)
    # Convert to float16 to save ~900MB RAM
    w["embed"] = w["embed"].astype(np.float16)
    w["out_w"] = w["out_w"].astype(np.float16)
    logger.info("Frozen embed + out_w (float16, no optimizer state)")

    # AdamW only for experts + routers + pos (small params)
    all_params = (
        [("pos", w["pos"])] +
        [(f"expert_{i}", e) for i, e in enumerate(w["experts"])] +
        [(f"router_W_{i}", r) for i, r in enumerate(w["routers_W"])] +
        [(f"router_b_{i}", b) for i, b in enumerate(w["routers_b"])]
    )
    adam_m = {name: np.zeros_like(p) for name, p in all_params}
    adam_v = {name: np.zeros_like(p) for name, p in all_params}
    b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01

    t0 = time.time()
    best_ppl = float("inf")
    step = 0

    logger.info("Starting distillation training...")

    while step < TRAIN_STEPS:
        # Shuffle training data each epoch
        indices = np.random.permutation(len(train_data))

        for idx in indices:
            if step >= TRAIN_STEPS:
                break

            ids, labels, teacher = train_data[idx]
            lr = lr_schedule(step)

            # Forward
            logits = gc.moe_forward(device, handle, ids)

            # Combined loss: 0.3 * CE + 0.6 * KL (sparse top-k)
            loss_ce, grad_ce = ce_with_grad(logits, labels)
            loss_kd, grad_kd = sparse_kl_with_grad(
                logits, teacher["top_k_indices"], teacher["top_k_logprobs"], KD_TEMP)
            total_loss = 0.3 * loss_ce + 0.6 * loss_kd
            grad_logits = (0.3 * grad_ce + 0.6 * grad_kd).astype(np.float32)
            del grad_ce, grad_kd  # free immediately

            # Backward
            grads = gc.moe_backward(device, handle, ids, grad_logits)

            # Gradient clipping
            gnorm_sq = sum(float(np.sum(g**2)) for g in [
                grads["grad_embed"], grads["grad_pos"], grads["grad_out_w"]])
            for ge in grads["grad_experts"]:
                gnorm_sq += float(np.sum(ge**2))
            for gr in grads["grad_routers_W"]:
                gnorm_sq += float(np.sum(gr**2))
            for gb in grads["grad_routers_b"]:
                gnorm_sq += float(np.sum(gb**2))
            gnorm = math.sqrt(gnorm_sq)

            if MAX_GRAD_NORM > 0 and gnorm > MAX_GRAD_NORM:
                clip = MAX_GRAD_NORM / (gnorm + 1e-8)
                grads["grad_embed"] *= clip
                grads["grad_pos"] *= clip
                grads["grad_out_w"] *= clip
                for j in range(len(grads["grad_experts"])):
                    grads["grad_experts"][j] *= clip
                for j in range(len(grads["grad_routers_W"])):
                    grads["grad_routers_W"][j] *= clip
                for j in range(len(grads["grad_routers_b"])):
                    grads["grad_routers_b"][j] *= clip

            # AdamW
            grad_map = {"embed": grads["grad_embed"], "pos": grads["grad_pos"],
                        "out_w": grads["grad_out_w"]}
            for j, ge in enumerate(grads["grad_experts"]):
                grad_map[f"expert_{j}"] = ge
            for j, gr in enumerate(grads["grad_routers_W"]):
                grad_map[f"router_W_{j}"] = gr
            for j, gb in enumerate(grads["grad_routers_b"]):
                grad_map[f"router_b_{j}"] = gb

            for name, param in all_params:
                g = grad_map.get(name)
                if g is None: continue
                adam_m[name] = b1 * adam_m[name] + (1 - b1) * g
                adam_v[name] = b2 * adam_v[name] + (1 - b2) * (g ** 2)
                mh = adam_m[name] / (1 - b1 ** (step + 1))
                vh = adam_v[name] / (1 - b2 ** (step + 1))
                param -= lr * (mh / (np.sqrt(vh) + eps) + wd * param)

            # Re-upload
            gc.moe_update_weights(device, handle,
                                   w["embed"].astype(np.float32), w["pos"],
                                   w["experts"], w["routers_W"], w["routers_b"],
                                   w["out_w"].astype(np.float32))

            # Log
            if step % VAL_EVERY == 0:
                ppl = compute_ppl(device, handle, val_data)
                el = time.time() - t0
                sps = (step + 1) / el if el > 0 else 0
                logger.info("step={:6d} lr={:.5f} | CE={:.3f} KD={:.3f} L={:.3f} gnorm={:.1f} | PPL={:.1f} | {:.2f} stp/s",
                            step, lr, loss_ce, loss_kd, total_loss, gnorm, ppl, sps)
                if ppl < best_ppl:
                    best_ppl = ppl
                    save_checkpoint(w, step, best_ppl)

            if step > 0 and step % SAVE_EVERY == 0:
                save_checkpoint(w, step, best_ppl)

            step += 1

    save_checkpoint(w, step, best_ppl)
    gc.moe_release(device, handle)

    print(f"\n{'='*55}")
    print(f"  MoQE Distillation Results")
    print(f"{'='*55}")
    print(f"  Best PPL: {best_ppl:.1f}")
    print(f"  Steps: {step}")
    print(f"  Time: {time.time() - t0:.0f}s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
