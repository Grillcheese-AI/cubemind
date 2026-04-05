"""MoQE training via grilly_core.moe_forward — fused C++ forward + Eigen backward.

Entire MoE forward pass in ONE C++ call (GPU matmuls + CPU routing).
Backward via analytical Eigen (CPU, GIL released).
No Python autograd overhead. Target: 5 stp/s.

Step 1: python -u sandbox/moqe_tinystories/prepare_data.py
Step 2: python -u sandbox/moqe_tinystories/train_fused.py
"""

import math
import os
import time

import numpy as np
from loguru import logger

import grilly_core as gc


# ── Config ───────────────────────────────────────────────────────────────────

D_MODEL = 768
N_LAYERS = 8
N_EXPERTS = 4
SEQ_LEN = 512
LR = 3e-4
LR_MIN = 1e-5
WARMUP = 1000
TRAIN_STEPS = 100000
VAL_EVERY = 50
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


# ── Weight initialization ───────────────────────────────────────────────────

def init_weights(vocab_size, d, n_layers, n_experts, seed=42):
    """Initialize all MoE weights. Returns dict of numpy arrays."""
    rng = np.random.default_rng(seed)
    std = 1.0 / math.sqrt(d)

    w = {
        "embed": (rng.normal(0, 0.02, (vocab_size, d))).astype(np.float32),
        "pos": (rng.normal(0, 0.02, (SEQ_LEN + 16, d))).astype(np.float32),
        "out_w": (rng.normal(0, 0.02, (vocab_size, d))).astype(np.float32),
        "experts": [rng.normal(0, std, (d, d)).astype(np.float32)
                     for _ in range(n_layers * n_experts)],
        "routers_W": [rng.standard_normal((n_experts, d)).astype(np.float32) * 0.01
                       for _ in range(n_layers)],
        "routers_b": [np.zeros(n_experts, dtype=np.float32)
                       for _ in range(n_layers)],
    }
    return w


# ── Cross-entropy (CPU, fast) ────────────────────────────────────────────────

def cross_entropy_with_grad(logits, labels):
    """CE loss + grad_logits. Returns (scalar_loss, grad array)."""
    seq_len, vocab = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    probs = exp_l / (exp_l.sum(axis=-1, keepdims=True) + 1e-8)
    loss = -np.mean(np.log(probs[np.arange(seq_len), labels] + 1e-8))
    grad = probs.copy()
    grad[np.arange(seq_len), labels] -= 1.0
    grad /= seq_len
    return float(loss), grad.astype(np.float32)


# ── PPL evaluation ───────────────────────────────────────────────────────────

def compute_ppl(device, handle, x_data, y_data, max_samples=200):
    total_loss, n = 0.0, 0
    for i in range(min(len(x_data), max_samples)):
        logits = gc.moe_forward(device, handle, x_data[i])
        p = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(p)
        p /= p.sum(axis=-1, keepdims=True) + 1e-8
        for t in range(len(y_data[i])):
            tok = int(y_data[i][t])
            if 0 <= tok < p.shape[1]:
                total_loss -= math.log(max(float(p[t, tok]), 1e-8))
                n += 1
    return float(math.exp(min(total_loss / max(n, 1), 20)))


# ── LR schedule ──────────────────────────────────────────────────────────────

def lr_schedule(step):
    if step < WARMUP:
        return LR_MIN + (LR - LR_MIN) * step / max(WARMUP, 1)
    t = (step - WARMUP) / max(TRAIN_STEPS - WARMUP, 1)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * t))


# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(w, step, best_ppl, adam_m=None, adam_v=None,
                     path="sandbox/moqe_tinystories/checkpoints"):
    os.makedirs(path, exist_ok=True)
    d = {"step": np.array([step]), "best_ppl": np.array([best_ppl]),
         "embed": w["embed"], "pos": w["pos"], "out_w": w["out_w"]}
    for i, ew in enumerate(w["experts"]):
        d[f"expert_{i}"] = ew
    for i, rw in enumerate(w["routers_W"]):
        d[f"router_W_{i}"] = rw
    for i, rb in enumerate(w["routers_b"]):
        d[f"router_b_{i}"] = rb
    if adam_m is not None:
        for k, v in adam_m.items():
            d[f"adam_m_{k}"] = v
        for k, v in adam_v.items():
            d[f"adam_v_{k}"] = v
    f = os.path.join(path, f"moqe_fused_step{step}.npz")
    np.savez_compressed(f, **d)
    logger.info("Saved: {} ({:.1f}MB)", f, os.path.getsize(f) / 1e6)


def load_checkpoint(w, path, adam_m=None, adam_v=None):
    data = np.load(path)
    w["embed"][:] = data["embed"]
    w["pos"][:] = data["pos"]
    w["out_w"][:] = data["out_w"]
    for i in range(len(w["experts"])):
        w["experts"][i][:] = data[f"expert_{i}"]
    for i in range(len(w["routers_W"])):
        w["routers_W"][i][:] = data[f"router_W_{i}"]
        w["routers_b"][i][:] = data[f"router_b_{i}"]
    if adam_m is not None:
        for k in list(adam_m.keys()):
            mk = f"adam_m_{k}"
            if mk in data:
                adam_m[k][:] = data[mk]
                adam_v[k][:] = data[f"adam_v_{k}"]
    step = int(data["step"][0])
    best_ppl = float(data["best_ppl"][0])
    logger.info("Loaded: {} (step={}, PPL={:.1f})", path, step, best_ppl)
    return step, best_ppl


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    vocab_size = int(np.load("sandbox/moqe_tinystories/data/vocab_size.npy")[0])
    logger.info("Vocab: {}", vocab_size)

    train_x, train_y, val_x, val_y = load_data("sandbox/moqe_tinystories/data", SEQ_LEN)

    # Init weights
    w = init_weights(vocab_size, D_MODEL, N_LAYERS, N_EXPERTS, SEED)
    n_params = (w["embed"].size + w["pos"].size + w["out_w"].size +
                sum(e.size for e in w["experts"]) +
                sum(r.size for r in w["routers_W"]) +
                sum(b.size for b in w["routers_b"]))
    logger.info("MoQE fused: d={}, layers={}, experts={}, vocab={}, params={:.1f}M",
                D_MODEL, N_LAYERS, N_EXPERTS, vocab_size, n_params / 1e6)

    # Upload to GPU
    device = gc.Device()
    device.load_shaders('C:/Users/grill/Documents/GitHub/grilly/shaders/spv')

    handle = gc.moe_upload(
        device, w["embed"], w["pos"], w["experts"],
        w["routers_W"], w["routers_b"], w["out_w"],
        N_LAYERS, N_EXPERTS)
    logger.success("GPU MoE uploaded: handle={}", handle)

    # AdamW state — keyed by weight name for checkpoint compatibility
    all_params = (
        [("embed", w["embed"]), ("pos", w["pos"]), ("out_w", w["out_w"])] +
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

    # Skip old checkpoints (fresh start for fused training)
    logger.info("Starting fresh (fused C++ path)")

    # Quick sanity check
    logger.info("Sanity: forward on first sequence...")
    test_logits = gc.moe_forward(device, handle, train_x[0])
    logger.info("Sanity OK: logits shape={}", test_logits.shape)

    # Sanity: backward
    logger.info("Sanity: backward...")
    test_grad = np.ones_like(test_logits, dtype=np.float32) * 0.001
    test_grads = gc.moe_backward(device, handle, train_x[0], test_grad)
    logger.info("Sanity backward OK: {} grads", len(test_grads))

    # Sanity: update
    logger.info("Sanity: update_weights...")
    gc.moe_update_weights(device, handle, w["embed"], w["pos"], w["experts"],
                           w["routers_W"], w["routers_b"], w["out_w"])
    logger.info("Sanity update OK")

    # Sanity: second forward after update
    logger.info("Sanity: second forward...")
    test_logits2 = gc.moe_forward(device, handle, train_x[0])
    logger.info("Sanity second forward OK: {}", test_logits2.shape)

    while step < TRAIN_STEPS:
        perm = np.random.permutation(len(train_x))
        train_x, train_y = train_x[perm], train_y[perm]

        for i in range(len(train_x)):
            if step >= TRAIN_STEPS:
                break

            lr = lr_schedule(step)

            # ── Forward (ONE C++ call — GPU matmuls + CPU routing) ──
            try:
                logits = gc.moe_forward(device, handle, train_x[i])
            except Exception as e:
                logger.error("Forward failed at step {}: {}", step, e)
                break

            # ── CE loss + grad (CPU) ──
            loss, grad_logits = cross_entropy_with_grad(
                logits, train_y[i].ravel().astype(np.int32))

            # ── Backward (ONE C++ call — Eigen on CPU, GIL released) ──
            grads = gc.moe_backward(device, handle, train_x[i], grad_logits)

            # ── Gradient clipping ──
            gnorm_sq = float(np.sum(grads["grad_embed"] ** 2))
            gnorm_sq += float(np.sum(grads["grad_pos"] ** 2))
            gnorm_sq += float(np.sum(grads["grad_out_w"] ** 2))
            for ge in grads["grad_experts"]:
                gnorm_sq += float(np.sum(ge ** 2))
            for gr in grads["grad_routers_W"]:
                gnorm_sq += float(np.sum(gr ** 2))
            for gb in grads["grad_routers_b"]:
                gnorm_sq += float(np.sum(gb ** 2))
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

            # ── AdamW step ──
            grad_map = {
                "embed": grads["grad_embed"],
                "pos": grads["grad_pos"],
                "out_w": grads["grad_out_w"],
            }
            for j, ge in enumerate(grads["grad_experts"]):
                grad_map[f"expert_{j}"] = ge
            for j, gr in enumerate(grads["grad_routers_W"]):
                grad_map[f"router_W_{j}"] = gr
            for j, gb in enumerate(grads["grad_routers_b"]):
                grad_map[f"router_b_{j}"] = gb

            for name, param in all_params:
                g = grad_map.get(name)
                if g is None:
                    continue
                adam_m[name] = b1 * adam_m[name] + (1 - b1) * g
                adam_v[name] = b2 * adam_v[name] + (1 - b2) * (g ** 2)
                mh = adam_m[name] / (1 - b1 ** (step + 1))
                vh = adam_v[name] / (1 - b2 ** (step + 1))
                param -= lr * (mh / (np.sqrt(vh) + eps) + wd * param)

            # ── Re-upload weights to GPU ──
            gc.moe_update_weights(device, handle, w["embed"], w["pos"], w["experts"],
                                   w["routers_W"], w["routers_b"], w["out_w"])

            # ── Log ──
            if step % VAL_EVERY == 0:
                ppl = compute_ppl(device, handle, val_x, val_y)
                el = time.time() - t0
                sps = (step + 1) / el if el > 0 else 0
                logger.info("step={:6d} lr={:.5f} | loss={:.3f} gnorm={:.1f} | PPL={:.1f} | {:.1f} stp/s",
                            step, lr, loss, gnorm, ppl, sps)
                if ppl < best_ppl:
                    best_ppl = ppl
                    save_checkpoint(w, step, best_ppl)

            if step > 0 and step % SAVE_EVERY == 0:
                save_checkpoint(w, step, best_ppl, adam_m, adam_v)

            step += 1

    save_checkpoint(w, step, best_ppl, adam_m, adam_v)
    gc.moe_release(device, handle)

    ppl = compute_ppl(device, handle, val_x, val_y) if False else best_ppl
    print(f"\n{'='*50}")
    print(f"  MoQE (fused C++) — Final PPL: {best_ppl:.1f}")
    print(f"  Steps: {step}  |  Time: {time.time() - t0:.0f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
