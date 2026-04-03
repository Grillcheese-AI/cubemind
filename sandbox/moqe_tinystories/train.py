"""MoQE on TinyStories — prove the architecture works on real text.

Uses existing MoQE with:
  1. Position embeddings (critical — without it, model can't tell position)
  2. SentencePiece BPE (1k–4k vocab, not char-level)
  3. Full backprop via moqe_backward + STE
  4. Cosine LR schedule with warmup
  5. Expert usage tracking per layer

Target: <2.0 PPL on TinyStories validation.

Run: python -u sandbox/moqe_tinystories/train.py
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from cubemind.execution.moqe import (
    ExpertSpec,
    MoQEModel,
    _quantize_symmetric,
    _dequant_weights,
)
from cubemind.training.moqe_distillation import (
    _cross_entropy_with_grad,
    _softmax,
)

# GPU training ops — persistent weights, fused GEMV
_moqe_gpu = None
try:
    import grilly_core as _gc
    if hasattr(_gc, 'moqe_layer_forward'):
        _moqe_gpu = _gc
except Exception:
    pass


@dataclass
class Config:
    vocab_size: int = 4000
    d_model: int = 256
    n_layers: int = 4
    top_k: int = 1
    block_size: int = 32
    seq_len: int = 128
    batch_size: int = 32
    max_stories: int = 30000
    train_steps: int = 20000
    warmup_steps: int = 1000
    val_every: int = 200
    lr: float = 5e-2
    lr_min: float = 1e-3
    max_grad_norm: float = 10.0
    seed: int = 42


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(cfg: Config):
    """Load TinyStories with SentencePiece BPE."""
    from datasets import load_dataset
    import sentencepiece as spm

    logger.info("Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    texts = []
    for i, item in enumerate(ds):
        if i >= cfg.max_stories:
            break
        texts.append(item["text"])
    logger.info("Loaded {} stories", len(texts))

    sp_model_prefix = os.path.join(tempfile.gettempdir(), "moqe_tinystories_sp")
    corpus_file = sp_model_prefix + "_corpus.txt"

    with open(corpus_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    logger.info("Training SentencePiece BPE (vocab={})", cfg.vocab_size)
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=sp_model_prefix,
        vocab_size=cfg.vocab_size,
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


# ── Model with position embeddings ──────────────────────────────────────────

class MoQEWithPositions(MoQEModel):
    """MoQEModel + learned position embeddings."""

    def __init__(self, max_seq_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        rng = np.random.default_rng(kwargs.get("seed", 42))
        self.pos_embed = (rng.standard_normal((max_seq_len, self.d_model)) * 0.02).astype(np.float32)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: np.ndarray):
        """Forward with position embeddings."""
        seq_len = len(input_ids)
        x = self.embedding[input_ids]
        x = x + self.pos_embed[:seq_len]  # position encoding

        layer_weights = []
        for layer in self.layers:
            outputs, indices, weights = layer.forward_batch(x)
            x = x + outputs
            layer_weights.append(weights)

        logits = (x @ self.out_proj.T).astype(np.float32)
        return logits, layer_weights


# ── Training step with manual backprop ──────────────────────────────────────

def train_step(
    model: MoQEWithPositions,
    input_ids: np.ndarray,
    labels: np.ndarray,
    shadow_weights: dict,
    lr: float,
    max_grad_norm: float = 1.0,
) -> tuple[float, dict]:
    """Forward + backward + update for one sequence.

    Uses Gumbel-Softmax for routing gradient flow.
    STE for quantization gradients.
    """
    seq_len = len(input_ids)
    gumbel_temp = getattr(model, '_gumbel_temperature', 0.5)

    # ── Forward ──
    x = model.embedding[input_ids].copy()
    x = x + model.pos_embed[:seq_len]
    activations = [x.copy()]

    layer_dequant = []     # [layer][expert] = w_fp
    layer_gumbel_w = []    # [layer] = (seq, n_experts)
    layer_expert_outs = [] # [layer][expert] = (seq, d)

    gpu_handle = getattr(model, '_gpu_train_handle', None)
    gpu_device = getattr(model, '_gpu_train_device', None)

    for l_idx, layer in enumerate(model.layers):
        n_exp = layer.n_experts
        weights, _ = layer.router.forward_gumbel(x, temperature=gumbel_temp)
        layer_gumbel_w.append(weights)

        # Collect shadow weights (needed for backward grad_W)
        experts_fp = []
        for e in range(n_exp):
            sw_key = id(layer) * 100 + e
            w_fp = shadow_weights.get(sw_key)
            if w_fp is None:
                w_fp = _dequant_weights(layer.expert_w_int[e],
                                         layer.expert_scales[e],
                                         layer.block_size)
            experts_fp.append(w_fp)

        # Expert forward: GPU persistent path or CPU BLAS
        expert_outs = []
        gpu_ok = False
        if gpu_handle is not None and _moqe_gpu is not None:
            try:
                x_c = np.ascontiguousarray(x)
                e0_gpu, e1_gpu = _moqe_gpu.moqe_layer_forward(
                    gpu_device, gpu_handle, l_idx, x_c, x_c)
                expert_outs.append(np.asarray(e0_gpu, dtype=np.float32))
                expert_outs.append(np.asarray(e1_gpu, dtype=np.float32))
                # Extra experts beyond the first pair: CPU fallback
                for e in range(2, n_exp):
                    expert_outs.append((x @ experts_fp[e].T).astype(np.float32))
                gpu_ok = True
            except Exception:
                pass

        if not gpu_ok:
            for e in range(n_exp):
                expert_outs.append((x @ experts_fp[e].T).astype(np.float32))

        out = np.zeros_like(x)
        for e in range(n_exp):
            out += weights[:, e:e+1] * expert_outs[e]

        layer_dequant.append(experts_fp)
        layer_expert_outs.append(expert_outs)
        x = x + out
        activations.append(x.copy())

    # Output logits
    logits = (x @ model.out_proj.T).astype(np.float32)

    # ── Loss ──
    loss, grad_logits = _cross_entropy_with_grad(logits, labels)

    # ── Backward: output projection ──
    grad_out_proj = (grad_logits.T @ activations[-1]).astype(np.float32)
    dx = (grad_logits @ model.out_proj).astype(np.float32)

    gradients = {}
    gradients[id(model.out_proj)] = grad_out_proj

    # ── Backward: MoQE layers (reverse) ──
    for l_idx in range(model.n_layers - 1, -1, -1):
        layer = model.layers[l_idx]
        n_exp = layer.n_experts
        x_in = activations[l_idx]
        weights = layer_gumbel_w[l_idx]
        experts_fp = layer_dequant[l_idx]
        expert_outs = layer_expert_outs[l_idx]

        d_out = dx.copy()

        # Expert weight gradients + dx through experts
        dx_expert = np.zeros_like(x_in)

        # Compute per-expert weighted gradients
        d_experts = []
        for e in range(n_exp):
            w_e = weights[:, e:e+1]
            d_e = d_out * w_e
            d_experts.append(d_e)
            grad_we = (d_e.T @ x_in).astype(np.float32)
            gradients[id(layer) * 100 + e] = grad_we

        # dx through experts: GPU path for first pair, CPU for rest
        gpu_bwd_ok = False
        if gpu_handle is not None and _moqe_gpu is not None and len(d_experts) >= 2:
            try:
                d_e0_c = np.ascontiguousarray(d_experts[0])
                d_e1_c = np.ascontiguousarray(d_experts[1])
                dx0_gpu, dx1_gpu = _moqe_gpu.moqe_layer_backward_dx(
                    gpu_device, gpu_handle, l_idx, d_e0_c, d_e1_c)
                dx_expert = np.asarray(dx0_gpu, dtype=np.float32) + \
                            np.asarray(dx1_gpu, dtype=np.float32)
                # Extra experts beyond pair: CPU
                for e in range(2, n_exp):
                    dx_expert += (d_experts[e] @ experts_fp[e]).astype(np.float32)
                gpu_bwd_ok = True
            except Exception:
                pass

        if not gpu_bwd_ok:
            dx_expert = np.zeros_like(x_in)
            for e in range(n_exp):
                dx_expert += (d_experts[e] @ experts_fp[e]).astype(np.float32)

        # Router gradient via Gumbel-Softmax Jacobian
        # For N-way softmax: d_loss/d_logit_e = sum_j w_j * (delta_ej - w_e) * d_scalar_j
        # Simplified: accumulate per-expert scalar gradients
        d_scalars = np.zeros((seq_len, n_exp), dtype=np.float32)
        for e in range(n_exp):
            d_scalars[:, e] = np.sum(d_out * expert_outs[e], axis=-1)

        # Softmax Jacobian: J[i,j] = w[i] * (delta[i,j] - w[j])
        # d_logit = J^T @ d_scalar = w * (d_scalar - sum(w * d_scalar))
        weighted_sum = np.sum(weights * d_scalars, axis=-1, keepdims=True)
        d_logits_router = weights * (d_scalars - weighted_sum) / max(gumbel_temp, 1e-6)

        # Balance loss gradient: push fractions toward targets
        target_frac = layer.get_target_fractions()
        actual_frac = weights.mean(axis=0)  # soft fractions
        balance_grad = 0.1 * 2.0 * (actual_frac - target_frac) / seq_len

        # d_logit/d_W_router = x_in, d_logit/d_b_router = 1
        total_d_logits = d_logits_router + balance_grad[None, :]
        # Router weights: W is (n_experts, d_model), b is (n_experts,)
        grad_router_W = (total_d_logits.T @ x_in).astype(np.float32)
        grad_router_b = total_d_logits.sum(axis=0).astype(np.float32)
        gradients[id(layer.router.W)] = grad_router_W
        gradients[id(layer.router.b)] = grad_router_b

        dx = dx + dx_expert

    # ── Backward: position embeddings ──
    grad_pos = np.zeros_like(model.pos_embed)
    grad_pos[:seq_len] = dx
    gradients[id(model.pos_embed)] = grad_pos

    # ── Backward: embedding ──
    grad_embedding = np.zeros_like(model.embedding)
    for i in range(seq_len):
        tok = int(input_ids[i])
        if 0 <= tok < model.vocab_size:
            grad_embedding[tok] += dx[i]
    gradients[id(model.embedding)] = grad_embedding

    # ── Global gradient norm clipping ──
    total_norm_sq = sum(float(np.sum(g * g)) for g in gradients.values())
    total_norm = math.sqrt(total_norm_sq)
    if max_grad_norm > 0 and total_norm > max_grad_norm:
        clip = max_grad_norm / (total_norm + 1e-8)
        for k in gradients:
            gradients[k] = gradients[k] * clip

    # ── SGD update ──
    # Expert shadow weights
    gpu_handle = getattr(model, '_gpu_train_handle', None)
    gpu_device = getattr(model, '_gpu_train_device', None)

    for l_idx, layer in enumerate(model.layers):
        for e in range(layer.n_experts):
            sw_key = id(layer) * 100 + e
            g = gradients.get(sw_key)
            if g is not None and sw_key in shadow_weights:
                shadow_weights[sw_key] -= lr * g
                # Re-quantize
                bits = layer.expert_specs[e].bits
                layer.expert_w_int[e], s_flat = _quantize_symmetric(
                    shadow_weights[sw_key], bits, layer.block_size)
                num_blocks = (layer.d_model + layer.block_size - 1) // layer.block_size
                layer.expert_scales[e] = s_flat[:num_blocks * layer.d_out].reshape(
                    layer.d_out, num_blocks)
                # Re-upload updated weights to GPU
                if gpu_handle is not None and _moqe_gpu is not None:
                    try:
                        _moqe_gpu.moqe_train_update_expert(
                            gpu_device, gpu_handle, l_idx, e,
                            shadow_weights[sw_key])
                    except Exception:
                        pass

        # Router
        g_W = gradients.get(id(layer.router.W))
        if g_W is not None:
            layer.router.W -= lr * g_W
        g_b = gradients.get(id(layer.router.b))
        if g_b is not None:
            layer.router.b -= lr * g_b

    # Embedding + pos_embed + out_proj
    g = gradients.get(id(model.embedding))
    if g is not None:
        model.embedding -= lr * g
    g = gradients.get(id(model.pos_embed))
    if g is not None:
        model.pos_embed -= lr * g
    g = gradients.get(id(model.out_proj))
    if g is not None:
        model.out_proj -= lr * g

    stats = {"loss": loss, "grad_norm": total_norm}
    return loss, stats


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_ppl(model, x_data, y_data, batch_size=32, max_batches=50):
    """Compute perplexity."""
    total_loss, total_tokens = 0.0, 0
    for i in range(0, min(len(x_data), max_batches * batch_size), batch_size):
        for j in range(i, min(i + batch_size, len(x_data))):
            xb = x_data[j]
            yb = y_data[j]
            logits, _ = model.forward(xb)
            probs = _softmax(logits)
            for t in range(len(yb)):
                tok = int(yb[t])
                if 0 <= tok < probs.shape[1]:
                    total_loss -= math.log(max(float(probs[t, tok]), 1e-8))
                    total_tokens += 1
    avg = total_loss / max(total_tokens, 1)
    return float(math.exp(min(avg, 20))), float(avg)


def save_checkpoint(model, shadow_weights, step, best_ppl,
                     save_dir="sandbox/moqe_tinystories/checkpoints"):
    """Save model + shadow weights to disk."""
    os.makedirs(save_dir, exist_ok=True)
    save_dict = {
        "embedding": model.embedding,
        "out_proj": model.out_proj,
        "pos_embed": model.pos_embed,
        "step": np.array([step], dtype=np.int64),
        "best_ppl": np.array([best_ppl], dtype=np.float32),
    }
    for li, layer in enumerate(model.layers):
        for e in range(layer.n_experts):
            sw_key = id(layer) * 100 + e
            save_dict[f"L{li}_E{e}_w_int"] = layer.expert_w_int[e]
            save_dict[f"L{li}_E{e}_scales"] = layer.expert_scales[e]
            if sw_key in shadow_weights:
                save_dict[f"L{li}_E{e}_shadow"] = shadow_weights[sw_key]
        save_dict[f"L{li}_router_W"] = layer.router.W
        save_dict[f"L{li}_router_b"] = layer.router.b
    path = os.path.join(save_dir, f"moqe_step{step}.npz")
    np.savez_compressed(path, **save_dict)
    logger.info("Saved checkpoint: {} ({:.1f}MB)", path, os.path.getsize(path) / 1e6)
    return path


def load_checkpoint(model, shadow_weights, path):
    """Load model + shadow weights from disk."""
    data = np.load(path)
    model.embedding[:] = data["embedding"]
    model.out_proj[:] = data["out_proj"]
    model.pos_embed[:] = data["pos_embed"]
    step = int(data["step"][0])
    best_ppl = float(data["best_ppl"][0])
    for li, layer in enumerate(model.layers):
        for e in range(layer.n_experts):
            layer.expert_w_int[e] = data[f"L{li}_E{e}_w_int"]
            layer.expert_scales[e] = data[f"L{li}_E{e}_scales"]
            sw_key = id(layer) * 100 + e
            shadow_key = f"L{li}_E{e}_shadow"
            if shadow_key in data:
                shadow_weights[sw_key] = data[shadow_key].copy()
        layer.router.W = data[f"L{li}_router_W"]
        layer.router.b = data[f"L{li}_router_b"]
    logger.info("Loaded checkpoint: {} (step={}, best_ppl={:.2f})", path, step, best_ppl)
    return step, best_ppl


def lr_schedule(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_min + (lr_max - lr_min) * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = Config(
        d_model=1024,
        n_layers=12,
        top_k=1,
        train_steps=200000,
        max_stories=300000,
        batch_size=64,
        seq_len=512,
    )

    train_x, train_y, val_x, val_y, vocab_size, sp = load_data(cfg)

    expert_specs = [
        ExpertSpec(bits=4, specialty="general", target_fraction=0.60),
        ExpertSpec(bits=4, specialty="code", target_fraction=0.20),
        ExpertSpec(bits=8, specialty="factual", target_fraction=0.15),
        ExpertSpec(bits=8, specialty="rare", target_fraction=0.05),
    ]

    model = MoQEWithPositions(
        max_seq_len=cfg.seq_len + 16,
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        expert_specs=expert_specs,
        top_k=cfg.top_k,
        block_size=cfg.block_size,
        seed=cfg.seed,
    )
    model._gumbel_temperature = 1.0

    # Initialize shadow weights (FP32 copies for gradient updates)
    shadow_weights = {}
    for layer in model.layers:
        for e in range(layer.n_experts):
            sw_key = id(layer) * 100 + e
            shadow_weights[sw_key] = _dequant_weights(
                layer.expert_w_int[e], layer.expert_scales[e], layer.block_size
            ).copy()

    # ── Upload expert weights to GPU (persistent W + W^T) ──
    if _moqe_gpu is not None:
        try:
            gpu_weight_list = []
            for layer in model.layers:
                for e in range(layer.n_experts):
                    sw_key = id(layer) * 100 + e
                    gpu_weight_list.append(shadow_weights[sw_key])

            gpu_device = _moqe_gpu.Device()
            spv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..', 'grilly', 'shaders', 'spv')
            # Try standard grilly shader path
            for candidate in [spv_dir,
                              'C:/Users/grill/Documents/GitHub/grilly/shaders/spv',
                              os.path.expanduser('~/Documents/GitHub/grilly/shaders/spv')]:
                if os.path.isdir(candidate):
                    gpu_device.load_shaders(candidate)
                    break

            gpu_handle = _moqe_gpu.moqe_train_upload(
                gpu_device, gpu_weight_list,
                cfg.d_model, cfg.n_layers, cfg.seq_len)
            model._gpu_train_handle = gpu_handle
            model._gpu_train_device = gpu_device

            vram_mb = cfg.n_layers * len(expert_specs) * 2 * cfg.d_model ** 2 * 4 / 1e6
            logger.success("GPU training: {} experts uploaded ({:.0f}MB VRAM)",
                           cfg.n_layers * len(expert_specs), vram_mb)
        except Exception as e:
            logger.warning("GPU training init failed: {} — using CPU", e)
            model._gpu_train_handle = None
            model._gpu_train_device = None
    else:
        logger.info("grilly_core.moqe_layer_forward not available — CPU only")
        model._gpu_train_handle = None
        model._gpu_train_device = None

    logger.info("MoQE TinyStories: vocab={}, d={}, layers={}, experts={}x{}, top_k={}",
                vocab_size, cfg.d_model, cfg.n_layers,
                len(expert_specs), cfg.n_layers, cfg.top_k)
    logger.info("  Experts: {}", [(s.bits, s.specialty) for s in expert_specs])

    t0 = time.time()
    best_ppl = float("inf")
    step = 0

    while step < cfg.train_steps:
        perm = np.random.permutation(len(train_x))
        train_x, train_y = train_x[perm], train_y[perm]

        for i in range(0, len(train_x), cfg.batch_size):
            if step >= cfg.train_steps:
                break

            # LR schedule
            lr = lr_schedule(step, cfg.warmup_steps, cfg.train_steps,
                             cfg.lr, cfg.lr_min)

            # Gumbel temperature annealing: 1.0 → 0.1
            progress = min(step / max(cfg.train_steps, 1), 1.0)
            model._gumbel_temperature = max(0.1, 1.0 - 0.9 * progress)

            # Train on one sequence at a time (online learning)
            idx = i % len(train_x)
            loss, stats = train_step(
                model, train_x[idx], train_y[idx].ravel(),
                shadow_weights, lr, cfg.max_grad_norm,
            )

            if step % cfg.val_every == 0:
                val_ppl, val_loss = compute_ppl(model, val_x, val_y)
                elapsed = time.time() - t0
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "step={:5d} lr={:.5f} T={:.2f} | loss={:.3f} gnorm={:.2f} "
                    "| val_ppl={:.2f} | {:.1f} stp/s",
                    step, lr, model._gumbel_temperature,
                    loss, stats["grad_norm"], val_ppl, sps)
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    save_checkpoint(model, shadow_weights, step, best_ppl)

            if step > 0 and step % 5000 == 0:
                save_checkpoint(model, shadow_weights, step, best_ppl)

            step += 1

    save_checkpoint(model, shadow_weights, step, best_ppl)
    elapsed = time.time() - t0
    val_ppl, _ = compute_ppl(model, val_x, val_y)

    # Expert usage
    usage = model.get_expert_usage(val_x[0])

    print(f"\n{'='*55}")
    print(f"  MoQE TinyStories Results")
    print(f"{'='*55}")
    print(f"  Final val PPL:  {val_ppl:.2f}")
    print(f"  Best val PPL:   {best_ppl:.2f}")
    print(f"  Steps:          {step}")
    print(f"  Time:           {elapsed:.0f}s")
    print(f"  Steps/sec:      {step / elapsed:.1f}")
    print(f"  Expert usage:")
    for name, frac in usage.items():
        print(f"    {name}: {frac*100:.1f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
