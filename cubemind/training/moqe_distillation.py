"""MoQE Offline Distillation Pipeline.

Two-phase distillation from dense teacher LLM(s) to a MoQE student:

Phase 1 (Colab/cloud): Extract teacher logits via llama-cpp-python
  Teacher GGUF → logits_all=True → save .npz to disk ("Logit Banking")
  Supports multi-teacher ensemble: run extraction for each teacher into
  separate directories, then merge their logits for distillation.

Phase 2 (Local/Grilly): Train MoQE student from saved logits
  Streaming DataLoader → CE + KL-div + router balancing → backprop
  Full gradient flow through experts via Straight-Through Estimator (STE).

Loss = 0.3 * CE(hard labels) + 0.6 * KL(soft teacher) + 0.1 * router_balance
Router target: 85% tokens → 4-bit expert, 15% → 8-bit expert.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ── Phase 1: Teacher Logit Extraction ─────────────────────────────────────

def extract_teacher_logits(
    model_path: str,
    texts: list[str],
    save_dir: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = 0,
) -> int:
    """Extract teacher logits from a GGUF model and save to .npz files."""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("llama-cpp-python required for logit extraction")

    os.makedirs(save_dir, exist_ok=True)

    teacher = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=True,
        verbose=False,
    )

    count = 0
    for i, text in enumerate(texts):
        tokens = teacher.tokenize(text.encode("utf-8"))
        if not tokens:
            continue

        teacher.reset()
        teacher.eval(tokens)

        logits = np.array(teacher.scores[: len(tokens)], dtype=np.float16)

        save_path = os.path.join(save_dir, f"sequence_{i:06d}.npz")
        np.savez_compressed(
            save_path,
            input_tokens=np.array(tokens, dtype=np.int32),
            logits=logits,
        )
        count += 1

        if (i + 1) % 100 == 0:
            print(f"  Extracted {i + 1}/{len(texts)} sequences "
                  f"(last: {logits.shape[0]} tokens, vocab={logits.shape[1]})")

    print(f"Saved {count} logit files to {save_dir}")
    return count


def merge_ensemble_logits(
    teacher_dirs: list[str],
    output_dir: str,
    weights: list[float] | None = None,
    temperature: float = 1.0,
) -> int:
    """Merge logits from multiple teachers into ensemble soft distributions."""
    n_teachers = len(teacher_dirs)
    if n_teachers == 0:
        raise ValueError("Need at least one teacher directory")

    if weights is None:
        weights = [1.0 / n_teachers] * n_teachers
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    os.makedirs(output_dir, exist_ok=True)

    file_sets = []
    for d in teacher_dirs:
        files = {os.path.basename(f) for f in glob.glob(os.path.join(d, "sequence_*.npz"))}
        file_sets.append(files)

    common_files = sorted(file_sets[0].intersection(*file_sets[1:]))
    print(f"Ensemble merge: {n_teachers} teachers, {len(common_files)} common sequences")

    count = 0
    for fname in common_files:
        try:
            data_0 = np.load(os.path.join(teacher_dirs[0], fname))
            tokens = data_0["input_tokens"]
            seq_len = len(tokens)

            ensemble_probs = None
            for t_idx, t_dir in enumerate(teacher_dirs):
                data = np.load(os.path.join(t_dir, fname))
                logits = data["logits"][:seq_len].astype(np.float32)

                scaled = logits / max(temperature, 1e-8)
                max_vals = np.max(scaled, axis=-1, keepdims=True)
                exp_vals = np.exp(scaled - max_vals)
                probs = exp_vals / (exp_vals.sum(axis=-1, keepdims=True) + 1e-8)

                if ensemble_probs is None:
                    ensemble_probs = weights[t_idx] * probs
                else:
                    min_vocab = min(ensemble_probs.shape[-1], probs.shape[-1])
                    ensemble_probs = ensemble_probs[:, :min_vocab]
                    ensemble_probs += weights[t_idx] * probs[:, :min_vocab]

            ensemble_logits = np.log(ensemble_probs + 1e-8).astype(np.float16)

            save_path = os.path.join(output_dir, fname)
            np.savez_compressed(
                save_path,
                input_tokens=tokens,
                logits=ensemble_logits,
                n_teachers=np.array([n_teachers], dtype=np.int32),
                weights=np.array(weights, dtype=np.float32),
            )
            count += 1

        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    print(f"Ensemble merge complete: {count} sequences saved to {output_dir}")
    return count


# ── Phase 2: Streaming DataLoader ─────────────────────────────────────────

class OfflineDistillationLoader:
    """Double-buffered prefetch loader for teacher logits.

    Loads files in chunks into RAM (default ~10GB). A background thread
    pre-loads the next chunk while the current chunk is being trained on.
    Result: near-zero I/O wait after the first chunk.

    Memory layout:
      [chunk A in RAM] ← training reads from here
      [chunk B loading] ← background thread fills this
      When A is exhausted, swap A↔B and start loading the next chunk.
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 1024,
        shuffle: bool = True,
        chunk_gb: float = 10.0,
    ) -> None:
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.chunk_gb = chunk_gb
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.file_list:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        # Estimate files per chunk from first file
        sample_size = os.path.getsize(self.file_list[0])
        # In-memory size is ~2x disk (float16→float32 + tokens)
        mem_per_file = sample_size * 2
        self.chunk_size = max(1, int(chunk_gb * 1e9 / max(mem_per_file, 1)))
        self.chunk_size = min(self.chunk_size, len(self.file_list))

        print(f"OfflineDistillationLoader: {len(self.file_list)} sequences, "
              f"chunk={self.chunk_size} files (~{chunk_gb:.0f}GB RAM)")

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def _load_chunk(file_paths: list[str], max_seq_len: int) -> list[tuple]:
        """Load a chunk of files into RAM. Runs in background thread."""
        chunk = []
        for path in file_paths:
            try:
                data = np.load(path)
                tokens = data["input_tokens"][:max_seq_len].astype(np.int32)
                logits = data["logits"][:max_seq_len].astype(np.float32)
                if len(tokens) >= 2:
                    chunk.append((tokens[:-1], tokens[1:], logits[:-1]))
            except Exception:
                pass
        return chunk

    def __iter__(self):
        import concurrent.futures

        files = list(self.file_list)
        if self.shuffle:
            np.random.shuffle(files)

        # Split into chunks
        chunks = [files[i:i + self.chunk_size]
                  for i in range(0, len(files), self.chunk_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            # Kick off first chunk load
            future = pool.submit(self._load_chunk, chunks[0], self.max_seq_len)

            for ci in range(len(chunks)):
                # Wait for current chunk
                current_data = future.result()

                # Start loading next chunk in background
                if ci + 1 < len(chunks):
                    future = pool.submit(
                        self._load_chunk, chunks[ci + 1], self.max_seq_len)

                # Yield from current chunk (all in RAM, zero I/O)
                if self.shuffle:
                    np.random.shuffle(current_data)
                for input_ids, labels, teacher_logits in current_data:
                    yield input_ids, labels, teacher_logits

                # Free RAM before next chunk arrives
                del current_data


# ── Phase 2: Loss Functions ───────────────────────────────────────────────

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - m)
    return ex / (ex.sum(axis=axis, keepdims=True) + 1e-8)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    logsumexp = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True) + 1e-8)
    return x - logsumexp


def _cross_entropy_with_grad(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Cross-entropy loss with gradient w.r.t. logits.

    Returns (loss, d_loss/d_logits) where grad shape = logits shape.
    """
    seq_len, vocab = logits.shape
    probs = _softmax(logits)

    # Loss: -mean(log(probs[i, label_i]))
    loss = 0.0
    for i in range(seq_len):
        label = int(labels[i])
        if 0 <= label < vocab:
            loss -= float(np.log(probs[i, label] + 1e-8))
    loss /= max(seq_len, 1)

    # Gradient: (probs - one_hot) / seq_len
    grad = probs.copy()
    for i in range(seq_len):
        label = int(labels[i])
        if 0 <= label < vocab:
            grad[i, label] -= 1.0
    grad /= max(seq_len, 1)

    return loss, grad


def _kl_divergence_with_grad(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float,
) -> tuple[float, np.ndarray]:
    """KL divergence with gradient w.r.t. student logits.

    KL(teacher || student) at given temperature.
    Returns (loss, d_loss/d_student_logits).
    """
    soft_teacher = _softmax(teacher_logits / temperature)
    soft_student = _softmax(student_logits / temperature)

    # KL = sum(P * (log P - log Q))
    log_teacher = np.log(soft_teacher + 1e-8)
    log_student = np.log(soft_student + 1e-8)
    kl = soft_teacher * (log_teacher - log_student)
    loss = float(np.mean(np.sum(kl, axis=-1))) * (temperature ** 2)

    # Gradient w.r.t. student logits:
    # d KL / d z_student = (soft_student - soft_teacher) * temperature
    grad = (soft_student - soft_teacher) * temperature
    grad *= (temperature ** 2) / max(student_logits.shape[0], 1)

    return loss, grad


# ── Phase 2: Full Backprop Through MoQE ──────────────────────────────────

# Native MoQE GPU training ops — persistent weights, batched dispatch
_moqe_gpu = None
try:
    import grilly_core as _gc
    if hasattr(_gc, 'moqe_layer_forward'):
        _moqe_gpu = _gc
except Exception:
    pass


def _dequant_weights(w_int: np.ndarray, scales: np.ndarray, block_size: int) -> np.ndarray:
    """Dequantize INT4/INT8 weights back to FP32. Fully vectorized."""
    d_out, d_in = w_int.shape
    num_blocks = scales.shape[1]
    # Reshape to (d_out, num_blocks, block_size), broadcast multiply, reshape back
    padded = d_in + (block_size - d_in % block_size) % block_size
    if padded == d_in:
        w_blocked = w_int.reshape(d_out, num_blocks, block_size).astype(np.float32)
    else:
        # Pad last block
        w_padded = np.zeros((d_out, padded), dtype=w_int.dtype)
        w_padded[:, :d_in] = w_int
        w_blocked = w_padded.reshape(d_out, num_blocks, block_size).astype(np.float32)
    # scales: (d_out, num_blocks) → (d_out, num_blocks, 1) for broadcast
    w_fp = (w_blocked * scales[:, :, np.newaxis]).reshape(d_out, -1)[:, :d_in]
    return np.ascontiguousarray(w_fp)


def moqe_backward(
    model,
    input_ids: np.ndarray,
    labels: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 2.0,
    target_8bit: float = 0.15,
    max_grad_norm: float = 1.0,
) -> tuple[dict, dict]:
    """Full forward + backward pass through MoQE with gradient computation.

    Uses Straight-Through Estimator (STE) for quantization:
    gradients flow through the dequantized weights as if quantization
    were identity. This is the standard approach for QAT.

    Gradient clipping: all gradients are globally norm-clipped to
    max_grad_norm to prevent loss explosions from outlier sequences.

    Returns:
        (gradients, stats): gradients maps id(param) → grad array,
        stats contains loss breakdown.
    """
    seq_len = len(input_ids)
    gradients = {}

    # ── Forward pass (store activations for backward) ─────────────────
    x = model.embedding[input_ids].copy()  # (seq, d_model)
    activations = [x.copy()]  # activation before each layer
    router_choices_per_layer = []
    router_probs_per_layer = []
    dequant_cache = []  # Cache dequantized weights per layer (avoid re-dequant)

    # Get GPU training handle if available
    gpu_handle = getattr(model, '_gpu_train_handle', None)
    gpu_device = getattr(model, '_gpu_train_device', None)

    # Temperature annealing for Gumbel-Softmax routing
    # Start at 1.0 (soft), anneal toward 0.1 over training
    gumbel_temp = getattr(model, '_gumbel_temperature', 1.0)
    router_weights_per_layer = []  # (batch, 2) soft weights

    for l_idx, layer in enumerate(model.layers):
        # Gumbel-Softmax soft routing: both experts get gradient
        weights, logits_2 = layer.router.forward_gumbel(x, temperature=gumbel_temp)
        router_weights_per_layer.append(weights)
        router_probs_per_layer.append(weights[:, 1])  # prob of expert 1

        # Hard choices for GPU dispatch (which expert gets each token)
        choices = (weights[:, 1] > 0.5).astype(np.int32)
        router_choices_per_layer.append(choices)

        w0_fp = _dequant_weights(layer.w0_int, layer.s0, layer.block_size)
        w1_fp = _dequant_weights(layer.w1_int, layer.s1, layer.block_size)
        dequant_cache.append((w0_fp, w1_fp))

        # Soft blending: out = w0_weight * expert0(x) + w1_weight * expert1(x)
        # Both experts process ALL tokens, weighted by Gumbel-Softmax
        e0_out = (x @ w0_fp.T).astype(np.float32)  # (seq, d)
        e1_out = (x @ w1_fp.T).astype(np.float32)  # (seq, d)

        # GPU path for expert matmuls
        if gpu_handle is not None and _moqe_gpu is not None:
            try:
                # Both experts process all tokens (Gumbel-Softmax blending)
                x_c = np.ascontiguousarray(x)
                e0_out_gpu, e1_out_gpu = _moqe_gpu.moqe_layer_forward(
                    gpu_device, gpu_handle, l_idx, x_c, x_c)
                e0_out = np.asarray(e0_out_gpu, dtype=np.float32)
                e1_out = np.asarray(e1_out_gpu, dtype=np.float32)
            except Exception:
                pass  # CPU fallback already computed above

        # Soft blend: weight each expert's output
        w0 = weights[:, 0:1]  # (seq, 1)
        w1 = weights[:, 1:2]  # (seq, 1)
        out = w0 * e0_out + w1 * e1_out

        x = x + out
        activations.append(x.copy())

    # Output logits: CPU BLAS (vocab too big for PCIe)
    student_logits = (x @ model.out_proj.T).astype(np.float32)

    # ── Loss computation ──────────────────────────────────────────────
    # Truncate to teacher vocab size if needed
    min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s_logits = student_logits[:, :min_vocab]
    t_logits = teacher_logits[:, :min_vocab]

    loss_ce, grad_ce = _cross_entropy_with_grad(s_logits, labels)
    loss_kd, grad_kd = _kl_divergence_with_grad(s_logits, t_logits, temperature)

    all_probs = np.stack(router_probs_per_layer)
    actual_8bit = float(np.mean(all_probs > 0.5))

    # ── Entropy-Gated Router Loss (Society of Thought) ───────────────
    # Instead of static balance penalty, use teacher entropy to identify
    # "conflict tokens" (high uncertainty = multi-perspective debate).
    # Only conflict tokens are allowed to use 8-bit experts.
    # Low-entropy tokens are aggressively pushed to 4-bit.
    #
    # Shannon entropy of teacher distribution: H = -sum(p * log(p))
    # High H → flat distribution → teacher is uncertain → needs 8-bit
    # Low H  → peaked distribution → teacher is confident → 4-bit is fine
    teacher_probs = _softmax(t_logits)
    teacher_entropy = -np.sum(
        teacher_probs * np.log(teacher_probs + 1e-8), axis=-1)  # (seq,)

    # Entropy threshold: tokens above this get relaxed 8-bit allowance
    # Use median entropy as adaptive threshold (robust to distribution shift)
    entropy_threshold = float(np.median(teacher_entropy)) + 0.5
    is_conflict = teacher_entropy > entropy_threshold  # (seq,) bool

    # Dynamic per-token target: conflict tokens get 8b_target up to 0.8,
    # normal tokens get base_target (0.15)
    dynamic_8b_target = np.where(
        is_conflict,
        np.clip(target_8bit + 0.5 * (teacher_entropy - entropy_threshold), 0, 0.8),
        target_8bit,
    ).astype(np.float32)  # (seq,)

    # Router loss: per-token MSE between actual routing and dynamic target
    # Averaged across all layers' routing decisions
    mean_router_8b = np.mean(all_probs, axis=0)  # (seq,) avg 8-bit prob across layers
    loss_router = float(np.mean((mean_router_8b - dynamic_8b_target) ** 2))
    mean_entropy = float(np.mean(teacher_entropy))
    conflict_frac = float(np.mean(is_conflict))

    # Combined gradient w.r.t. output logits
    grad_logits = np.zeros_like(student_logits)
    grad_logits[:, :min_vocab] = 0.3 * grad_ce + 0.6 * grad_kd

    total_loss = 0.3 * loss_ce + 0.6 * loss_kd + 0.1 * loss_router

    # ── Backward: output projection (CPU BLAS — vocab too large for PCIe) ──
    # logits = x @ out_proj.T
    # d_loss/d_out_proj = grad_logits^T @ x  → (vocab, d_model)
    # d_loss/d_x = grad_logits @ out_proj   → (seq, d_model)
    grad_out_proj = (grad_logits.T @ activations[-1]).astype(np.float32)
    gradients[id(model.out_proj)] = grad_out_proj

    dx = (grad_logits @ model.out_proj).astype(np.float32)

    # ── Backward: MoQE layers (reverse order) ────────────────────────
    for l_idx in range(model.n_layers - 1, -1, -1):
        layer = model.layers[l_idx]
        choices = router_choices_per_layer[l_idx]
        probs = router_probs_per_layer[l_idx]
        x_in = activations[l_idx]
        w0_fp, w1_fp = dequant_cache[l_idx]  # Cached from forward pass

        # Residual: x_out = x_in + expert(x_in)
        d_expert_out = dx.copy()

        # ── Expert backward with Gumbel-Softmax soft gradient flow ────────
        # Forward was: out = w0 * expert0(x) + w1 * expert1(x)
        # d_out flows to both experts proportional to their Gumbel weights.
        weights = router_weights_per_layer[l_idx]
        w0_weight = weights[:, 0:1]  # (seq, 1)
        w1_weight = weights[:, 1:2]  # (seq, 1)

        # d_expert0 = d_out * w0_weight, d_expert1 = d_out * w1_weight
        d_e0 = d_expert_out * w0_weight  # (seq, d) — gradient to expert 0
        d_e1 = d_expert_out * w1_weight  # (seq, d) — gradient to expert 1

        # grad_W for each expert: full batch (not masked — Gumbel gives all tokens)
        grad_w0 = (d_e0.T @ x_in).astype(np.float32)
        grad_w1 = (d_e1.T @ x_in).astype(np.float32)

        # dx through experts: dx = w0 * (d_out @ W0) + w1 * (d_out @ W1)
        if gpu_handle is not None and _moqe_gpu is not None:
            try:
                d_e0_c = np.ascontiguousarray(d_e0)
                d_e1_c = np.ascontiguousarray(d_e1)
                dx0_gpu, dx1_gpu = _moqe_gpu.moqe_layer_backward_dx(
                    gpu_device, gpu_handle, l_idx, d_e0_c, d_e1_c)
                dx_expert = np.asarray(dx0_gpu, dtype=np.float32) + np.asarray(dx1_gpu, dtype=np.float32)
            except Exception:
                dx_expert = (d_e0 @ w0_fp + d_e1 @ w1_fp).astype(np.float32)
        else:
            dx_expert = (d_e0 @ w0_fp + d_e1 @ w1_fp).astype(np.float32)

        # Gradient to router weights: d_loss/d_weights
        # out = w0 * e0_out + w1 * e1_out
        # Need e0_out and e1_out from forward — reconstruct from dequant cache
        e0_out = (x_in @ w0_fp.T).astype(np.float32)
        e1_out = (x_in @ w1_fp.T).astype(np.float32)
        # d_loss/d_w0 = sum(d_out * e0_out), d_loss/d_w1 = sum(d_out * e1_out)
        d_w0_scalar = np.sum(d_expert_out * e0_out, axis=-1)  # (seq,)
        d_w1_scalar = np.sum(d_expert_out * e1_out, axis=-1)  # (seq,)
        # Gumbel-Softmax gradient: d_weights/d_logits (Jacobian of softmax)
        # For 2-class: d_softmax/d_logit = w0*w1 (both directions)
        gs_deriv = w0_weight.ravel() * w1_weight.ravel() / max(gumbel_temp, 1e-6)
        d_logit_1 = (d_w1_scalar - d_w0_scalar) * gs_deriv  # (seq,)

        # Store expert weight gradients (STE: applied to FP32 shadow weights)
        gradients[id(layer) * 10 + 0] = grad_w0  # Expert 0
        gradients[id(layer) * 10 + 1] = grad_w1  # Expert 1

        # ── Router gradient (entropy-gated Gumbel-Softmax) ─────────────
        # d_loss/d_logit flows through the Gumbel-Softmax weights.
        # d_logit_1 was computed above from expert output gradients.
        # Balance gradient is now per-token based on teacher entropy.
        probs = router_probs_per_layer[l_idx]
        # Per-token balance gradient: 2 * (actual_prob - dynamic_target)
        per_token_balance_grad = 2.0 * (probs - dynamic_8b_target)  # (seq,)
        balance_d_logit = 0.1 * per_token_balance_grad * gs_deriv / seq_len

        total_d_logit = d_logit_1 + balance_d_logit  # (seq,)

        # d_logit/d_w = x_in, d_logit/d_b = 1
        d_router_w = (total_d_logit[:, np.newaxis] * x_in).sum(axis=0).astype(np.float32)
        d_router_b = np.atleast_1d(np.float32(total_d_logit.sum()))

        gradients[id(layer.router.w)] = d_router_w
        gradients[id(layer.router.b)] = d_router_b

        # Propagate gradient through residual: dx += dx_expert
        dx = dx + dx_expert

    # ── Backward: embedding ──────────────────────────────────────────
    # x = embedding[input_ids] → scatter-add dx into embedding gradient
    grad_embedding = np.zeros_like(model.embedding)
    for i in range(seq_len):
        tok = int(input_ids[i])
        if 0 <= tok < model.vocab_size:
            grad_embedding[tok] += dx[i]
    gradients[id(model.embedding)] = grad_embedding

    # ── Global gradient norm clipping ─────────────────────────────────
    # Prevents loss explosions from outlier sequences (e.g. rare tokens
    # causing huge embedding gradients at vocab=151K).
    if max_grad_norm > 0:
        total_norm_sq = 0.0
        for g in gradients.values():
            total_norm_sq += float(np.sum(g * g))
        total_norm = np.sqrt(total_norm_sq)
        if total_norm > max_grad_norm:
            clip_coef = max_grad_norm / (total_norm + 1e-8)
            for pid in gradients:
                gradients[pid] = gradients[pid] * clip_coef

    stats = {
        "total_loss": total_loss,
        "loss_ce": loss_ce,
        "loss_kd": loss_kd,
        "loss_router": loss_router,
        "actual_8bit_frac": actual_8bit,
        "grad_norm": float(total_norm) if max_grad_norm > 0 else 0.0,
        "entropy": mean_entropy,
        "conflict_frac": conflict_frac,
    }
    return gradients, stats


# ── Phase 2: Full Distillation Training with Backprop ─────────────────────

def run_offline_distillation(
    model,
    data_dir: str,
    epochs: int = 3,
    max_seq_len: int = 512,
    temperature: float = 2.0,
    target_8bit: float = 0.15,
    lr: float = 3e-4,
    save_dir: str | None = None,
    save_every: int = 200,
) -> list[dict]:
    """Run offline MoQE distillation with full backprop.

    Uses AutoHypergradientAdamW for adaptive learning rate.
    STE (Straight-Through Estimator) for quantization gradients.
    Expert weights are maintained as FP32 shadow copies, re-quantized
    after each optimizer step.

    Args:
        model:       MoQEModel instance.
        data_dir:    Directory with .npz logit files.
        epochs:      Number of training epochs.
        max_seq_len: Max sequence length per sample.
        temperature: KL divergence temperature.
        target_8bit: Target 8-bit routing fraction.
        lr:          Initial learning rate.
        save_dir:    Directory to save checkpoints (None = no save).
        save_every:  Save checkpoint every N batches.

    Returns:
        List of per-epoch summary dicts.
    """
    from cubemind.execution.moqe import (
        _quantize_weights_int4,
        _quantize_weights_int8,
    )

    # Import optimizer (prefer auto-hypergradient if available)
    try:
        from grilly.optim.hypergradient import AutoHypergradientAdamW
        OptimizerClass = AutoHypergradientAdamW
        optim_kwargs = dict(
            lr=lr, hyper_lr=0.005, warmup_steps=20,
            lr_max=lr * 3,  # Don't let lr run away beyond 3x initial
            track_surprise=True, surprise_gamma=0.95,
        )
        log.info("Using AutoHypergradientAdamW with surprise tracking")
    except ImportError:
        try:
            from grilly.optim.adamw import AdamW
            OptimizerClass = AdamW
            optim_kwargs = dict(lr=lr)
            log.info("Using AdamW")
        except ImportError:
            OptimizerClass = None

    loader = OfflineDistillationLoader(data_dir, max_seq_len=max_seq_len)

    # ── Initialize FP32 shadow weights for STE training ───────────────
    # Expert weights live as INT4/INT8 for inference, but we maintain
    # FP32 copies for gradient accumulation. After each optimizer step,
    # we re-quantize back to INT for the next forward pass.
    #
    # Embedding + out_proj are FROZEN: they're (vocab, d_model) matrices
    # that amplify gradients by vocab/d_model ratio (~1000x for 151K vocab).
    # Training them with a d_model=128 bottleneck causes instant divergence.
    # Only expert weights and routers are trained.
    shadow_weights = {}
    all_params = []

    gpu_weight_list = []  # For GPU upload: [w0_L0, w1_L0, w0_L1, ...]

    for layer in model.layers:
        w0_fp = _dequant_weights(layer.w0_int, layer.s0, layer.block_size)
        w1_fp = _dequant_weights(layer.w1_int, layer.s1, layer.block_size)
        shadow_weights[id(layer) * 10 + 0] = w0_fp
        shadow_weights[id(layer) * 10 + 1] = w1_fp
        gpu_weight_list.append(w0_fp)
        gpu_weight_list.append(w1_fp)
        all_params.append(w0_fp)
        all_params.append(w1_fp)
        all_params.append(layer.router.w)
        all_params.append(np.atleast_1d(layer.router.b))

    # Embedding + out_proj frozen (not in optimizer)

    # Create optimizer
    optimizer = None
    if OptimizerClass is not None:
        optimizer = OptimizerClass(
            [{"params": all_params}],
            **optim_kwargs,
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ── Upload expert weights to GPU (persistent W + W^T) ────────────
    if _moqe_gpu is not None:
        try:
            gpu_device = _moqe_gpu.Device()
            gpu_device.load_shaders(
                'C:/Users/grill/Documents/GitHub/grilly/shaders/spv')
            gpu_handle = _moqe_gpu.moqe_train_upload(
                gpu_device, gpu_weight_list,
                model.d_model, model.n_layers, max_seq_len)
            model._gpu_train_handle = gpu_handle
            model._gpu_train_device = gpu_device
            vram_mb = model.n_layers * 2 * 2 * model.d_model**2 * 4 / 1e6
            log.info("MoQE GPU training: %d experts on GPU (%.0fMB VRAM)",
                     model.n_layers * 2, vram_mb)
        except Exception as e:
            log.warning("GPU training init failed: %s — using CPU", e)
            model._gpu_train_handle = None
            model._gpu_train_device = None
    else:
        model._gpu_train_handle = None
        model._gpu_train_device = None

    epoch_stats = []

    # Gumbel-Softmax temperature annealing schedule
    total_steps = epochs * len(loader)
    step_count = 0

    for epoch in range(epochs):
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_8bit = 0.0
        n_batches = 0

        for input_ids, labels, teacher_logits in loader:
            # Anneal temperature: 1.0 → 0.1 over training
            progress = min(step_count / max(total_steps, 1), 1.0)
            model._gumbel_temperature = 1.0 - 0.9 * progress
            step_count += 1
            # Forward + backward
            grads, stats = moqe_backward(
                model, input_ids, labels, teacher_logits,
                temperature=temperature, target_8bit=target_8bit,
            )

            total_loss += stats["total_loss"]
            total_ce += stats["loss_ce"]
            total_kd += stats["loss_kd"]
            total_8bit += stats["actual_8bit_frac"]
            n_batches += 1

            if optimizer is not None:

                # Map shadow weight gradients to their param ids
                # (embedding + out_proj frozen — not in optimizer)
                opt_grads = {}
                for layer in model.layers:
                    sw0 = shadow_weights[id(layer) * 10 + 0]
                    sw1 = shadow_weights[id(layer) * 10 + 1]
                    opt_grads[id(sw0)] = grads.get(id(layer) * 10 + 0, np.zeros_like(sw0))
                    opt_grads[id(sw1)] = grads.get(id(layer) * 10 + 1, np.zeros_like(sw1))
                    opt_grads[id(layer.router.w)] = grads.get(id(layer.router.w),
                        np.zeros(layer.d_model, dtype=np.float32))
                    b_arr = np.atleast_1d(layer.router.b)
                    opt_grads[id(b_arr)] = grads.get(id(layer.router.b),
                        np.zeros(1, dtype=np.float32))

                optimizer.step(gradients=opt_grads)

                # Re-quantize shadow weights back to INT4/INT8
                for li, layer in enumerate(model.layers):
                    sw0 = shadow_weights[id(layer) * 10 + 0]
                    sw1 = shadow_weights[id(layer) * 10 + 1]

                    layer.w0_int, s0_flat = _quantize_weights_int4(sw0, layer.block_size)
                    layer.w1_int, s1_flat = _quantize_weights_int8(sw1, layer.block_size)

                    num_blocks = (layer.d_model + layer.block_size - 1) // layer.block_size
                    layer.s0 = s0_flat[:num_blocks * layer.d_out].reshape(layer.d_out, num_blocks)
                    layer.s1 = s1_flat[:num_blocks * layer.d_out].reshape(layer.d_out, num_blocks)

                    # Re-upload updated weights to GPU (persistent W + W^T)
                    if model._gpu_train_handle is not None and _moqe_gpu is not None:
                        _moqe_gpu.moqe_train_update_expert(
                            model._gpu_train_device, model._gpu_train_handle,
                            li, 0, sw0)
                        _moqe_gpu.moqe_train_update_expert(
                            model._gpu_train_device, model._gpu_train_handle,
                            li, 1, sw1)

            if n_batches % 1 == 0:
                avg_loss = total_loss / n_batches
                avg_8bit = total_8bit / n_batches * 100
                extra = ""
                if hasattr(optimizer, 'current_lr'):
                    extra += f" | lr={optimizer.current_lr:.6f}"
                if hasattr(optimizer, 'current_surprise_gain'):
                    extra += f" | S={optimizer.current_surprise_gain:.3f}"
                extra += f" | gnorm={stats.get('grad_norm', 0):.2f}"
                extra += f" | T={model._gumbel_temperature:.2f}"
                extra += f" | H={stats.get('entropy', 0):.1f}"
                extra += f" | cfl={stats.get('conflict_frac', 0)*100:.0f}%"
                print(f"  E{epoch+1} B{n_batches:>4} | L={avg_loss:.4f} "
                      f"| 8b={avg_8bit:.1f}%{extra}")

            if save_dir and n_batches % save_every == 0:
                _save_checkpoint(model, shadow_weights, optimizer,
                                 epoch, n_batches, save_dir)

        n = max(n_batches, 1)
        summary = {
            "epoch": epoch + 1,
            "avg_loss": total_loss / n,
            "avg_ce": total_ce / n,
            "avg_kd": total_kd / n,
            "avg_8bit_frac": total_8bit / n,
            "n_batches": n_batches,
        }
        if hasattr(optimizer, 'current_lr'):
            summary["final_lr"] = optimizer.current_lr
        epoch_stats.append(summary)
        print(f"Epoch {epoch + 1}: loss={summary['avg_loss']:.4f}, "
              f"CE={summary['avg_ce']:.4f}, KD={summary['avg_kd']:.4f}, "
              f"8-bit={summary['avg_8bit_frac'] * 100:.1f}%")

        if save_dir:
            _save_checkpoint(model, shadow_weights, optimizer,
                             epoch + 1, 0, save_dir)

    return epoch_stats


def _save_checkpoint(model, shadow_weights, optimizer, epoch, batch, save_dir):
    """Save model + optimizer state."""
    path = os.path.join(save_dir, f"checkpoint_e{epoch}_b{batch}.npz")
    save_dict = {
        "embedding": model.embedding,
        "out_proj": model.out_proj,
    }
    for i, layer in enumerate(model.layers):
        save_dict[f"layer{i}_w0_int"] = layer.w0_int
        save_dict[f"layer{i}_s0"] = layer.s0
        save_dict[f"layer{i}_w1_int"] = layer.w1_int
        save_dict[f"layer{i}_s1"] = layer.s1
        save_dict[f"layer{i}_router_w"] = layer.router.w
        save_dict[f"layer{i}_router_b"] = np.array([layer.router.b])
        sw_key_0 = id(layer) * 10 + 0
        sw_key_1 = id(layer) * 10 + 1
        if sw_key_0 in shadow_weights:
            save_dict[f"layer{i}_shadow_w0"] = shadow_weights[sw_key_0]
            save_dict[f"layer{i}_shadow_w1"] = shadow_weights[sw_key_1]

    if hasattr(optimizer, 'lr_history'):
        save_dict["lr_history"] = np.array(optimizer.lr_history)
    if hasattr(optimizer, 'surprise_history'):
        save_dict["surprise_history"] = np.array(optimizer.surprise_history)

    np.savez_compressed(path, **save_dict)
    log.info("Checkpoint saved: %s", path)


def load_checkpoint(model, path: str) -> dict:
    """Load model weights from checkpoint."""
    data = np.load(path)

    model.embedding[:] = data["embedding"]
    model.out_proj[:] = data["out_proj"]

    for i, layer in enumerate(model.layers):
        layer.w0_int = data[f"layer{i}_w0_int"]
        layer.s0 = data[f"layer{i}_s0"]
        layer.w1_int = data[f"layer{i}_w1_int"]
        layer.s1 = data[f"layer{i}_s1"]
        layer.router.w = data[f"layer{i}_router_w"]
        layer.router.b = np.float32(data[f"layer{i}_router_b"][0])

    info = {}
    if "lr_history" in data:
        info["lr_history"] = data["lr_history"]
    if "surprise_history" in data:
        info["surprise_history"] = data["surprise_history"]
    return info
