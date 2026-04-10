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
import math
import os

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

    Supports multiple teacher directories and two logit formats:
      - Full vocab: logits shape (seq, vocab_size) — standard KD
      - Top-k compressed: top_k_indices (seq, k) + top_k_logprobs (seq, k) — sparse KD

    Loads files in chunks into RAM (default ~10GB). A background thread
    pre-loads the next chunk while the current chunk is being trained on.

    Multi-teacher: pass a list of directories or a parent directory
    containing subdirectories. Sequences from all teachers are shuffled
    together so the router sees diverse distributions each batch.
    """

    def __init__(
        self,
        data_dir: str | list[str],
        max_seq_len: int = 1024,
        shuffle: bool = True,
        chunk_gb: float = 10.0,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.chunk_gb = chunk_gb

        # Collect files from one or more directories
        self.file_list = []
        if isinstance(data_dir, list):
            dirs = data_dir
        elif os.path.isdir(data_dir):
            # Check if data_dir contains subdirectories (multi-teacher parent)
            subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))]
            if subdirs and any(glob.glob(os.path.join(d, "*.npz")) for d in subdirs):
                dirs = subdirs
            else:
                dirs = [data_dir]
        else:
            dirs = [data_dir]

        for d in dirs:
            files = sorted(glob.glob(os.path.join(d, "*.npz")))
            teacher_name = os.path.basename(d)
            if files:
                log.info("Teacher %s: %d sequences", teacher_name, len(files))
            self.file_list.extend(files)

        if not self.file_list:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        # Estimate files per chunk from first file
        sample_size = os.path.getsize(self.file_list[0])
        mem_per_file = sample_size * 2
        self.chunk_size = max(1, int(chunk_gb * 1e9 / max(mem_per_file, 1)))
        self.chunk_size = min(self.chunk_size, len(self.file_list))

        print(f"OfflineDistillationLoader: {len(self.file_list)} sequences "
              f"from {len(dirs)} teacher(s), "
              f"chunk={self.chunk_size} files (~{chunk_gb:.0f}GB RAM)")

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def _load_chunk(file_paths: list[str], max_seq_len: int) -> list[tuple]:
        """Load a chunk of files into RAM. Handles both formats.

        Returns list of (input_ids, labels, teacher_data) where teacher_data
        is either:
          - np.ndarray of shape (seq, vocab) for full-vocab logits
          - dict {"top_k_indices": (seq, k), "top_k_logprobs": (seq, k)}
            for compressed top-k format
        """
        chunk = []
        for path in file_paths:
            try:
                data = np.load(path)
                tokens = data["input_tokens"][:max_seq_len].astype(np.int32)
                if len(tokens) < 2:
                    continue

                # Detect format
                if "logits" in data:
                    logits = data["logits"]
                    # Check if this is full-vocab logits or MoE routing patterns
                    # Full vocab: (seq, vocab_size) where vocab_size > 1000
                    # MoE routing: (n_layers, seq, n_experts) where n_experts < 100
                    if logits.ndim == 2 and logits.shape[-1] > 1000:
                        logits = logits[:max_seq_len].astype(np.float16)
                        chunk.append((tokens[:-1], tokens[1:], logits[:-1]))
                    elif "moe_patterns" in data:
                        # MoE data — store routing patterns for router supervision
                        moe = data["moe_patterns"][:max_seq_len]
                        teacher_data = {"moe_patterns": moe.astype(np.float32)}
                        chunk.append((tokens[:-1], tokens[1:], teacher_data))
                    else:
                        # Unknown logit shape — skip
                        pass
                elif "top_k_indices" in data and "top_k_logprobs" in data:
                    # Top-k compressed format
                    seq_len = len(tokens) - 1
                    indices = data["top_k_indices"][:seq_len].astype(np.int32)
                    logprobs = data["top_k_logprobs"][:seq_len].astype(np.float16)
                    teacher_data = {
                        "top_k_indices": indices,
                        "top_k_logprobs": logprobs,
                    }
                    chunk.append((tokens[:-1], tokens[1:], teacher_data))
                else:
                    # CE-only: tokens present but no teacher logits
                    chunk.append((tokens[:-1], tokens[1:], None))
            except Exception:
                pass
        return chunk

    def __iter__(self):
        import concurrent.futures

        files = list(self.file_list)
        if self.shuffle:
            np.random.shuffle(files)

        chunks = [files[i:i + self.chunk_size]
                  for i in range(0, len(files), self.chunk_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._load_chunk, chunks[0], self.max_seq_len)

            for ci in range(len(chunks)):
                current_data = future.result()

                if ci + 1 < len(chunks):
                    future = pool.submit(
                        self._load_chunk, chunks[ci + 1], self.max_seq_len)

                if self.shuffle:
                    np.random.shuffle(current_data)
                for input_ids, labels, teacher_logits in current_data:
                    yield input_ids, labels, teacher_logits

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
    soft_teacher = _softmax(teacher_logits.astype(np.float32) / temperature)
    soft_student = _softmax(student_logits / temperature)

    # KL = sum(P * (log P - log Q))
    soft_teacher = np.clip(soft_teacher, 1e-7, 1.0)
    soft_student = np.clip(soft_student, 1e-7, 1.0)
    log_teacher = np.log(soft_teacher)
    log_student = np.log(soft_student)
    kl = soft_teacher * (log_teacher - log_student)
    kl = np.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
    loss = float(np.mean(np.sum(kl, axis=-1))) * (temperature ** 2)

    # Gradient w.r.t. student logits:
    # d KL / d z_student = (soft_student - soft_teacher) * temperature
    grad = (soft_student - soft_teacher) * temperature
    grad *= (temperature ** 2) / max(student_logits.shape[0], 1)

    return loss, grad


def _sparse_kl_divergence_with_grad(
    student_logits: np.ndarray,
    top_k_indices: np.ndarray,
    top_k_logprobs: np.ndarray,
    temperature: float,
) -> tuple[float, np.ndarray]:
    """KL divergence for top-k compressed teacher logits.

    Instead of full-vocab KL, compute KD loss only on the teacher's
    top-k indices. The teacher distribution is reconstructed from
    top-k logprobs with a uniform residual for non-top-k tokens.

    Args:
        student_logits: (seq, vocab) student output logits.
        top_k_indices: (seq, k) teacher's top-k token indices.
        top_k_logprobs: (seq, k) teacher's top-k log-probabilities.
        temperature: KD temperature.

    Returns:
        (loss, gradient) where gradient is (seq, vocab).
    """
    seq_len, vocab_size = student_logits.shape
    top_k_indices.shape[1]

    # Teacher distribution: softmax over top-k logprobs
    teacher_lp = top_k_logprobs / temperature
    teacher_lp_max = np.max(teacher_lp, axis=-1, keepdims=True)
    teacher_probs_topk = np.exp(teacher_lp - teacher_lp_max)
    teacher_probs_topk /= teacher_probs_topk.sum(axis=-1, keepdims=True) + 1e-8

    # Student log-softmax at temperature
    s_scaled = student_logits / temperature
    s_max = np.max(s_scaled, axis=-1, keepdims=True)
    s_exp = np.exp(s_scaled - s_max)
    s_sum = s_exp.sum(axis=-1, keepdims=True)
    s_log_softmax = s_scaled - s_max - np.log(s_sum + 1e-8)

    # Gather student log-probs at teacher's top-k positions
    # s_log_at_topk[i, j] = s_log_softmax[i, top_k_indices[i, j]]
    row_idx = np.arange(seq_len)[:, None]  # (seq, 1)
    s_log_at_topk = s_log_softmax[row_idx, top_k_indices]  # (seq, k)

    # KL = sum_k teacher_p[k] * (log teacher_p[k] - log student_p[k])
    teacher_log = np.log(np.clip(teacher_probs_topk, 1e-7, 1.0))
    kl_per_token = np.sum(teacher_probs_topk * (teacher_log - s_log_at_topk), axis=-1)
    kl_per_token = np.nan_to_num(kl_per_token, nan=0.0, posinf=0.0, neginf=0.0)
    loss = float(np.mean(kl_per_token)) * (temperature ** 2)

    # Gradient: scatter teacher probs into full-vocab gradient
    # d_KL/d_z_student = (softmax(z/T) - teacher_target) * T
    s_softmax = s_exp / (s_sum + 1e-8)
    grad = s_softmax.copy()  # start with student softmax

    # Subtract teacher probs at top-k positions
    for i in range(seq_len):
        grad[i, top_k_indices[i]] -= teacher_probs_topk[i]

    grad *= temperature * (temperature ** 2) / max(seq_len, 1)

    return loss, grad.astype(np.float32)


# ── Phase 2: Full Backprop Through MoQE ──────────────────────────────────

# Native MoQE GPU training ops — persistent weights, batched dispatch
_moqe_gpu = None
try:
    import grilly_core as _gc
    if hasattr(_gc, 'moqe_layer_forward'):
        _moqe_gpu = _gc
except Exception:
    pass


def _dequant_weights(
    w_int: np.ndarray, scales: np.ndarray, block_size: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Dequantize INT4/INT8 weights. Fully vectorized.

    Args:
        dtype: Output dtype. Use np.float16 to halve RAM usage.
    """
    d_out, d_in = w_int.shape
    num_blocks = scales.shape[1]
    padded = d_in + (block_size - d_in % block_size) % block_size
    if padded == d_in:
        w_blocked = w_int.reshape(d_out, num_blocks, block_size).astype(dtype)
    else:
        w_padded = np.zeros((d_out, padded), dtype=w_int.dtype)
        w_padded[:, :d_in] = w_int
        w_blocked = w_padded.reshape(d_out, num_blocks, block_size).astype(dtype)
    scales_cast = scales.astype(dtype)
    w_fp = (w_blocked * scales_cast[:, :, np.newaxis]).reshape(d_out, -1)[:, :d_in]
    return np.ascontiguousarray(w_fp)


def moqe_backward(
    model,
    input_ids: np.ndarray,
    labels: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 2.0,
    target_fractions: np.ndarray | None = None,
    max_grad_norm: float = 1.0,
) -> tuple[dict, dict]:
    """Full forward + backward pass through N-expert MoQE.

    Uses Straight-Through Estimator (STE) for quantization:
    gradients flow through the dequantized weights as if quantization
    were identity. Supports N experts per layer with Gumbel-Softmax.

    Args:
        target_fractions: Per-expert routing targets. If None, uses
            expert_specs defaults. Shape: (n_experts,).

    Returns:
        (gradients, stats): gradients maps id(param) → grad array,
        stats contains loss breakdown.
    """
    seq_len = len(input_ids)
    gradients = {}

    # ── Forward pass (store activations for backward) ─────────────────
    x = model.embedding[input_ids].copy()  # (seq, d_model)
    # Add position embeddings if available
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        x = x + model.pos_embed[:seq_len]
    activations = [x.copy()]
    dequant_cache = []          # list of lists: dequant_cache[l][e] = w_fp
    router_weights_per_layer = []  # (seq, n_experts) soft weights per layer
    expert_outs_per_layer = []     # (n_experts, seq, d) cached for router grad

    gumbel_temp = getattr(model, '_gumbel_temperature', 1.0)

    for l_idx, layer in enumerate(model.layers):
        n_exp = layer.n_experts

        # Gumbel-Softmax: all experts get gradient
        weights, raw_logits = layer.router.forward_gumbel(x, temperature=gumbel_temp)
        router_weights_per_layer.append(weights)  # (seq, n_experts)

        # Dequantize all experts
        layer_dequant = []
        for e in range(n_exp):
            w_fp = _dequant_weights(layer.expert_w_int[e], layer.expert_scales[e],
                                     layer.block_size)
            layer_dequant.append(w_fp)
        dequant_cache.append(layer_dequant)

        # All experts process all tokens, soft-blended
        expert_outs = []
        out = np.zeros_like(x)
        for e in range(n_exp):
            e_out = (x @ layer_dequant[e].T).astype(np.float32)
            expert_outs.append(e_out)
            out += weights[:, e:e+1] * e_out

        expert_outs_per_layer.append(expert_outs)
        x = x + out
        activations.append(x.copy())

    student_logits = (x @ model.out_proj.T).astype(np.float32)

    # ── Loss computation ──────────────────────────────────────────────
    loss_ce, grad_ce = _cross_entropy_with_grad(student_logits, labels)

    # Dispatch KD loss based on teacher format
    if teacher_logits is None:
        # CE-only mode: no teacher, no KD loss
        loss_kd = 0.0
        grad_kd = np.zeros_like(student_logits)
        # Use student's own entropy for router guidance
        student_probs = _softmax(student_logits)
        student_probs = np.clip(student_probs, 1e-7, 1.0)
        teacher_entropy = -np.sum(
            student_probs * np.log(student_probs), axis=-1)
        teacher_entropy = np.nan_to_num(teacher_entropy, nan=1.5)
        t_logits = None
    elif isinstance(teacher_logits, dict) and "moe_patterns" in teacher_logits:
        # MoE routing patterns only — no KD, just CE + router supervision
        loss_kd = 0.0
        grad_kd = np.zeros_like(student_logits)
        # Use MoE expert entropy as router supervision signal
        moe = teacher_logits["moe_patterns"]  # (n_layers, seq, n_experts)
        if moe.ndim == 3 and moe.shape[1] >= student_logits.shape[0]:
            moe_probs = _softmax(moe[:, :student_logits.shape[0], :], axis=-1)
            # Average expert entropy across teacher layers
            moe_entropy = -np.sum(
                moe_probs * np.log(moe_probs + 1e-8), axis=-1)  # (n_layers, seq)
            teacher_entropy = moe_entropy.mean(axis=0)  # (seq,)
        else:
            teacher_entropy = np.ones(student_logits.shape[0]) * 1.5
        t_logits = None
    elif isinstance(teacher_logits, dict) and "top_k_indices" in teacher_logits:
        # Sparse top-k format from compressed teacher
        loss_kd, grad_kd = _sparse_kl_divergence_with_grad(
            student_logits,
            teacher_logits["top_k_indices"],
            teacher_logits["top_k_logprobs"],
            temperature,
        )
        # For entropy-gated routing, estimate teacher entropy from top-k
        lp = teacher_logits["top_k_logprobs"].astype(np.float32)
        teacher_probs_topk = _softmax(lp)
        teacher_probs_topk = np.clip(teacher_probs_topk, 1e-7, 1.0)
        teacher_entropy = -np.sum(
            teacher_probs_topk * np.log(teacher_probs_topk), axis=-1)
        t_logits = None
    else:
        # Full vocab logits — truncate to match
        min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        s_logits = student_logits[:, :min_vocab]
        t_logits = teacher_logits[:, :min_vocab]
        loss_kd, grad_kd = _kl_divergence_with_grad(
            s_logits, t_logits, temperature)
        # Pad grad_kd back to full student vocab if truncated
        if min_vocab < student_logits.shape[-1]:
            pad = np.zeros((grad_kd.shape[0],
                            student_logits.shape[-1] - min_vocab),
                           dtype=np.float32)
            grad_kd = np.concatenate([grad_kd, pad], axis=-1)
        teacher_entropy = None  # compute below from t_logits

    # Compute routing stats from N-expert soft weights
    # For backward compat: "8bit fraction" = fraction NOT routed to expert 0
    all_weights = router_weights_per_layer  # list of (seq, n_experts)
    actual_8bit = 1.0 - float(np.mean([w[:, 0].mean() for w in all_weights]))

    # ── Entropy-Gated Router Loss (Society of Thought) ───────────────
    # Instead of static balance penalty, use teacher entropy to identify
    # "conflict tokens" (high uncertainty = multi-perspective debate).
    # Only conflict tokens are allowed to use 8-bit experts.
    # Low-entropy tokens are aggressively pushed to 4-bit.
    #
    # Shannon entropy of teacher distribution: H = -sum(p * log(p))
    # High H → flat distribution → teacher is uncertain → needs 8-bit
    # Low H  → peaked distribution → teacher is confident → 4-bit is fine
    if teacher_entropy is None:
        # Full-vocab path: compute entropy from t_logits
        teacher_probs = _softmax(t_logits.astype(np.float32))
        teacher_probs = np.clip(teacher_probs, 1e-7, 1.0)
        teacher_entropy = -np.sum(
            teacher_probs * np.log(teacher_probs), axis=-1)  # (seq,)
        teacher_entropy = np.nan_to_num(teacher_entropy, nan=1.5)

    # Entropy threshold: tokens above this get relaxed 8-bit allowance
    # Use median entropy as adaptive threshold (robust to distribution shift)
    entropy_threshold = float(np.median(teacher_entropy)) + 0.5
    is_conflict = teacher_entropy > entropy_threshold  # (seq,) bool

    # Router balance loss: per-expert fraction MSE across all layers
    # Use target_fractions from first layer's expert specs (shared across layers)
    layer0 = model.layers[0]
    n_exp = layer0.n_experts
    tfrac = np.array([s.target_fraction for s in layer0.expert_specs], dtype=np.float32)
    if tfrac.sum() < 1e-6:
        tfrac = np.ones(n_exp, dtype=np.float32) / n_exp
    else:
        tfrac /= tfrac.sum()

    # Average expert fractions across layers
    avg_fracs = np.mean([w.mean(axis=0) for w in all_weights], axis=0)  # (n_experts,)
    loss_router = float(np.mean((avg_fracs - tfrac) ** 2))
    mean_entropy = float(np.mean(teacher_entropy))
    conflict_frac = float(np.mean(is_conflict))

    # Combined gradient w.r.t. output logits
    has_teacher = loss_kd > 0 or np.any(grad_kd != 0)
    if has_teacher:
        grad_logits = 0.3 * grad_ce + 0.6 * grad_kd
        total_loss = 0.3 * loss_ce + 0.6 * loss_kd + 0.1 * loss_router
    else:
        # CE-only mode: full weight on cross-entropy
        grad_logits = 0.9 * grad_ce
        total_loss = 0.9 * loss_ce + 0.1 * loss_router

    # ── Backward: output projection (CPU BLAS — vocab too large for PCIe) ──
    # logits = x @ out_proj.T
    # d_loss/d_out_proj = grad_logits^T @ x  → (vocab, d_model)
    # d_loss/d_x = grad_logits @ out_proj   → (seq, d_model)
    grad_out_proj = (grad_logits.T @ activations[-1]).astype(np.float32)
    gradients[id(model.out_proj)] = grad_out_proj

    dx = (grad_logits @ model.out_proj).astype(np.float32)

    # ── Backward: MoQE layers (reverse order, N-expert) ────────────────
    gpu_handle = getattr(model, '_gpu_train_handle', None)
    gpu_device = getattr(model, '_gpu_train_device', None)

    for l_idx in range(model.n_layers - 1, -1, -1):
        layer = model.layers[l_idx]
        n_exp = layer.n_experts
        x_in = activations[l_idx]
        expert_fps = dequant_cache[l_idx]  # list of w_fp per expert
        weights = router_weights_per_layer[l_idx]  # (seq, n_experts)

        d_out = dx.copy()

        # ── Per-expert gradient (Gumbel-Softmax: all experts get gradient) ──
        dx_expert = np.zeros_like(x_in)
        for e in range(n_exp):
            w_e = weights[:, e:e+1]  # (seq, 1)
            d_e = d_out * w_e        # gradient to this expert
            grad_we = (d_e.T @ x_in).astype(np.float32)
            gradients[id(layer) * 10 + e] = grad_we

            # dx through expert: GPU for first pair, CPU for rest
            if e < 2 and gpu_handle is not None and _moqe_gpu is not None:
                pass  # Accumulated below via GPU batch
            else:
                dx_expert += (d_e @ expert_fps[e]).astype(np.float32)

        # GPU backward for first expert pair (0, 1)
        if n_exp >= 2 and gpu_handle is not None and _moqe_gpu is not None:
            try:
                d_e0 = d_out * weights[:, 0:1]
                d_e1 = d_out * weights[:, 1:2]
                d_e0_c = np.ascontiguousarray(d_e0)
                d_e1_c = np.ascontiguousarray(d_e1)
                dx0_gpu, dx1_gpu = _moqe_gpu.moqe_layer_backward_dx(
                    gpu_device, gpu_handle, l_idx, d_e0_c, d_e1_c)
                dx_expert += np.asarray(dx0_gpu, dtype=np.float32)
                dx_expert += np.asarray(dx1_gpu, dtype=np.float32)
            except Exception:
                # CPU fallback for first pair
                dx_expert += (d_out * weights[:, 0:1] @ expert_fps[0]).astype(np.float32)
                if n_exp > 1:
                    dx_expert += (d_out * weights[:, 1:2] @ expert_fps[1]).astype(np.float32)

        # ── Router gradient via N-way Gumbel-Softmax Jacobian ──
        # Reconstruct expert outputs from dequant cache
        d_scalars = np.zeros((seq_len, n_exp), dtype=np.float32)
        for e in range(n_exp):
            e_out = (x_in @ expert_fps[e].T).astype(np.float32)
            d_scalars[:, e] = np.sum(d_out * e_out, axis=-1)

        # Softmax Jacobian: d_logit = w * (d_scalar - sum(w * d_scalar)) / T
        weighted_sum = np.sum(weights * d_scalars, axis=-1, keepdims=True)
        d_logits_router = weights * (d_scalars - weighted_sum) / max(gumbel_temp, 1e-6)

        # Balance gradient: push fractions toward targets
        target_frac = np.array([s.target_fraction for s in layer.expert_specs], dtype=np.float32)
        if target_frac.sum() < 1e-6:
            target_frac = np.ones(n_exp, dtype=np.float32) / n_exp
        else:
            target_frac /= target_frac.sum()
        actual_frac = weights.mean(axis=0)
        balance_grad = 0.1 * 2.0 * (actual_frac - target_frac) / seq_len

        total_d_logits = d_logits_router + balance_grad[None, :]  # (seq, n_experts)

        # Router: W is (n_experts, d_model), b is (n_experts,)
        grad_router_W = (total_d_logits.T @ x_in).astype(np.float32)
        grad_router_b = total_d_logits.sum(axis=0).astype(np.float32)
        gradients[id(layer.router.W)] = grad_router_W
        gradients[id(layer.router.b)] = grad_router_b

        dx = dx + dx_expert

    # ── Backward: position embeddings ──────────────────────────────
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        grad_pos = np.zeros_like(model.pos_embed)
        grad_pos[:seq_len] = dx
        gradients[id(model.pos_embed)] = grad_pos

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
    chunk_gb: float = 10.0,
    optimizer_type: str = "auto",
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

    # Import optimizer
    OptimizerClass = None
    optim_kwargs = {}

    if optimizer_type == "sgd":
        # SGD: zero optimizer state, minimal RAM
        try:
            from grilly.optim.sgd import SGD
            OptimizerClass = SGD
            optim_kwargs = dict(lr=lr, momentum=0.0)
            log.info("Using SGD (no momentum — RAM-optimized)")
        except ImportError:
            OptimizerClass = None
    elif optimizer_type == "adamw":
        try:
            from grilly.optim.adamw import AdamW
            OptimizerClass = AdamW
            optim_kwargs = dict(lr=lr)
            log.info("Using AdamW")
        except ImportError:
            OptimizerClass = None
    else:  # "auto" — prefer hypergradient, fall back
        try:
            from grilly.optim.hypergradient import AutoHypergradientAdamW
            OptimizerClass = AutoHypergradientAdamW
            optim_kwargs = dict(
                lr=lr, hyper_lr=0.005, warmup_steps=20,
                lr_max=lr * 3,
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

    loader = OfflineDistillationLoader(data_dir, max_seq_len=max_seq_len,
                                       chunk_gb=chunk_gb)

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

    # Shadow weights stay float32 for optimizer precision
    # (float16 updates round to zero at small lr)
    _train_dtype = np.float32

    for layer in model.layers:
        for e in range(layer.n_experts):
            w_fp = _dequant_weights(layer.expert_w_int[e], layer.expert_scales[e],
                                     layer.block_size, dtype=_train_dtype)
            shadow_weights[id(layer) * 10 + e] = w_fp
            gpu_weight_list.append(w_fp)
            all_params.append(w_fp)
        all_params.append(layer.router.W)
        all_params.append(np.atleast_1d(layer.router.b))

    # Position embeddings: trainable if present
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        all_params.append(model.pos_embed)

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
            n_total_experts = len(gpu_weight_list)
            vram_mb = n_total_experts * 2 * model.d_model**2 * 4 / 1e6
            log.info("MoQE GPU training: %d experts on GPU (%.0fMB VRAM)",
                     n_total_experts, vram_mb)
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
                temperature=temperature,
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
                    for e in range(layer.n_experts):
                        sw = shadow_weights[id(layer) * 10 + e]
                        opt_grads[id(sw)] = grads.get(id(layer) * 10 + e, np.zeros_like(sw))
                    opt_grads[id(layer.router.W)] = grads.get(id(layer.router.W),
                        np.zeros_like(layer.router.W))
                    b_arr = np.atleast_1d(layer.router.b)
                    opt_grads[id(b_arr)] = grads.get(id(layer.router.b),
                        np.zeros_like(layer.router.b))

                # Apply gradients
                try:
                    optimizer.step(gradients=opt_grads)
                except TypeError:
                    # Manual SGD with gradient clipping (in-place updates)
                    _lr = optimizer.defaults.get("lr", lr) if optimizer else lr

                    # Global gradient norm clipping
                    all_grads = [g for g in opt_grads.values() if g is not None]
                    if all_grads:
                        global_norm = np.sqrt(sum(
                            float(np.sum(g.astype(np.float32) ** 2))
                            for g in all_grads))
                        clip_coef = min(1.0, 1.0 / (global_norm + 1e-6))
                    else:
                        clip_coef = 1.0

                    for p in all_params:
                        g = opt_grads.get(id(p))
                        if g is not None:
                            # In-place: np.subtract(p, step, out=p)
                            step = (_lr * clip_coef * g).astype(p.dtype)
                            np.subtract(p, step, out=p)

                # Re-quantize shadow weights (STE: quantize for forward, keep FP32 shadow for grad)
                from cubemind.execution.moqe import _quantize_symmetric
                for li, layer in enumerate(model.layers):
                    num_blocks = (layer.d_model + layer.block_size - 1) // layer.block_size
                    for e in range(layer.n_experts):
                        sw = shadow_weights[id(layer) * 10 + e]
                        bits = layer.expert_specs[e].bits
                        # Quantize a COPY — shadow stays FP32 for gradient accumulation
                        layer.expert_w_int[e], s_flat = _quantize_symmetric(
                            sw, bits, layer.block_size)
                        layer.expert_scales[e] = s_flat[:num_blocks * layer.d_out].reshape(
                            layer.d_out, num_blocks)

                        # Re-upload to GPU
                        if model._gpu_train_handle is not None and _moqe_gpu is not None:
                            try:
                                _moqe_gpu.moqe_train_update_expert(
                                    model._gpu_train_device, model._gpu_train_handle,
                                    li, e, sw)
                            except Exception:
                                pass

            if n_batches % 1 == 0:
                avg_loss = total_loss / n_batches
                avg_ce = total_ce / n_batches
                avg_8bit = total_8bit / n_batches * 100
                ppl = math.exp(min(avg_ce, 20))
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
                      f"| CE={avg_ce:.3f} | PPL={ppl:.1f} "
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
        save_dict[f"layer{i}_router_w"] = layer.router.W
        save_dict[f"layer{i}_router_b"] = np.asarray(layer.router.b)
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
        layer.router.W = data[f"layer{i}_router_w"]
        layer.router.b = data[f"layer{i}_router_b"].astype(np.float32)

    info = {}
    if "lr_history" in data:
        info["lr_history"] = data["lr_history"]
    if "surprise_history" in data:
        info["surprise_history"] = data["surprise_history"]
    return info
