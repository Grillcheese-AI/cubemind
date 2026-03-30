"""MoQE Offline Distillation Pipeline.

Two-phase distillation from dense teacher LLM(s) to a MoQE student:

Phase 1 (Colab/cloud): Extract teacher logits via llama-cpp-python
  Teacher GGUF → logits_all=True → save .npz to disk ("Logit Banking")
  Supports multi-teacher ensemble: run extraction for each teacher into
  separate directories, then merge their logits for distillation.

Phase 2 (Local/Grilly): Train MoQE student from saved logits
  Streaming DataLoader → CE + KL-div + router balancing → backprop

Loss = 0.3 * CE(hard labels) + 0.6 * KL(soft teacher) + 0.1 * router_balance
Router target: 85% tokens → 4-bit expert, 15% → 8-bit expert.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np


# ── Phase 1: Teacher Logit Extraction ─────────────────────────────────────

def extract_teacher_logits(
    model_path: str,
    texts: list[str],
    save_dir: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = 0,
) -> int:
    """Extract teacher logits from a GGUF model and save to .npz files.

    Run this on a machine with enough RAM/VRAM for the teacher model.
    The student training (Phase 2) does NOT need the teacher loaded.

    Args:
        model_path:    Path to GGUF teacher model.
        texts:         List of training text sequences.
        save_dir:      Directory to save .npz logit files.
        n_ctx:         Context window size.
        n_gpu_layers:  GPU layers to offload (0=CPU only).

    Returns:
        Number of sequences processed.
    """
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

        # Extract logits: (seq_len, vocab_size)
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
    """Merge logits from multiple teachers into ensemble soft distributions.

    Each teacher's logits are softmax'd at the given temperature, then
    the resulting probabilities are weighted-averaged across teachers.
    The ensemble distribution is saved as log-probabilities.

    This compiles the strengths of multiple models:
      - Teacher A might be strong at reasoning
      - Teacher B might be strong at factual recall
      - Teacher C might be strong at code generation
    The student learns from all three simultaneously.

    Args:
        teacher_dirs: List of directories, one per teacher, each containing
                      sequence_NNNNNN.npz files with matching filenames.
        output_dir:   Directory to write merged ensemble .npz files.
        weights:      Per-teacher weights (default: uniform). Must sum to 1.
        temperature:  Softmax temperature for averaging (default: 1.0).

    Returns:
        Number of sequences merged.
    """
    n_teachers = len(teacher_dirs)
    if n_teachers == 0:
        raise ValueError("Need at least one teacher directory")

    if weights is None:
        weights = [1.0 / n_teachers] * n_teachers
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    os.makedirs(output_dir, exist_ok=True)

    # Find common sequence files across all teachers
    file_sets = []
    for d in teacher_dirs:
        files = {os.path.basename(f) for f in glob.glob(os.path.join(d, "sequence_*.npz"))}
        file_sets.append(files)

    common_files = sorted(file_sets[0].intersection(*file_sets[1:]))
    print(f"Ensemble merge: {n_teachers} teachers, {len(common_files)} common sequences")
    print(f"  Weights: {[f'{w:.2f}' for w in weights]}")

    count = 0
    for fname in common_files:
        try:
            # Load tokens from first teacher (should be identical across teachers)
            data_0 = np.load(os.path.join(teacher_dirs[0], fname))
            tokens = data_0["input_tokens"]
            seq_len = len(tokens)

            # Accumulate weighted soft distributions
            ensemble_probs = None

            for t_idx, t_dir in enumerate(teacher_dirs):
                data = np.load(os.path.join(t_dir, fname))
                logits = data["logits"][:seq_len].astype(np.float32)

                # Softmax at temperature
                scaled = logits / max(temperature, 1e-8)
                max_vals = np.max(scaled, axis=-1, keepdims=True)
                exp_vals = np.exp(scaled - max_vals)
                probs = exp_vals / (exp_vals.sum(axis=-1, keepdims=True) + 1e-8)

                if ensemble_probs is None:
                    ensemble_probs = weights[t_idx] * probs
                else:
                    # Vocab sizes might differ — take the minimum
                    min_vocab = min(ensemble_probs.shape[-1], probs.shape[-1])
                    ensemble_probs = ensemble_probs[:, :min_vocab]
                    ensemble_probs += weights[t_idx] * probs[:, :min_vocab]

            # Convert back to logits (log of averaged probs)
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

            if count % 100 == 0:
                print(f"  Merged {count}/{len(common_files)} sequences")

        except Exception as e:
            print(f"  Skipping {fname}: {e}")

    print(f"Ensemble merge complete: {count} sequences saved to {output_dir}")
    return count


# ── Phase 2: Streaming DataLoader ─────────────────────────────────────────

class OfflineDistillationLoader:
    """Streams pre-calculated teacher logits from .npz files.

    Memory-efficient: loads one file at a time, converts to float32,
    yields (input_ids, labels, teacher_logits) batches.

    Args:
        data_dir:    Directory with .npz files from Phase 1.
        max_seq_len: Truncate sequences to this length.
        shuffle:     Shuffle file order each epoch.
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 1024,
        shuffle: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.file_list:
            raise FileNotFoundError(f"No .npz files in {data_dir}")
        print(f"OfflineDistillationLoader: {len(self.file_list)} sequences")

    def __len__(self) -> int:
        return len(self.file_list)

    def __iter__(self):
        files = list(self.file_list)
        if self.shuffle:
            np.random.shuffle(files)

        for path in files:
            try:
                data = np.load(path)
                tokens = data["input_tokens"][:self.max_seq_len].astype(np.int32)
                logits = data["logits"][:self.max_seq_len].astype(np.float32)

                if len(tokens) < 2:
                    continue

                # Shift: input = tokens[:-1], labels = tokens[1:]
                input_ids = tokens[:-1]
                labels = tokens[1:]
                teacher_logits = logits[:-1]

                yield input_ids, labels, teacher_logits

            except Exception as e:
                print(f"Skipping corrupted file {path}: {e}")


# ── Phase 2: Distillation Training Loop ───────────────────────────────────

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - m)
    return ex / (ex.sum(axis=axis, keepdims=True) + 1e-8)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    logsumexp = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True) + 1e-8)
    return x - logsumexp


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy loss: logits (seq, vocab), labels (seq,)."""
    log_probs = _log_softmax(logits, axis=-1)
    seq_len = len(labels)
    loss = 0.0
    for i in range(seq_len):
        label = int(labels[i])
        if 0 <= label < logits.shape[1]:
            loss -= float(log_probs[i, label])
    return loss / max(seq_len, 1)


def _kl_divergence(student_logits: np.ndarray, teacher_logits: np.ndarray, temperature: float) -> float:
    """KL divergence between teacher and student soft distributions."""
    soft_teacher = _softmax(teacher_logits / temperature, axis=-1)
    log_soft_student = _log_softmax(student_logits / temperature, axis=-1)

    # KL(P || Q) = sum(P * (log P - log Q))
    kl = soft_teacher * (np.log(soft_teacher + 1e-8) - log_soft_student)
    return float(np.mean(np.sum(kl, axis=-1))) * (temperature ** 2)


def distill_step(
    model,
    input_ids: np.ndarray,
    labels: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 2.0,
    target_8bit: float = 0.15,
    lr: float = 3e-4,
) -> dict:
    """Single distillation step (forward + loss computation).

    Note: Full backprop requires grilly autograd. This computes the
    forward pass and losses for monitoring. Weight updates need grilly.

    Args:
        model:          MoQEModel instance.
        input_ids:      (seq_len,) int32.
        labels:         (seq_len,) int32.
        teacher_logits: (seq_len, vocab_size) float32.
        temperature:    KL divergence temperature.
        target_8bit:    Target fraction of tokens routed to 8-bit expert.
        lr:             Learning rate (for router SGD update).

    Returns:
        Dict with loss breakdown and routing stats.
    """
    # Forward pass
    student_logits, router_probs = model.forward(input_ids)

    # Match sequence lengths (student may be shorter due to input_ids)
    min_len = min(student_logits.shape[0], teacher_logits.shape[0], len(labels))
    s_logits = student_logits[:min_len]
    t_logits = teacher_logits[:min_len]
    lbls = labels[:min_len]

    # A. Cross-entropy (hard labels)
    loss_ce = _cross_entropy(s_logits, lbls)

    # B. Knowledge distillation (soft labels)
    loss_kd = _kl_divergence(s_logits, t_logits, temperature)

    # C. Router load balancing
    actual_8bit = float(np.mean(router_probs > 0.5))
    loss_router = (actual_8bit - target_8bit) ** 2

    # Combined loss
    total_loss = 0.3 * loss_ce + 0.6 * loss_kd + 0.1 * loss_router

    # Simple SGD router update (push toward target balance)
    for layer in model.layers:
        router = layer.router
        grad_direction = 1.0 if actual_8bit > target_8bit else -1.0
        router.b -= np.float32(lr * grad_direction * 0.1)

    return {
        "total_loss": total_loss,
        "loss_ce": loss_ce,
        "loss_kd": loss_kd,
        "loss_router": loss_router,
        "actual_8bit_frac": actual_8bit,
    }


def run_offline_distillation(
    model,
    data_dir: str,
    epochs: int = 3,
    max_seq_len: int = 512,
    temperature: float = 2.0,
    target_8bit: float = 0.15,
    lr: float = 3e-4,
) -> list[dict]:
    """Run offline MoQE distillation from saved teacher logits.

    Args:
        model:       MoQEModel instance.
        data_dir:    Directory with .npz logit files.
        epochs:      Number of training epochs.
        max_seq_len: Max sequence length per sample.
        temperature: KL divergence temperature.
        target_8bit: Target 8-bit routing fraction.
        lr:          Learning rate.

    Returns:
        List of per-epoch summary dicts.
    """
    loader = OfflineDistillationLoader(data_dir, max_seq_len=max_seq_len)
    epoch_stats = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_8bit = 0.0
        n_batches = 0

        for input_ids, labels, teacher_logits in loader:
            stats = distill_step(
                model, input_ids, labels, teacher_logits,
                temperature=temperature, target_8bit=target_8bit, lr=lr,
            )
            total_loss += stats["total_loss"]
            total_ce += stats["loss_ce"]
            total_kd += stats["loss_kd"]
            total_8bit += stats["actual_8bit_frac"]
            n_batches += 1

            if n_batches % 50 == 0:
                avg_loss = total_loss / n_batches
                avg_8bit = total_8bit / n_batches * 100
                print(f"  Epoch {epoch + 1} | Batch {n_batches} | "
                      f"Loss: {avg_loss:.4f} | 8-bit: {avg_8bit:.1f}%")

        n = max(n_batches, 1)
        summary = {
            "epoch": epoch + 1,
            "avg_loss": total_loss / n,
            "avg_ce": total_ce / n,
            "avg_kd": total_kd / n,
            "avg_8bit_frac": total_8bit / n,
            "n_batches": n_batches,
        }
        epoch_stats.append(summary)
        print(f"Epoch {epoch + 1}: loss={summary['avg_loss']:.4f}, "
              f"CE={summary['avg_ce']:.4f}, KD={summary['avg_kd']:.4f}, "
              f"8-bit={summary['avg_8bit_frac'] * 100:.1f}%")

    return epoch_stats
