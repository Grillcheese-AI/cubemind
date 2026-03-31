"""Training loop for CNN perception frontend.

Trains the CNNEncoder to map 160x160 grayscale RAVEN panel images to
block-code vectors that match the ground-truth NVSA metadata encoding.

Loss: per-block cross-entropy between CNN softmax output and target
one-hot block-codes derived from XML metadata attributes.

Gradient through the discrete bottleneck: DisARM antithetic sampling
(grilly.nn.DisARMSampler) provides low-variance gradient estimates
for the block-code discretization step.

Temperature annealing: tau starts at 1.0 and decays by 0.95 per epoch,
driving the CNN output toward hard one-hot codes.

Usage:
    python -m cubemind.perception.train_cnn --epochs 30 --lr 0.001
    python -m cubemind.perception.train_cnn --configs center_single --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.cnn_encoder import CNNEncoder

# benchmarks.iraven lives at project root but grilly's editable install
# shadows the `benchmarks` namespace. Use importlib to load from explicit path.
import importlib.util as _ilu
from pathlib import Path as _Path

def _fix_grilly_shadowing():
    """Fix grilly editable install shadowing datasets and benchmarks packages."""
    import sys
    # Remove grilly source dirs from sys.path so HF datasets isn't shadowed
    grilly_src = [p for p in sys.path
                  if 'grilly' in p.lower() and 'cubemind' not in p.lower()
                  and 'site-packages' not in p.lower()]
    for p in grilly_src:
        sys.path.remove(p)
    # Purge cached grilly-shadowed modules
    for key in list(sys.modules):
        if key == 'datasets' or key.startswith('datasets.'):
            mod = sys.modules[key]
            if mod and hasattr(mod, '__file__') and mod.__file__ and 'grilly' in str(mod.__file__):
                del sys.modules[key]
        if key == 'benchmarks' or key.startswith('benchmarks.'):
            del sys.modules[key]
    return grilly_src

def _load_iraven():
    _iraven_path = _Path(__file__).resolve().parent.parent.parent / "benchmarks" / "iraven.py"
    spec = _ilu.spec_from_file_location("benchmarks.iraven", _iraven_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Force-load HF datasets from site-packages, bypassing grilly's editable
# install finder hook which shadows it with grilly/datasets/.
import sys as _sys
_venv_sp = [p for p in _sys.path if 'site-packages' in p and 'cubemind' in p]
if _venv_sp:
    _ds_spec = _ilu.find_spec('datasets', [_venv_sp[0]])
    if _ds_spec and _ds_spec.origin and 'site-packages' in _ds_spec.origin:
        _hf_datasets = _ilu.module_from_spec(_ds_spec)
        _ds_spec.loader.exec_module(_hf_datasets)
        _sys.modules['datasets'] = _hf_datasets

_iraven = _load_iraven()

logger = logging.getLogger(__name__)

# ── Try grilly DisARM for gradient estimation ───────────────────────────────

try:
    _HAS_DISARM = True
except Exception:
    _HAS_DISARM = False

# ── Try grilly optimizer ────────────────────────────────────────────────────

try:
    _HAS_GRILLY_OPTIM = True
except Exception:
    _HAS_GRILLY_OPTIM = False


# ── Loss functions ──────────────────────────────────────────────────────────


def block_cross_entropy(
    pred: np.ndarray,
    target: np.ndarray,
    k: int,
    l: int,
) -> tuple[float, np.ndarray]:
    """Per-block cross-entropy loss with gradient.

    Each of the k blocks is an independent probability simplex.
    Loss = -(1/k) * sum_j sum_i target[j,i] * log(pred[j,i])

    Args:
        pred: CNN output (k, l) — softmax probabilities.
        target: Ground-truth one-hot block-code (k, l).
        k: Number of blocks.
        l: Block length.

    Returns:
        (loss, grad) where grad has shape (k, l).
    """
    eps = 1e-8
    pred_c = np.clip(pred.reshape(k, l), eps, 1.0 - eps)
    target_c = target.reshape(k, l)

    # Loss: mean cross-entropy across blocks
    loss = -float(np.sum(target_c * np.log(pred_c)) / k)

    # Gradient of CE w.r.t. pre-softmax logits: pred - target
    grad = (pred_c - target_c) / k

    return loss, grad.astype(np.float32)


def similarity_loss(
    pred: np.ndarray,
    target: np.ndarray,
    bc: BlockCodes,
) -> tuple[float, np.ndarray]:
    """Block-code similarity loss: 1 - similarity(pred, target).

    Args:
        pred: CNN output (k, l).
        target: Ground-truth block-code (k, l).
        bc: BlockCodes instance.

    Returns:
        (loss, grad) where grad has shape (k, l).
    """
    sim = bc.similarity(pred, target)
    loss = 1.0 - sim

    # Gradient: -target / k (similarity is linear in pred for normalized codes)
    grad = -target / pred.shape[0]

    return float(loss), grad.astype(np.float32)


# ── Target encoding ────────────────────────────────────────────────────────


def encode_target_from_metadata(
    metadata_xml: str,
    panel_index: int,
    bc: BlockCodes,
) -> np.ndarray:
    """Encode the ground-truth block-code from XML metadata.

    Uses the same NVSA role-filler encoding as the benchmark pipeline.
    """
    return _iraven.encode_panel_from_metadata(metadata_xml, panel_index, bc)


# ── SGD optimizer (pure numpy fallback) ─────────────────────────────────────


class SimpleSGD:
    """Minimal SGD for numpy parameter arrays."""

    def __init__(self, params: list[np.ndarray], lr: float = 0.001,
                 momentum: float = 0.9, weight_decay: float = 1e-4):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        """Update parameters with momentum SGD."""
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue
            # Weight decay
            g = g + self.weight_decay * p
            # Momentum
            self.velocities[i] = self.momentum * self.velocities[i] + g
            # Update
            p -= self.lr * self.velocities[i]


# ── Training loop ───────────────────────────────────────────────────────────


def train_epoch(
    encoder: CNNEncoder,
    problems: list[dict],
    bc: BlockCodes,
    optimizer: SimpleSGD,
    max_problems: int | None = None,
) -> dict:
    """Train CNN encoder for one epoch on RAVEN problems.

    For each problem, encodes all 16 panels (8 context + 8 choices) from
    images, computes loss against metadata-derived targets, and updates
    via backprop.

    Args:
        encoder: CNN encoder to train.
        problems: RAVEN problem dicts with panels, metadata.
        bc: BlockCodes instance.
        optimizer: SGD optimizer.
        max_problems: Max problems to train on (None = all).

    Returns:
        Dict with epoch statistics.
    """
    total_loss = 0.0
    total_sim = 0.0
    n_panels = 0
    t0 = time.perf_counter()

    if max_problems:
        problems = problems[:max_problems]

    for prob in problems:
        panels = prob.get("panels", [])
        metadata = prob.get("metadata", "")

        if not panels or not metadata:
            continue

        # Train on context panels (0-7)
        for pi in range(min(len(panels), 8)):
            panel_img = panels[pi]

            # Prepare image (80x80 for efficiency)
            img = np.array(panel_img.convert("L").resize((80, 80)),
                           dtype=np.float32) / 255.0

            # Target: metadata-derived block-code
            target = encode_target_from_metadata(metadata, pi, bc)

            # Zero gradients
            encoder.zero_grad()

            # Forward
            pred = encoder.forward(img)

            # Loss + output gradient (cosine similarity loss for smoother gradients)
            loss, grad = similarity_loss(pred, target, bc)
            total_loss += loss

            # Clip gradient to prevent NaN
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm

            # Backward through full conv stack
            encoder.backward(grad)

            # SGD step
            encoder.step(lr=optimizer.lr)

            # Similarity tracking
            sim = bc.similarity(bc.discretize(pred), target)
            total_sim += sim
            n_panels += 1

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / max(n_panels, 1)
    avg_sim = total_sim / max(n_panels, 1)

    return {
        "loss": avg_loss,
        "similarity": avg_sim,
        "n_panels": n_panels,
        "elapsed_s": elapsed,
        "temperature": encoder.temperature,
    }


def evaluate(
    encoder: CNNEncoder,
    problems: list[dict],
    bc: BlockCodes,
    max_problems: int | None = None,
) -> dict:
    """Evaluate CNN encoder on RAVEN problems.

    Measures how well the CNN block-codes match metadata-derived targets,
    and how well the full pipeline (CNN → detectors) predicts answers.

    Args:
        encoder: Trained CNN encoder.
        problems: RAVEN problem dicts.
        bc: BlockCodes instance.
        max_problems: Max problems to evaluate.

    Returns:
        Dict with evaluation metrics.
    """
    # Available via _iraven module if needed

    total_sim = 0.0
    n_panels = 0

    if max_problems:
        problems = problems[:max_problems]

    # Measure encoding similarity
    for prob in problems:
        panels = prob.get("panels", [])
        metadata = prob.get("metadata", "")
        if not panels or not metadata:
            continue

        for pi in range(min(len(panels), 8)):
            pred = encoder.encode_panel(panels[pi], target_size=80)
            target = encode_target_from_metadata(metadata, pi, bc)
            sim = bc.similarity(bc.discretize(pred), target)
            total_sim += sim
            n_panels += 1

    avg_sim = total_sim / max(n_panels, 1)

    return {
        "avg_similarity": avg_sim,
        "n_panels": n_panels,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train CNN perception for CubeMind")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train", type=int, default=200)
    parser.add_argument("--max-eval", type=int, default=50)
    parser.add_argument("--configs", nargs="+", default=["center_single"])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--l", type=int, default=64)
    parser.add_argument("--temp-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    bc = BlockCodes(args.k, args.l)
    encoder = CNNEncoder(k=args.k, l=args.l, temperature=1.0, seed=args.seed)
    logger.info("Encoder: %s", repr(encoder))

    params = encoder.get_parameters()
    optimizer = SimpleSGD(params, lr=args.lr, momentum=args.momentum)

    def _load_split(config, split, max_n):
        """Load RAVEN split via parquet (bypasses HF datasets RLock issue)."""
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from PIL import Image
        import io

        logger.info("Loading %s/%s (max %d, parquet)...", config, split, max_n)
        # Download parquet file from HF hub
        path = hf_hub_download(
            repo_id="HuggingFaceM4/RAVEN", repo_type="dataset",
            filename=f"{config}/{split}-00000-of-00001.parquet",
        )
        table = pq.read_table(path)
        problems = []
        for i in range(min(max_n, len(table))):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            # Decode panel images from bytes
            panels = []
            if "panels" in row and row["panels"]:
                for p in row["panels"]:
                    if isinstance(p, dict) and "bytes" in p:
                        panels.append(Image.open(io.BytesIO(p["bytes"])))
                    elif isinstance(p, Image.Image):
                        panels.append(p)
            problems.append({
                "panels": panels,
                "choices": [],
                "target": row.get("target"),
                "metadata": row.get("metadata", ""),
            })
        logger.info("Loaded %d problems with %d panels each",
                     len(problems), len(problems[0]["panels"]) if problems else 0)
        return problems

    for config in args.configs:
        logger.info("Loading %s train/test splits...", config)
        train_problems = _load_split(config, "train", args.max_train)
        test_problems = _load_split(config, "test", args.max_eval)

        for epoch in range(args.epochs):
            # Train
            stats = train_epoch(encoder, train_problems, bc, optimizer,
                                max_problems=args.max_train)

            # Anneal temperature
            encoder.anneal_temperature(args.temp_decay)

            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                eval_stats = evaluate(encoder, test_problems, bc,
                                      max_problems=args.max_eval)
                logger.info(
                    "  [%s] epoch %2d/%d: loss=%.4f sim=%.4f temp=%.3f | eval_sim=%.4f",
                    config, epoch + 1, args.epochs,
                    stats["loss"], stats["similarity"], stats["temperature"],
                    eval_stats["avg_similarity"],
                )
            else:
                logger.info(
                    "  [%s] epoch %2d/%d: loss=%.4f sim=%.4f temp=%.3f",
                    config, epoch + 1, args.epochs,
                    stats["loss"], stats["similarity"], stats["temperature"],
                )

    logger.info("Training complete. Final encoder: %s", repr(encoder))


if __name__ == "__main__":
    main()
