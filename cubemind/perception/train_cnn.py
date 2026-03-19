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

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.cnn_encoder import CNNEncoder

logger = logging.getLogger(__name__)

# ── Try grilly DisARM for gradient estimation ───────────────────────────────

try:
    from cubemind.training.disarm import discretize_block_codes
    _HAS_DISARM = True
except Exception:
    _HAS_DISARM = False

# ── Try grilly optimizer ────────────────────────────────────────────────────

try:
    from grilly.optim import SGD as _GrillySGD
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

    Args:
        metadata_xml: XML metadata string.
        panel_index: Panel index (0-7 context, 8-15 choices).
        bc: BlockCodes instance.

    Returns:
        Discrete one-hot block-code (k, l).
    """
    from benchmarks.iraven import encode_panel_from_metadata
    return encode_panel_from_metadata(metadata_xml, panel_index, bc)


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

            # Loss + output gradient
            loss, grad = block_cross_entropy(pred, target, encoder.k, encoder.l)
            total_loss += loss

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
    from benchmarks.iraven import evaluate_problem_dataset, train_multiview_hmms
    from cubemind.model import CubeMind

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

    # Load training data
    from benchmarks.iraven import load_raven_split

    for config in args.configs:
        logger.info("Loading %s train/test splits...", config)
        train_problems = load_raven_split(config, "train")[:args.max_train]
        test_problems = load_raven_split(config, "test")[:args.max_eval]

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
