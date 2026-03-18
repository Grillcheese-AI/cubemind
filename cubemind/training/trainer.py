"""End-to-end training loop for CubeMind.

Orchestrates: forward pass → loss → gradient estimation → optimizer step.
Records telemetry at every step for live dashboard and paper-quality plots.

Usage:
    model = CubeMind(...)
    optimizer = SurpriseMomentumOptimizer(model_params, hippo=model.hippocampal)
    trainer = Trainer(model, optimizer, loss_fn=mse_loss)

    for epoch in range(10):
        stats = trainer.train_epoch(dataset)
        print(f"Epoch {epoch}: loss={stats['mean_loss']:.4f}")
"""

from __future__ import annotations

import time

import numpy as np

from cubemind.ops.block_codes import BlockCodes
from cubemind.telemetry import metrics


class Trainer:
    """End-to-end training loop for CubeMind.

    Orchestrates forward pass through the CubeMind pipeline, loss
    computation, gradient estimation (finite-difference or DisARM),
    and optimizer step. Records telemetry at every step.

    Args:
        model: CubeMind instance.
        optimizer: SurpriseMomentumOptimizer, HopfieldSurpriseOptimizer, or any
            object with a step(params, grads) method.
        loss_fn: Callable(prediction, target) -> float. Default: MSE.
        grad_method: "finite_diff" or "disarm". Default: "finite_diff".
        grad_eps: Perturbation size for finite-difference gradients.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        loss_fn=None,
        grad_method: str = "finite_diff",
        grad_eps: float = 1e-3,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or self._default_mse
        self.grad_method = grad_method
        self.grad_eps = grad_eps
        self._step = 0
        self._epoch = 0

    @staticmethod
    def _default_mse(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))

    def train_step(
        self,
        observations: list[np.ndarray],
        target: np.ndarray,
        lr: float = 0.01,
    ) -> dict:
        """One training step: forward → loss → gradients → update.

        Args:
            observations: Sequence of block-code observations (each k, l).
            target: Target block-code vector (k, l).
            lr: Learning rate (used if no optimizer attached).

        Returns:
            Dict with: loss, surprise, step, elapsed_ms.
        """
        self._step += 1
        t0 = time.perf_counter()

        # Forward: HMM prediction
        hmm_pred, hmm_weights = self.model.hmm.predict(observations)

        # Loss
        loss = self.loss_fn(hmm_pred, target)
        metrics.record("training.loss", loss)

        # Get surprise from last observation
        phi_flat = observations[-1].ravel().astype(np.float32)
        surprise = self.model.cache.surprise(phi_flat)
        metrics.record("training.surprise", surprise)

        # HMM training step (uses internal finite-diff gradients)
        hmm_loss = self.model.train_step(observations, target, lr=lr)

        # Optimizer step (if attached)
        effective_lr = lr
        if self.optimizer is not None:
            # Compute gradient as direction of loss decrease
            grad = (hmm_pred - target).ravel().astype(np.float32)
            result = self.optimizer.step(grad)
            effective_lr = result.get("effective_lr", lr) if isinstance(result, dict) else lr

        metrics.record("training.effective_lr", effective_lr)

        elapsed = (time.perf_counter() - t0) * 1000
        metrics.record("training.step_ms", elapsed)

        return {
            "loss": loss,
            "hmm_loss": hmm_loss,
            "surprise": surprise,
            "effective_lr": effective_lr,
            "step": self._step,
            "elapsed_ms": elapsed,
        }

    def train_epoch(
        self,
        dataset: list[tuple[list[np.ndarray], np.ndarray]],
        lr: float = 0.01,
        shuffle: bool = True,
    ) -> dict:
        """Train over a full dataset.

        Args:
            dataset: List of (observations, target) tuples.
            lr: Learning rate.
            shuffle: Whether to shuffle dataset order.

        Returns:
            Dict with: mean_loss, min_loss, max_loss, epoch, n_samples, elapsed_s.
        """
        self._epoch += 1
        t0 = time.perf_counter()

        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)

        losses = []
        for idx in indices:
            obs, target = dataset[int(idx)]
            result = self.train_step(obs, target, lr=lr)
            losses.append(result["loss"])

        elapsed = time.perf_counter() - t0

        stats = {
            "mean_loss": float(np.mean(losses)),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "epoch": self._epoch,
            "n_samples": len(dataset),
            "elapsed_s": elapsed,
        }
        metrics.record("training.epoch_loss", stats["mean_loss"])
        return stats

    def evaluate(
        self,
        dataset: list[tuple[list[np.ndarray], np.ndarray]],
    ) -> dict:
        """Evaluate without gradient updates.

        Args:
            dataset: List of (observations, target) tuples.

        Returns:
            Dict with: mean_loss, accuracy (fraction of correct top-1 predictions).
        """
        losses = []
        correct = 0
        bc = BlockCodes(self.model.k, self.model.l)

        for obs, target in dataset:
            pred, _ = self.model.hmm.predict(obs)
            loss = self.loss_fn(pred, target)
            losses.append(loss)

            # Accuracy: discretize prediction, check if it matches target
            pred_disc = bc.discretize(pred)
            if bc.similarity(pred_disc, target) > 0.99:
                correct += 1

        return {
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": correct / len(dataset) if dataset else 0.0,
            "n_samples": len(dataset),
        }

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def epoch_count(self) -> int:
        return self._epoch
