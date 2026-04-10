"""Neurogenesis Controller — Dynamic network growth and pruning.

Manages the lifecycle of neurons in spiking networks:
  - Growth: spawn new neurons when residual error exceeds threshold
  - Pruning: remove neurons that never fire (zero utility)
  - Maturation: new neurons start as progenitors, mature over time
  - Oja normalization: self-normalizing weights via Oja's rule

Ported from:
  - superfast-neuro algo/oja_sanger_whitener.py (OjaLayer with auto-growth)
  - aura-hybrid neuron lifecycle states (PROGENITOR → MYELINATED)
  - AURA_GENESIS hippocampus.py (stimulate_neurogenesis)

The controller wraps any layer that has .weight and .bias arrays,
monitoring activity and growing/pruning as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import numpy as np


class MaturationStage(Enum):
    """Neuron lifecycle stages."""
    PROGENITOR = "progenitor"      # Just born, random weights
    MIGRATING = "migrating"        # Finding its place in the network
    DIFFERENTIATED = "differentiated"  # Specialized, learning
    MYELINATED = "myelinated"      # Mature, fast conduction


@dataclass
class NeuronState:
    """State tracking for a single neuron."""
    stage: MaturationStage = MaturationStage.PROGENITOR
    age: int = 0
    total_spikes: int = 0
    recent_spikes: float = 0.0  # EMA of recent activity
    merit: float = 1.0


class NeurogenesisController:
    """Dynamic network growth and pruning controller.

    Monitors neuron activity, grows new neurons when the network can't
    represent something (high residual), prunes neurons that never fire.

    Args:
        initial_neurons: Starting number of neurons.
        max_neurons: Maximum neurons allowed.
        feature_dim: Input feature dimension.
        growth_threshold: Residual EMA above this triggers growth.
        prune_threshold: Neurons with recent_spikes below this get pruned.
        growth_rate: Number of neurons to add per growth event.
        prune_cooldown: Steps between prune checks.
        growth_cooldown: Steps between growth events.
        maturation_steps: Steps for PROGENITOR → DIFFERENTIATED.
        myelination_steps: Steps for DIFFERENTIATED → MYELINATED.
        oja_lr: Oja normalization learning rate.
        activity_ema: EMA decay for recent spike tracking.
        seed: Random seed.
    """

    def __init__(
        self,
        initial_neurons: int = 64,
        max_neurons: int = 10000,
        feature_dim: int = 32,
        growth_threshold: float = 0.35,
        prune_threshold: float = 0.001,
        growth_rate: int = 8,
        prune_cooldown: int = 200,
        growth_cooldown: int = 100,
        maturation_steps: int = 50,
        myelination_steps: int = 200,
        oja_lr: float = 0.01,
        activity_ema: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.max_neurons = max_neurons
        self.feature_dim = feature_dim
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold
        self.growth_rate = growth_rate
        self.prune_cooldown = prune_cooldown
        self.growth_cooldown = growth_cooldown
        self.maturation_steps = maturation_steps
        self.myelination_steps = myelination_steps
        self.oja_lr = oja_lr
        self.activity_ema = activity_ema

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_growth_step = -growth_cooldown
        self._last_prune_step = -prune_cooldown

        # Residual tracking
        self.residual_ema = 0.0

        # Neuron states
        self.neuron_count = initial_neurons
        self.states: List[NeuronState] = [
            NeuronState(stage=MaturationStage.DIFFERENTIATED)
            for _ in range(initial_neurons)
        ]

        # Weight matrix (out_dim × in_dim) — the neurons' receptive fields
        std = np.sqrt(2.0 / (feature_dim + initial_neurons))
        self.weights = self._rng.normal(
            0, std, (initial_neurons, feature_dim)
        ).astype(np.float32)
        self._renorm_rows()

    def _renorm_rows(self) -> None:
        """Row-wise L2 normalization for stability."""
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights /= np.maximum(norms, 1e-8)

    def step(
        self,
        input_vec: np.ndarray,
        spike_counts: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """One step of the neurogenesis controller.

        Args:
            input_vec: Current input (feature_dim,).
            spike_counts: Per-neuron spike counts this step (neuron_count,).
                If None, computes activity from projection.

        Returns:
            Dict with: grew (bool), pruned (int), residual, neuron_count, etc.
        """
        self._step_count += 1
        x = np.asarray(input_vec, dtype=np.float32).ravel()

        # Project input through current weights
        y = self.weights @ x  # (neuron_count,)

        # Reconstruction
        x_hat = self.weights.T @ y  # (feature_dim,)
        x_norm = float(np.dot(x, x) + 1e-12)
        explained = float(np.dot(x_hat, x_hat) / x_norm)
        residual = max(0.0, 1.0 - explained)

        # Update residual EMA
        self.residual_ema = (1.0 - self.activity_ema) * self.residual_ema + \
                            self.activity_ema * residual

        # Update neuron activity tracking
        if spike_counts is not None:
            for i in range(min(len(spike_counts), self.neuron_count)):
                s = self.states[i]
                s.total_spikes += int(spike_counts[i])
                s.recent_spikes = ((1.0 - self.activity_ema) * s.recent_spikes +
                                   self.activity_ema * float(spike_counts[i]))
                s.age += 1
                self._advance_maturation(s)
        else:
            # Use projection magnitude as proxy for activity
            for i in range(self.neuron_count):
                activity = abs(float(y[i]))
                s = self.states[i]
                s.recent_spikes = ((1.0 - self.activity_ema) * s.recent_spikes +
                                   self.activity_ema * activity)
                s.age += 1
                self._advance_maturation(s)

        # Oja update: w += lr * y * (x - W.T @ y)
        residual_vec = x - x_hat
        dW = self.oja_lr * np.outer(y, residual_vec)
        self.weights[:self.neuron_count] += dW[:self.neuron_count]
        self._renorm_rows()

        # Growth check
        grew = False
        if (self.residual_ema > self.growth_threshold and
                self.neuron_count < self.max_neurons and
                self._step_count - self._last_growth_step >= self.growth_cooldown):
            grew = self._grow(residual_vec)

        # Prune check
        pruned = 0
        if self._step_count - self._last_prune_step >= self.prune_cooldown:
            pruned = self._prune()

        return {
            "grew": grew,
            "pruned": pruned,
            "residual": residual,
            "residual_ema": self.residual_ema,
            "neuron_count": self.neuron_count,
            "explained": explained,
            "step": self._step_count,
        }

    def _advance_maturation(self, state: NeuronState) -> None:
        """Advance neuron through lifecycle stages based on age."""
        if state.stage == MaturationStage.PROGENITOR and state.age >= self.maturation_steps // 2:
            state.stage = MaturationStage.MIGRATING
        elif state.stage == MaturationStage.MIGRATING and state.age >= self.maturation_steps:
            state.stage = MaturationStage.DIFFERENTIATED
        elif state.stage == MaturationStage.DIFFERENTIATED and state.age >= self.myelination_steps:
            state.stage = MaturationStage.MYELINATED

    def _grow(self, residual_vec: np.ndarray) -> bool:
        """Grow new neurons along the residual direction."""
        n_new = min(self.growth_rate, self.max_neurons - self.neuron_count)
        if n_new <= 0:
            return False

        # New neurons initialized along residual + small noise
        residual_norm = np.linalg.norm(residual_vec) + 1e-8
        base_dir = residual_vec / residual_norm

        new_weights = np.zeros((n_new, self.feature_dim), dtype=np.float32)
        for i in range(n_new):
            noise = self._rng.standard_normal(self.feature_dim).astype(np.float32) * 0.1
            new_weights[i] = base_dir + noise
            norm = np.linalg.norm(new_weights[i])
            new_weights[i] /= max(norm, 1e-8)

        # Expand weight matrix
        self.weights = np.concatenate([self.weights, new_weights], axis=0)

        # Create new neuron states
        for _ in range(n_new):
            self.states.append(NeuronState(stage=MaturationStage.PROGENITOR))

        self.neuron_count += n_new
        self._last_growth_step = self._step_count
        return True

    def _prune(self) -> int:
        """Remove neurons with near-zero activity."""
        self._last_prune_step = self._step_count

        # Find candidates: low activity + past progenitor stage
        to_remove = []
        for i in range(self.neuron_count):
            s = self.states[i]
            if (s.recent_spikes < self.prune_threshold and
                    s.age > self.maturation_steps and
                    s.stage != MaturationStage.PROGENITOR):
                to_remove.append(i)

        if not to_remove:
            return 0

        # Keep at least some minimum neurons
        min_neurons = max(8, self.neuron_count // 4)
        if self.neuron_count - len(to_remove) < min_neurons:
            to_remove = to_remove[:self.neuron_count - min_neurons]

        if not to_remove:
            return 0

        # Remove from weights and states
        keep = [i for i in range(self.neuron_count) if i not in set(to_remove)]
        self.weights = self.weights[keep]
        self.states = [self.states[i] for i in keep]
        self.neuron_count = len(keep)

        return len(to_remove)

    def get_mature_mask(self) -> np.ndarray:
        """Boolean mask of mature (differentiated+) neurons."""
        return np.array([
            s.stage in (MaturationStage.DIFFERENTIATED, MaturationStage.MYELINATED)
            for s in self.states
        ], dtype=bool)

    def stats(self) -> Dict[str, Any]:
        """Network statistics."""
        stages = {}
        for s in self.states:
            stages[s.stage.value] = stages.get(s.stage.value, 0) + 1
        avg_activity = np.mean([s.recent_spikes for s in self.states]) if self.states else 0
        return {
            "neuron_count": self.neuron_count,
            "max_neurons": self.max_neurons,
            "stages": stages,
            "avg_activity": float(avg_activity),
            "residual_ema": self.residual_ema,
            "step": self._step_count,
        }
