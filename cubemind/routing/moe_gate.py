"""DSelect-k Mixture of Experts gate with smooth-step selection.

Provides sparse expert selection via differentiable binary encoding:
  - SmoothStep: cubic polynomial approximation of the Heaviside step function
  - DSelectKGate: selects exactly k out of n experts using smooth binary codes

Reference: Hazimeh et al., "DSelect-k: Differentiable Selection in the
Mixture of Experts with Applications to Multi-Task Learning", NeurIPS 2021.

Pure numpy implementation, no TF/JAX/torch dependency.
"""

from __future__ import annotations

import math

import numpy as np

from cubemind.core.registry import register

# Small constant for numerical stability.
EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Smooth-step function
# ---------------------------------------------------------------------------


def smooth_step(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Smooth approximation of the Heaviside step function.

    For scalar x::

        0                                       if x <= -gamma/2
        1                                       if x >= gamma/2
        a3*x^3 + a1*x + 0.5                    otherwise

    where a3 = -2/gamma^3 and a1 = 3/(2*gamma).

    Args:
        x: Input array of any shape.
        gamma: Width of the polynomial transition region.

    Returns:
        Array of same shape as *x* with values in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    lower = -gamma / 2.0
    upper = gamma / 2.0
    a3 = -2.0 / (gamma**3)
    a1 = 3.0 / (2.0 * gamma)

    return np.where(
        x <= lower,
        np.zeros_like(x),
        np.where(
            x >= upper,
            np.ones_like(x),
            a3 * (x**3) + a1 * x + 0.5,
        ),
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Softmax utility
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along *axis*."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# DSelect-k gate
# ---------------------------------------------------------------------------


@register("router", "dselect_k")
class DSelectKGate:
    """Differentiable top-k expert selection gate.

    Uses binary encoding of expert indices and smooth-step activations to
    produce a sparse, differentiable mixture over *num_experts* experts,
    selecting approximately *k* of them.

    Args:
        num_experts: Total number of available experts.
        k: Number of experts to select (number of non-zero weights).
        gamma: Width parameter for the smooth-step function.
        seed: Random seed for weight initialization.
    """

    def __init__(
        self,
        num_experts: int,
        k: int = 2,
        gamma: float = 1.0,
        seed: int = 42,
    ) -> None:
        if k > num_experts:
            raise ValueError(
                f"k ({k}) cannot exceed num_experts ({num_experts})"
            )
        self.num_experts = num_experts
        self.k = k
        self.gamma = gamma

        rng = np.random.default_rng(seed)

        # Number of binary digits needed to encode expert indices.
        self._num_binary = max(1, math.ceil(math.log2(num_experts)))
        self._power_of_2 = num_experts == 2**self._num_binary

        # Binary encoding matrix: (num_experts, num_binary).
        # Row i is the binary representation of integer i.
        self._binary_codes = np.array(
            [
                [int(c) for c in np.binary_repr(val, width=self._num_binary)]
                for val in range(num_experts)
            ],
            dtype=bool,
        )  # (num_experts, num_binary)

        # Learnable parameters ------------------------------------------
        # z_logits: (k, num_binary) — selects which binary code to activate.
        self.z_logits = rng.uniform(
            -gamma / 100, gamma / 100, size=(k, self._num_binary)
        ).astype(np.float64)

        # w_logits: (k,) — mixing weights across the k selectors.
        self.w_logits = rng.uniform(size=(k,)).astype(np.float64)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute sparse k-hot gate weights over experts.

        For task-only routing (the default), *scores* is ignored and the
        gate uses only its internal learned parameters. When *scores* is
        provided it is added to each row of ``z_logits`` to provide
        input-dependent conditioning.

        Args:
            scores: Optional input array of shape ``(input_dim,)`` or
                ``(batch, input_dim)``. Used as an additive bias to
                ``z_logits`` (projected to ``num_binary`` dims via
                mean-pooling if ``input_dim != num_binary``).

        Returns:
            1-D array of length ``num_experts`` with non-negative weights
            that sum to ~1. Most entries will be near zero (sparse).
        """
        z = self.z_logits.copy()

        # Optional input conditioning: add a bias derived from scores.
        if scores is not None:
            x = np.asarray(scores, dtype=np.float64).ravel()
            # Simple projection: tile or truncate to num_binary length.
            if x.shape[0] != self._num_binary:
                repeats = math.ceil(self._num_binary / max(x.shape[0], 1))
                x_tiled = np.tile(x, repeats)[: self._num_binary]
            else:
                x_tiled = x
            z = z + x_tiled[np.newaxis, :]  # broadcast over k selectors

        # Smooth-step activations: (k, num_binary)
        ss = smooth_step(z, self.gamma)

        # Selector outputs: (k, num_experts)
        # For each selector and each expert, compute the product of
        # matching bits. If the binary code bit is 1, use ss; else 1-ss.
        codes = self._binary_codes[np.newaxis, :, :]  # (1, E, B)
        ss_expanded = ss[:, np.newaxis, :]  # (k, 1, B)
        matched = np.where(codes, ss_expanded, 1.0 - ss_expanded)  # (k, E, B)
        selector_outputs = np.prod(matched, axis=2)  # (k, E)

        # Selector weights via softmax: (k,)
        selector_weights = _softmax(self.w_logits, axis=0)  # (k,)

        # Final expert weights: weighted sum over selectors -> (E,)
        expert_weights = np.sum(
            selector_weights[:, np.newaxis] * selector_outputs, axis=0
        )

        return expert_weights

    # ------------------------------------------------------------------
    # Entropy regularization
    # ------------------------------------------------------------------

    def entropy_regularization(self, expert_weights: np.ndarray) -> float:
        """Compute entropy penalty to encourage binary (sparse) selections.

        Returns 0 when all weights are exactly 0 or 1, and a positive
        value when weights are soft / non-binary.

        Args:
            expert_weights: 1-D array of expert weights (typically from
                :meth:`forward`).

        Returns:
            Non-negative scalar entropy value.
        """
        w = np.asarray(expert_weights, dtype=np.float64)
        w_clamped = np.clip(w, EPSILON, 1.0 - EPSILON)
        entropy = -np.sum(
            w_clamped * np.log(w_clamped)
            + (1.0 - w_clamped) * np.log(1.0 - w_clamped)
        )
        return float(entropy)
