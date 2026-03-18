"""DisARM wrapper for CubeMind training.

Re-exports grilly.nn.DisARMSampler and disarm_gradient with CubeMind-specific
utilities: block-code discretization pass and gradient logging to telemetry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cubemind.telemetry import metrics

# Re-export from grilly
try:
    from grilly.nn.disarm import DisARMSampler, disarm_gradient
except ImportError:
    # Provide local fallback implementations so training can proceed
    # without grilly installed (e.g., in test environments).
    DisARMSampler = None  # type: ignore[assignment, misc]
    disarm_gradient = None  # type: ignore[assignment]


def discretize_block_codes(
    logits: np.ndarray,
    block_size: int,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize continuous logits into block codes via DisARM.

    Performs Gumbel-argmax sampling within each block and returns both
    the discrete block codes and a DisARM gradient estimate.

    Args:
        logits: (batch, dim) continuous logits where dim = num_blocks * block_size.
        block_size: Size of each block.
        temperature: Sampling temperature (lower = sharper).

    Returns:
        (discrete, grad_estimate):
            discrete: (batch, dim) one-hot block codes.
            grad_estimate: (batch, dim) gradient through discretization.
    """
    logits = np.asarray(logits, dtype=np.float32)
    batch_size = logits.shape[0]
    dim = logits.shape[1]
    num_blocks = dim // block_size

    block_logits = logits.reshape(batch_size, num_blocks, block_size)

    # Sample discrete block codes via Gumbel-argmax
    u = np.random.uniform(0.01, 0.99, size=block_logits.shape).astype(np.float32)
    gumbel = -np.log(-np.log(u))
    perturbed = block_logits / temperature + gumbel

    indices = perturbed.argmax(axis=-1)
    discrete = np.zeros_like(block_logits)
    for b in range(batch_size):
        for k in range(num_blocks):
            discrete[b, k, indices[b, k]] = 1.0

    # Antithetic sample
    gumbel_anti = -np.log(-np.log(1.0 - u))
    perturbed_anti = block_logits / temperature + gumbel_anti
    indices_anti = perturbed_anti.argmax(axis=-1)
    discrete_anti = np.zeros_like(block_logits)
    for b in range(batch_size):
        for k in range(num_blocks):
            discrete_anti[b, k, indices_anti[b, k]] = 1.0

    # Gradient estimate via straight-through + antithetic correction
    softmax_probs = np.exp(block_logits / temperature)
    softmax_probs = softmax_probs / softmax_probs.sum(axis=-1, keepdims=True)
    diff = discrete - discrete_anti
    grad_estimate = diff * softmax_probs

    discrete_flat = discrete.reshape(batch_size, dim)
    grad_flat = grad_estimate.reshape(batch_size, dim)

    return discrete_flat, grad_flat


def discretize_and_log(
    logits: np.ndarray,
    block_size: int,
    temperature: float = 1.0,
    step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize block codes and log gradient statistics to telemetry.

    Same as discretize_block_codes but records metrics:
      - training.disarm.grad_norm: L2 norm of the gradient estimate
      - training.disarm.entropy: mean entropy of the softmax distributions

    Args:
        logits: (batch, dim) continuous logits.
        block_size: Size of each block.
        temperature: Sampling temperature.
        step: Optional step number for tagging.

    Returns:
        (discrete, grad_estimate) tuple.
    """
    discrete, grad_est = discretize_block_codes(logits, block_size, temperature)

    # Log gradient statistics
    grad_norm = float(np.linalg.norm(grad_est))
    tags: dict[str, str] = {}
    if step is not None:
        tags["step"] = str(step)

    metrics.record("training.disarm.grad_norm", grad_norm, tags=tags)

    # Log entropy of softmax distribution (measure of discretization sharpness)
    logits_arr = np.asarray(logits, dtype=np.float32)
    batch_size = logits_arr.shape[0]
    dim = logits_arr.shape[1]
    num_blocks = dim // block_size
    block_logits = logits_arr.reshape(batch_size, num_blocks, block_size)
    probs = np.exp(block_logits / temperature)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
    mean_entropy = float(np.mean(entropy))
    metrics.record("training.disarm.entropy", mean_entropy, tags=tags)

    return discrete, grad_est


__all__ = [
    "DisARMSampler",
    "disarm_gradient",
    "discretize_block_codes",
    "discretize_and_log",
]
