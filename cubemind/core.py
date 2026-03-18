"""Shared types, constants, and initialization functions for CubeMind."""

from enum import Enum

import numpy as np


class Strategy(Enum):
    """VSA encoding strategy."""

    BLAKE3 = "blake3"
    BLOCK_CODE = "block_code"


# Convenience aliases at module level
BLAKE3 = Strategy.BLAKE3
BLOCK_CODE = Strategy.BLOCK_CODE

# Default dimensions (paper: k=16, l=128, d=2048)
K_BLOCKS: int = 16
L_BLOCK: int = 128
D_VSA: int = K_BLOCKS * L_BLOCK  # 2048

EPS: float = 1e-20

# Activation correction factors (He et al., 2015 extended)
_ACTIVATION_FACTORS: dict[str, float] = {
    "linear": 1.0,
    "relu": 2.0,
    "gelu": 1.7,  # empirical approximation
    "tanh": 1.0,  # odd function, unit derivative at 0
}


def hyperfan_in_variance(
    fan_in: int,
    d_k: int,
    l: int,
    has_bias: bool = False,
    activation: str = "gelu",
) -> float:
    """Hyperfan-in variance formula (Chang et al., 2020) adapted for block codes.

    For block-code embeddings with block length l, Var(e) = 1/l.  The formula
    preserves signal variance through the hypernet output layer:

        Var(W) = act_factor / (bias_factor * fan_in * d_k * var_e)

    where var_e = 1/l and bias_factor = 2 when the hypernet also generates a
    mainnet bias (halves the per-weight budget), 1 otherwise.

    Args:
        fan_in: Mainnet layer fan-in (d_j in the paper).
        d_k: Hypernet hidden dimension.
        l: Block length (determines embedding variance via var_e = 1/l).
        has_bias: Whether the hypernet generates a mainnet bias vector too.
        activation: Activation function used in the hypernet hidden layers.

    Returns:
        Scalar variance for initialising hypernet output-layer weights.
    """
    act_factor = _ACTIVATION_FACTORS.get(activation, 1.0)
    bias_factor = 2.0 if has_bias else 1.0
    var_e = 1.0 / l  # block-code embedding variance
    return act_factor / (bias_factor * fan_in * d_k * var_e)


def hyperfan_out_variance(
    fan_in: int,
    fan_out: int,
    d_k: int,
    l: int,
    has_bias: bool = False,
    activation: str = "gelu",
) -> float:
    """Hyperfan-out variance formula adapted for block codes.

    Like hyperfan_in_variance but the denominator uses fan_out (d_i) instead
    of fan_in, preserving output signal variance rather than input variance.

    Args:
        fan_in: Mainnet layer fan-in (used for bias-only path; ignored here).
        fan_out: Mainnet layer fan-out (d_i in the paper).
        d_k: Hypernet hidden dimension.
        l: Block length.
        has_bias: Whether the hypernet generates a mainnet bias vector too.
        activation: Activation function used in the hypernet hidden layers.

    Returns:
        Scalar variance for initialising hypernet output-layer weights.
    """
    act_factor = _ACTIVATION_FACTORS.get(activation, 1.0)
    bias_factor = 2.0 if has_bias else 1.0
    var_e = 1.0 / l
    return act_factor / (bias_factor * fan_out * d_k * var_e)


def hyperfan_init(
    fan_out: int,
    fan_in: int,
    d_k: int,
    l: int,
    has_bias: bool = False,
    activation: str = "gelu",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Initialise hypernet output-layer weights with Hyperfan-in (Theorem 3).

    Samples W ~ N(0, Var) where Var is given by hyperfan_in_variance, then
    returns the weight matrix reshaped for the hypernet output projection.

    Args:
        fan_out: Mainnet output features.
        fan_in: Mainnet input features.
        d_k: Hypernet hidden dimension.
        l: Block length.
        has_bias: Whether the hypernet also generates a mainnet bias.
        activation: Hypernet activation function.
        rng: NumPy random Generator; defaults to a fresh default_rng().

    Returns:
        float32 array of shape (fan_out * fan_in, d_k).
    """
    if rng is None:
        rng = np.random.default_rng()
    var = hyperfan_in_variance(fan_in, d_k, l, has_bias, activation)
    std = float(np.sqrt(var))
    return rng.normal(0.0, std, size=(fan_out * fan_in, d_k)).astype(np.float32)
