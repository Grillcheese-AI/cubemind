"""MoQE — Mixture of Quantization Experts.

Hard-routed mixture of two experts with different quantization levels:
  - Expert 0: 4-bit weights (high compression, ~85% of tokens)
  - Expert 1: 8-bit weights (high precision, ~15% of tokens)

Each token is routed to exactly one expert via a learned sigmoid gate.
The chosen expert runs a fused dynamic-quantization GEMV:
  activations (FP32) → quantize on-the-fly → INT matmul → dequantize → FP32

GPU shaders: moqe-fused-gemv-dp4a.glsl (DP4a hardware, Wave64 adaptive)
CPU fallback: numpy quantize + integer dot product via grilly bridge.

Architecture per layer:
  input (d_model,) → router → expert_0 or expert_1 → output (d_model,)
  router: Linear(d_model, 1) → sigmoid → threshold(0.5)

Training:
  Router loss = MSE(actual_8bit_fraction, target=0.15)
  Forces 4-bit expert to handle majority of tokens.
"""

from __future__ import annotations

import math

import numpy as np

# GPU bridge
_bridge_moqe = None
try:
    from grilly.backend._bridge import (
        moqe_dynamic_quantize,
        moqe_fused_gemv,
        moqe_route_and_gemv,
    )
    _bridge_moqe = True
except ImportError:
    pass


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))).astype(np.float32)


def _quantize_weights_int4(weights: np.ndarray, block_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weights to INT4 (symmetric, per-block).

    Returns:
        (quantized_int8, scales) — stored as int8 with range [-7, 7].
    """
    w = weights.astype(np.float32).ravel()
    n = len(w)
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    blocks = w.reshape(-1, block_size)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.where(absmax < 1e-7, 1e-7, absmax)
    scales = absmax / 7.0
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -7, 7).astype(np.int8)
    return quantized.ravel()[:n].reshape(weights.shape), scales.astype(np.float32)


def _quantize_weights_int8(weights: np.ndarray, block_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weights to INT8 (symmetric, per-block)."""
    w = weights.astype(np.float32).ravel()
    n = len(w)
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    blocks = w.reshape(-1, block_size)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.where(absmax < 1e-7, 1e-7, absmax)
    scales = absmax / 127.0
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    return quantized.ravel()[:n].reshape(weights.shape), scales.astype(np.float32)


def _gumbel_softmax_2way(logits_2: np.ndarray, temperature: float,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Gumbel-Softmax for 2-expert routing. Fully differentiable.

    Args:
        logits_2: (batch, 2) raw router logits for expert 0 and 1.
        temperature: Annealing param. High=soft blend, low=hard argmax.
        rng: Random generator for Gumbel noise.

    Returns:
        (batch, 2) soft weights summing to 1.0 per token.
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(1e-7, 1.0 - 1e-7, size=logits_2.shape).astype(np.float32)
    gumbel = -np.log(-np.log(u))
    noisy = (logits_2 + gumbel) / max(temperature, 1e-6)
    # Stable softmax
    noisy -= np.max(noisy, axis=-1, keepdims=True)
    ex = np.exp(noisy)
    return (ex / (ex.sum(axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


class MoQERouter:
    """Learned 2-expert routing gate with Gumbel-Softmax training support.

    Inference: hard argmax routing (expert 0 or 1).
    Training: Gumbel-Softmax soft blending with temperature annealing.
    Both experts receive gradient proportional to their selection weight.

    Args:
        d_model: Input dimension.
        seed:    Random seed.
    """

    def __init__(self, d_model: int, seed: int = 42) -> None:
        self.d_model = d_model
        self.rng = np.random.default_rng(seed)
        self.w = (self.rng.standard_normal(d_model) * 0.01).astype(np.float32)
        self.b = np.float32(0.0)

    def forward(self, x: np.ndarray, hard: bool = True) -> tuple[int, float]:
        """Route a single token (inference)."""
        logit = float(np.dot(x.ravel(), self.w) + self.b)
        prob = float(_sigmoid(np.array([logit]))[0])
        return (1 if prob > 0.5 else 0), prob

    def forward_batch(self, X: np.ndarray, hard: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Route a batch of tokens (inference: hard argmax)."""
        logits = X @ self.w + self.b  # (batch,)
        probs = _sigmoid(logits)
        choices = (probs > 0.5).astype(np.int32)
        return choices, probs

    def forward_gumbel(self, X: np.ndarray,
                        temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Gumbel-Softmax routing (training: soft blending).

        Returns:
            (weights, logits_2):
                weights: (batch, 2) soft expert weights summing to 1.
                logits_2: (batch, 2) raw logits (for gradient computation).
        """
        # 2-way logits: [logit_expert0, logit_expert1]
        # Expert 0 = 4-bit (default), Expert 1 = 8-bit
        logit_1 = X @ self.w + self.b  # prob of expert 1
        logit_0 = -logit_1             # symmetric: prob of expert 0
        logits_2 = np.stack([logit_0, logit_1], axis=-1).astype(np.float32)

        weights = _gumbel_softmax_2way(logits_2, temperature, self.rng)
        return weights, logits_2


class MoQELayer:
    """Single MoQE layer: router + dual quantized experts.

    Expert 0: 4-bit weights (high compression)
    Expert 1: 8-bit weights (high precision)

    Both experts share the same weight matrix shape (d_out, d_model),
    but are quantized to different bit widths.

    Args:
        d_model:    Input dimension.
        d_out:      Output dimension.
        block_size: Quantization block size.
        seed:       Random seed.
    """

    def __init__(
        self, d_model: int, d_out: int, block_size: int = 32, seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.d_out = d_out
        self.block_size = block_size

        rng = np.random.default_rng(seed)
        self.router = MoQERouter(d_model, seed=seed)

        # Initialize FP32 weights, then quantize
        std = 1.0 / math.sqrt(d_model)
        weights_fp32 = rng.normal(0, std, (d_out, d_model)).astype(np.float32)

        # Expert 0: 4-bit quantization
        self.w0_int, self.s0 = _quantize_weights_int4(weights_fp32, block_size)
        # Expert 1: 8-bit quantization
        self.w1_int, self.s1 = _quantize_weights_int8(weights_fp32, block_size)

        # Per-row scales: (d_out, num_blocks)
        num_blocks = (d_model + block_size - 1) // block_size
        self.s0 = self.s0[:num_blocks * d_out].reshape(d_out, num_blocks)
        self.s1 = self.s1[:num_blocks * d_out].reshape(d_out, num_blocks)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass: route + expert GEMV.

        Args:
            x: (d_model,) float32 input vector.

        Returns:
            (output, router_prob): output (d_out,) float32, prob of 8-bit expert.
        """
        choice, prob = self.router.forward(x)

        # Try grilly GPU path
        if _bridge_moqe:
            try:
                output = moqe_route_and_gemv(
                    x,
                    choice,
                    [self.w0_int, self.w1_int],
                    [self.s0, self.s1],
                    self.block_size,
                )
                return output, prob
            except Exception:
                pass

        # CPU fallback: select expert and compute
        if choice == 0:
            w_int, scales = self.w0_int, self.s0
        else:
            w_int, scales = self.w1_int, self.s1

        # Dequantize and matmul (simple path)
        output = self._dequant_matmul(x, w_int, scales)
        return output, prob

    def forward_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch forward: each token routed independently.

        Args:
            X: (batch, d_model) float32.

        Returns:
            (outputs, router_probs): (batch, d_out), (batch,).
        """
        batch = X.shape[0]
        outputs = np.zeros((batch, self.d_out), dtype=np.float32)
        choices, probs = self.router.forward_batch(X)

        for i in range(batch):
            out, _ = self.forward(X[i])
            outputs[i] = out

        return outputs, probs

    def _dequant_matmul(
        self, x: np.ndarray, w_int: np.ndarray, scales: np.ndarray,
    ) -> np.ndarray:
        """Dequantize weights and compute matmul."""
        bs = self.block_size
        d_in = self.d_model
        num_blocks = scales.shape[1]

        output = np.zeros(self.d_out, dtype=np.float32)
        for row in range(self.d_out):
            for b in range(num_blocks):
                start = b * bs
                end = min(start + bs, d_in)
                w_block = w_int[row, start:end].astype(np.float32) * scales[row, b]
                output[row] += float(np.dot(x[start:end], w_block))
        return output


class MoQEModel:
    """Multi-layer MoQE model for LLM distillation.

    Stack of MoQE layers with embedding and output projection.
    Each layer independently routes tokens to 4-bit or 8-bit experts.

    Args:
        vocab_size: Vocabulary size.
        d_model:    Model dimension.
        n_layers:   Number of MoQE layers.
        block_size: Quantization block size.
        seed:       Random seed.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 4,
        block_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)

        # Token embedding (FP32, not quantized)
        self.embedding = (
            rng.standard_normal((vocab_size, d_model)) * 0.02
        ).astype(np.float32)

        # MoQE layers
        self.layers = [
            MoQELayer(d_model, d_model, block_size, seed=seed + i)
            for i in range(n_layers)
        ]

        # Output projection (FP32)
        self.out_proj = (
            rng.standard_normal((vocab_size, d_model)) * 0.02
        ).astype(np.float32)

    def forward(self, input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass: input_ids → logits + router probs.

        Args:
            input_ids: (seq_len,) int32 token IDs.

        Returns:
            (logits, router_probs):
                logits: (seq_len, vocab_size) float32.
                router_probs: (n_layers, seq_len) float32 — prob of 8-bit per layer.
        """
        seq_len = len(input_ids)
        x = self.embedding[input_ids]  # (seq_len, d_model)

        all_probs = []
        for layer in self.layers:
            outputs, probs = layer.forward_batch(x)
            x = x + outputs  # Residual connection
            all_probs.append(probs)

        # Output logits
        logits = (x @ self.out_proj.T).astype(np.float32)

        router_probs = np.stack(all_probs)  # (n_layers, seq_len)
        return logits, router_probs

    def get_8bit_fraction(self, router_probs: np.ndarray) -> float:
        """Average fraction of tokens routed to the 8-bit expert."""
        return float(np.mean(router_probs > 0.5))
