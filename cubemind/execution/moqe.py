"""MoQE — Mixture of Quantization Experts (N-expert).

N experts with configurable quantization bit-widths and specializations.
Top-k routing via learned softmax gate with Gumbel-Softmax training.

Expert types:
  - Quantization levels: 2-bit, 4-bit, 6-bit, 8-bit (compression vs precision)
  - Specializations: dense (general), sparse (rare tokens), low-rank (compression)

Architecture per layer:
  input (d_model,) → router → top-k experts → weighted sum → output (d_model,)
  router: Linear(d_model, n_experts) → softmax → top-k selection

Training:
  Gumbel-Softmax soft routing (all experts get gradient proportional to weight).
  Router balance loss = MSE(actual_fractions, target_fractions).
  Load balancing via auxiliary entropy loss.

GPU shaders: moqe-fused-gemv-dp4a.glsl (2-expert fast path still supported).
CPU fallback: vectorized dequant + BLAS matmul.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# GPU bridge (2-expert fast path)
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


# ── Quantization ─────────────────────────────────────────────────────────────

def _quantize_symmetric(weights: np.ndarray, bits: int,
                         block_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Quantize FP32 weights to signed integer (symmetric, per-block).

    Supports 2/4/6/8-bit quantization. Returns (quantized_int8, scales).
    """
    max_val = (1 << (bits - 1)) - 1  # 1→0, 2→1, 4→7, 6→31, 8→127
    w = weights.astype(np.float32).ravel()
    n = len(w)
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    blocks = w.reshape(-1, block_size)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.where(absmax < 1e-7, 1e-7, absmax)
    scales = absmax / max_val
    quantized = np.clip(np.round(blocks / scales[:, None]), -max_val, max_val).astype(np.int8)
    return quantized.ravel()[:n].reshape(weights.shape), scales.astype(np.float32)


def _dequant_weights(w_int: np.ndarray, scales: np.ndarray,
                      block_size: int) -> np.ndarray:
    """Dequantize INT weights to FP32. Fully vectorized."""
    d_out, d_in = w_int.shape
    num_blocks = scales.shape[1]
    padded = d_in + (block_size - d_in % block_size) % block_size
    if padded == d_in:
        w_blocked = w_int.reshape(d_out, num_blocks, block_size).astype(np.float32)
    else:
        w_padded = np.zeros((d_out, padded), dtype=w_int.dtype)
        w_padded[:, :d_in] = w_int
        w_blocked = w_padded.reshape(d_out, num_blocks, block_size).astype(np.float32)
    w_fp = (w_blocked * scales[:, :, None]).reshape(d_out, -1)[:, :d_in]
    return np.ascontiguousarray(w_fp)


# Backward-compat aliases for code importing old function names
def _quantize_weights_int4(weights, block_size=32):
    return _quantize_symmetric(weights, bits=4, block_size=block_size)

def _quantize_weights_int8(weights, block_size=32):
    return _quantize_symmetric(weights, bits=8, block_size=block_size)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / (e.sum(axis=axis, keepdims=True) + 1e-8)).astype(np.float32)


def _gumbel_softmax(logits: np.ndarray, temperature: float,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Gumbel-Softmax for N-expert routing.

    Args:
        logits: (..., n_experts) raw router logits.
        temperature: Annealing param. High=soft, low=hard.
        rng: Random generator for Gumbel noise.

    Returns:
        (..., n_experts) soft weights summing to 1.0 per token.
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(1e-7, 1.0 - 1e-7, size=logits.shape).astype(np.float32)
    gumbel = -np.log(-np.log(u))
    noisy = (logits + gumbel) / max(temperature, 1e-6)
    return _softmax(noisy, axis=-1)


# ── Expert Specification ─────────────────────────────────────────────────────

@dataclass
class ExpertSpec:
    """Configuration for a single expert."""
    bits: int = 8                 # Quantization bit-width (2, 4, 6, 8)
    specialty: str = "general"    # "general", "rare", "factual", "code", "dialogue"
    target_fraction: float = 0.0  # Target routing fraction (0 = auto-balance)
    rank: int = 0                 # Low-rank factor (0 = full rank)


# ── Router ───────────────────────────────────────────────────────────────────

class MoQERouter:
    """N-expert routing gate with top-k selection.

    Inference: hard top-k via argpartition.
    Training: Gumbel-Softmax soft blending with temperature annealing.

    Args:
        d_model:   Input dimension.
        n_experts: Number of experts to route to.
        top_k:     Number of experts selected per token.
        seed:      Random seed.
    """

    def __init__(self, d_model: int, n_experts: int = 2,
                 top_k: int = 1, seed: int = 42) -> None:
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.rng = np.random.default_rng(seed)
        # Learnable routing weights
        self.W = (self.rng.standard_normal((n_experts, d_model)) * 0.01).astype(np.float32)
        self.b = np.zeros(n_experts, dtype=np.float32)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Compute raw routing logits. x: (..., d_model) → (..., n_experts)."""
        return (x @ self.W.T + self.b).astype(np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hard top-k routing (inference).

        Args:
            x: (d_model,) or (batch, d_model).

        Returns:
            (indices, weights): top-k expert indices and softmax weights.
            For single input: indices (top_k,), weights (top_k,).
            For batch: indices (batch, top_k), weights (batch, top_k).
        """
        raw = self.logits(x)
        probs = _softmax(raw, axis=-1)

        if raw.ndim == 1:
            idx = np.argsort(probs)[-self.top_k:][::-1]
            w = probs[idx]
            w /= w.sum() + 1e-8
            return idx, w

        # Batch
        idx = np.argpartition(-probs, self.top_k, axis=-1)[:, :self.top_k]
        # Sort within top-k
        for i in range(len(idx)):
            order = np.argsort(-probs[i, idx[i]])
            idx[i] = idx[i, order]
        w = np.take_along_axis(probs, idx, axis=-1)
        w /= w.sum(axis=-1, keepdims=True) + 1e-8
        return idx, w

    def forward_gumbel(self, X: np.ndarray,
                        temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Gumbel-Softmax routing (training: soft blending over ALL experts).

        Returns:
            (weights, raw_logits):
                weights: (batch, n_experts) soft weights summing to 1.
                raw_logits: (batch, n_experts) pre-Gumbel logits.
        """
        raw = self.logits(X)  # (batch, n_experts)
        weights = _gumbel_softmax(raw, temperature, self.rng)
        return weights, raw


# ── MoQE Layer ───────────────────────────────────────────────────────────────

class MoQELayer:
    """Single MoQE layer with N quantized experts.

    Each expert has its own bit-width and specialty. Tokens are routed
    to top-k experts via the learned router.

    Args:
        d_model:    Input dimension.
        d_out:      Output dimension.
        expert_specs: List of ExpertSpec configurations.
        top_k:      Number of experts per token.
        block_size: Quantization block size.
        seed:       Random seed.
    """

    def __init__(
        self,
        d_model: int,
        d_out: int,
        expert_specs: list[ExpertSpec] | None = None,
        top_k: int = 1,
        block_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.d_out = d_out
        self.block_size = block_size

        # Default: 2 experts (4-bit + 8-bit) for backward compat
        if expert_specs is None:
            expert_specs = [
                ExpertSpec(bits=4, specialty="general", target_fraction=0.85),
                ExpertSpec(bits=8, specialty="general", target_fraction=0.15),
            ]
        self.expert_specs = expert_specs
        self.n_experts = len(expert_specs)

        rng = np.random.default_rng(seed)
        self.router = MoQERouter(d_model, self.n_experts, top_k=top_k, seed=seed)

        # Initialize + quantize each expert
        std = 1.0 / math.sqrt(d_model)
        num_blocks = (d_model + block_size - 1) // block_size

        self.expert_w_int = []   # quantized weights per expert
        self.expert_scales = []  # quantization scales per expert

        for i, spec in enumerate(expert_specs):
            w_fp32 = rng.normal(0, std, (d_out, d_model)).astype(np.float32)
            w_int, s_flat = _quantize_symmetric(w_fp32, spec.bits, block_size)
            scales = s_flat[:num_blocks * d_out].reshape(d_out, num_blocks)
            self.expert_w_int.append(w_int)
            self.expert_scales.append(scales)

    # Backward-compat properties for code that uses w0_int/w1_int/s0/s1
    @property
    def w0_int(self): return self.expert_w_int[0]
    @w0_int.setter
    def w0_int(self, v): self.expert_w_int[0] = v
    @property
    def w1_int(self): return self.expert_w_int[1] if len(self.expert_w_int) > 1 else self.expert_w_int[0]
    @w1_int.setter
    def w1_int(self, v):
        if len(self.expert_w_int) > 1: self.expert_w_int[1] = v
    @property
    def s0(self): return self.expert_scales[0]
    @s0.setter
    def s0(self, v): self.expert_scales[0] = v
    @property
    def s1(self): return self.expert_scales[1] if len(self.expert_scales) > 1 else self.expert_scales[0]
    @s1.setter
    def s1(self, v):
        if len(self.expert_scales) > 1: self.expert_scales[1] = v

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass: route + expert GEMV.

        Args:
            x: (d_model,) float32 input vector.

        Returns:
            (output, expert_indices, expert_weights).
        """
        indices, weights = self.router.forward(x)

        output = np.zeros(self.d_out, dtype=np.float32)
        for j, idx in enumerate(indices):
            w_fp = _dequant_weights(self.expert_w_int[idx],
                                     self.expert_scales[idx],
                                     self.block_size)
            output += weights[j] * (w_fp @ x)

        return output, indices, weights

    def forward_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch forward: route each token, weighted expert sum.

        Args:
            X: (batch, d_model) float32.

        Returns:
            (outputs, indices, weights):
                outputs: (batch, d_out)
                indices: (batch, top_k)
                weights: (batch, top_k)
        """
        batch = X.shape[0]
        indices, weights = self.router.forward(X)  # (batch, top_k), (batch, top_k)

        # Pre-dequantize all experts that are used
        used_experts = set(indices.ravel().tolist())
        dequant_cache = {}
        for eidx in used_experts:
            dequant_cache[eidx] = _dequant_weights(
                self.expert_w_int[eidx], self.expert_scales[eidx], self.block_size)

        # Compute weighted expert outputs
        outputs = np.zeros((batch, self.d_out), dtype=np.float32)
        for j in range(self.router.top_k):
            for eidx in used_experts:
                mask = indices[:, j] == eidx  # tokens routed to this expert
                if not mask.any():
                    continue
                w_fp = dequant_cache[eidx]
                expert_out = X[mask] @ w_fp.T  # (n_tokens, d_out)
                outputs[mask] += weights[mask, j:j+1] * expert_out

        return outputs, indices, weights

    def forward_gumbel_batch(self, X: np.ndarray,
                              temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gumbel-Softmax forward (training): all experts get gradient.

        Args:
            X: (batch, d_model) float32.
            temperature: Gumbel temperature.

        Returns:
            (outputs, soft_weights, raw_logits):
                outputs: (batch, d_out) — soft-blended expert outputs.
                soft_weights: (batch, n_experts) — Gumbel-Softmax weights.
                raw_logits: (batch, n_experts) — pre-noise logits.
        """
        soft_weights, raw_logits = self.router.forward_gumbel(X, temperature)

        # All experts process all tokens (weighted by Gumbel-Softmax)
        outputs = np.zeros((X.shape[0], self.d_out), dtype=np.float32)
        for eidx in range(self.n_experts):
            w_fp = _dequant_weights(self.expert_w_int[eidx],
                                     self.expert_scales[eidx],
                                     self.block_size)
            expert_out = (X @ w_fp.T).astype(np.float32)
            outputs += soft_weights[:, eidx:eidx+1] * expert_out

        return outputs, soft_weights, raw_logits

    def get_expert_fractions(self, indices: np.ndarray) -> np.ndarray:
        """Compute actual routing fractions per expert."""
        counts = np.bincount(indices.ravel(), minlength=self.n_experts)
        return counts.astype(np.float32) / max(counts.sum(), 1)

    def get_target_fractions(self) -> np.ndarray:
        """Get target routing fractions from expert specs."""
        targets = np.array([s.target_fraction for s in self.expert_specs], dtype=np.float32)
        if targets.sum() < 1e-6:
            # Auto-balance: equal fractions
            targets = np.ones(self.n_experts, dtype=np.float32) / self.n_experts
        else:
            targets /= targets.sum() + 1e-8
        return targets


# ── MoQE Model ───────────────────────────────────────────────────────────────

class MoQEModel:
    """Multi-layer MoQE model with N configurable experts.

    Stack of MoQE layers with embedding and output projection.
    Each layer routes tokens to top-k of N experts.

    Args:
        vocab_size:   Vocabulary size.
        d_model:      Model dimension.
        n_layers:     Number of MoQE layers.
        expert_specs: Per-expert configuration. Applied to all layers.
        top_k:        Experts selected per token per layer.
        block_size:   Quantization block size.
        seed:         Random seed.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 4,
        expert_specs: list[ExpertSpec] | None = None,
        top_k: int = 1,
        block_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)

        self.embedding = (rng.standard_normal((vocab_size, d_model)) * 0.02).astype(np.float32)

        self.layers = [
            MoQELayer(d_model, d_model, expert_specs=expert_specs,
                      top_k=top_k, block_size=block_size, seed=seed + i)
            for i in range(n_layers)
        ]

        self.out_proj = (rng.standard_normal((vocab_size, d_model)) * 0.02).astype(np.float32)

        self.n_experts = self.layers[0].n_experts if self.layers else 0

    def forward(self, input_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Forward pass: input_ids → logits + per-layer routing info.

        Args:
            input_ids: (seq_len,) int32 token IDs.

        Returns:
            (logits, layer_weights):
                logits: (seq_len, vocab_size) float32.
                layer_weights: list of (seq_len, top_k) weight arrays.
        """
        seq_len = len(input_ids)
        x = self.embedding[input_ids]

        layer_weights = []
        for layer in self.layers:
            outputs, indices, weights = layer.forward_batch(x)
            x = x + outputs
            layer_weights.append(weights)

        logits = (x @ self.out_proj.T).astype(np.float32)
        return logits, layer_weights

    def get_expert_usage(self, input_ids: np.ndarray) -> dict:
        """Profile expert usage across layers."""
        seq_len = len(input_ids)
        x = self.embedding[input_ids]

        usage = {}
        for li, layer in enumerate(self.layers):
            outputs, indices, weights = layer.forward_batch(x)
            x = x + outputs
            fractions = layer.get_expert_fractions(indices)
            for ei, spec in enumerate(layer.expert_specs):
                usage[f"L{li}_E{ei}_{spec.bits}bit_{spec.specialty}"] = float(fractions[ei])
        return usage
