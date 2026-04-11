"""MindForge — VSA-conditioned hypernetwork that forges LoRA adapters from block-code context.

Extends HYLA's factored weight generation with:
  1. Layer-ID conditioning — single generator for all target model layers
  2. LoRA-style injection — generates low-rank (A, B) adapters for a frozen base
  3. Shared basis with mixing coefficients — predicts continuous linear combinations
     of a trainable basis set (NO softmax — prevents mode collapse)
  4. Symbolic context — input is VSA block-code from bind/bundle/permute ops
  5. Cleanup verification — SDLS purification validates symbolic consistency
  6. Full backward pass ��� analytical gradients through basis mixing, MLP, LayerNorm

Architecture::

    context_hv = bind(task_hv, personality_hv)          # VSA symbolic context
    layer_emb  = layer_embeddings[layer_id]             # per-layer coordinate
    ctx_proj   = LayerNorm(context_flat @ W_proj)       # stabilize VSA noise
    h          = GELU(concat(ctx_proj, layer_emb) @ W_h + b_h)
    coeffs     = h @ W_coeff                            # continuous mixing (NO softmax)
    A          = sum(coeffs[i] * basis_A[i])            # (rank, d_in) adapter down
    B          = sum(coeffs[i] * basis_B[i])            # (d_out, rank) adapter up
    output     = frozen_base(x) + scale * (x @ A.T @ B.T)  # LoRA injection

B_basis initialized to ZERO — initial adapter output is exactly 0 (identity).

All ops use grilly GPU backend when available, numpy fallback otherwise.

Part of the CubeMind cognitive architecture.
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes
from cubemind.core.registry import register

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS: int = 80
    L_BLOCK: int = 128

# ── Grilly GPU ops with numpy fallback ───────────────────────────────────

_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


def _gelu(x: np.ndarray) -> np.ndarray:
    if _bridge is not None:
        return np.asarray(_bridge.gelu(x), dtype=np.float32)
    return (0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))).astype(np.float32)


def _gelu_backward(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
    if _bridge is not None:
        return np.asarray(_bridge.gelu_backward(grad, x), dtype=np.float32)
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
    return (grad * (cdf + x * pdf)).astype(np.float32)


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(x))
    var = float(np.var(x))
    if _bridge is not None:
        normed = np.asarray(
            _bridge.layernorm(x.reshape(1, -1), gamma, beta, eps),
            dtype=np.float32,
        ).ravel()
        return normed, mean, var
    normed = gamma * (x - mean) / np.sqrt(var + eps) + beta
    return normed.astype(np.float32), mean, var


def _layernorm_backward(dout: np.ndarray, x: np.ndarray, mean: float,
                         var: float, gamma: np.ndarray, eps: float = 1e-5):
    if _bridge is not None:
        dx, dg, db = _bridge.layernorm_backward(
            dout.reshape(1, -1), x.reshape(1, -1), gamma,
            np.array([mean], dtype=np.float32),
            np.array([var], dtype=np.float32), eps,
        )
        return (np.asarray(dx, dtype=np.float32).ravel(),
                np.asarray(dg, dtype=np.float32).ravel(),
                np.asarray(db, dtype=np.float32).ravel())
    N = len(x)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * std_inv
    dx_hat = dout * gamma
    dvar = float(np.sum(dx_hat * (x - mean) * -0.5 * (std_inv ** 3)))
    dmean = float(np.sum(dx_hat * -std_inv)) + dvar * float(np.mean(-2.0 * (x - mean)))
    dx = dx_hat * std_inv + dvar * 2.0 * (x - mean) / N + dmean / N
    dg = (dout * x_hat).astype(np.float32)
    db = dout.astype(np.float32)
    return dx.astype(np.float32), dg, db


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    if _bridge is not None:
        return np.asarray(_bridge.linear(x, w, b), dtype=np.float32)
    out = (x @ w.T).astype(np.float32)
    if b is not None:
        out += b
    return out


# Keep backward-compat name for external callers
def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (grilly GPU when available)."""
    return _gelu(x)


@register("executor", "mindforge")
class MindForge:
    """VSA-conditioned hypernetwork that forges LoRA adapters on the fly.

    Args:
        k: Number of VSA blocks.
        l: Block length.
        n_layers: Number of target model layers to generate adapters for.
        d_target: Dimension of the target model layers.
        rank: LoRA adapter rank.
        n_basis: Number of shared basis adapter pairs.
        d_hidden: Hidden dimension of the generator MLP.
        scale: LoRA scaling factor (alpha / rank).
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        n_layers: int = 12,
        d_target: int = 2048,
        rank: int = 8,
        n_basis: int = 16,
        d_hidden: int = 256,
        scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.n_layers = n_layers
        self.d_target = d_target
        self.rank = rank
        self.n_basis = n_basis
        self.d_hidden = d_hidden
        self.scale = scale

        self.bc = BlockCodes(k=k, l=l)
        rng = np.random.default_rng(seed)

        # ── Layer-ID embeddings ──────────────────────────────────────────────
        self.layer_embeddings = rng.standard_normal(
            (n_layers, d_hidden)
        ).astype(np.float32) * 0.02

        # ── Generator MLP ────────────────────────────────────────────────────
        input_dim = d_hidden * 2
        self.W_h = rng.normal(
            0, np.sqrt(2.0 / (input_dim + d_hidden)),
            size=(d_hidden, input_dim),
        ).astype(np.float32)
        self.b_h = np.zeros(d_hidden, dtype=np.float32)

        self.W_coeff = rng.normal(
            0, np.sqrt(2.0 / (d_hidden + n_basis)),
            size=(n_basis, d_hidden),
        ).astype(np.float32)
        self.b_coeff = np.zeros(n_basis, dtype=np.float32)

        # ── Context projection: d_vsa → d_hidden ─────────────────────────────
        self.W_proj = rng.normal(
            0, np.sqrt(2.0 / (self.d_vsa + d_hidden)),
            size=(d_hidden, self.d_vsa),
        ).astype(np.float32)

        # ── LayerNorm on projection (stabilizes VSA high-frequency noise) ────
        self.ln_g = np.ones(d_hidden, dtype=np.float32)
        self.ln_b = np.zeros(d_hidden, dtype=np.float32)

        # ── Shared basis adapters ────────────────────────────────────────────
        basis_std = np.sqrt(2.0 / (d_target + rank))
        self.A_basis = rng.normal(
            0, basis_std, size=(n_basis, rank, d_target),
        ).astype(np.float32)
        # FIX: B_basis initialized to ZERO — initial LoRA output is exactly 0
        self.B_basis = np.zeros(
            (n_basis, d_target, rank), dtype=np.float32,
        )

        # ── Gradient accumulators ────────────────────────────────────────────
        self.grads = self.zero_grads()

    # ── Gradient Management ──────────────────────────────────────────────────

    def zero_grads(self) -> dict[str, np.ndarray]:
        return {
            "A_basis": np.zeros_like(self.A_basis),
            "B_basis": np.zeros_like(self.B_basis),
            "W_coeff": np.zeros_like(self.W_coeff),
            "b_coeff": np.zeros_like(self.b_coeff),
            "W_h": np.zeros_like(self.W_h),
            "b_h": np.zeros_like(self.b_h),
            "W_proj": np.zeros_like(self.W_proj),
            "layer_embeddings": np.zeros_like(self.layer_embeddings),
            "ln_g": np.zeros_like(self.ln_g),
            "ln_b": np.zeros_like(self.ln_b),
        }

    # ── SDLS Duality Gate ────────────────────────────────────────────────────

    def register_context(self, name: str, context: np.ndarray) -> None:
        if not hasattr(self, "_cleanup_mem"):
            from cubemind.reasoning.vm import CleanupMemory
            self._cleanup_mem = CleanupMemory(self.bc)
        self._cleanup_mem.store(name, context)

    def sdls_purify(self, context: np.ndarray, threshold: float = 0.85) -> np.ndarray:
        if not hasattr(self, "_cleanup_mem") or self._cleanup_mem.size == 0:
            return self._default_context()
        name, clean = self._cleanup_mem.cleanup(context)
        sim = float(self.bc.similarity(context, clean))
        if sim < threshold:
            return self._default_context()
        return clean

    def verify_duality(self, context: np.ndarray, role: np.ndarray,
                        value: np.ndarray) -> float:
        rv = self.bc.unbind(context, role)
        rr = self.bc.unbind(context, value)
        return (float(self.bc.similarity(rv, value))
                + float(self.bc.similarity(rr, role))) / 2.0

    def forge_with_sdls(self, context: np.ndarray, layer_id: int,
                         threshold: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
        clean = self.sdls_purify(context, threshold)
        return self.forge(clean, layer_id)

    def _default_context(self) -> np.ndarray:
        ctx = np.zeros((self.k, self.l), dtype=np.float32)
        ctx[:, 0] = 1.0
        return self.bc.discretize(ctx)

    # ── Forward (with cache for backward) ────────────────────────────────────

    def forge(
        self, context: np.ndarray, layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forge adapter. Returns (A, B). Use forge_with_cache for training."""
        A, B, _ = self.forge_with_cache(context, layer_id)
        return A, B

    def forge_with_cache(
        self, context: np.ndarray, layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple]:
        """Forge adapter with cache for backward pass.

        Returns (A, B, cache) where cache is needed by backward().
        """
        ctx_flat = self.bc.to_flat(context).astype(np.float32)

        # Project + LayerNorm (grilly GPU)
        ctx_proj_raw = _linear(ctx_flat, self.W_proj)
        ctx_proj, ln_mean, ln_var = _layernorm(ctx_proj_raw, self.ln_g, self.ln_b)

        # Concat with layer embedding
        layer_emb = self.layer_embeddings[layer_id]
        combined = np.concatenate([ctx_proj, layer_emb])

        # Generator MLP (grilly GPU)
        pre_act = _linear(combined, self.W_h, self.b_h)
        h = _gelu(pre_act)

        # Continuous mixing (NO softmax — prevents mode collapse)
        coeffs = _linear(h, self.W_coeff, self.b_coeff)

        # Mix basis adapters
        A = np.tensordot(coeffs, self.A_basis, axes=([0], [0])).astype(np.float32)
        B = np.tensordot(coeffs, self.B_basis, axes=([0], [0])).astype(np.float32)

        cache = (ctx_flat, ctx_proj_raw, ln_mean, ln_var,
                 layer_emb, combined, pre_act, h, coeffs)
        return A, B, cache

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(
        self, d_A: np.ndarray, d_B: np.ndarray,
        layer_id: int, cache: tuple,
    ) -> None:
        """Accumulate gradients from adapter loss.

        Args:
            d_A: Gradient w.r.t. forged A matrix (rank, d_target).
            d_B: Gradient w.r.t. forged B matrix (d_target, rank).
            layer_id: Which layer this adapter was forged for.
            cache: Cache from forge_with_cache().
        """
        (ctx_flat, ctx_proj_raw, ln_mean, ln_var,
         layer_emb, combined, pre_act, h, coeffs) = cache

        # ── Backprop through basis mixing ────────────────────────────────
        d_coeffs = np.zeros_like(coeffs)
        for i in range(self.n_basis):
            d_coeffs[i] = float(np.sum(d_A * self.A_basis[i])
                                + np.sum(d_B * self.B_basis[i]))
            self.grads["A_basis"][i] += coeffs[i] * d_A
            self.grads["B_basis"][i] += coeffs[i] * d_B

        # ── Backprop through coefficient layer ───────────────────────────
        d_h = (d_coeffs @ self.W_coeff).astype(np.float32)
        self.grads["W_coeff"] += np.outer(d_coeffs, h)
        self.grads["b_coeff"] += d_coeffs

        # ── Backprop through GELU (grilly GPU) ───────────────────────────
        d_pre_act = _gelu_backward(d_h, pre_act)

        # ── Backprop through MLP hidden layer ────────────────────────────
        self.grads["W_h"] += np.outer(d_pre_act, combined)
        self.grads["b_h"] += d_pre_act

        # ── Backprop to combined input ───────────────────────────────────
        d_combined = _linear(d_pre_act, self.W_h.T).ravel()
        d_ctx_proj = d_combined[:self.d_hidden]
        self.grads["layer_embeddings"][layer_id] += d_combined[self.d_hidden:]

        # ── Backprop through LayerNorm (grilly GPU) ──────────────────────
        d_ctx_proj_raw, d_ln_g, d_ln_b = _layernorm_backward(
            d_ctx_proj, ctx_proj_raw, ln_mean, ln_var, self.ln_g,
        )
        self.grads["ln_g"] += d_ln_g
        self.grads["ln_b"] += d_ln_b

        # ── Backprop through projection ──────────────────────────────────
        self.grads["W_proj"] += np.outer(d_ctx_proj_raw, ctx_flat)

    # ── Convenience ──────────────────────────────────────────────────────────

    def forge_all_layers(self, context: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        return [self.forge(context, i) for i in range(self.n_layers)]

    def apply_adapter(self, x: np.ndarray, base_output: np.ndarray,
                       A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if _bridge is not None:
            lora_out = np.asarray(
                _bridge.linear(np.asarray(_bridge.linear(x, A), dtype=np.float32), B),
                dtype=np.float32,
            )
        else:
            lora_out = ((x @ A.T) @ B.T).astype(np.float32)
        return (base_output + self.scale * lora_out).astype(np.float32)

    def memory_bytes(self) -> int:
        basis_mem = self.n_basis * self.rank * self.d_target * 4 * 2
        layer_emb_mem = self.n_layers * self.d_hidden * 4
        proj_mem = self.d_vsa * self.d_hidden * 4
        mlp_mem = (self.d_hidden * 2 * self.d_hidden + self.d_hidden) * 4
        coeff_mem = (self.n_basis * self.d_hidden + self.n_basis) * 4
        ln_mem = self.d_hidden * 4 * 2
        return basis_mem + layer_emb_mem + proj_mem + mlp_mem + coeff_mem + ln_mem

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)
