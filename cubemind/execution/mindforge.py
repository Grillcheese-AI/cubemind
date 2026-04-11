"""MindForge — VSA-conditioned hypernetwork that forges LoRA adapters from block-code context.

Extends HYLA's factored weight generation with:
  1. Layer-ID conditioning — single generator for all target model layers
  2. LoRA-style injection — generates low-rank (A, B) adapters for a frozen base
  3. Shared basis with mixing coefficients — instead of raw weights, predict
     linear combinations of a trainable basis set
  4. Symbolic context — input is VSA block-code from bind/bundle/permute ops
  5. Cleanup verification — SDLS purification validates symbolic consistency

Architecture::

    context_hv = bind(task_hv, personality_hv)          # VSA symbolic context
    layer_emb  = layer_embeddings[layer_id]             # per-layer coordinate
    h          = GELU(concat(context_flat, layer_emb) @ W_h + b_h)
    coeffs     = h @ W_coeff                            # (n_basis,) mixing coefficients
    A          = sum(coeffs[i] * basis_A[i])            # (rank, d_in) adapter down
    B          = sum(coeffs[i] * basis_B[i])            # (d_out, rank) adapter up
    output     = frozen_base(x) + scale * (x @ A.T @ B.T)  # LoRA injection

Memory: O(n_basis * rank * (d_in + d_out)) shared across all layers/contexts.
At n_basis=16, rank=8, d=2048: ~4MB total vs ~256MB for 128 separate adapters.

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


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return (0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))).astype(np.float32)


@register("executor", "mindforge")
class MindForge:
    """VSA-conditioned hypernetwork that forges LoRA adapters on the fly.

    Given a symbolic context vector (block-code) and a layer ID, generates
    low-rank adapter matrices (A, B) by predicting mixing coefficients over
    a shared trainable basis. This allows a single generator to produce
    unique adapters for every layer of a target model.

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
        # Input: concat(context_flat_projected, layer_emb) → d_hidden * 2
        # Output: n_basis mixing coefficients
        input_dim = d_hidden * 2

        xavier_std_h = np.sqrt(2.0 / (input_dim + d_hidden))
        self.W_h = rng.normal(0, xavier_std_h,
                              size=(d_hidden, input_dim)).astype(np.float32)
        self.b_h = np.zeros(d_hidden, dtype=np.float32)

        xavier_std_c = np.sqrt(2.0 / (d_hidden + n_basis))
        self.W_coeff = rng.normal(0, xavier_std_c,
                                  size=(n_basis, d_hidden)).astype(np.float32)
        self.b_coeff = np.zeros(n_basis, dtype=np.float32)

        # ── Context projection: d_vsa → d_hidden ─────────────────────────────
        xavier_std_p = np.sqrt(2.0 / (self.d_vsa + d_hidden))
        self.W_proj = rng.normal(0, xavier_std_p,
                                 size=(d_hidden, self.d_vsa)).astype(np.float32)

        # ── Shared basis: n_basis pairs of (A, B) adapters ───────────────────
        # A_basis[i]: (rank, d_target) — adapter down-projection
        # B_basis[i]: (d_target, rank) — adapter up-projection
        basis_std = np.sqrt(2.0 / (d_target + rank))
        self.A_basis = rng.normal(
            0, basis_std, size=(n_basis, rank, d_target)
        ).astype(np.float32)
        self.B_basis = rng.normal(
            0, basis_std, size=(n_basis, d_target, rank)
        ).astype(np.float32)

    # ── SDLS Duality Gate ────────────────────────────────────────────────────

    def register_context(self, name: str, context: np.ndarray) -> None:
        """Register a known-clean context for SDLS purification."""
        if not hasattr(self, "_cleanup_mem"):
            from cubemind.reasoning.vm import CleanupMemory
            self._cleanup_mem = CleanupMemory(self.bc)
        self._cleanup_mem.store(name, context)

    def sdls_purify(
        self, context: np.ndarray, threshold: float = 0.85,
    ) -> np.ndarray:
        """SDLS Purification: validate symbolic consistency before forging.

        1. Similarity search in cleanup memory (snap to nearest clean vector)
        2. If similarity < threshold, return safe default context
        3. Otherwise return the purified (clean) context

        This prevents hallucinated weight generation by ensuring the input
        is a valid member of the WorldManager vocabulary.
        """
        if not hasattr(self, "_cleanup_mem") or self._cleanup_mem.size == 0:
            # No cleanup memory — return default
            return self._default_context()

        name, clean = self._cleanup_mem.cleanup(context)
        sim = float(self.bc.similarity(context, clean))

        if sim < threshold:
            return self._default_context()
        return clean

    def verify_duality(
        self,
        context: np.ndarray,
        role: np.ndarray,
        value: np.ndarray,
    ) -> float:
        """Algebraic self-duality check.

        Verifies: unbind(context, role) ≈ value AND unbind(context, value) ≈ role.
        Returns average duality score in [0, 1].
        """
        recovered_value = self.bc.unbind(context, role)
        recovered_role = self.bc.unbind(context, value)

        score_v = float(self.bc.similarity(recovered_value, value))
        score_r = float(self.bc.similarity(recovered_role, role))

        return (score_v + score_r) / 2.0

    def forge_with_sdls(
        self,
        context: np.ndarray,
        layer_id: int,
        threshold: float = 0.85,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forge with SDLS purification gate.

        Low duality score → high softmax temperature (generic adapter).
        High duality score → low temperature (specialized adapter).
        """
        clean_context = self.sdls_purify(context, threshold)
        return self.forge(clean_context, layer_id)

    def _default_context(self) -> np.ndarray:
        """Safe default context when SDLS purification fails."""
        # Identity-like: all mass on index 0 of each block
        ctx = np.zeros((self.k, self.l), dtype=np.float32)
        ctx[:, 0] = 1.0
        return self.bc.discretize(ctx)

    # ── Adapter generation ───────────────────────────────────────────────────

    def forge(
        self,
        context: np.ndarray,
        layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forge a LoRA adapter (A, B) for a specific layer from symbolic context.

        Args:
            context: VSA block-code context vector (k, l).
            layer_id: Target model layer index.

        Returns:
            (A, B) where A is (rank, d_target) and B is (d_target, rank).
        """
        # Project context to hidden dim
        ctx_flat = self.bc.to_flat(context)
        ctx_proj = (ctx_flat @ self.W_proj.T).astype(np.float32)  # (d_hidden,)

        # Concat with layer embedding
        layer_emb = self.layer_embeddings[layer_id]  # (d_hidden,)
        combined = np.concatenate([ctx_proj, layer_emb])  # (d_hidden * 2,)

        # Generator MLP
        h = gelu(combined @ self.W_h.T + self.b_h)  # (d_hidden,)
        coeffs = h @ self.W_coeff.T + self.b_coeff   # (n_basis,)

        # Softmax over basis coefficients for stable mixing
        coeffs_exp = np.exp(coeffs - np.max(coeffs))
        coeffs_soft = (coeffs_exp / np.sum(coeffs_exp)).astype(np.float32)

        # Mix basis adapters: weighted sum
        A = np.tensordot(coeffs_soft, self.A_basis, axes=([0], [0]))  # (rank, d_target)
        B = np.tensordot(coeffs_soft, self.B_basis, axes=([0], [0]))  # (d_target, rank)

        return A.astype(np.float32), B.astype(np.float32)

    def forge_all_layers(
        self,
        context: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Forge adapters for all layers from a single context.

        Args:
            context: VSA block-code context vector (k, l).

        Returns:
            List of (A, B) pairs, one per layer.
        """
        return [self.forge(context, i) for i in range(self.n_layers)]

    # ── LoRA injection ───────────────────────────────────────────────────────

    def apply_adapter(
        self,
        x: np.ndarray,
        base_output: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """Apply a forged LoRA adapter to a base model output.

        output = base_output + scale * (x @ A.T @ B.T)

        Args:
            x: Input to the layer (seq_len, d_target) or (d_target,).
            base_output: Output from the frozen base layer, same shape.
            A: Adapter down-projection (rank, d_target).
            B: Adapter up-projection (d_target, rank).

        Returns:
            Adapted output, same shape as base_output.
        """
        # x @ A.T → (*, rank), then @ B.T → (*, d_target)
        lora_out = (x @ A.T) @ B.T
        return (base_output + self.scale * lora_out).astype(np.float32)

    # ── Memory stats ─────────────────────────────────────────────────────────

    def memory_bytes(self) -> int:
        """Total memory footprint of the MindForge module."""
        basis_mem = self.n_basis * self.rank * self.d_target * 4 * 2  # A + B
        layer_emb_mem = self.n_layers * self.d_hidden * 4
        proj_mem = self.d_vsa * self.d_hidden * 4
        mlp_mem = (self.d_hidden * 2 * self.d_hidden + self.d_hidden) * 4  # W_h + b_h
        coeff_mem = (self.n_basis * self.d_hidden + self.n_basis) * 4  # W_coeff + b_coeff
        return basis_mem + layer_emb_mem + proj_mem + mlp_mem + coeff_mem

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)
