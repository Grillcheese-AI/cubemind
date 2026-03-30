"""Perceiver cross-attention encoder for image feature extraction.

Funnels high-dimensional image patches (e.g. 4096 patches × 256D) into a
compact latent representation via cross-attention, then pools to a single
dense vector for downstream VSA projection.

Pipeline:
  image_patches (N, d_patch) → cross-attention with learned latents (n_latents, d_model)
  → pooled dense vector (d_model,)

Uses grilly flash_attention2 when available, numpy fallback otherwise.
"""

from __future__ import annotations

import math

import numpy as np

# GPU bridge (grilly)
_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - m)
    return ex / (ex.sum(axis=axis, keepdims=True) + 1e-8)


class PerceiverEncoder:
    """Perceiver cross-attention bottleneck for image → dense vector.

    Instead of letting all image patches attend to each other (O(N^2)),
    a small fixed set of latent vectors (Q) cross-attends to the image
    patches (K, V). This is O(n_latents * N) — linear in patch count.

    Args:
        d_model:      Latent/patch dimension.
        n_latents:    Number of latent vectors (bottleneck size).
        n_heads:      Number of attention heads.
        n_layers:     Number of cross-attention + self-attention layers.
        max_patches:  Max number of patches (for positional encoding).
        seed:         Random seed for weight initialization.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_latents: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_patches: int = 4096,
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_layers = n_layers
        self.max_patches = max_patches

        rng = np.random.default_rng(seed)

        # Learned latent array (the "bottleneck")
        self.latents = (
            rng.standard_normal((n_latents, d_model)) * 0.02
        ).astype(np.float32)

        # Learned positional embeddings for image patches
        # Stamps each patch with its spatial location so the Perceiver
        # knows geometry (attention is a set operation by default)
        self.position_embeddings = (
            rng.standard_normal((max_patches, d_model)) * 0.02
        ).astype(np.float32)

        # Per-layer weights
        std = 1.0 / math.sqrt(d_model)
        self.layers = []
        for _ in range(n_layers):
            layer = {
                # Cross-attention: latents attend to image patches
                'W_Q_cross': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_K_cross': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_V_cross': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                # Self-attention among latents
                'W_Q_self': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_K_self': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_V_self': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
            }
            self.layers.append(layer)

    def _cross_attention(
        self, Q_input: np.ndarray, KV_input: np.ndarray, layer: dict,
    ) -> np.ndarray:
        """Cross-attention: Q from latents, K/V from image patches."""
        # Try grilly GPU path
        if _bridge is not None:
            try:
                result = _bridge.flash_attention2(
                    q=Q_input @ layer['W_Q_cross'],
                    k=KV_input @ layer['W_K_cross'],
                    v=KV_input @ layer['W_V_cross'],
                )
                if result is not None:
                    return np.asarray(result, dtype=np.float32)
            except Exception:
                pass

        # CPU fallback: standard scaled dot-product attention
        Q = Q_input @ layer['W_Q_cross']   # (n_latents, d_model)
        K = KV_input @ layer['W_K_cross']  # (N_patches, d_model)
        V = KV_input @ layer['W_V_cross']  # (N_patches, d_model)

        scale = math.sqrt(self.d_head)
        scores = (Q @ K.T) / scale  # (n_latents, N_patches)
        attn = _softmax(scores, axis=-1)
        return (attn @ V).astype(np.float32)

    def _self_attention(
        self, X: np.ndarray, layer: dict,
    ) -> np.ndarray:
        """Self-attention among latent vectors."""
        if _bridge is not None:
            try:
                result = _bridge.flash_attention2(
                    q=X @ layer['W_Q_self'],
                    k=X @ layer['W_K_self'],
                    v=X @ layer['W_V_self'],
                )
                if result is not None:
                    return np.asarray(result, dtype=np.float32)
            except Exception:
                pass

        Q = X @ layer['W_Q_self']
        K = X @ layer['W_K_self']
        V = X @ layer['W_V_self']

        scale = math.sqrt(self.d_head)
        scores = (Q @ K.T) / scale
        attn = _softmax(scores, axis=-1)
        return (attn @ V).astype(np.float32)

    def encode(self, image_patches: np.ndarray) -> np.ndarray:
        """Encode image patches into a single dense vector.

        Args:
            image_patches: (N_patches, d_model) float32 array.
                           Can be raw pixel patches or CNN features.

        Returns:
            Dense vector (d_model,) float32 — pooled latent representation.
        """
        image_patches = np.asarray(image_patches, dtype=np.float32)
        n_patches = image_patches.shape[0]

        # Inject positional encoding (spatial awareness)
        if n_patches <= self.max_patches:
            image_patches = image_patches + self.position_embeddings[:n_patches]

        latents = self.latents.copy()

        for layer in self.layers:
            # Cross-attention: latents query the image
            cross_out = self._cross_attention(latents, image_patches, layer)
            latents = latents + cross_out  # Residual

            # Self-attention among latents
            self_out = self._self_attention(latents, layer)
            latents = latents + self_out  # Residual

        # Mean pool latents → single dense vector
        return np.mean(latents, axis=0).astype(np.float32)

    def encode_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode multiple images.

        Args:
            images: List of (N_patches, d_model) arrays.

        Returns:
            (batch, d_model) float32 array.
        """
        return np.stack([self.encode(img) for img in images])
