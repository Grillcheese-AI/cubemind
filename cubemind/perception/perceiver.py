"""Perceiver cross-attention encoder for image feature extraction.

Funnels high-dimensional image patches (e.g. 4096 patches × 256D) into a
compact latent representation via cross-attention, then pools to a single
dense vector for downstream VSA projection.

Pipeline:
  image_patches (N, d_patch) → cross-attention with learned latents (n_latents, d_model)
  → pooled dense vector (d_model,)

GPU paths (in priority order):
  1. grilly perceiver_encode (native batched) — entire N-layer pipeline in
     ONE Vulkan command buffer submission. Ping-pong VRAM, zero Python
     round-trips. 10-50x faster than per-layer dispatch.
  2. grilly perceiver_cross_attn_gpu — dedicated register-pinned shader,
     1 thread per latent, streaming K/V, online softmax, zero LDS.
  3. grilly flash_attention2 — generic flash attention fallback.
  4. numpy CPU — standard scaled dot-product.
"""

from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger(__name__)

# GPU bridge (grilly)
_bridge = None
_perceiver_gpu = None   # Dedicated perceiver cross-attention shader
_perceiver_native = None  # Native batched encoder (single submit)
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
    import grilly_core as _gc
    if hasattr(_gc, 'perceiver_cross_attn_gpu'):
        _perceiver_gpu = _gc
    if hasattr(_gc, 'perceiver_encode'):
        _perceiver_native = _gc
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
        self.position_embeddings = (
            rng.standard_normal((max_patches, d_model)) * 0.02
        ).astype(np.float32)

        # Per-layer weights
        std = 1.0 / math.sqrt(d_model)
        self.layers = []
        for _ in range(n_layers):
            layer = {
                'W_Q_cross': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_K_cross': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_V_cross': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_Q_self': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_K_self': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
                'W_V_self': rng.normal(0, std, (d_model, d_model)).astype(np.float32),
            }
            self.layers.append(layer)

        # Native GPU handle (lazy upload on first encode)
        self._gpu_handle = None
        self._gpu_device = None

    def _upload_to_gpu(self) -> bool:
        """Upload all weights to GPU for native batched encoding."""
        if _perceiver_native is None:
            return False

        try:
            device = _perceiver_native.Device()
            device.load_shaders('C:/Users/grill/Documents/GitHub/grilly/shaders/spv')

            # Build weight list in expected order
            weights = []
            for layer in self.layers:
                weights.append(layer['W_Q_cross'].ravel())
                weights.append(layer['W_K_cross'].ravel())
                weights.append(layer['W_V_cross'].ravel())
                weights.append(layer['W_Q_self'].ravel())
                weights.append(layer['W_K_self'].ravel())
                weights.append(layer['W_V_self'].ravel())
            weights.append(self.latents.ravel())
            weights.append(self.position_embeddings.ravel())

            self._gpu_handle = _perceiver_native.perceiver_upload_weights(
                device, weights,
                n_latents=self.n_latents,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                max_patches=self.max_patches,
            )
            self._gpu_device = device
            log.info("Perceiver weights uploaded to GPU (native batched encoder)")
            return True
        except Exception as e:
            log.debug("Native perceiver upload failed: %s", e)
            return False

    def _cross_attention(
        self, Q_input: np.ndarray, KV_input: np.ndarray, layer: dict,
    ) -> np.ndarray:
        """Cross-attention: Q from latents, K/V from image patches."""
        Q = (Q_input @ layer['W_Q_cross']).astype(np.float32)
        K = (KV_input @ layer['W_K_cross']).astype(np.float32)
        V = (KV_input @ layer['W_V_cross']).astype(np.float32)

        # Path 1: Dedicated perceiver shader (register-pinned Q, streaming K/V)
        if _perceiver_gpu is not None and self.d_head <= 64 and self.d_head % 4 == 0:
            try:
                results = []
                for h in range(self.n_heads):
                    s = h * self.d_head
                    e = s + self.d_head
                    out_h = _perceiver_gpu.perceiver_cross_attn_gpu(
                        _perceiver_gpu.Device(), Q[:, s:e], K[:, s:e], V[:, s:e],
                    )
                    results.append(np.asarray(out_h, dtype=np.float32))
                return np.concatenate(results, axis=-1)
            except Exception:
                pass

        # Path 2: grilly flash_attention2 (generic)
        if _bridge is not None:
            try:
                result = _bridge.flash_attention2(q=Q, k=K, v=V)
                if result is not None:
                    return np.asarray(result, dtype=np.float32)
            except Exception:
                pass

        # Path 3: CPU fallback
        scale = math.sqrt(self.d_head)
        scores = (Q @ K.T) / scale
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

        Returns:
            Dense vector (d_model,) float32 — pooled latent representation.
        """
        image_patches = np.asarray(image_patches, dtype=np.float32)
        n_patches = image_patches.shape[0]

        # Inject positional encoding (spatial awareness)
        if n_patches <= self.max_patches:
            image_patches = image_patches + self.position_embeddings[:n_patches]

        # ── Path 0: Native batched encoder (single GPU submit) ────────
        # Entire pipeline in ONE Vulkan command buffer. Zero Python loops.
        if _perceiver_native is not None and n_patches <= self.max_patches:
            if self._gpu_handle is None:
                self._upload_to_gpu()
            if self._gpu_handle is not None:
                try:
                    result = _perceiver_native.perceiver_encode(
                        self._gpu_device, self._gpu_handle, image_patches,
                    )
                    return np.asarray(result, dtype=np.float32)
                except Exception as e:
                    log.debug("Native perceiver encode failed: %s", e)

        # ── Fallback: per-layer Python dispatch ───────────────────────
        latents = self.latents.copy()

        for layer in self.layers:
            cross_out = self._cross_attention(latents, image_patches, layer)
            latents = latents + cross_out

            self_out = self._self_attention(latents, layer)
            latents = latents + self_out

        return np.mean(latents, axis=0).astype(np.float32)

    def encode_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode multiple images.

        Args:
            images: List of (N_patches, d_model) arrays.

        Returns:
            (batch, d_model) float32 array.
        """
        return np.stack([self.encode(img) for img in images])
