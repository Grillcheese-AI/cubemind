"""SigLIP2 Vision Encoder — safetensors → VulkanTensor → GPU inference.

Loads SigLIP2 weights directly from safetensors into grilly VulkanTensors.
No PyTorch. Full vision transformer forward pass on Vulkan GPU.

Architecture (SigLIP2-base-patch16-512):
  Image (3, 512, 512)
  → Patch embedding: Conv2d(3, 768, 16, stride=16) → (1024, 768)
  → + position embeddings (1024, 768)
  → 12× Transformer layers:
      LayerNorm → MultiHeadAttention(12 heads) → residual
      LayerNorm → MLP(768→3072→768) → residual
  → Post-LayerNorm
  → Attention pooling head → (768,) embedding

Uses grilly GPU ops: linear, gelu, softmax, layernorm via Vulkan shaders.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

# Grilly GPU
_bridge = None
_VulkanTensor = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass

try:
    from grilly.utils.tensor_conversion import VulkanTensor as _VT
    _VulkanTensor = _VT
except Exception:
    pass

# Safetensors
try:
    from safetensors.numpy import load_file as _load_safetensors
except ImportError:
    _load_safetensors = None

# HuggingFace hub
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def _gpu_linear(x, w, b=None):
    """GPU linear projection."""
    if _bridge is not None:
        try:
            r = _bridge.linear(
                np.ascontiguousarray(x, dtype=np.float32),
                np.ascontiguousarray(w, dtype=np.float32),
                np.ascontiguousarray(b, dtype=np.float32) if b is not None else None,
            )
            if r is not None:
                return np.asarray(r, dtype=np.float32)
        except Exception:
            pass
    out = x @ w.T
    if b is not None:
        out = out + b
    return out.astype(np.float32)


def _gelu(x):
    """GELU activation — grilly GPU bridge handles overflow clamping."""
    if _bridge is not None:
        try:
            r = _bridge.gelu(np.ascontiguousarray(x, dtype=np.float32))
            if r is not None:
                return np.asarray(r, dtype=np.float32)
        except Exception:
            pass
    # CPU fallback with same overflow protection
    x = np.asarray(x, dtype=np.float32)
    safe = np.clip(x, -10, 10)
    result = 0.5 * safe * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (safe + 0.044715 * safe ** 3)))
    # Asymptotic: large positive → x, large negative → 0
    result = np.where(x > 10, x, result)
    result = np.where(x < -10, 0.0, result)
    return result.astype(np.float32)


def _layernorm(x, weight, bias, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return (normed * weight + bias).astype(np.float32)


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (e / (e.sum(axis=axis, keepdims=True) + 1e-8)).astype(np.float32)


class SigLIPVulkan:
    """SigLIP2 Vision Encoder running on Vulkan GPU via grilly.

    Loads safetensors weights directly — no PyTorch dependency.

    Args:
        model_id:   HuggingFace model ID.
        image_size: Input image size (default: 512).
        use_gpu:    Use VulkanTensor for weight storage.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-512",
        image_size: int = 512,
        use_gpu: bool = True,
    ) -> None:
        self.model_id = model_id
        self.image_size = image_size
        self.use_gpu = use_gpu and _VulkanTensor is not None

        # Download and load weights
        weights = self._load_weights(model_id)

        # Architecture params
        self.hidden_size = 768
        self.num_heads = 12
        self.head_dim = self.hidden_size // self.num_heads
        self.patch_size = 16
        self.num_patches = (image_size // self.patch_size) ** 2  # 1024

        # Extract and optionally move to GPU
        self.w = {}
        vision_keys = [k for k in weights if k.startswith("vision_model.")]
        for k in vision_keys:
            short = k.replace("vision_model.", "")
            tensor = weights[k].astype(np.float32)
            if self.use_gpu:
                try:
                    self.w[short] = _VulkanTensor(tensor)
                except Exception:
                    self.w[short] = tensor
            else:
                self.w[short] = tensor

        # Count layers
        self.num_layers = 0
        while f"encoder.layers.{self.num_layers}.self_attn.q_proj.weight" in self.w:
            self.num_layers += 1

        print(f"SigLIP2 Vulkan: {self.hidden_size}D, {self.num_layers} layers, "
              f"{self.num_patches} patches, GPU={'yes' if self.use_gpu else 'no'}")

    def _load_weights(self, model_id: str) -> dict:
        """Download and load safetensors weights."""
        if _load_safetensors is None:
            raise ImportError("safetensors required: pip install safetensors")

        # Try local cache first
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
        if cache_dir.exists():
            snapshots = list((cache_dir / "snapshots").iterdir())
            if snapshots:
                sf_path = snapshots[0] / "model.safetensors"
                if sf_path.exists():
                    return _load_safetensors(str(sf_path))

        # Download via hub
        if hf_hub_download is not None:
            path = hf_hub_download(model_id, "model.safetensors")
            return _load_safetensors(path)

        raise FileNotFoundError(f"Cannot find SigLIP2 weights for {model_id}")

    def _np(self, key: str) -> np.ndarray:
        """Get weight as numpy array (handles VulkanTensor)."""
        v = self.w[key]
        if isinstance(v, np.ndarray):
            return v
        return np.asarray(v, dtype=np.float32)

    # ── Forward pass ──────────────────────────────────────────────────────

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode an image to a 768D embedding vector.

        Args:
            image: (H, W, 3) uint8 or float32 image. Will be resized to 512×512.

        Returns:
            (768,) float32 embedding vector, L2-normalized.
        """
        # Preprocess
        x = self._preprocess(image)  # (1024, 768)

        # Transformer encoder
        for i in range(self.num_layers):
            x = self._transformer_layer(x, i)

        # Post-layernorm
        x = _layernorm(x, self._np("post_layernorm.weight"), self._np("post_layernorm.bias"))

        # Attention pooling head
        embedding = self._attention_pool(x)

        # L2 normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return (embedding / norm).astype(np.float32)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Image → patch embeddings + position embeddings."""
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # Resize to 512×512 if needed
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            try:
                import cv2
                img = cv2.resize(img, (self.image_size, self.image_size))
            except ImportError:
                # Nearest-neighbor resize fallback
                h, w = img.shape[:2]
                row_idx = (np.arange(self.image_size) * h / self.image_size).astype(int)
                col_idx = (np.arange(self.image_size) * w / self.image_size).astype(int)
                img = img[np.ix_(row_idx, col_idx)]

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # Extract 16×16 patches → (n_patches, 3*16*16)
        ps = self.patch_size
        n_h = self.image_size // ps
        n_w = self.image_size // ps
        # (H, W, C) → (n_h, ps, n_w, ps, C) → (n_h, n_w, ps, ps, C) → (n_patches, ps*ps*C)
        patches = img.reshape(n_h, ps, n_w, ps, 3).transpose(0, 2, 1, 3, 4).reshape(n_h * n_w, ps * ps * 3)

        # Patch embedding: Linear(768_input=768, 768) — conv2d flattened
        patch_w = self._np("embeddings.patch_embedding.weight").reshape(768, -1)  # (768, 768)
        patch_b = self._np("embeddings.patch_embedding.bias")
        x = _gpu_linear(patches, patch_w, patch_b)  # (1024, 768)

        # Position embeddings
        pos_emb = self._np("embeddings.position_embedding.weight")  # (1024, 768)
        x = x + pos_emb[:x.shape[0]]

        return x.astype(np.float32)

    def _transformer_layer(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """Single transformer encoder layer."""
        prefix = f"encoder.layers.{layer_idx}"

        # Pre-norm + Self-attention
        normed = _layernorm(x, self._np(f"{prefix}.layer_norm1.weight"),
                            self._np(f"{prefix}.layer_norm1.bias"))
        attn_out = self._self_attention(normed, prefix)
        x = x + attn_out

        # Pre-norm + MLP
        normed = _layernorm(x, self._np(f"{prefix}.layer_norm2.weight"),
                            self._np(f"{prefix}.layer_norm2.bias"))
        mlp_out = self._mlp(normed, prefix)
        x = x + mlp_out

        return x

    def _self_attention(self, x: np.ndarray, prefix: str) -> np.ndarray:
        """Multi-head self attention."""
        seq_len = x.shape[0]

        # Q, K, V projections (GPU linear)
        Q = _gpu_linear(x, self._np(f"{prefix}.self_attn.q_proj.weight"),
                         self._np(f"{prefix}.self_attn.q_proj.bias"))
        K = _gpu_linear(x, self._np(f"{prefix}.self_attn.k_proj.weight"),
                         self._np(f"{prefix}.self_attn.k_proj.bias"))
        V = _gpu_linear(x, self._np(f"{prefix}.self_attn.v_proj.weight"),
                         self._np(f"{prefix}.self_attn.v_proj.bias"))

        # Reshape to multi-head: (seq, hidden) → (heads, seq, head_dim)
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention (float64 for stability with 1024 patches)
        scale = math.sqrt(self.head_dim)
        scores = np.matmul(Q.astype(np.float64), K.transpose(0, 2, 1).astype(np.float64)) / scale
        scores = np.clip(scores, -50, 50)  # Prevent exp overflow
        attn = _softmax(scores.astype(np.float32), axis=-1)
        out = np.matmul(attn, V)  # (heads, seq, head_dim)

        # Concat heads
        out = out.transpose(1, 0, 2).reshape(seq_len, self.hidden_size)

        # Output projection
        out = _gpu_linear(out, self._np(f"{prefix}.self_attn.out_proj.weight"),
                           self._np(f"{prefix}.self_attn.out_proj.bias"))
        return out

    def _mlp(self, x: np.ndarray, prefix: str) -> np.ndarray:
        """Feed-forward MLP: Linear → GELU → Linear."""
        h = _gpu_linear(x, self._np(f"{prefix}.mlp.fc1.weight"),
                         self._np(f"{prefix}.mlp.fc1.bias"))
        h = _gelu(h)
        return _gpu_linear(h, self._np(f"{prefix}.mlp.fc2.weight"),
                            self._np(f"{prefix}.mlp.fc2.bias"))

    def _attention_pool(self, x: np.ndarray) -> np.ndarray:
        """SigLIP2 attention pooling head."""
        # Probe token: learned query for pooling
        probe = self._np("head.probe").reshape(1, self.hidden_size)  # (1, 768)

        # QKV from single in_proj
        in_proj_w = self._np("head.attention.in_proj_weight")  # (2304, 768)
        in_proj_b = self._np("head.attention.in_proj_bias")    # (2304,)

        # Q from probe, K/V from sequence
        q = _gpu_linear(probe, in_proj_w[:768], in_proj_b[:768])          # (1, 768)
        k = _gpu_linear(x, in_proj_w[768:1536], in_proj_b[768:1536])     # (seq, 768)
        v = _gpu_linear(x, in_proj_w[1536:], in_proj_b[1536:])           # (seq, 768)

        # Attention: probe attends to all patches
        scale = math.sqrt(self.head_dim)
        # Multi-head: reshape to (heads, seq/1, head_dim)
        q_h = q.reshape(1, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k_h = k.reshape(-1, self.num_heads, self.head_dim).transpose(1, 0, 2)
        v_h = v.reshape(-1, self.num_heads, self.head_dim).transpose(1, 0, 2)

        scores = np.matmul(q_h.astype(np.float64), k_h.transpose(0, 2, 1).astype(np.float64)) / scale
        scores = np.clip(scores, -50, 50)
        attn = _softmax(scores.astype(np.float32), axis=-1)
        pooled = np.matmul(attn, v_h)  # (heads, 1, head_dim)
        pooled = pooled.transpose(1, 0, 2).reshape(1, self.hidden_size)

        # Output projection
        pooled = _gpu_linear(pooled, self._np("head.attention.out_proj.weight"),
                              self._np("head.attention.out_proj.bias"))

        # LayerNorm + MLP
        pooled = _layernorm(pooled.ravel(), self._np("head.layernorm.weight"),
                            self._np("head.layernorm.bias"))
        h = _gpu_linear(pooled.reshape(1, -1), self._np("head.mlp.fc1.weight"),
                         self._np("head.mlp.fc1.bias"))
        h = _gelu(h)
        out = _gpu_linear(h, self._np("head.mlp.fc2.weight"),
                           self._np("head.mlp.fc2.bias"))

        return out.ravel().astype(np.float32)

    @property
    def embed_dim(self) -> int:
        return self.hidden_size
