"""Image VSA Pipeline — full cognitive image processing engine.

End-to-end pipeline:
  Raw image → patch extraction → Perceiver bottleneck → LSH projection
  → binarize & pack → store/retrieve from ContinuousItemMemory.

No gradient training for the VSA side. The Perceiver weights can be
frozen (pretrained CNN features) or learned separately.

This is the module that enables I-RAVEN-X image processing with cubemind.
"""

from __future__ import annotations

import math

import numpy as np

from cubemind.ops.vsa_bridge import (
    LSHProjector,
    ContinuousItemMemory,
    binarize_and_pack,
    unpack_to_float,
    hamming_similarity,
)
from cubemind.perception.perceiver import PerceiverEncoder

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 80
    L_BLOCK = 128


class ImageVSAPipeline:
    """Full image → binary VSA pipeline with continuous learning.

    Architecture:
      1. Patch extraction: image → (N_patches, d_patch)
      2. Perceiver: (N_patches, d_patch) → (d_model,) dense vector
      3. LSH: (d_model,) → (d_vsa,) continuous projection
      4. Binarize & pack: (d_vsa,) → (words_per_vec,) uint32
      5. Store/retrieve from ContinuousItemMemory

    Args:
        d_patch:      Dimension of each image patch.
        d_model:      Perceiver latent dimension.
        d_vsa:        Binary VSA dimension (K_BLOCKS * L_BLOCK).
        n_latents:    Number of Perceiver latent vectors.
        n_heads:      Number of attention heads in Perceiver.
        n_layers:     Number of Perceiver layers.
        max_concepts: Maximum concepts in item memory.
        patch_size:   Pixel patch size (patch_size × patch_size).
        seed:         Random seed.
    """

    def __init__(
        self,
        d_patch: int = 256,
        d_model: int = 256,
        d_vsa: int = K_BLOCKS * L_BLOCK,
        n_latents: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_concepts: int = 100000,
        patch_size: int = 8,
        seed: int = 42,
    ) -> None:
        self.d_patch = d_patch
        self.d_model = d_model
        self.d_vsa = d_vsa
        self.patch_size = patch_size

        # Perceiver bottleneck
        self.perceiver = PerceiverEncoder(
            d_model=d_model,
            n_latents=n_latents,
            n_heads=n_heads,
            n_layers=n_layers,
            seed=seed,
        )

        # LSH random projection: d_model → d_vsa
        self.lsh = LSHProjector(d_input=d_model, d_output=d_vsa, seed=seed + 1)

        # Continuous-learning item memory
        self.memory = ContinuousItemMemory(d_vsa=d_vsa, max_capacity=max_concepts)

    # ── ImageNet normalization constants ─────────────────────────────────
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ── Patch extraction ──────────────────────────────────────────────────

    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """Extract non-overlapping patches from an image with normalization.

        Applies ImageNet normalization for color images, then slices into
        a grid and projects each patch to d_patch dimensions.

        Args:
            image: (H, W) grayscale or (H, W, C) color image.
                   Values should be in [0, 1] or [0, 255] range.

        Returns:
            (N_patches, d_patch) float32 array.
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        H, W, C = image.shape

        # Normalize to [0, 1] if needed
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # ImageNet normalization for RGB images
        if C == 3:
            img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD

        ps = self.patch_size

        # Crop to multiple of patch_size
        H_crop = (H // ps) * ps
        W_crop = (W // ps) * ps
        img = img[:H_crop, :W_crop, :]

        # Reshape into patches: (n_h, n_w, ps, ps, C) → (N_patches, ps*ps*C)
        n_h, n_w = H_crop // ps, W_crop // ps
        patches = img.reshape(n_h, ps, n_w, ps, C)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(n_h * n_w, ps * ps * C)

        # Linear projection: raw_patch_dim → d_patch
        patch_dim = ps * ps * C
        if patch_dim == self.d_patch:
            return patches.astype(np.float32)

        if not hasattr(self, '_patch_proj'):
            rng = np.random.default_rng(99)
            std = 1.0 / math.sqrt(patch_dim)
            self._patch_proj = rng.normal(
                0, std, (patch_dim, self.d_patch),
            ).astype(np.float32)

        return (patches @ self._patch_proj).astype(np.float32)

    # ── Core pipeline ─────────────────────────────────────────────────────

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Full pipeline: image → packed binary VSA vector.

        Args:
            image: (H, W) or (H, W, C) numpy array.

        Returns:
            (words_per_vec,) uint32 — packed binary VSA vector.
        """
        patches = self.extract_patches(image)
        dense = self.perceiver.encode(patches)
        projected = self.lsh.project(dense)
        return binarize_and_pack(projected)

    def encode_patches(self, patches: np.ndarray) -> np.ndarray:
        """Pipeline from pre-extracted patches → packed binary VSA.

        Args:
            patches: (N_patches, d_patch) float32.

        Returns:
            (words_per_vec,) uint32.
        """
        dense = self.perceiver.encode(patches)
        projected = self.lsh.project(dense)
        return binarize_and_pack(projected)

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        """Pipeline from pre-computed dense features → packed binary VSA.

        Skips the Perceiver — useful when features come from an external
        CNN (ResNet, DenseNet) or are already pooled.

        Args:
            features: (d_model,) float32 dense vector.

        Returns:
            (words_per_vec,) uint32.
        """
        projected = self.lsh.project(features)
        return binarize_and_pack(projected)

    # ── Learn / Retrieve ──────────────────────────────────────────────────

    def learn(self, image: np.ndarray, label: str = "") -> int:
        """Learn a new concept from an image. No backprop.

        Args:
            image: (H, W) or (H, W, C) numpy array.
            label: Human-readable label.

        Returns:
            Concept ID.
        """
        packed = self.encode_image(image)
        return self.memory.learn(packed, label)

    def learn_from_patches(self, patches: np.ndarray, label: str = "") -> int:
        """Learn from pre-extracted patches."""
        packed = self.encode_patches(patches)
        return self.memory.learn(packed, label)

    def learn_from_features(self, features: np.ndarray, label: str = "") -> int:
        """Learn from a pre-computed dense feature vector."""
        packed = self.encode_features(features)
        return self.memory.learn(packed, label)

    def retrieve(self, image: np.ndarray, k: int = 1) -> list[tuple[int, float, str]]:
        """Retrieve the k nearest concepts for a query image.

        Args:
            image: (H, W) or (H, W, C) numpy array.
            k:     Number of results.

        Returns:
            List of (concept_id, similarity, label).
        """
        packed = self.encode_image(image)
        return self.memory.retrieve(packed, k=k)

    def retrieve_from_patches(
        self, patches: np.ndarray, k: int = 1,
    ) -> list[tuple[int, float, str]]:
        """Retrieve from pre-extracted patches."""
        packed = self.encode_patches(patches)
        return self.memory.retrieve(packed, k=k)

    def similarity(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        """Compute Hamming similarity between two images.

        Returns:
            Similarity in [0, 1]. 1.0 = identical encoding.
        """
        packed_a = self.encode_image(image_a)
        packed_b = self.encode_image(image_b)
        return hamming_similarity(packed_a, packed_b, self.d_vsa)
