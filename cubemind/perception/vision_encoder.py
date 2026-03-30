"""VisionEncoder — pretrained VL model → block-code VSA projection.

Model-agnostic: supports any vision encoder that produces a dense
embedding vector from an image. Backends loaded lazily to avoid
import-time overhead.

Supported backends (priority order):
  1. transformers (SigLIP, CLIP, DINOv2, InternVL, etc.)
  2. open_clip (CLIP variants)
  3. llama-cpp-python (LLaVA vision tower via GGUF)
  4. Fallback: raw pixel statistics (mean/std per channel)

Pipeline:
  image → VL model (e.g. 768D) → orthogonal projection P → (k, l) block-code

Drop-in compatible with SemanticEncoder: .encode_image(img) → (k, l)
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes
from cubemind.ops.vsa_bridge import LSHProjector, binarize_and_pack

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 80
    L_BLOCK = 128


class VisionEncoder:
    """Encode images into block-code VSA vectors via pretrained VL models.

    Automatically selects the best available backend. All backends
    produce a dense float embedding which is then projected to (k, l)
    block-code space via orthogonal projection.

    Args:
        k: Number of blocks.
        l: Block length.
        model_name: HuggingFace model name or path.
        backend: Force a specific backend ("transformers", "open_clip",
                 "llama_cpp", None=auto-detect).
        device: Device for inference ("cpu", "cuda", "vulkan").
        seed: Random seed for projection matrix.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        model_name: str = "google/siglip-base-patch16-224",
        backend: str | None = None,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self._model_name = model_name
        self._device = device
        self._embed_dim: int = 0
        self._mode = "none"

        # Lazy-loaded model references
        self._hf_model = None
        self._hf_processor = None
        self._clip_model = None
        self._clip_preprocess = None
        self._llama_model = None

        # Try backends in priority order
        if backend in (None, "transformers"):
            self._try_transformers(model_name, device)
        if self._mode == "none" and backend in (None, "open_clip"):
            self._try_open_clip(model_name, device)
        if self._mode == "none" and backend in (None, "llama_cpp"):
            self._try_llama_cpp(model_name)
        if self._mode == "none":
            self._mode = "fallback"
            self._embed_dim = 768  # Dummy for projection matrix

        # Build projection matrix: (k*l, embed_dim)
        vsa_dim = k * l
        rng = np.random.default_rng(seed)
        if vsa_dim >= self._embed_dim:
            Q, _ = np.linalg.qr(
                rng.standard_normal((vsa_dim, self._embed_dim)),
            )
            self._P = Q.astype(np.float32)
        else:
            scale = 1.0 / np.sqrt(vsa_dim)
            self._P = (
                scale * rng.standard_normal((vsa_dim, self._embed_dim))
            ).astype(np.float32)

        # LSH projector for binary VSA path
        self._lsh = LSHProjector(
            d_input=self._embed_dim, d_output=vsa_dim, seed=seed + 1,
        )

    # ── Backend initialization ────────────────────────────────────────────

    def _try_transformers(self, model_name: str, device: str) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
            self._hf_processor = AutoProcessor.from_pretrained(model_name)
            self._hf_model = AutoModel.from_pretrained(model_name)
            self._hf_model.eval()
            if device != "cpu" and hasattr(self._hf_model, "to"):
                try:
                    self._hf_model = self._hf_model.to(device)
                except Exception:
                    pass

            # Probe embed dim
            self._embed_dim = getattr(
                self._hf_model.config, "hidden_size",
                getattr(self._hf_model.config, "vision_config", {})
                .__dict__.get("hidden_size", 768),
            )
            self._mode = "transformers"
        except Exception:
            pass

    def _try_open_clip(self, model_name: str, device: str) -> None:
        try:
            import open_clip
            # Parse model_name like "ViT-B-32/openai"
            parts = model_name.split("/")
            arch = parts[0] if parts else "ViT-B-32"
            pretrained = parts[1] if len(parts) > 1 else "openai"
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained, device=device,
            )
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._embed_dim = model.visual.output_dim
            self._mode = "open_clip"
        except Exception:
            pass

    def _try_llama_cpp(self, model_path: str) -> None:
        try:
            import importlib.util
            if importlib.util.find_spec("llama_cpp") is None:
                return
            from llama_cpp import Llama
            self._llama_model = Llama(
                model_path=model_path,
                embedding=True,
                n_ctx=64,
                n_gpu_layers=0,
                verbose=False,
            )
            self._embed_dim = self._llama_model.n_embd()
            self._mode = "llama_cpp"
        except Exception:
            pass

    # ── Embedding extraction ──────────────────────────────────────────────

    def _embed_image(self, image: np.ndarray) -> np.ndarray | None:
        """Get raw embedding vector from the underlying VL model.

        Args:
            image: (H, W) or (H, W, C) numpy array, values in [0, 255] or [0, 1].

        Returns:
            (embed_dim,) float32 vector, or None if no backend available.
        """
        if self._mode == "transformers":
            return self._embed_transformers(image)
        if self._mode == "open_clip":
            return self._embed_open_clip(image)
        if self._mode == "llama_cpp":
            return self._embed_llama_cpp(image)
        return self._embed_fallback(image)

    def _embed_transformers(self, image: np.ndarray) -> np.ndarray | None:
        try:
            import torch
            from PIL import Image as PILImage

            # Convert numpy to PIL
            img = self._numpy_to_pil(image)
            inputs = self._hf_processor(images=img, return_tensors="pt")
            if self._device != "cpu":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._hf_model(**inputs)

            # Try common output formats
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output[0]
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state[:, 0, :]  # CLS token
                emb = emb[0]
            elif hasattr(outputs, "image_embeds"):
                emb = outputs.image_embeds[0]
            else:
                return None

            emb = emb.cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception:
            return None

    def _embed_open_clip(self, image: np.ndarray) -> np.ndarray | None:
        try:
            import torch
            from PIL import Image as PILImage

            img = self._numpy_to_pil(image)
            img_tensor = self._clip_preprocess(img).unsqueeze(0)
            if self._device != "cpu":
                img_tensor = img_tensor.to(self._device)

            with torch.no_grad():
                emb = self._clip_model.encode_image(img_tensor)

            emb = emb[0].cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception:
            return None

    def _embed_llama_cpp(self, image: np.ndarray) -> np.ndarray | None:
        # llama-cpp vision embedding is model-specific; skip for now
        return None

    def _embed_fallback(self, image: np.ndarray) -> np.ndarray:
        """Fallback: simple pixel statistics as features."""
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        # Compute spatial statistics per channel
        features = []
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            features.extend([
                float(np.mean(ch)),
                float(np.std(ch)),
                float(np.median(ch)),
                float(np.percentile(ch, 25)),
                float(np.percentile(ch, 75)),
            ])

        # Pad/truncate to embed_dim
        feat = np.array(features, dtype=np.float32)
        if len(feat) < self._embed_dim:
            feat = np.pad(feat, (0, self._embed_dim - len(feat)))
        else:
            feat = feat[:self._embed_dim]
        return feat

    @staticmethod
    def _numpy_to_pil(image: np.ndarray):
        """Convert numpy array to PIL Image."""
        from PIL import Image as PILImage
        img = image.copy()
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255)
        img = img.astype(np.uint8)
        if img.ndim == 2:
            return PILImage.fromarray(img, mode="L")
        return PILImage.fromarray(img, mode="RGB")

    # ── Public API ────────────────────────────────────────────────────────

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode an image into a (k, l) block-code vector.

        Args:
            image: (H, W) or (H, W, C) numpy array.

        Returns:
            Block-code vector of shape (k, l), dtype float32.
        """
        embedding = self._embed_image(image)
        if embedding is None:
            embedding = self._embed_fallback(image)

        projected = self._P @ embedding  # (k*l,)
        return projected.reshape(self.k, self.l).astype(np.float32)

    def encode_image_binary(self, image: np.ndarray) -> np.ndarray:
        """Encode an image into a packed binary VSA vector.

        Uses LSH projection → binarize → pack for Hamming retrieval.

        Args:
            image: (H, W) or (H, W, C) numpy array.

        Returns:
            (words_per_vec,) uint32 packed binary vector.
        """
        embedding = self._embed_image(image)
        if embedding is None:
            embedding = self._embed_fallback(image)

        projected = self._lsh.project(embedding)
        return binarize_and_pack(projected)

    def encode_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Encode multiple images to (k, l) block-code vectors."""
        return [self.encode_image(img) for img in images]

    def encode_batch_binary(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Encode multiple images to packed binary VSA vectors."""
        return [self.encode_image_binary(img) for img in images]

    def similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Cosine similarity between two images in embedding space."""
        emb_a = self._embed_image(img_a)
        emb_b = self._embed_image(img_b)
        if emb_a is None or emb_b is None:
            return 0.0
        dot = float(np.dot(emb_a, emb_b))
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def mode(self) -> str:
        """Active backend name."""
        return self._mode

    @property
    def embed_dim(self) -> int:
        """Embedding dimension of the active backend."""
        return self._embed_dim
