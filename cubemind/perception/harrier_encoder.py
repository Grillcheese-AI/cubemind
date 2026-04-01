"""Harrier-OSS encoder — Microsoft's open embedding model for CubeMind.

Loads Harrier-OSS-v1-0.6b (Qwen3 base, 1024-dim embeddings) from local
safetensors weights. Encodes text to dense embeddings, then projects to
block-code VSA vectors.

Three-level fallback:
  1. Harrier via safetensors + tokenizers (fast, GPU-ready)
  2. Harrier via sentence-transformers (convenient, heavier)
  3. Hash encoding fallback (always works)

Usage:
    encoder = HarrierEncoder()
    hv = encoder.encode("The knight entered the dark forest")
    # hv.shape == (K, L) block-code vector

    batch = encoder.encode_batch(["hello", "world"])
    # batch.shape == (2, K, L)

    # Raw dense embeddings (1024-dim) without block-code projection
    dense = encoder.embed("some text")
    # dense.shape == (1024,)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS: int = 80
    L_BLOCK: int = 128

# Default local weight path
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "data" / "harrier-0.6b"


class HarrierEncoder:
    """Text encoder using Microsoft Harrier-OSS-v1-0.6b.

    Produces 1024-dim dense embeddings projected to (K, L) block-codes.

    Args:
        weights_dir: Path to local Harrier weights (model.safetensors + config.json).
        hf_model_id: HuggingFace model ID (fallback if weights_dir not found).
        k: Number of VSA blocks.
        l: Block length.
        device: Device for inference ('cpu', 'cuda', 'auto').
        seed: Random seed for the projection matrix.
    """

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        hf_model_id: str = "microsoft/harrier-oss-v1-0.6b",
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.bc = BlockCodes(k=k, l=l)
        self.device = device

        self._model = None
        self._tokenizer = None
        self._embed_dim = 1024  # Harrier-OSS-v1-0.6b hidden_size

        weights_path = Path(weights_dir) if weights_dir else _DEFAULT_WEIGHTS

        # Try loading from local safetensors
        if weights_path.exists() and (weights_path / "model.safetensors").exists():
            self._init_from_safetensors(weights_path)
        else:
            # Fallback: sentence-transformers
            self._init_from_sentence_transformers(hf_model_id)

        # Learned projection: dense embedding → VSA space
        # Fixed random projection preserves cosine similarity (JL lemma)
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal(
            (self.d_vsa, self._embed_dim)
        ).astype(np.float32) * np.sqrt(2.0 / self._embed_dim)

    def _init_from_safetensors(self, weights_dir: Path) -> None:
        """Load model from local safetensors weights."""
        try:
            from safetensors.numpy import load_file
            from tokenizers import Tokenizer

            config_path = weights_dir / "config.json"
            with open(config_path) as f:
                self._config = json.load(f)
            self._embed_dim = self._config.get("hidden_size", 1024)

            # Load tokenizer
            tok_path = weights_dir / "tokenizer.json"
            if tok_path.exists():
                self._tokenizer = Tokenizer.from_file(str(tok_path))
            else:
                # Fallback to transformers tokenizer
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(weights_dir), trust_remote_code=True)

            # Load weights as numpy arrays
            self._weights = load_file(str(weights_dir / "model.safetensors"))
            self._backend = "safetensors"

            # For mean-pooling we only need the embedding layer + final norm
            # Full transformer forward is expensive — use sentence-transformers
            # if available, otherwise extract embedding layer only
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    str(weights_dir), trust_remote_code=True, device=self.device)
                self._backend = "sentence_transformers_local"
            except ImportError:
                # Use raw weights for a simple embedding lookup
                self._backend = "safetensors_raw"

        except ImportError as e:
            raise ImportError(
                f"Need safetensors + tokenizers: pip install safetensors tokenizers. {e}"
            )

    def _init_from_sentence_transformers(self, model_id: str) -> None:
        """Load via sentence-transformers from HuggingFace."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                model_id, trust_remote_code=True, device=self.device)
            self._embed_dim = self._model.get_sentence_embedding_dimension()
            self._backend = "sentence_transformers"
        except ImportError:
            self._backend = "none"

    # ── Dense embeddings ─────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Get dense embedding for a single text.

        Args:
            text: Input text string.

        Returns:
            Dense embedding vector (embed_dim,).
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Get dense embeddings for a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            Embeddings array (N, embed_dim).
        """
        if self._model is not None:
            embeddings = self._model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=True)
            return embeddings.astype(np.float32)

        if self._backend == "safetensors_raw" and self._weights is not None:
            # Simple mean-of-token-embeddings (not ideal but works)
            embed_key = None
            for k in self._weights:
                if "embed_tokens" in k or "word_embeddings" in k:
                    embed_key = k
                    break
            if embed_key is None:
                raise RuntimeError("Cannot find embedding weights in safetensors")

            embed_matrix = self._weights[embed_key]  # (vocab, hidden)
            results = []
            for text in texts:
                if hasattr(self._tokenizer, 'encode'):
                    if hasattr(self._tokenizer, 'encode_batch'):
                        # tokenizers library
                        enc = self._tokenizer.encode(text)
                        ids = enc.ids
                    else:
                        ids = self._tokenizer.encode(text, add_special_tokens=True)
                else:
                    ids = list(range(min(len(text.split()), 512)))

                ids = [i for i in ids if i < embed_matrix.shape[0]]
                if not ids:
                    results.append(np.zeros(self._embed_dim, dtype=np.float32))
                    continue

                token_embeds = embed_matrix[ids].astype(np.float32)
                # Mean pooling
                pooled = token_embeds.mean(axis=0)
                # L2 normalize
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled /= norm
                results.append(pooled)

            return np.stack(results)

        # No model available — return random (hash-based) embeddings
        results = []
        for text in texts:
            import hashlib
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**31))
            vec = rng.standard_normal(self._embed_dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            results.append(vec)
        return np.stack(results)

    # ── Block-code projection ────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a block-code vector.

        Args:
            text: Input text string.

        Returns:
            Discrete block-code (k, l).
        """
        dense = self.embed(text)
        return self._project_to_block_code(dense)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode batch of texts to block-code vectors.

        Args:
            texts: List of input texts.

        Returns:
            Block-codes (N, k, l).
        """
        if not texts:
            return np.empty((0, self.k, self.l), dtype=np.float32)
        dense = self.embed_batch(texts)
        return np.stack([self._project_to_block_code(d) for d in dense])

    def _project_to_block_code(self, dense: np.ndarray) -> np.ndarray:
        """Project dense embedding to discrete block-code.

        Uses a fixed random projection (preserves cosine similarity
        via Johnson-Lindenstrauss lemma), then discretizes per block.

        Args:
            dense: Dense embedding (embed_dim,).

        Returns:
            Discrete block-code (k, l).
        """
        # Project: (d_vsa, embed_dim) @ (embed_dim,) → (d_vsa,)
        projected = (self._proj @ dense).reshape(self.k, self.l)
        return self.bc.discretize(projected)

    # ── Info ─────────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def __repr__(self) -> str:
        return (f"HarrierEncoder(backend={self._backend!r}, "
                f"embed_dim={self._embed_dim}, k={self.k}, l={self.l})")
