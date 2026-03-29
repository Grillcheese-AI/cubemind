# cubemind/perception/semantic_encoder.py
"""SemanticEncoder — sentence-transformers → block-code VSA projection.

Replaces hash-based WorldEncoder for text retrieval tasks where semantic
similarity matters (e.g., "merchant vessels" should match "ships in combat").

Pipeline:
  text → SentenceTransformer (384D) → orthogonal projection P (k*l, 384)
  → reshape to (k, l) block-code

The projection matrix P is computed via QR decomposition (k*l >= 384)
or Johnson-Lindenstrauss random projection (k*l < 384).

Drop-in compatible with WorldEncoder: .encode_action(text) → (k, l)
"""
from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS = 80
    L_BLOCK = 128

# Try llama-cpp-python for GGUF models (preferred — fast, no torch overhead)
_LLAMA = None
try:
    from llama_cpp import Llama as _LlamaCpp
    _LLAMA = _LlamaCpp
except ImportError:
    pass

# Fallback: sentence-transformers
_SBERT = None
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer
except ImportError:
    pass


class SemanticEncoder:
    """Encode text into block-code VSA vectors with semantic similarity.

    Uses all-MiniLM-L6-v2 (384D, local, fast) for sentence embedding,
    then projects into (k, l) block-code space via orthogonal matrix P.

    Semantically similar texts produce similar block-code vectors:
      "merchant vessels" ≈ "ships in naval combat"

    Args:
        k: Number of blocks.
        l: Block length.
        model_name: SentenceTransformer model name.
        seed: Random seed for projection matrix.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,  # noqa: E741
        gguf_path: str | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self._embed_dim = 1024  # BGE-M3 default
        self._gguf_model = None
        self._sbert_model = None
        self._mode = "none"

        # Priority 1: GGUF model (BGE-M3, fast, no torch)
        if gguf_path and _LLAMA is not None:
            try:
                self._gguf_model = _LLAMA(
                    model_path=gguf_path,
                    embedding=True,
                    n_ctx=512,
                    n_batch=512,
                    verbose=False,
                )
                self._embed_dim = self._gguf_model.n_embd()
                self._mode = "gguf"
            except Exception:
                pass

        # Priority 2: sentence-transformers (384D, slower but works)
        if self._mode == "none" and _SBERT is not None:
            try:
                self._sbert_model = _SBERT(model_name, device="cpu")
                self._embed_dim = (
                    self._sbert_model.get_sentence_embedding_dimension()
                )
                self._mode = "sbert"
            except Exception:
                pass

        # Build projection matrix P: (k*l, embed_dim)
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

    def _embed_text(self, text: str) -> np.ndarray:
        """Get raw embedding vector from the underlying model."""
        if self._mode == "gguf" and self._gguf_model is not None:
            emb = np.array(
                self._gguf_model.embed(text), dtype=np.float32,
            )
            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb

        if self._mode == "sbert" and self._sbert_model is not None:
            return self._sbert_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

        return None

    def encode_action(self, text: str) -> np.ndarray:
        """Encode text into a (k, l) block-code vector with semantic meaning.

        Drop-in replacement for WorldEncoder.encode_action().

        Args:
            text: Input text string (any length).

        Returns:
            Block-code vector of shape (k, l), dtype float32.
        """
        embedding = self._embed_text(text)
        if embedding is None:
            # Fallback: hash-based (no semantic understanding)
            return self.bc.random_discrete(
                seed=hash(text) % (2**31),
            )

        # Project to VSA space and reshape
        projected = self._P @ embedding  # (k*l,)
        return projected.reshape(self.k, self.l).astype(np.float32)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts efficiently.

        Args:
            texts: List of text strings.

        Returns:
            List of (k, l) block-code vectors.
        """
        if not texts:
            return []

        if self._mode == "sbert" and self._sbert_model is not None:
            embeddings = self._sbert_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10,
                batch_size=64,
            ).astype(np.float32)
            projected = (self._P @ embeddings.T).T
            return [
                projected[i].reshape(self.k, self.l).astype(np.float32)
                for i in range(len(texts))
            ]

        # GGUF or fallback: encode one at a time
        return [self.encode_action(t) for t in texts]

    def encode_state(self, attributes: dict[str, str]) -> np.ndarray:
        """Encode structured state as semantic text.

        Drop-in for WorldEncoder.encode_state().
        Converts dict to "key: value, key: value" text and encodes.
        """
        text = ", ".join(f"{k}: {v}" for k, v in attributes.items())
        return self.encode_action(text)

    @property
    def available(self) -> bool:
        """Whether the sentence encoder is loaded."""
        return self._encoder is not None
