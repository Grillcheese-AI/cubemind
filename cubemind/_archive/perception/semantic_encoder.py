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

# llama-cpp-python is imported lazily inside __init__ to avoid
# Vulkan backend allocating tens of GB of system RAM on import.
_LLAMA_AVAILABLE = False
try:
    import importlib.util
    _LLAMA_AVAILABLE = importlib.util.find_spec("llama_cpp") is not None
except Exception:
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
        # Imported lazily to avoid Vulkan backend eating 50+GB on import
        if gguf_path and _LLAMA_AVAILABLE:
            try:
                from llama_cpp import Llama as _LlamaCpp
                self._gguf_model = _LlamaCpp(
                    model_path=gguf_path,
                    embedding=True,
                    n_ctx=64,
                    n_batch=64,
                    n_gpu_layers=0,
                    verbose=True,
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

    # ── SDLS Purification ──────────────────────────────────────────────

    def encode_corpus(
        self, texts: list[str],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Encode a document corpus with SDLS noise purification.

        1. Embed all chunks to float space (1024D)
        2. Compute document mean (the noise axis)
        3. Orthogonal projection: remove noise component from each chunk
        4. Project purified embeddings to (k, l) block-code

        This is the SDLS (Semantically Decoupled Latent Steering) approach:
        instead of `X - 0.99*mean` (crude, loses magnitude), we project
        onto the null-space of the mean: `X - (X·m̂)m̂`

        Args:
            texts: List of document chunks.

        Returns:
            (vectors, noise_axis):
                vectors: List of purified (k, l) block-code vectors.
                noise_axis: The unit mean vector that was removed (1024D).
        """
        if not texts:
            return [], np.zeros(self._embed_dim, dtype=np.float32)

        # 1. Get raw float embeddings
        raw_embeddings = []
        for t in texts:
            emb = self._embed_text(t)
            if emb is None:
                emb = np.zeros(self._embed_dim, dtype=np.float32)
            raw_embeddings.append(emb)

        raw_matrix = np.stack(raw_embeddings)  # (N, embed_dim)

        # 2. Compute and remove noise axis (SDLS purification)
        purified, noise_axis = self._sdls_purify(raw_matrix)

        # 3. Project to block-code space
        vectors = []
        for i in range(len(texts)):
            projected = self._P @ purified[i]
            vectors.append(
                projected.reshape(self.k, self.l).astype(np.float32),
            )

        return vectors, noise_axis

    def encode_query_purified(
        self, text: str, noise_axis: np.ndarray,
    ) -> np.ndarray:
        """Encode a query with the same SDLS purification as the corpus.

        Must use the same noise_axis returned by encode_corpus().

        Args:
            text: Query text.
            noise_axis: Unit noise vector from encode_corpus().

        Returns:
            Purified (k, l) block-code vector.
        """
        emb = self._embed_text(text)
        if emb is None:
            return self.bc.random_discrete(seed=hash(text) % (2**31))

        # Remove noise component
        projection = float(np.dot(emb, noise_axis))
        purified = emb - projection * noise_axis

        # Re-normalize
        norm = np.linalg.norm(purified)
        if norm > 1e-8:
            purified = purified / norm

        # Project to block-code
        projected = self._P @ purified.astype(np.float32)
        return projected.reshape(self.k, self.l).astype(np.float32)

    @staticmethod
    def _sdls_purify(
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SDLS: project embeddings onto the null-space of their mean.

        Removes the document-level noise floor (common vocabulary)
        while preserving all discriminative signal at full magnitude.

        Math: X_purified = X - (X · m̂) m̂
        where m̂ = mean(X) / ||mean(X)||

        Args:
            embeddings: (N, D) float32 matrix.

        Returns:
            (purified, noise_axis):
                purified: (N, D) denoised embeddings, L2-normalized.
                noise_axis: (D,) unit mean vector.
        """
        # Compute noise axis (unit document mean)
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-8:
            return embeddings, mean

        noise_axis = (mean / norm).astype(np.float32)

        # Orthogonal projection: remove noise component
        projections = embeddings @ noise_axis  # (N,)
        purified = embeddings - np.outer(projections, noise_axis)

        # Re-normalize each vector
        norms = np.linalg.norm(purified, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        purified = (purified / norms).astype(np.float32)

        return purified, noise_axis

    @property
    def available(self) -> bool:
        """Whether the sentence encoder is loaded."""
        return self._encoder is not None
