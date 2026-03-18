"""VSA Cache with FAISS binary Hamming-distance lookup (with numpy fallback).

Implements the surprise-driven cache expansion and stress-driven eviction
from CubeMind Section 2.1-2.2:

    cache = {phi(g_i) -> (irrep(g_i), emotion(g_i))  for i = 1..N}

Keys are stored as int8 bipolar {-1,+1} and searched via FAISS
IndexBinaryFlat (popcount Hamming distance) when available.  When FAISS
is not installed, a pure-numpy fallback using ``np.packbits`` + XOR +
popcount provides identical results.

For bipolar vectors, Hamming ranking is equivalent to cosine ranking:

    cosine_dist = 2 * hamming_dist / d_vsa

Reference: CubeMind Paper Section 2.1 (VSA-Compressed Superposition)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

# GPU bridge
_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _to_packed(bipolar: np.ndarray) -> np.ndarray:
    """Convert bipolar {-1,+1} int8/float to bitpacked uint8 for FAISS binary."""
    binary = ((np.asarray(bipolar).ravel().reshape(bipolar.shape) + 1) // 2).astype(
        np.uint8
    )
    return np.packbits(binary, axis=-1)


def _hamming_distance_matrix(
    queries_packed: np.ndarray, keys_packed: np.ndarray
) -> np.ndarray:
    """Pure-numpy Hamming distance between packed query and key matrices.

    Args:
        queries_packed: (Q, d_packed) uint8
        keys_packed:    (N, d_packed) uint8

    Returns:
        (Q, N) int32 Hamming distances
    """
    xor = np.bitwise_xor(
        queries_packed[:, None, :], keys_packed[None, :, :]
    )  # (Q, N, d_packed)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    return lut[xor].sum(axis=-1)  # (Q, N)


# ------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------


class VSACache:
    """Hash table of VSA hypervectors with binary Hamming lookup.

    Keys are int8 bipolar {-1, +1}.  When FAISS is available a
    ``IndexBinaryFlat`` over bitpacked representations provides fast
    popcount-based search; otherwise a pure-numpy fallback is used.

    Args:
        max_size: Maximum number of cached prototypes (paper: 490 000).
        d_vsa: Hypervector dimension (paper: 10 240).
        initial_capacity: Pre-allocated rows (grows lazily).
    """

    def __init__(
        self,
        max_size: int = 490_000,
        d_vsa: int = 10_240,
        initial_capacity: int = 256,
    ) -> None:
        self.max_size = max_size
        self.d_vsa = d_vsa

        cap = min(max_size, initial_capacity)
        self.keys = np.zeros((cap, d_vsa), dtype=np.int8)
        self.emotions = np.zeros((cap, 2), dtype=np.float32)  # (surprise, stress)
        self.utility = np.zeros(cap, dtype=np.float32)
        self._size = 0
        self._cap = cap

        # FAISS binary index (rebuilt lazily, only when faiss is available)
        self._index: object | None = None
        self._index_size = 0

    # ------------------------------------------------------------------
    # size property
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        self._size = value

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """(Re)build the FAISS binary index from current cache keys."""
        if self._size == 0:
            self._index = None
            self._index_size = 0
            return
        if _HAS_FAISS:
            self._index = faiss.IndexBinaryFlat(self.d_vsa)
            packed = _to_packed(self.keys[: self._size])
            self._index.add(packed)
            self._index_size = self._size
        else:
            # For the numpy fallback we just cache the packed keys
            self._index = _to_packed(self.keys[: self._size])
            self._index_size = self._size

    def _ensure_index(self) -> None:
        """Rebuild index if cache has changed since last build."""
        if self._index is None or self._index_size != self._size:
            self._rebuild_index()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def lookup(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Binary Hamming lookup (equivalent to cosine for bipolar).

        Args:
            query: (batch, d_vsa) or (d_vsa,) -- bipolar or float.
            k: Number of nearest neighbours.

        Returns:
            similarities (batch, k)   cosine similarities in [0, 1]
            keys         (batch, k, d_vsa)
            indices      (batch, k)
        """
        q = np.atleast_2d(query)
        batch = q.shape[0]

        if self._size == 0:
            return (
                np.zeros((batch, k), dtype=np.float32),
                np.zeros((batch, k, self.d_vsa), dtype=np.int8),
                np.zeros((batch, k), dtype=np.int32),
            )

        k_actual = min(k, self._size)

        # GPU fast path: float dot-product similarity (equivalent to Hamming
        # for bipolar vectors since cosine = dot / d_vsa).
        if _bridge is not None:
            try:
                q_f32 = np.sign(q).astype(np.float32)
                keys_f32 = self.keys[: self._size].astype(np.float32)
                # linear(q, keys, None) = q @ keys^T → (batch, size)
                dots = _bridge.linear(q_f32, keys_f32, None)
                if dots is not None:
                    dots = np.asarray(dots, dtype=np.float32)
                    sims_full = dots / self.d_vsa
                    if k_actual < sims_full.shape[1]:
                        part_idx = np.argpartition(
                            -sims_full, k_actual, axis=1
                        )[:, :k_actual]
                        row_idx = np.arange(batch)[:, None]
                        part_sims = sims_full[row_idx, part_idx]
                        sorted_order = np.argsort(-part_sims, axis=1)
                        indices = part_idx[row_idx, sorted_order]
                        sims = part_sims[row_idx, sorted_order]
                    else:
                        sorted_order = np.argsort(-sims_full, axis=1)
                        indices = sorted_order[:, :k_actual]
                        row_idx = np.arange(batch)[:, None]
                        sims = sims_full[row_idx, indices]
                    indices = indices.astype(np.int32)
                    if k_actual < k:
                        pad = k - k_actual
                        indices = np.pad(indices, ((0, 0), (0, pad)))
                        sims = np.pad(
                            sims, ((0, 0), (0, pad)), constant_values=0.0
                        )
                    retrieved_keys = self.keys[indices]
                    return sims, retrieved_keys, indices
            except Exception:
                pass

        # CPU fallback: FAISS or numpy Hamming
        self._ensure_index()

        q_packed = _to_packed(np.sign(q).astype(np.int8))

        if _HAS_FAISS:
            ham_dists, indices = self._index.search(q_packed, k_actual)
        else:
            ham_dists_full = _hamming_distance_matrix(
                q_packed, self._index
            )  # (batch, size)
            if k_actual < ham_dists_full.shape[1]:
                part_idx = np.argpartition(ham_dists_full, k_actual, axis=1)[
                    :, :k_actual
                ]
                row_idx = np.arange(batch)[:, None]
                part_dists = ham_dists_full[row_idx, part_idx]
                sorted_order = np.argsort(part_dists, axis=1)
                indices = part_idx[row_idx, sorted_order]
                ham_dists = part_dists[row_idx, sorted_order]
            else:
                sorted_order = np.argsort(ham_dists_full, axis=1)
                indices = sorted_order[:, :k_actual]
                row_idx = np.arange(batch)[:, None]
                ham_dists = ham_dists_full[row_idx, indices]

        # Convert Hamming distance to cosine similarity: sim = 1 - 2*hamming/d_vsa
        sims = 1.0 - (2.0 * ham_dists.astype(np.float32)) / self.d_vsa

        indices = indices.astype(np.int32)

        if k_actual < k:
            pad = k - k_actual
            indices = np.pad(indices, ((0, 0), (0, pad)))
            sims = np.pad(sims, ((0, 0), (0, pad)), constant_values=0.0)

        retrieved_keys = self.keys[indices]
        return sims, retrieved_keys, indices

    def add(self, phi: np.ndarray, emotion: np.ndarray) -> bool:
        """Add a single hypervector.  Returns False if cache is full."""
        if self._size >= self.max_size:
            return False
        if self._size >= self._cap:
            self._grow()

        self.keys[self._size] = np.sign(phi).astype(np.int8)
        self.emotions[self._size] = np.asarray(emotion, dtype=np.float32)
        self.utility[self._size] = 1.0
        self._size += 1
        # Index rebuilt lazily on next lookup
        return True

    def add_batch(self, phis: np.ndarray, emotions: np.ndarray) -> int:
        """Add a batch of hypervectors.  Returns number added."""
        added = 0
        for phi, emo in zip(phis, emotions):
            if not self.add(phi, emo):
                break
            added += 1
        return added

    def evict(self, n: int = 1_000) -> None:
        """Evict the *n* lowest-utility entries (stress-driven pruning)."""
        if self._size <= n:
            return
        keep = np.argsort(self.utility[: self._size])[n:]
        count = len(keep)
        self.keys[:count] = self.keys[keep]
        self.emotions[:count] = self.emotions[keep]
        self.utility[:count] = self.utility[keep]
        self._size = count
        self._index = None
        self._index_size = 0

    def update_utility(self, indices: np.ndarray, decay: float = 0.99) -> None:
        """Decay all utility; boost recently-accessed entries."""
        self.utility[: self._size] *= decay
        valid = indices[(indices >= 0) & (indices < self._size)]
        self.utility[valid] += 1.0

    # ------------------------------------------------------------------
    # Surprise and stress signals
    # ------------------------------------------------------------------

    def surprise(self, query: np.ndarray) -> float:
        """Surprise signal: 1 - max_similarity to any entry in the cache.

        Returns 1.0 for empty cache (total surprise) and approaches 0.0
        when the query is already well-represented.

        Args:
            query: Bipolar hypervector (d_vsa,).

        Returns:
            Scalar in [0, 1].
        """
        if self._size == 0:
            return 1.0
        sims, _, _ = self.lookup(query, k=1)
        max_sim = float(sims[0, 0])
        return float(np.clip(1.0 - max_sim, 0.0, 1.0))

    def stress(self) -> float:
        """Stress signal: cache pressure = size / max_size.

        Returns:
            Scalar in [0, 1].
        """
        return self._size / self.max_size

    # ------------------------------------------------------------------
    # Float32 access (for model layers that need float input)
    # ------------------------------------------------------------------

    def keys_float(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Return float32 view of keys (or a subset by indices)."""
        if indices is not None:
            return self.keys[indices].astype(np.float32)
        return self.keys[: self._size].astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Save cache to disk.

        Creates:
            <directory>/cache_keys.bin        (size, d_vsa) int8 raw
            <directory>/cache_emotions.npy    (size, 2) float32
            <directory>/cache_utility.npy     (size,) float32
            <directory>/cache_meta.json
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        # Keys: chunked memmap write
        keys_path = out / "cache_keys.bin"
        nbytes = self._size * self.d_vsa * np.dtype(np.int8).itemsize
        with open(str(keys_path), "wb") as f:
            f.truncate(nbytes)
        mm = np.memmap(
            str(keys_path),
            dtype=np.int8,
            mode="r+",
            shape=(self._size, self.d_vsa),
        )
        chunk = 50_000
        for i in range(0, self._size, chunk):
            end = min(i + chunk, self._size)
            mm[i:end] = self.keys[i:end]
        mm.flush()
        del mm

        np.save(str(out / "cache_emotions.npy"), self.emotions[: self._size])
        np.save(str(out / "cache_utility.npy"), self.utility[: self._size])

        meta = {
            "size": self._size,
            "max_size": self.max_size,
            "d_vsa": self.d_vsa,
            "dtype": "int8",
        }
        (out / "cache_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> VSACache:
        """Load a previously saved cache from disk."""
        root = Path(directory)
        meta = json.loads((root / "cache_meta.json").read_text(encoding="utf-8"))
        size = meta["size"]
        d_vsa = meta["d_vsa"]
        key_dtype = meta.get("dtype", "float32")

        # Keys: raw memmap binary or .npy fallback
        keys_bin = root / "cache_keys.bin"
        if keys_bin.exists():
            keys_mm = np.memmap(
                str(keys_bin),
                dtype=key_dtype,
                mode="r",
                shape=(size, d_vsa),
            )
        else:
            keys_mm = np.load(str(root / "cache_keys.npy"))

        emotions = np.load(str(root / "cache_emotions.npy"))
        utility = np.load(str(root / "cache_utility.npy"))

        cache = cls(
            max_size=meta["max_size"],
            d_vsa=d_vsa,
            initial_capacity=max(size, 1),
        )
        cache.keys = np.zeros((max(size, cache._cap), d_vsa), dtype=np.int8)
        cache.emotions = np.zeros((max(size, cache._cap), 2), dtype=np.float32)
        cache.utility = np.zeros(max(size, cache._cap), dtype=np.float32)

        # Copy in chunks (handles both int8 and float32 source)
        chunk = 50_000
        for i in range(0, size, chunk):
            end = min(i + chunk, size)
            src = keys_mm[i:end]
            if key_dtype == "float32":
                cache.keys[i:end] = np.sign(src).astype(np.int8)
            else:
                cache.keys[i:end] = src

        cache.emotions[:size] = emotions[:size]
        cache.utility[:size] = utility[:size]
        cache._size = size
        cache._cap = max(size, cache._cap)
        return cache

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _grow(self) -> None:
        new_cap = min(self._cap * 2, self.max_size)
        new_keys = np.zeros((new_cap, self.d_vsa), dtype=np.int8)
        new_emotions = np.zeros((new_cap, 2), dtype=np.float32)
        new_utility = np.zeros(new_cap, dtype=np.float32)
        new_keys[: self._size] = self.keys[: self._size]
        new_emotions[: self._size] = self.emotions[: self._size]
        new_utility[: self._size] = self.utility[: self._size]
        self.keys = new_keys
        self.emotions = new_emotions
        self.utility = new_utility
        self._cap = new_cap
