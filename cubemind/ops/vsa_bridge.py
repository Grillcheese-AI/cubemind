"""VSA Bridge — binarization, packing, and continuous-learning item memory.

Bridges the continuous neural world (Perceiver output) with the discrete
binary VSA world (packed uint32 hypervectors).

Pipeline:
  dense_vector (d_model,) → LSH random projection (d_vsa,) → binarize → pack → uint32[]

GPU shader: vsa-binarize-pack.glsl (threshold + pack in one dispatch).
CPU fallback: numpy vectorized operations.

ContinuousItemMemory provides over-provisioned storage with instant
learn/retrieve — no catastrophic forgetting, no gradient descent.
"""

from __future__ import annotations

import numpy as np

# GPU bridge (grilly)
_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


def _popcount_uint32(arr: np.ndarray) -> int:
    """Count total set bits in a uint32 array."""
    total = 0
    for shift in range(32):
        total += int(np.sum((arr >> np.uint32(shift)) & np.uint32(1)))
    return total


class LSHProjector:
    """Locality-Sensitive Hashing: dense float → high-dim float for binarization.

    Uses a fixed random projection matrix to expand a compact dense vector
    into the VSA dimensionality. Similar inputs produce similar projections.

    Args:
        d_input:  Input dimension (e.g. 256 from Perceiver).
        d_output: Output dimension (VSA dim, e.g. 10240).
        seed:     Random seed for reproducibility.
    """

    def __init__(self, d_input: int, d_output: int, seed: int = 42) -> None:
        self.d_input = d_input
        self.d_output = d_output
        rng = np.random.default_rng(seed)
        # Bipolar random projection: {-1, +1} for maximum entropy
        self.P = rng.choice(
            [-1.0, 1.0], size=(d_input, d_output),
        ).astype(np.float32)

    def project(self, dense_vector: np.ndarray) -> np.ndarray:
        """Project dense vector to VSA dimensionality.

        Args:
            dense_vector: (d_input,) float32.

        Returns:
            (d_output,) float32 — continuous, pre-binarization.
        """
        dense_vector = np.asarray(dense_vector, dtype=np.float32)
        if _bridge is not None:
            try:
                result = _bridge.linear(
                    dense_vector.reshape(1, -1), self.P.T, None,
                )
                if result is not None:
                    return np.asarray(result, dtype=np.float32).ravel()
            except Exception:
                pass
        return (dense_vector @ self.P).astype(np.float32)

    def project_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Project a batch of dense vectors.

        Args:
            vectors: (batch, d_input) float32.

        Returns:
            (batch, d_output) float32.
        """
        return (np.asarray(vectors, dtype=np.float32) @ self.P).astype(np.float32)


def binarize_and_pack(continuous: np.ndarray) -> np.ndarray:
    """Binarize (threshold > 0) and pack into uint32 array.

    GPU shader: vsa-binarize-pack.glsl.
    CPU fallback: numpy packbits.

    Args:
        continuous: (d_vsa,) float32 — LSH projection output.

    Returns:
        (ceil(d_vsa/32),) uint32 — packed binary VSA vector.
    """
    continuous = np.asarray(continuous, dtype=np.float32)
    d = len(continuous)
    # Pad to multiple of 32
    pad = (32 - d % 32) % 32
    if pad > 0:
        continuous = np.concatenate([continuous, np.zeros(pad, dtype=np.float32)])

    # Threshold: > 0 → 1
    bits = (continuous > 0).astype(np.uint8)

    # Pack 32 bits per uint32 (little-endian bit order)
    words_per_vec = len(bits) // 32
    packed = np.zeros(words_per_vec, dtype=np.uint32)
    for i in range(32):
        packed |= bits[i::32].astype(np.uint32) << np.uint32(i)

    return packed


def unpack_to_float(packed: np.ndarray, d_vsa: int) -> np.ndarray:
    """Unpack uint32 array back to bipolar {-1, +1} float32.

    Args:
        packed: (words_per_vec,) uint32.
        d_vsa:  Original dimension.

    Returns:
        (d_vsa,) float32 with values in {-1.0, +1.0}.
    """
    words = len(packed)
    result = np.zeros(words * 32, dtype=np.float32)
    for i in range(32):
        bits = ((packed >> np.uint32(i)) & np.uint32(1)).astype(np.float32)
        result[i::32] = bits * 2.0 - 1.0  # 0→-1, 1→+1
    return result[:d_vsa]


def hamming_similarity(a: np.ndarray, b: np.ndarray, dim: int) -> float:
    """Hamming similarity between two packed binary vectors.

    Args:
        a, b: (words_per_vec,) uint32.
        dim:  Original vector dimension.

    Returns:
        Similarity in [0, 1]. 1.0 = identical, 0.5 = random.
    """
    xored = np.bitwise_xor(a, b)
    hamming_dist = _popcount_uint32(xored)
    return 1.0 - hamming_dist / float(dim)


class ContinuousItemMemory:
    """Over-provisioned VSA item memory with instant learn/retrieve.

    Allocates capacity up front. Learning = append (no backprop).
    Retrieval = Hamming distance search against active entries.

    No catastrophic forgetting: new concepts are stored without
    disturbing existing ones.

    Args:
        d_vsa:        VSA dimension (e.g. 10240).
        max_capacity: Maximum number of concepts.
    """

    def __init__(self, d_vsa: int = 10240, max_capacity: int = 100000) -> None:
        self.d_vsa = d_vsa
        self.words_per_vec = int(np.ceil(d_vsa / 32))
        self.max_capacity = max_capacity

        # Over-provisioned storage
        self._memory = np.zeros(
            (max_capacity, self.words_per_vec), dtype=np.uint32,
        )
        self.num_active: int = 0
        self._labels: list[str] = []

    def learn(self, packed_vector: np.ndarray, label: str = "") -> int:
        """Store a new concept in the next free slot.

        Args:
            packed_vector: (words_per_vec,) uint32.
            label:         Human-readable label for this concept.

        Returns:
            Concept ID (index).

        Raises:
            MemoryError: If capacity is full.
        """
        if self.num_active >= self.max_capacity:
            raise MemoryError(
                f"Item memory full ({self.max_capacity} concepts). "
                "Increase max_capacity or implement eviction."
            )
        self._memory[self.num_active] = packed_vector
        self._labels.append(label)
        idx = self.num_active
        self.num_active += 1
        return idx

    def retrieve(self, query: np.ndarray, k: int = 1) -> list[tuple[int, float, str]]:
        """Find the k nearest concepts by Hamming distance.

        Args:
            query: (words_per_vec,) uint32 — packed query vector.
            k:     Number of results.

        Returns:
            List of (concept_id, similarity, label) tuples, descending by similarity.
        """
        if self.num_active == 0:
            return []

        active = self._memory[:self.num_active]

        # Try grilly GPU path
        try:
            from grilly.backend._bridge import hamming_topk
            result = hamming_topk(query, active, self.d_vsa, k=k)
            return [
                (int(result['indices'][i]),
                 float(result['similarities'][i]),
                 self._labels[int(result['indices'][i])])
                for i in range(len(result['indices']))
            ]
        except Exception:
            pass

        # CPU fallback: compute all Hamming distances
        xored = np.bitwise_xor(active, query[np.newaxis, :])
        distances = np.zeros(self.num_active, dtype=np.int32)
        for shift in range(32):
            distances += (
                (xored >> np.uint32(shift)) & np.uint32(1)
            ).astype(np.int32).sum(axis=1)

        sims = 1.0 - distances.astype(np.float32) / float(self.d_vsa)

        k = min(k, self.num_active)
        if k < self.num_active:
            topk_idx = np.argpartition(-sims, k)[:k]
        else:
            topk_idx = np.arange(self.num_active)
        topk_idx = topk_idx[np.argsort(-sims[topk_idx])]

        return [
            (int(i), float(sims[i]), self._labels[i])
            for i in topk_idx
        ]

    def retrieve_best(self, query: np.ndarray) -> tuple[int, float, str]:
        """Find the single nearest concept.

        Returns:
            (concept_id, similarity, label) or (-1, 0.0, "") if empty.
        """
        results = self.retrieve(query, k=1)
        if not results:
            return (-1, 0.0, "")
        return results[0]

    def save(self, path: str) -> None:
        """Save active concepts to disk (binary dump, not the empty capacity).

        Only saves the active slice — if 1000 of 100000 slots are used,
        only ~1000 * words_per_vec * 4 bytes are written.

        Args:
            path: File path prefix (saves {path}.npz).
        """
        if self.num_active == 0:
            return
        active_data = self._memory[:self.num_active]
        np.savez_compressed(
            f"{path}.npz",
            vsa_data=active_data,
            num_active=np.array([self.num_active]),
            d_vsa=np.array([self.d_vsa]),
            labels=np.array(self._labels, dtype=object),
        )

    def load(self, path: str) -> None:
        """Load saved concepts back into memory.

        Args:
            path: File path prefix (loads {path}.npz).
        """
        data = np.load(f"{path}.npz", allow_pickle=True)
        active_data = data['vsa_data']
        self.num_active = int(data['num_active'][0])
        self.d_vsa = int(data['d_vsa'][0])
        self.words_per_vec = int(np.ceil(self.d_vsa / 32))

        # Resize memory if needed
        if active_data.shape[0] > self._memory.shape[0]:
            self._memory = np.zeros(
                (active_data.shape[0] * 2, self.words_per_vec), dtype=np.uint32,
            )
        self._memory[:self.num_active] = active_data

        labels = data.get('labels', np.array([]))
        self._labels = list(labels) if len(labels) > 0 else [""] * self.num_active

    @property
    def size(self) -> int:
        return self.num_active
