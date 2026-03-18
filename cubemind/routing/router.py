"""CubeMindRouter — prototype similarity routing with save/load.

Routes queries to topics by comparing block-code vectors against
stored topic prototypes. Uses BlockCodes.similarity_batch for
efficient comparison and DSelectKGate for sparse expert selection.

Usage:
    router = CubeMindRouter.from_categories(categories, embedder)
    topic, score = router.route_vector(query_vec)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.ops.block_codes import BlockCodes
from cubemind.routing.moe_gate import DSelectKGate


class CubeMindRouter:
    """MoWM-based topic router using block-code prototypes + DSelect-k gating.

    Each topic is represented by a prototype block-code vector (the bundle
    of representative sentence embeddings). Routing is a single similarity
    computation: compare the query embedding against all prototypes.

    Args:
        topic_names: List of N topic names.
        prototypes: Block-code prototypes, shape (N, k, l).
        k: Number of blocks per vector.
        l: Block length.
        top_k: Number of experts to activate per query.
    """

    def __init__(
        self,
        topic_names: list[str],
        prototypes: np.ndarray,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        top_k: int = 3,
    ) -> None:
        self.topic_names = topic_names
        self.prototypes = prototypes.astype(np.float32)
        self.k = k
        self.l = l
        self.top_k = top_k
        self.bc = BlockCodes(k, l)

        # Name-to-index mapping
        self._name_to_idx = {name: i for i, name in enumerate(topic_names)}

        # DSelect-k gate for sparse expert selection
        self.gate = DSelectKGate(
            num_experts=len(topic_names),
            k=top_k,
        )

    @property
    def topic_count(self) -> int:
        """Number of topics in the router."""
        return len(self.topic_names)

    def route_vector(self, query_vec: np.ndarray) -> tuple[str, float]:
        """Route a block-code vector to the best-matching topic.

        Args:
            query_vec: Block-code vector of shape (k, l).

        Returns:
            (topic_name, similarity_score) for the best match.
        """
        sims = self.bc.similarity_batch(query_vec, self.prototypes)
        best_idx = int(np.argmax(sims))
        return self.topic_names[best_idx], float(sims[best_idx])

    def route_topk_vector(
        self, query_vec: np.ndarray
    ) -> list[tuple[str, float]]:
        """Route to top-k topics with scores.

        Args:
            query_vec: Block-code vector of shape (k, l).

        Returns:
            List of (topic_name, score) sorted by descending score.
        """
        sims = self.bc.similarity_batch(query_vec, self.prototypes)
        top_indices = np.argsort(sims)[::-1][: self.top_k]
        return [(self.topic_names[i], float(sims[i])) for i in top_indices]

    @classmethod
    def from_categories(
        cls,
        categories: dict[str, list[str]],
        embedder,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        max_samples_per_topic: int = 50,
        seed: int = 42,
        top_k: int = 3,
    ) -> CubeMindRouter:
        """Build router from a categories dict by computing topic prototypes.

        For each topic, encodes up to max_samples_per_topic representative
        sentences and bundles them into a single prototype vector.

        Args:
            categories: Dict mapping topic name -> list of representative texts.
            embedder: Object with encode_one(text) -> np.ndarray method.
            k: Number of blocks per vector.
            l: Block length.
            max_samples_per_topic: Maximum sentences to sample per topic.
            seed: Random seed for reproducibility.
            top_k: Number of experts to activate per query.

        Returns:
            Configured CubeMindRouter instance.
        """
        rng = np.random.default_rng(seed)
        bc = BlockCodes(k, l)
        topic_names = sorted(categories.keys())
        prototypes = np.zeros((len(topic_names), k, l), dtype=np.float32)

        for i, topic in enumerate(topic_names):
            sentences = categories[topic]
            n_sample = min(max_samples_per_topic, len(sentences))

            if n_sample == 0:
                # Random prototype for empty topics
                prototypes[i] = bc.random_discrete(seed=seed + i)
                continue

            # Sample representative sentences
            indices = rng.choice(len(sentences), size=n_sample, replace=False)
            sample = [sentences[idx] for idx in indices]

            # Encode and bundle
            vecs = []
            for sent in sample:
                dense = embedder.encode_one(sent)
                d = len(dense)
                if d >= k * l:
                    proj = dense[: k * l].reshape(k, l)
                else:
                    padded = np.zeros(k * l, dtype=np.float32)
                    padded[:d] = dense
                    proj = padded.reshape(k, l)
                exp = np.exp(proj - proj.max(axis=1, keepdims=True))
                vecs.append(
                    (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
                )

            prototypes[i] = bc.bundle(vecs, normalize=True)

        return cls(topic_names, prototypes, k=k, l=l, top_k=top_k)

    def save(self, path: str | Path) -> None:
        """Save router to compressed numpy archive.

        Args:
            path: Output file path (will use .npz extension).
        """
        p = Path(path)
        np.savez_compressed(
            p,
            topic_names=np.array(self.topic_names, dtype=object),
            prototypes=self.prototypes,
            k=np.array([self.k]),
            l=np.array([self.l]),
            top_k=np.array([self.top_k]),
        )

    @classmethod
    def load(cls, path: str | Path) -> CubeMindRouter:
        """Load router from a compressed numpy archive.

        Args:
            path: Path to the .npz file saved by save().

        Returns:
            Loaded CubeMindRouter instance.
        """
        data = np.load(str(path), allow_pickle=True)
        return cls(
            topic_names=list(data["topic_names"]),
            prototypes=data["prototypes"],
            k=int(data["k"][0]),
            l=int(data["l"][0]),
            top_k=int(data["top_k"][0]),
        )
