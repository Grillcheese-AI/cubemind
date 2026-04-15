"""
WorldEncoder — encode natural language states, actions, and narratives
into block-code VSA vectors.

Uses BLAKE2b-seeded RNG for deterministic text-to-vector hashing, with
role binding for structured state representations and positional binding
for ordered narrative encoding.

Part of the Decision Oracle pipeline (Task 2).
"""

from __future__ import annotations

import hashlib

import numpy as np

from cubemind.core.registry import register
from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS: int = 80
    L_BLOCK: int = 128


@register("encoder", "world")
class WorldEncoder:
    """Encode text (states, actions, narratives) into block-code VSA vectors.

    All outputs are (k, l) float32 arrays compatible with BlockCodes operations.
    Hashing is deterministic: the same input text always produces the same vector.

    Args:
        k: Number of blocks (default: K_BLOCKS from cubemind.core).
        l: Block length (default: L_BLOCK from cubemind.core).
    """

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK) -> None:  # noqa: E741
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self._role_cache: dict[str, np.ndarray] = {}

    # ── Core hashing ─────────────────────────────────────────────────────────

    def _hash_to_vec(self, text: str) -> np.ndarray:
        """Deterministic text -> one-hot block-code via BLAKE2b-seeded RNG.

        Hashes the text with BLAKE2b to get a 64-bit seed, then generates
        a discrete one-hot block-code vector using that seed.

        Args:
            text: Arbitrary text string.

        Returns:
            One-hot block-code vector of shape (k, l), dtype float32.
        """
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little")
        # numpy seed must fit in uint64
        seed = seed % (2**63)
        return self.bc.random_discrete(seed=seed)

    # ── Role vectors ─────────────────────────────────────────────────────────

    def _role_vec(self, role: str) -> np.ndarray:
        """Return a cached role vector for the given role name.

        Role vectors are generated via _hash_to_vec with a 'role:' prefix
        to avoid collisions with value vectors.

        Args:
            role: Role name (e.g. "color", "size", "agent").

        Returns:
            Cached one-hot block-code vector of shape (k, l), dtype float32.
        """
        if role not in self._role_cache:
            self._role_cache[role] = self._hash_to_vec(f"__role__:{role}")
        return self._role_cache[role]

    # ── State encoding ───────────────────────────────────────────────────────

    def encode_state(self, attributes: dict[str, str]) -> np.ndarray:
        """Encode a structured state as a bundle of role-bound attribute pairs.

        For each (role, value) pair, binds the role vector with the value
        vector, then bundles all pairs into a single superposition vector.

        Args:
            attributes: Mapping of role names to value strings,
                        e.g. {"color": "red", "size": "large"}.

        Returns:
            Bundled block-code vector of shape (k, l), dtype float32.
        """
        pairs: list[np.ndarray] = []
        for role, value in attributes.items():
            role_v = self._role_vec(role)
            value_v = self._hash_to_vec(value)
            pairs.append(self.bc.bind(role_v, value_v))
        return self.bc.bundle(pairs, normalize=True)

    # ── Action encoding ──────────────────────────────────────────────────────

    def encode_action(self, action_text: str) -> np.ndarray:
        """Encode an action description as a block-code vector.

        Args:
            action_text: Natural language action description.

        Returns:
            One-hot block-code vector of shape (k, l), dtype float32.
        """
        return self._hash_to_vec(action_text)

    # ── Narrative encoding ───────────────────────────────────────────────────

    def encode_narrative(self, text: str) -> np.ndarray:
        """Encode a multi-sentence narrative with positional binding.

        Splits text on sentence boundaries ('. '), assigns each sentence a
        positional vector, binds sentence with position, then bundles all.

        Args:
            text: Multi-sentence narrative text.

        Returns:
            Bundled block-code vector of shape (k, l), dtype float32.
        """
        # Split on '. ' but also handle trailing '.'
        sentences = [s.strip() for s in text.replace(".\n", ". ").split(". ")]
        sentences = [s.rstrip(".") for s in sentences if s]

        bound_sentences: list[np.ndarray] = []
        for i, sentence in enumerate(sentences):
            sent_vec = self._hash_to_vec(sentence)
            pos_vec = self._hash_to_vec(f"__pos__:{i}")
            bound_sentences.append(self.bc.bind(sent_vec, pos_vec))

        return self.bc.bundle(bound_sentences, normalize=True)

    # ── Action variant generation ────────────────────────────────────────────

    def generate_action_variants(
        self, base: np.ndarray, n: int = 128
    ) -> np.ndarray:
        """Generate n personality-flavored variants of a base action vector.

        Each variant is the base action bound with a unique personality
        vector, producing n distinct but related action representations.

        Args:
            base: Base action block-code vector of shape (k, l).
            n: Number of variants to generate (default 128).

        Returns:
            Array of shape (n, k, l), dtype float32.
        """
        variants = np.zeros((n, self.k, self.l), dtype=np.float32)
        for i in range(n):
            personality = self._hash_to_vec(f"__personality__:{i}")
            variants[i] = self.bc.bind(base, personality)
        return variants
