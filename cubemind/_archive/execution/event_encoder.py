"""
EventEncoder — encode rich JSONL historical events into block-code VSA vectors.

Converts events with participants, affect_tags, topic_tags, and causal_links into
deterministic (k, l) float32 block-code vectors using BLAKE2b-seeded RNG.

Uses the same BLAKE2b hashing pattern as WorldEncoder:
    digest = BLAKE2b(text) -> seed -> bc.random_discrete(seed=seed)

Role binding encodes structured relationships within events:
    bind(entity_vec, role_vec)  — attaches a role to each entity/topic
    bundle([...], normalize=True) — superposes all components into one vector

Part of the WorldManager pipeline (Task 3).
"""

from __future__ import annotations

import hashlib

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS: int = 80
    L_BLOCK: int = 128


class EventEncoder:
    """Encode historical JSONL events into block-code VSA vectors.

    Each event is converted into a single (k, l) float32 block-code that
    bundles together: the summary text, participant bindings, affect-tag
    weighted contributions, topic bindings, and scalar energy/pleasantness.

    All hashing is deterministic via BLAKE2b: the same event dict always
    produces the same vector. A text->vector cache prevents redundant work.

    Args:
        k: Number of blocks (default: K_BLOCKS from cubemind.core).
        l: Block length (default: L_BLOCK from cubemind.core).
    """

    def __init__(self, k: int = K_BLOCKS, l: int = L_BLOCK) -> None:  # noqa: E741
        self.k = k
        self.l = l
        self.bc = BlockCodes(k=k, l=l)
        self._cache: dict[str, np.ndarray] = {}  # text -> vector cache

    # ── Core hashing ─────────────────────────────────────────────────────────

    def _hash_vec(self, text: str) -> np.ndarray:
        """Deterministic text -> one-hot block-code via BLAKE2b-seeded RNG.

        Result is cached so repeated calls for the same text return the same
        object (identity equality), which is used by tests to verify caching.

        Args:
            text: Arbitrary UTF-8 string.

        Returns:
            Cached one-hot block-code of shape (k, l), dtype float32.
        """
        if text not in self._cache:
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            seed = int.from_bytes(digest, "little") % (2**63)
            self._cache[text] = self.bc.random_discrete(seed=seed)
        return self._cache[text]

    # ── Role vectors ─────────────────────────────────────────────────────────

    def _role_vec(self, role: str) -> np.ndarray:
        """Return a cached role vector for the given role name.

        Uses the '__role__:{role}' prefix (matching WorldEncoder) to avoid
        collisions with plain value vectors.

        Args:
            role: Role name (e.g. "Participant", "HasTopic", "Causes").

        Returns:
            Cached one-hot block-code of shape (k, l), dtype float32.
        """
        return self._hash_vec(f"__role__:{role}")

    # ── Event encoding ────────────────────────────────────────────────────────

    def encode_event(self, event: dict) -> np.ndarray:
        """Encode a historical event dict into a single block-code vector.

        The resulting vector is a normalised bundle of:
          1. Summary text as the base semantic vector.
          2. Participants: bind(entity_name_vec, entity_type_role_vec) per entry.
          3. Affect tags: score * hash_vec(tag_id) per entry, bundled.
          4. Topic tags: bind(topic_vec, HasTopic_role_vec) per tag.
          5. Energy scalar: energy * hash_vec("__energy__").
          6. Pleasantness scalar: pleasantness * hash_vec("__pleasantness__").

        Missing optional fields are silently skipped.

        Args:
            event: Dict with fields:
                - summary (str, required)
                - participants (list of {name, entity_type}, optional)
                - affect_tags (list of {id, score}, optional)
                - topic_tags (list of str, optional)
                - energy (float, optional)
                - pleasantness (float, optional)
                - causal_link (dict, optional — not used here, see encode_causal_edge)

        Returns:
            Normalised block-code vector of shape (k, l), dtype float32.
        """
        components: list[np.ndarray] = []

        # 1. Summary — base semantic vector
        summary = event.get("summary", "")
        if summary:
            components.append(self._hash_vec(summary))

        # 2. Participants — bind entity name with its type role
        participant_role = self._role_vec("Participant")
        for participant in event.get("participants", []):
            name = participant.get("name", "")
            entity_type = participant.get("entity_type", "ENTITY")
            if name:
                name_vec = self._hash_vec(name)
                type_role = self._role_vec(entity_type)
                # bind entity name with type role, then bind with participant role
                typed_entity = self.bc.bind(name_vec, type_role)
                components.append(self.bc.bind(typed_entity, participant_role))

        # 3. Affect tags — score-weighted hash vectors
        for tag in event.get("affect_tags", []):
            tag_id = tag.get("id", "")
            score = float(tag.get("score", 1.0))
            if tag_id:
                components.append(score * self._hash_vec(tag_id))

        # 4. Topic tags — bind topic with HasTopic role
        has_topic_role = self._role_vec("HasTopic")
        for topic in event.get("topic_tags", []):
            if topic:
                topic_vec = self._hash_vec(topic)
                components.append(self.bc.bind(topic_vec, has_topic_role))

        # 5. Energy scalar
        if "energy" in event:
            energy = float(event["energy"])
            components.append(energy * self._hash_vec("__energy__"))

        # 6. Pleasantness scalar
        if "pleasantness" in event:
            pleasantness = float(event["pleasantness"])
            components.append(pleasantness * self._hash_vec("__pleasantness__"))

        # If no components were produced (empty event), return a zero vector
        if not components:
            return np.zeros((self.k, self.l), dtype=np.float32)

        return self.bc.bundle(components, normalize=True)

    # ── Causal edge encoding ──────────────────────────────────────────────────

    def encode_causal_edge(
        self, event_a: dict, event_b: dict, weight: float = 1.0
    ) -> np.ndarray:
        """Encode a directed causal edge between two events.

        Computes:
            edge = weight * bind(bind(V_a, Causes_role), V_b)

        where V_a and V_b are the full event vectors for event_a and event_b.
        The Causes_role vector distinguishes the causal direction.

        Args:
            event_a: The causing event dict.
            event_b: The effect event dict.
            weight: Scalar weight for the causal influence (default 1.0).

        Returns:
            Block-code vector of shape (k, l), dtype float32.
        """
        v_a = self.encode_event(event_a)
        v_b = self.encode_event(event_b)
        causes_role = self._role_vec("Causes")
        bound_a = self.bc.bind(v_a, causes_role)
        edge = self.bc.bind(bound_a, v_b)
        return (weight * edge).astype(np.float32)
