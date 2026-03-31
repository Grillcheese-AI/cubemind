"""
WorldManager — self-organizing specialist world models.

Dynamically spawns and consolidates specialist world models from observed
state transitions.  Each specialist is a rule vector R = unbind(s_after, s_before)
stored in a pre-allocated arena.

Spawn / Consolidate decision:
  - Compute batch cosine similarity between the observed rule and all active
    specialist vectors.
  - max_sim < tau  → spawn a new specialist (copy r_obs into the next free slot).
  - max_sim >= tau → consolidate the best-matching specialist via Oja's rule.

Oja's rule (Oja, 1982):
    w_new = w_old + eta * y * (x - y * w_old)
  where y = w_old · x (the projection) and x = r_observed.
After the update each block of w_new is L2-normalised to keep the specialist
vector on the unit sphere.

This module replaces the 128 static personality vectors in DecisionOracle
with dynamically discovered specialists.
"""

from __future__ import annotations

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS: int = 80
    L_BLOCK: int = 128


class WorldManager:
    """Self-organizing specialist world models via spawn-or-consolidate.

    Maintains a pre-allocated arena of specialist vectors.  Each specialist
    represents a recurring transition rule discovered from (state_before,
    state_after) pairs.

    Args:
        k:          Number of VSA blocks.
        l:          Block length.
        max_worlds: Maximum number of simultaneously active specialists.
        tau:        Similarity threshold in [0, 1].  Observations above tau
                    consolidate into the best match; observations below tau
                    spawn a new specialist.
        oja_lr:     Oja's rule learning rate (eta).
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,  # noqa: E741
        max_worlds: int = 1024,
        tau: float = 0.65,
        oja_lr: float = 0.01,
    ) -> None:
        self.k = k
        self.l = l
        self.tau = tau
        self.oja_lr = oja_lr

        self.bc = BlockCodes(k=k, l=l)

        # Pre-allocated arena: (max_worlds, k, l)
        self._arena = np.zeros((max_worlds, k, l), dtype=np.float32)
        self._obs_counts = np.zeros(max_worlds, dtype=np.int32)
        self.active_worlds: int = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def process_transition(
        self,
        state_before: np.ndarray,
        state_after: np.ndarray,
    ) -> dict:
        """Process a (state_before → state_after) transition.

        Extracts the underlying rule vector, then either spawns a new
        specialist or consolidates the best-matching one via Oja's rule.

        Args:
            state_before: Block-code vector (k, l) before the transition.
            state_after:  Block-code vector (k, l) after the transition.

        Returns:
            Dict with keys:
              - "action":     "spawned" | "consolidated"
              - "world_id":   int — index of the affected specialist
              - "similarity": float — max similarity at decision time
                              (0.0 when spawning from empty arena)

        Raises:
            RuntimeError: When active_worlds == max_worlds and a new specialist
                          would need to be spawned.
        """
        # Step 1: extract the rule as an unbind of the transition
        r_obs: np.ndarray = self.bc.unbind(state_after, state_before)

        # Step 2: first specialist always spawns immediately
        if self.active_worlds == 0:
            return self._spawn(r_obs, similarity=0.0)

        # Step 3: batch similarity against all active specialists
        active_codebook = self._arena[: self.active_worlds]  # (n, k, l)
        similarities = self.bc.similarity_batch(r_obs, active_codebook)  # (n,)
        best_id = int(np.argmax(similarities))
        max_sim = float(similarities[best_id])

        # Step 4: spawn or consolidate
        if max_sim < self.tau:
            return self._spawn(r_obs, similarity=max_sim)
        return self._consolidate(best_id, r_obs, similarity=max_sim)

    def get_specialists(self) -> list[np.ndarray]:
        """Return a list of copies of the active specialist vectors.

        Returns:
            List of float32 arrays, each of shape (k, l).  Modifying the
            returned arrays does not affect the internal arena.
        """
        return [self._arena[i].copy() for i in range(self.active_worlds)]

    def get_obs_count(self, world_id: int) -> int:
        """Return the observation count for a specialist.

        Args:
            world_id: Index of the specialist (0-based).

        Returns:
            Total number of times this specialist has been updated
            (spawned = 1, each consolidation adds 1).

        Raises:
            IndexError: If world_id is out of range [0, active_worlds).
        """
        if world_id < 0 or world_id >= self.active_worlds:
            raise IndexError(
                f"world_id {world_id} out of range [0, {self.active_worlds})"
            )
        return int(self._obs_counts[world_id])

    # ── Private methods ───────────────────────────────────────────────────────

    def _spawn(self, r_observed: np.ndarray, similarity: float) -> dict:
        """Allocate the next free slot and copy r_observed into it.

        Args:
            r_observed: Observed rule vector (k, l).
            similarity: The max similarity computed before deciding to spawn.

        Returns:
            Dict with action="spawned", world_id, and similarity.

        Raises:
            RuntimeError: When the arena is already full.
        """
        max_worlds = self._arena.shape[0]
        if self.active_worlds >= max_worlds:
            raise RuntimeError(
                f"WorldManager arena is full ({max_worlds} specialists). "
                "Increase max_worlds or implement eviction."
            )
        world_id = self.active_worlds
        # Normalise r_observed per-block before storing so arena is consistent
        self._arena[world_id] = self._per_block_l2_normalize(r_observed)
        self._obs_counts[world_id] = 1
        self.active_worlds += 1
        return {"action": "spawned", "world_id": world_id, "similarity": similarity}

    def _consolidate(
        self,
        world_id: int,
        r_observed: np.ndarray,
        similarity: float,
    ) -> dict:
        """
        Updated: Semantically Decoupled Consolidation.
        Uses Oja's rule + Gram-Schmidt Orthogonalization to keep specialists distinct.
        """
        w = self._arena[world_id]  # (k, l)
        x = r_observed.astype(np.float32)

        # 1. STANDARD OJA UPDATE (The 'Learning' Step)
        y = np.einsum("bl,bl->b", w, x)  # (k,)
        w_new = w + self.oja_lr * y[:, np.newaxis] * (x - y[:, np.newaxis] * w)

        # 2. GEOMETRIC DISENTANGLEMENT (The 'SDLS' Step)
        # If we have other specialists, we 'push' this update away from them.
        # This prevents 'Warfare' jargon from leaking into 'Merchant Vessel' logic.
        if self.active_worlds > 1:
            # Flatten for geometric operations
            w_flat = w_new.ravel()
            
            # Get the 'Consensus Direction' of all other specialists
            # This represents the 'Prior/Bias Subspace' from the paper
            others_idx = [i for i in range(self.active_worlds) if i != world_id]
            others = self._arena[others_idx].reshape(len(others_idx), -1)
            bias_axis = np.mean(others, axis=0)
            bias_axis /= (np.linalg.norm(bias_axis) + 1e-6)
            
            # Project w_new onto the orthogonal complement of the bias
            # Formula: w_pure = w_new - (w_new dot bias) * bias
            projection = np.dot(w_flat, bias_axis) * bias_axis
            w_new = (w_flat - projection).reshape(self.k, self.l)

        # 3. PER-BLOCK NORMALIZATION
        self._arena[world_id] = self._per_block_l2_normalize(w_new)
        self._obs_counts[world_id] += 1

        return {"action": "consolidated", "world_id": world_id, "similarity": similarity}
    
    def purify_arena(self):
        """
        Global QR-based Purification.
        Call this after document ingestion to perfectly decouple all specialists.
        Matches the paper's 'QR-based orthogonalization' logic.
        """
        if self.active_worlds < 2: return
        # 1. Get the active slice
        active_slice = self._arena[:self.active_worlds] # (n, k, l)
        n = self.active_worlds
        
        # 2. Dynamically calculate flat dimension D
        D = active_slice.shape[1] * active_slice.shape[2] 
        
        # 3. Reshape to (D, n) for QR
        V = active_slice.reshape(n, D).T 
        # 2. QR Decomposition
        # Q is the orthonormal basis (The Purified Specialists)
        # R is the entanglement matrix (The Bias/Prior)
        Q, _ = np.linalg.qr(V)
        
        # 3. Update arena with purified vectors
        purified = Q.T # (N, D)
        for i in range(self.active_worlds):
            self._arena[i] = purified[i].reshape(self.k, self.l)
            # Re-normalize to keep L2 block properties
            self._arena[i] = self._per_block_l2_normalize(self._arena[i])
            
        print(f"✨ World Arena purified via QR. {self.active_worlds} specialists decoupled.")

    @staticmethod
    def _per_block_l2_normalize(v: np.ndarray) -> np.ndarray:
        """Normalise each block of v to unit L2 norm.

        Args:
            v: Block-code array (k, l).

        Returns:
            Normalised float32 array of the same shape.
        """
        norms = np.linalg.norm(v, axis=-1, keepdims=True)  # (k, 1)
        # Avoid division by zero: blocks with zero norm stay zero
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return (v / safe_norms).astype(np.float32)
