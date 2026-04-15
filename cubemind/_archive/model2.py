"""CubeMind v2.1 — Oja-Plastic NVSA architecture.

Extends CubeMind with neuroplastic VSA memory via Oja's rule.
Standard VSA codebooks are frozen — once generated, they never change.
This model treats VSA hypervectors as pseudo-neurons and applies Oja's
learning rule for continuous unsupervised adaptation:

    Δm = η·y·(x - y·m)

where y = m·x is the VSA similarity (activation), x is the input,
and m is the memory vector. This update has two key properties:

1. **Self-normalizing**: converges to ||m|| = 1.0, preserving the
   L1 block-code normalization required for probability distributions.

2. **PCA extraction**: over many updates, m converges to the principal
   component of the input stream, actively decaying superposition noise
   while amplifying consistent semantic signals.

Applications in CubeMind:
- Hippocampal consolidation: recalled memories morph into archetypes
- Dynamic codebooks: concept vectors adapt to environment statistics
- Cache plasticity: frequently-accessed entries sharpen over time

Uses grilly._bridge.hebbian_learning for GPU-accelerated updates
when available, with pure-Python fallback.

Reference: Oja, E. (1982). "Simplified neuron model as a principal
component analyzer." Journal of Mathematical Biology, 15(3), 267-273.
"""

from __future__ import annotations

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.execution.cvl import ContrastiveValueEstimator
from cubemind.execution.decoder import Decoder
from cubemind.execution.hyla import HYLA
from cubemind.memory.cache import VSACache
from cubemind.memory.hippocampal import HippocampalMemory
from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.encoder import Encoder
from cubemind.reasoning.combiner import CombinerAxialAttention
from cubemind.reasoning.hmm_rule import HMMEnsemble
from cubemind.telemetry import metrics




# ── GPU bridge for Oja updates ──────────────────────────────────────────────

_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass



# ── Oja's Rule ──────────────────────────────────────────────────────────────


def oja_update(
    m: np.ndarray,
    x: np.ndarray,
    eta: float = 0.01,
) -> np.ndarray:
    """Apply Oja's learning rule to a VSA memory vector.

    Updates m in the direction of x, with self-normalizing decay that
    prevents magnitude explosion and extracts the principal component.

    Δm = η·y·(x - y·m)  where y = m·x (VSA similarity/activation)

    Args:
        m: Memory vector (d,) or (k, l) — gets updated.
        x: Input vector, same shape as m.
        eta: Learning rate. Lower = more stable, higher = faster adaptation.

    Returns:
        Updated memory vector, same shape as input.
    """
    m_flat = m.ravel().astype(np.float32)
    x_flat = x.ravel().astype(np.float32)

    # Activation: dot product (VSA similarity without normalization)
    y = float(np.dot(m_flat, x_flat))

    # Oja update: Δm = η·y·(x - y·m)
    eta_y = eta * y
    m_flat = m_flat + eta_y * (x_flat - y * m_flat)

    return m_flat.reshape(m.shape)


def oja_update_batch(
    memories: np.ndarray,
    inputs: np.ndarray,
    eta: float = 0.01,
) -> np.ndarray:
    """Batch Oja update for multiple memory-input pairs.

    Uses grilly GPU bridge (oja-learning.spv) when available.

    Args:
        memories: (n, d) memory vectors.
        inputs: (n, d) input vectors.
        eta: Learning rate.

    Returns:
        Updated memory vectors (n, d).
    """
    n, d = memories.shape

    # GPU path: dispatch oja-learning.spv via grilly bridge
    if _bridge is not None:
        try:
            result = _bridge.oja_learning(
                memories.astype(np.float32),
                inputs.astype(np.float32),
                n, d, eta,
            )
            if result is not None:
                return np.asarray(result, dtype=np.float32).reshape(n, d)
        except Exception:
            pass

    # CPU fallback: vectorized numpy
    y = np.sum(memories * inputs, axis=1, keepdims=True)  # (n, 1)
    eta_y = eta * y
    updated = memories + eta_y * (inputs - y * memories)
    return updated.astype(np.float32)


def oja_update_blockcode(
    m: np.ndarray,
    x: np.ndarray,
    bc: BlockCodes,
    eta: float = 0.01,
) -> np.ndarray:
    """Oja update for block-code vectors with per-block normalization.

    After the Oja update, re-normalizes each block to maintain valid
    probability distributions (sum-to-one per block).

    Args:
        m: Block-code memory vector (k, l).
        x: Block-code input vector (k, l).
        bc: BlockCodes instance for normalization.
        eta: Learning rate.

    Returns:
        Updated and normalized block-code vector (k, l).
    """
    updated = oja_update(m, x, eta)

    # Re-normalize per block to maintain probability distribution
    block_sums = updated.sum(axis=-1, keepdims=True)
    block_sums = np.where(block_sums == 0, 1.0, block_sums)
    normalized = (updated / block_sums).astype(np.float32)

    # Clamp negatives (Oja can produce small negatives during convergence)
    normalized = np.maximum(normalized, 0.0)
    block_sums = normalized.sum(axis=-1, keepdims=True)
    block_sums = np.where(block_sums == 0, 1.0, block_sums)

    return (normalized / block_sums).astype(np.float32)


# ── Plastic Codebook ────────────────────────────────────────────────────────


class PlasticCodebook:
    """VSA codebook with Oja-driven adaptation.

    Standard codebooks are frozen at initialization. This codebook
    allows concept vectors to slowly rotate toward the actual data
    distribution via Oja's rule, enabling unsupervised domain adaptation.

    Args:
        bc: BlockCodes instance.
        n_entries: Number of codebook entries.
        eta: Oja learning rate for codebook adaptation.
        seed: Random seed for initial codebook.
    """

    def __init__(
        self,
        bc: BlockCodes,
        n_entries: int = 16,
        eta: float = 0.005,
        seed: int = 42,
    ) -> None:
        self.bc = bc
        self.eta = eta
        self.entries = bc.codebook_discrete(n_entries, seed=seed)
        self.access_count = np.zeros(n_entries, dtype=np.int64)
        self._n = n_entries

    @property
    def n_entries(self) -> int:
        return self._n

    def lookup(self, query: np.ndarray) -> tuple[int, float]:
        """Find the nearest codebook entry and optionally adapt it.

        Args:
            query: Block-code query vector (k, l).

        Returns:
            (best_index, similarity)
        """
        sims = self.bc.similarity_batch(query, self.entries)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        self.access_count[best_idx] += 1
        return best_idx, best_sim

    def adapt(self, idx: int, observation: np.ndarray) -> None:
        """Apply Oja update to a codebook entry.

        Call this after a successful lookup to adapt the matched
        concept vector toward the observed input.

        Args:
            idx: Codebook entry index.
            observation: Observed block-code vector (k, l).
        """
        self.entries[idx] = oja_update_blockcode(
            self.entries[idx], observation, self.bc, self.eta
        )

    def adapt_nearest(self, observation: np.ndarray) -> tuple[int, float]:
        """Lookup + adapt in one step.

        Args:
            observation: Observed block-code vector (k, l).

        Returns:
            (best_index, similarity)
        """
        idx, sim = self.lookup(observation)
        self.adapt(idx, observation)
        return idx, sim


# ── CubeMind v2.1: Oja-Plastic Architecture ────────────────────────────────


class CubeMindPlastic:
    """CubeMind with Oja-plastic VSA memory and codebooks.

    Extends the base CubeMind architecture with:
    1. Plastic codebook: concept vectors adapt via Oja's rule
    2. Hippocampal consolidation: recalled memories sharpen over time
    3. Cache plasticity: frequently-accessed entries converge to archetypes

    Args:
        k: Number of blocks per vector.
        l: Block length.
        n_experts: Number of topic experts.
        n_hmm_rules: Number of HMM rules per expert.
        n_codebook: Codebook entries for HMM states.
        d_hidden: Hidden dimension for HYLA hypernetwork.
        cache_size: Maximum VSACache entries.
        gamma: Discount factor for CVL.
        oja_eta: Oja learning rate for memory plasticity.
        codebook_eta: Oja learning rate for codebook adaptation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        n_experts: int = 8,
        n_hmm_rules: int = 4,
        n_codebook: int = 16,
        d_hidden: int = 128,
        cache_size: int = 1000,
        gamma: float = 0.99,
        oja_eta: float = 0.01,
        codebook_eta: float = 0.005,
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.seed = seed
        self.oja_eta = oja_eta

        # ── Ops ───────────────────────────────────────────────
        self.bc = BlockCodes(k, l)

        # ── Perception ────────────────────────────────────────
        self.encoder = Encoder(k=k, l=l)

        # ── Routing (attached externally) ─────────────────────
        self.router = None
        self.n_experts = n_experts

        # ── Plastic codebook ──────────────────────────────────
        self.plastic_codebook = PlasticCodebook(
            self.bc, n_entries=n_codebook,
            eta=codebook_eta, seed=seed,
        )
        # Also keep a frozen codebook for HMM (stability)
        self.codebook = self.bc.codebook_discrete(n_codebook, seed=seed)

        # ── Detection: HMM ensemble ──────────────────────────
        self.hmm = HMMEnsemble(self.codebook, n_rules=n_hmm_rules, seed=seed)

        # ── Execution: HYLA hypernetwork ─────────────────────
        self.hyla = HYLA(
            d_vsa=self.d_vsa,
            d_hidden=d_hidden,
            d_out=self.d_vsa,
            k=k, l=l, seed=seed,
        )

        # ── Execution: Contrastive Value Learning ────────────
        self.cvl = ContrastiveValueEstimator(
            d_state=self.d_vsa,
            d_action=self.d_vsa,
            d_latent=d_hidden,
            gamma=gamma,
            seed=seed,
        )

        # ── Execution: Decoder ───────────────────────────────
        self.decoder = Decoder(self.codebook)

        # ── Memory ───────────────────────────────────────────
        self.cache = VSACache(max_size=cache_size, d_vsa=self.d_vsa)
        self.hippocampal = HippocampalMemory(d_model=self.d_vsa, capacity=cache_size)

        # ── Context aggregation ──────────────────────────────
        self.combiner = CombinerAxialAttention(d_model=self.d_vsa)
        

        # ── State ────────────────────────────────────────────
        self._step = 0

    def attach_router(self, router) -> None:
        """Attach a CubeMindRouter for topic routing."""
        self.router = router
        self.n_experts = router.topic_count

    # ══════════════════════════════════════════════════════════
    # Pipeline
    # ══════════════════════════════════════════════════════════

    def forward(
        self,
        text: str | None = None,
        phi: np.ndarray | None = None,
        embedder=None,
        context: list[np.ndarray] | None = None,
    ) -> dict:
        """Full pipeline with Oja-plastic memory consolidation.

        Same interface as CubeMind.forward, but with three additions:
        1. Plastic codebook lookup adapts concept vectors
        2. Cache hits trigger Oja consolidation of the matched memory
        3. Hippocampal recall sharpens retrieved episodes
        """
        self._step += 1

        # ── 1. Perception ─────────────────────────────────────
        with metrics.record_timing("perception.latency_ms"):
            if phi is None:
                if text is not None:
                    phi = self.encoder.encode(text)
                else:
                    phi = self.bc.random_discrete(seed=self._step)

        phi_flat = phi.ravel().astype(np.float32)

        # ── 2. Routing ────────────────────────────────────────
        topic, topic_score = None, 0.0
        with metrics.record_timing("routing.latency_ms"):
            if self.router is not None:
                topic, topic_score = self.router.route_vector(phi)
                metrics.record("routing.top_score", topic_score)

        # ── 3. Memory: surprise + stress + Oja consolidation ─
        with metrics.record_timing("memory.latency_ms"):
            surprise = self.cache.surprise(phi_flat)
            stress = self.cache.stress()

            # Oja consolidation: if cache hit is strong, adapt the memory
            if self.cache.size > 0:
                sims, matched_keys, indices = self.cache.lookup(phi_flat, k=1)
                if sims.size > 0 and float(sims[0, 0]) > 0.7:
                    # Strong match — apply Oja to sharpen the stored memory
                    idx = int(indices[0, 0])
                    stored = self.cache.keys[idx].astype(np.float32)
                    updated = oja_update(stored, phi_flat, self.oja_eta)
                    self.cache.keys[idx] = np.sign(updated).astype(np.int8)
                    metrics.record("memory.oja_consolidation", 1.0)

            self.cache.add(phi_flat, np.array([surprise, stress], dtype=np.float32))
            metrics.record("memory.surprise", surprise)
            metrics.record("memory.stress", stress)
            metrics.record("memory.cache_size", float(self.cache.size))

        # ── 4. Plastic codebook lookup ────────────────────────
        with metrics.record_timing("codebook.latency_ms"):
            cb_idx, cb_sim = self.plastic_codebook.adapt_nearest(phi)
            metrics.record("codebook.best_sim", cb_sim)
            metrics.record("codebook.best_idx", float(cb_idx))

        # ── 5. Detection: HMM ensemble ────────────────────────
        with metrics.record_timing("detection.latency_ms"):
            obs_sequence = (context or []) + [phi]
            hmm_pred, hmm_weights = self.hmm.predict(obs_sequence)
            metrics.record("detection.log_likelihood", float(hmm_weights.max()))

        # ── 6. Execution: HYLA + CVL ──────────────────────────
        with metrics.record_timing("execution.latency_ms"):
            hyla_out = self.hyla.forward(phi_flat, phi_flat)
            action = hyla_out[: self.cvl.d_action]
            q_value = self.cvl.q_value(
                phi_flat[: self.cvl.d_state], action
            )
            metrics.record("execution.q_value", q_value)

        # ── 7. Answer: decode output ──────────────────────────
        with metrics.record_timing("answer.latency_ms"):
            output_bc = self.bc.discretize(
                self.bc.from_flat(hyla_out, self.k)
            )
            answer_label, answer_sim, answer_idx = self.decoder.decode(output_bc)
            metrics.record("answer.confidence", answer_sim)

        # ── 8. Store in hippocampal memory ────────────────────
        self.hippocampal.store(phi_flat, content_tag=text or "")

        return {
            "topic": topic,
            "topic_score": topic_score,
            "surprise": surprise,
            "stress": stress,
            "hmm_prediction": hmm_pred,
            "hmm_weights": hmm_weights,
            "q_value": q_value,
            "output": output_bc,
            "answer": answer_label,
            "answer_confidence": answer_sim,
            "codebook_idx": cb_idx,
            "codebook_sim": cb_sim,
            "step": self._step,
        }

    # ══════════════════════════════════════════════════════════
    # Hippocampal Consolidation (offline)
    # ══════════════════════════════════════════════════════════

    def consolidate_memories(self, n_passes: int = 1) -> dict:
        """Run Oja consolidation over all cached memories.

        This is the "sleep" phase — iterate over cached memories and
        apply Oja updates between similar entries. Over multiple passes,
        each memory converges toward the archetype of its cluster.

        Args:
            n_passes: Number of consolidation sweeps.

        Returns:
            Dict with consolidation statistics.
        """
        if self.cache.size < 2:
            return {"n_updates": 0, "mean_sim": 0.0}

        n_updates = 0
        total_sim = 0.0

        for _ in range(n_passes):
            keys = self.cache.keys[: self.cache.size].astype(np.float32)

            for i in range(self.cache.size):
                # Find nearest neighbor (excluding self)
                sims, _, indices = self.cache.lookup(keys[i], k=2)
                if indices.shape[1] < 2:
                    continue

                # Skip self-match (index 0 is usually self)
                j_idx = int(indices[0, 1])
                j_sim = float(sims[0, 1])

                if j_sim > 0.3:  # Only consolidate similar memories
                    neighbor = self.cache.keys[j_idx].astype(np.float32)
                    updated = oja_update(keys[i], neighbor, self.oja_eta)
                    self.cache.keys[i] = np.sign(updated).astype(np.int8)
                    n_updates += 1
                    total_sim += j_sim

        mean_sim = total_sim / max(n_updates, 1)
        return {"n_updates": n_updates, "mean_sim": mean_sim}

    # ══════════════════════════════════════════════════════════
    # Introspection
    # ══════════════════════════════════════════════════════════

    @property
    def stats(self) -> dict:
        """Current model statistics including plasticity metrics."""
        return {
            "step": self._step,
            "cache_size": self.cache.size,
            "cache_stress": self.cache.stress(),
            "n_experts": self.n_experts,
            "has_router": self.router is not None,
            "hippocampal_episodes": len(self.hippocampal._episodes),
            "codebook_access": self.plastic_codebook.access_count.tolist(),
            "oja_eta": self.oja_eta,
        }

    def __repr__(self) -> str:
        return (
            f"CubeMindPlastic(k={self.k}, l={self.l}, d_vsa={self.d_vsa}, "
            f"experts={self.n_experts}, step={self._step}, "
            f"oja_eta={self.oja_eta})"
        )
