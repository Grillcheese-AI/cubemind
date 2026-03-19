"""CubeMind v2 — full MoWM pipeline orchestrator.

Wires all modules into the complete pipeline:
    Input → Perception → Routing → Memory → Detection → Execution → Answer

Every stage emits telemetry for live monitoring and paper-quality plots.
All tensor operations route through grilly's GPU backend where available.

Usage:
    model = CubeMind(n_experts=8, n_codebook=16)
    model.attach_router(router)
    result = model.forward("what is the capital of France?", embedder=enc)
"""

from __future__ import annotations

import numpy as np

from cubemind.core import D_VSA, K_BLOCKS, L_BLOCK
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


class CubeMind:
    """Full CubeMind MoWM architecture.

    Pipeline: Input → Perception → Route → Detect → Execute → Answer

    Args:
        k: Number of blocks per vector.
        l: Block length.
        n_experts: Number of topic experts (set by router when attached).
        n_hmm_rules: Number of HMM rules per expert.
        n_codebook: Codebook entries for HMM states.
        d_hidden: Hidden dimension for HYLA hypernetwork.
        cache_size: Maximum VSACache entries.
        gamma: Discount factor for CVL.
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
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.seed = seed

        # ── Ops ───────────────────────────────────────────────
        self.bc = BlockCodes(k, l)

        # ── Perception ────────────────────────────────────────
        self.encoder = Encoder(k=k, l=l)

        # ── Routing (attached externally) ─────────────────────
        self.router = None
        self.n_experts = n_experts

        # ── Detection: HMM ensemble ──────────────────────────
        codebook = self.bc.codebook_discrete(n_codebook, seed=seed)
        self.codebook = codebook
        self.hmm = HMMEnsemble(codebook, n_rules=n_hmm_rules, seed=seed)

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
        self.decoder = Decoder(codebook)

        # ── Memory ───────────────────────────────────────────
        self.cache = VSACache(max_size=cache_size, d_vsa=self.d_vsa)
        self.hippocampal = HippocampalMemory(d_model=self.d_vsa, capacity=cache_size)

        # ── Context aggregation ──────────────────────────────
        self.combiner = CombinerAxialAttention(d_model=self.d_vsa)

        # ── State ────────────────────────────────────────────
        self._step = 0

    def attach_router(self, router) -> None:
        """Attach a CubeMindRouter for topic routing.

        Args:
            router: CubeMindRouter instance with topic prototypes.
        """
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
        """Full pipeline: Input → Perceive → Route → Detect → Execute → Answer.

        Provide either `text` (with optional `embedder`) or a pre-computed
        block-code `phi`.

        Args:
            text: Input text string (requires embedder for GPU path).
            phi: Pre-computed block-code vector (k, l). Overrides text.
            embedder: Object with encode_one(text) method for perception.
            context: Previous block-code vectors for HMM context window.

        Returns:
            Dict with keys: topic, topic_score, surprise, stress,
            hmm_prediction, hmm_weights, q_value, output, answer, step.
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

        # ── 3. Memory: surprise + stress ──────────────────────
        with metrics.record_timing("memory.latency_ms"):
            surprise = self.cache.surprise(phi_flat)
            stress = self.cache.stress()
            self.cache.add(phi_flat, np.array([surprise, stress], dtype=np.float32))
            metrics.record("memory.surprise", surprise)
            metrics.record("memory.stress", stress)
            metrics.record("memory.cache_size", float(self.cache.size))

        # ── 4. Detection: HMM ensemble ────────────────────────
        with metrics.record_timing("detection.latency_ms"):
            obs_sequence = (context or []) + [phi]
            hmm_pred, hmm_weights = self.hmm.predict(obs_sequence)
            metrics.record("detection.log_likelihood", float(hmm_weights.max()))

        # ── 5. Execution: HYLA + CVL ──────────────────────────
        with metrics.record_timing("execution.latency_ms"):
            hyla_out = self.hyla.forward(phi_flat, phi_flat)
            action = hyla_out[: self.cvl.d_action]
            q_value = self.cvl.q_value(
                phi_flat[: self.cvl.d_state], action
            )
            metrics.record("execution.q_value", q_value)

        # ── 6. Answer: decode output ──────────────────────────
        with metrics.record_timing("answer.latency_ms"):
            output_bc = self.bc.discretize(
                self.bc.from_flat(hyla_out, self.k)
            )
            answer_label, answer_sim, answer_idx = self.decoder.decode(output_bc)
            metrics.record("answer.confidence", answer_sim)

        # ── 7. Store in hippocampal memory ────────────────────
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
            "step": self._step,
        }

    # ══════════════════════════════════════════════════════════
    # Training
    # ══════════════════════════════════════════════════════════

    def train_step(
        self,
        observations: list[np.ndarray],
        target: np.ndarray,
        lr: float = 0.01,
    ) -> float:
        """One training step: HMM + HYLA gradient update.

        Args:
            observations: Sequence of block-code vectors (each k, l).
            target: Target block-code vector (k, l).
            lr: Learning rate.

        Returns:
            Training loss (MSE between prediction and target).
        """
        # HMM training
        hmm_losses = self.hmm.train_step(observations, target, lr=lr)
        hmm_loss = float(np.mean(hmm_losses))
        metrics.record("training.loss", hmm_loss)
        return hmm_loss

    def train_step_em(
        self,
        sequences: list[list[np.ndarray]],
        smoothing: float = 0.01,
    ) -> float:
        """Baum-Welch EM training step on multiple sequences.

        Args:
            sequences: List of observation sequences.
            smoothing: Laplace smoothing for transition updates.

        Returns:
            Total log-likelihood across all sequences.
        """
        total_ll = self.hmm.rules[0].train_step_em(sequences, smoothing)
        metrics.record("training.em_log_likelihood", total_ll)
        return total_ll

    # ══════════════════════════════════════════════════════════
    # Introspection
    # ══════════════════════════════════════════════════════════

    @property
    def stats(self) -> dict:
        """Current model statistics."""
        return {
            "step": self._step,
            "cache_size": self.cache.size,
            "cache_stress": self.cache.stress(),
            "n_experts": self.n_experts,
            "has_router": self.router is not None,
            "hippocampal_episodes": len(self.hippocampal._episodes),
        }

    def __repr__(self) -> str:
        return (
            f"CubeMind(k={self.k}, l={self.l}, d_vsa={self.d_vsa}, "
            f"experts={self.n_experts}, step={self._step})"
        )
