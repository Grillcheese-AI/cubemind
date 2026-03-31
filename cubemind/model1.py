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


import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.execution.cvl import ContrastiveValueEstimator
from cubemind.execution.decoder import Decoder
from cubemind.execution.hyla import HYLA
from cubemind.memory.cache import VSACache
from cubemind.memory.hippocampal import HippocampalMemory
from cubemind.ops.block_codes import BlockCodes
from cubemind.perception.encoder import Encoder
from cubemind.reasoning.combiner import CombinerAxialAttention, HyperAxialAttention
from cubemind.reasoning.hmm_rule import HMMEnsemble
from cubemind.telemetry import metrics


class CubeMind:
    """Full CubeMind MoWM architecture with Hyper-Axial Attention support."""

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
        use_hyper_attention: bool = True,  # <--- New Toggle
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.seed = seed

        # ── Ops & Perception ──
        self.bc = BlockCodes(k, l)
        self.encoder = Encoder(k=k, l=l)

        # ── Routing & Detection ──
        self.router = None
        self.n_experts = n_experts
        codebook = self.bc.codebook_discrete(n_codebook, seed=seed)
        self.codebook = codebook
        self.hmm = HMMEnsemble(codebook, n_rules=n_hmm_rules, seed=seed)

        # ── Execution ──
        self.hyla = HYLA(d_vsa=self.d_vsa, d_hidden=d_hidden, d_out=self.d_vsa, k=k, l=l, seed=seed)
        self.cvl = ContrastiveValueEstimator(d_state=self.d_vsa, d_action=self.d_vsa, d_latent=d_hidden, gamma=gamma, seed=seed)
        self.decoder = Decoder(codebook)

        # ── Memory ──
        self.cache = VSACache(max_size=cache_size, d_vsa=self.d_vsa)
        self.hippocampal = HippocampalMemory(d_model=self.d_vsa, capacity=cache_size)

        # ── Context Aggregation (The "Combiner") ──
        if use_hyper_attention:
            # Linear-time O(L) Attention
            self.combiner = HyperAxialAttention(
                d_model=self.d_vsa, 
                num_heads=4, 
                bucket_size=128
            )
        else:
            # Sub-quadratic O(L*sqrt(L)) Attention
            self.combiner = CombinerAxialAttention(d_model=self.d_vsa)

        self._step = 0

    def forward(
        self,
        text: str | None = None,
        phi: np.ndarray | None = None,
        context: list[np.ndarray] | None = None,
    ) -> dict:
        self._step += 1

        # ── 1. Perception ──
        if phi is None:
            phi = self.encoder.encode(text) if text else self.bc.random_discrete(seed=self._step)
        
        phi_flat = phi.ravel().astype(np.float32)

        # ── 2. Routing ──
        topic, topic_score = None, 0.0
        if self.router is not None:
            topic, topic_score = self.router.route_vector(phi)

        # ── 3. Memory ──
        surprise = self.cache.surprise(phi_flat)
        stress = self.cache.stress()
        self.cache.add(phi_flat, np.array([surprise, stress], dtype=np.float32))

        # ── 4. Detection (HMM) ──
        obs_sequence = (context or []) + [phi]
        hmm_pred, hmm_weights = self.hmm.predict(obs_sequence)

        # ── 5. NEW: Contextual Aggregation (Combiner) ──
        # This is where we test the HyperAxialAttention. 
        # We aggregate the history (context) with the current state (phi).
        with metrics.record_timing("combiner.latency_ms"):
            # Prepare sequence: (L, d_vsa)
            if context and len(context) > 0:
                # Stack context + current phi
                history_stack = np.stack([c.ravel() for c in context] + [phi_flat])
                # Run through HyperAxialAttention
                combined_seq = self.combiner.forward(history_stack)
                # Take the last vector as our "informed" state
                phi_integrated = combined_seq[-1]
            else:
                phi_integrated = phi_flat

        # ── 6. Execution (HYLA + CVL) ──
        # We now pass 'phi_integrated' instead of 'phi_flat'
        with metrics.record_timing("execution.latency_ms"):
            hyla_out = self.hyla.forward(phi_integrated, phi_integrated)
            action = hyla_out[: self.cvl.d_action]
            q_value = self.cvl.q_value(phi_integrated[: self.cvl.d_state], action)

        # ── 7. Answer ──
        output_bc = self.bc.discretize(self.bc.from_flat(hyla_out, self.k))
        answer_label, answer_sim, _ = self.decoder.decode(output_bc)

        # ── 8. Store ──
        self.hippocampal.store(phi_flat, content_tag=text or "")

        return {
            "topic": topic,
            "phi_integrated": phi_integrated, # For debugging
            "q_value": q_value,
            "answer": answer_label,
            "confidence": answer_sim,
            "step": self._step,
        }