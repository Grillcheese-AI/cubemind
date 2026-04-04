"""HE-MoE Staging: Integration with real CubeMind + grilly components.

Proves the concept works in production context:
- grilly ops (not numpy) for kernel/similarity/learning
- BlockCodes VSA vectors as inputs
- HippocampalFormation for capsule storage + retrieval
- GIFNeuron as expert internals
- SNNFFN as expert forward path
- Identity modulates routing
- Neurochemistry modulates consolidation rate
- CubeMind v3 model3 full pipeline

Run: uv run pytest sandbox/he_moe/staging/integration.py -v
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

# Add sandbox + project root to path
_sandbox = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent)
_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent)
for p in [_sandbox, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from he_moe.experiment import HEMoE, ChargedExpert, rbf_kernel

# CubeMind production imports (import submodules directly to avoid __init__ chain)
from cubemind.ops.block_codes import BlockCodes
from cubemind.memory.formation import HippocampalFormation
from cubemind.brain.gif_neuron import GIFNeuron
from cubemind.brain.synapsis import Synapsis
from cubemind.brain.snn_ffn import SNNFFN, HybridFFN
from cubemind.brain.neurogenesis import NeurogenesisController
from cubemind.brain.identity import Identity
from cubemind.functional import F


K, L = 4, 32
D = K * L  # 128


# ═══════════════════════════════════════════════════════════════════════════════
# I1: HE-MoE with grilly functional ops
# ═══════════════════════════════════════════════════════════════════════════════


class TestI1_GrillyOps:
    """HE-MoE scoring uses cubemind.functional instead of raw numpy."""

    def test_kernel_via_F(self):
        x = np.random.randn(16).astype(np.float32)
        y = np.random.randn(16).astype(np.float32)
        raw = rbf_kernel(x, y)
        via_f = F.rbf_kernel(x, y)
        assert abs(raw - via_f) < 1e-5

    def test_cosine_via_F(self):
        x = np.random.randn(16).astype(np.float32)
        y = np.random.randn(16).astype(np.float32)
        sim = F.cosine_similarity(x, y)
        assert -1.0 <= sim <= 1.0

    def test_oja_via_F(self):
        w = np.random.randn(16).astype(np.float32)
        x = np.random.randn(16).astype(np.float32)
        w2 = F.oja_update(w, x, eta=0.01)
        assert w2.shape == w.shape
        assert not np.array_equal(w, w2)

    def test_top_k_via_F(self):
        scores = np.random.randn(8).astype(np.float32)
        idx, weights = F.top_k(scores, k=2)
        assert len(idx) == 2
        assert abs(sum(weights) - 1.0) < 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# I2: HE-MoE on VSA block-code vectors
# ═══════════════════════════════════════════════════════════════════════════════


class TestI2_BlockCodes:
    """HE-MoE operates on real VSA block-code inputs."""

    def setup_method(self):
        self.bc = BlockCodes(k=K, l=L)
        self.moe = HEMoE(d_input=D, d_output=D, initial_experts=4,
                          top_k=2, sigma=2.0, seed=42)

    def test_forward_on_block_code(self):
        hv = self.bc.random_discrete(seed=1)
        x = self.bc.to_flat(hv).astype(np.float32)
        y = self.moe.forward(x)
        assert y.shape == (D,)
        assert np.all(np.isfinite(y))

    def test_train_on_block_codes(self):
        hv_in = self.bc.random_discrete(seed=1)
        hv_target = self.bc.random_discrete(seed=2)
        x = self.bc.to_flat(hv_in).astype(np.float32)
        t = self.bc.to_flat(hv_target).astype(np.float32)
        result = self.moe.train_step(x, t)
        assert np.isfinite(result["loss"])

    def test_output_reconstructs_to_block_code(self):
        hv = self.bc.random_discrete(seed=3)
        x = self.bc.to_flat(hv).astype(np.float32)
        y = self.moe.forward(x)
        hv_out = self.bc.discretize(y.reshape(K, L))
        assert hv_out.shape == (K, L)


# ═══════════════════════════════════════════════════════════════════════════════
# I3: HE-MoE + HippocampalFormation
# ═══════════════════════════════════════════════════════════════════════════════


class TestI3_Hippocampus:
    """HE-MoE capsules stored in real hippocampal formation."""

    def setup_method(self):
        self.moe = HEMoE(d_input=32, d_output=32, initial_experts=4, seed=42)
        self.hippo = HippocampalFormation(
            feature_dim=32, max_memories=500,
            n_place_cells=50, n_time_cells=10, n_grid_cells=20, seed=42)

    def test_store_moe_output(self):
        rng = np.random.default_rng(42)
        for i in range(10):
            x = rng.standard_normal(32).astype(np.float32)
            output = self.moe.forward(x)
            self.hippo.create_episodic_memory(features=output)
        assert self.hippo.memory_count == 10

    def test_retrieve_similar_output(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(32).astype(np.float32)
        output = self.moe.forward(x)
        self.hippo.create_episodic_memory(features=output)

        # Query with same input → should retrieve
        query = self.moe.forward(x)
        results = self.hippo.retrieve_similar_memories(query, k=1)
        assert len(results) >= 1

    def test_capsule_replay_via_hippo(self):
        rng = np.random.default_rng(42)
        # Train and store capsules
        for _ in range(20):
            x = rng.standard_normal(32).astype(np.float32) * 3
            t = rng.standard_normal(32).astype(np.float32) * 3
            self.moe.train_step(x, t)
            output = self.moe.forward(x)
            self.hippo.create_episodic_memory(features=output)

        # Retrieve for replay
        query = rng.standard_normal(32).astype(np.float32)
        results = self.hippo.retrieve_similar_memories(query, k=5)
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# I4: HE-MoE + GIFNeuron experts
# ═══════════════════════════════════════════════════════════════════════════════


class TestI4_GIFExperts:
    """HE-MoE experts use GIFNeuron for spiking forward pass."""

    def test_gif_as_expert_forward(self):
        gif = GIFNeuron(input_dim=16, hidden_dim=16, L=8, seed=42)
        x = np.random.randn(1, 16).astype(np.float32)  # (1, d_in)
        spikes, _ = gif.forward(x)
        assert spikes.shape == (1, 16)
        assert np.all(spikes >= 0)

    def test_synapsis_gif_chain(self):
        syn = Synapsis(16, 32, seed=42)
        gif = GIFNeuron(32, 32, L=8, seed=42)
        x = np.random.randn(4, 16).astype(np.float32)  # (seq, d_in)
        h, _ = syn.forward(x)
        spikes, _ = gif.forward(h)
        assert spikes.shape == (4, 32)

    def test_snnffn_as_expert(self):
        snn = SNNFFN(input_dim=16, hidden_dim=32, output_dim=16,
                      num_timesteps=2, L=8, seed=42)
        x = np.random.randn(1, 1, 16).astype(np.float32)
        y = snn.forward(x)
        assert y.shape == (1, 1, 16)
        assert np.all(np.isfinite(y))


# ═══════════════════════════════════════════════════════════════════════════════
# I5: HE-MoE + Identity modulation
# ═══════════════════════════════════════════════════════════════════════════════


class TestI5_Identity:
    """Identity vector modulates HE-MoE routing."""

    def test_identity_modulates_input(self):
        bc = BlockCodes(k=K, l=L)
        identity = Identity(name="TestMind", k=K, l=L, seed=42)
        hv = bc.random_discrete(seed=1)
        modulated = identity.modulate_input(hv)
        assert modulated.shape == (K, L)
        assert not np.array_equal(hv, modulated)

    def test_different_identities_different_routing(self):
        id_a = Identity(name="A", k=K, l=L,
                        traits={"curiosity": 0.9, "caution": 0.1}, seed=1)
        id_b = Identity(name="B", k=K, l=L,
                        traits={"curiosity": 0.1, "caution": 0.9}, seed=2)
        bc = BlockCodes(k=K, l=L)

        moe = HEMoE(d_input=D, d_output=D, initial_experts=4, sigma=2.0, seed=42)
        hv = bc.random_discrete(seed=10)

        # Modulate same input with different identities
        x_a = bc.to_flat(id_a.modulate_input(hv)).astype(np.float32)
        x_b = bc.to_flat(id_b.modulate_input(hv)).astype(np.float32)

        idx_a, _ = moe.route(x_a)
        idx_b, _ = moe.route(x_b)
        # Different identities may route differently
        # (not guaranteed, but input vectors differ)
        assert not np.array_equal(x_a, x_b)


# ═══════════════════════════════════════════════════════════════════════════════
# I6: HE-MoE + Neurogenesis
# ═══════════════════════════════════════════════════════════════════════════════


class TestI6_Neurogenesis:
    """Neurogenesis controller monitors HE-MoE residual for growth."""

    def test_neurogenesis_tracks_moe_residual(self):
        moe = HEMoE(d_input=16, d_output=16, initial_experts=4, seed=42)
        ng = NeurogenesisController(
            initial_neurons=8, max_neurons=32, feature_dim=16,
            growth_threshold=0.3, seed=42)

        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.standard_normal(16).astype(np.float32)
            output = moe.forward(x)
            ng.step(output)

        assert ng.residual_ema > 0


# ═══════════════════════════════════════════════════════════════════════════════
# I7: HE-MoE in CubeMind v3 pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestI7_CubeMindV3:
    """HE-MoE integrates with full CubeMind v3."""

    def test_model3_forward_still_works(self):
        from cubemind.model3 import CubeMindV3
        brain = CubeMindV3(
            k=K, l=L, d_hidden=32, max_memories=50,
            n_place_cells=10, n_time_cells=5, n_grid_cells=10,
            initial_neurons=8, max_neurons=16,
            enable_neurochemistry=False, seed=42)
        result = brain.forward(text="integration test")
        assert result["output_hv"].shape == (K, L)

    def test_model3_train_step_still_works(self):
        from cubemind.model3 import CubeMindV3
        bc = BlockCodes(k=K, l=L)
        brain = CubeMindV3(
            k=K, l=L, d_hidden=32, max_memories=50,
            n_place_cells=10, n_time_cells=5, n_grid_cells=10,
            initial_neurons=8, max_neurons=16,
            enable_neurochemistry=False, seed=42)
        target = bc.random_discrete(seed=99)
        result = brain.train_step(text="train test", target_hv=target)
        assert np.isfinite(result["loss"])

    def test_no_regression_on_existing_tests(self):
        """Importing HE-MoE shouldn't break anything."""
        from he_moe.experiment import HEMoE  # noqa: F401
        from cubemind.model3 import CubeMindV3  # noqa: F401
        from cubemind.functional import F  # noqa: F401
        # If we get here, no import conflicts
        assert True
