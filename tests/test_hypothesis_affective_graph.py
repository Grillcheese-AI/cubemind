"""Hypothesis 1: Affective Graph Message Passing via Hormonal Modulation.

Tests that the 4-hormone ODE system can dynamically control the blending
factor alpha in VS-Graph message passing, creating an emotionally-modulated
graph reasoner.

High dopamine → alpha ~0.7 (trust neighbors, explore)
High cortisol → alpha ~0.3 (trust self, consolidate)

All tests are self-contained — no modifications to existing code.
"""

import numpy as np
import pytest

from cubemind.experimental.vs_graph import (
    associative_message_passing,
    spike_diffusion,
    VSGraph,
)
from cubemind.perception.snn import NeurochemicalState


# ── The new function under test ──────────────────────────────────────

def affective_alpha(state: NeurochemicalState) -> float:
    """Compute dynamic blending alpha from neurochemical state.

    alpha controls self-vs-neighbor blend: alpha*self + (1-alpha)*neighbor
    High alpha → trust self (consolidate under stress)
    Low alpha → trust neighbors (explore under curiosity)

    High cortisol → alpha → 0.7 (consolidate, trust self)
    High dopamine → alpha → 0.3 (explore, trust neighbors)
    """
    d = state.dopamine
    c = state.cortisol
    # Cortisol increases alpha (self-trust), dopamine decreases it (neighbor-trust)
    return 0.3 + 0.4 * (c / (d + c + 1e-8))


def affective_message_passing(
    node_hvs: np.ndarray,
    adjacency: np.ndarray,
    state: NeurochemicalState,
    L: int = 2,
) -> np.ndarray:
    """Message passing with hormone-modulated alpha."""
    alpha = affective_alpha(state)
    return associative_message_passing(node_hvs, adjacency, L=L, alpha=alpha)


# ── Tests ─────────────────────────────────────────────────────────────

class TestAffectiveAlpha:
    """Test that neurochemistry correctly modulates alpha."""

    def test_resting_state_not_extreme(self):
        state = NeurochemicalState()
        alpha = affective_alpha(state)
        # Resting: d=0.4, c=0.2 → alpha = 0.3 + 0.4*(0.2/0.6) = 0.433
        assert 0.3 <= alpha <= 0.7, f"Resting alpha should be moderate, got {alpha}"

    def test_threat_increases_alpha(self):
        """High threat → cortisol → higher alpha (trust self, consolidate)."""
        state = NeurochemicalState()
        for _ in range(10):
            state.update(novelty=0.0, threat=1.0, focus=0.0, valence=-0.5)
        alpha = affective_alpha(state)
        # After threat: d≈0.015, c=1.0 → alpha ≈ 0.694
        assert alpha > 0.6, f"Threat should push alpha > 0.6, got {alpha}"

    def test_pure_dopamine_decreases_alpha(self):
        """Pure dopamine (no cortisol coupling) → lower alpha (explore)."""
        state = NeurochemicalState()
        state.dopamine = 0.9
        state.cortisol = 0.1
        alpha = affective_alpha(state)
        assert alpha < 0.4, f"High d/c ratio should give low alpha, got {alpha}"

    def test_alpha_bounded(self):
        state = NeurochemicalState()
        # Extreme cortisol
        state.dopamine = 0.0
        state.cortisol = 1.0
        assert affective_alpha(state) >= 0.69

        # Extreme dopamine
        state.dopamine = 1.0
        state.cortisol = 0.0
        assert affective_alpha(state) <= 0.31

    def test_alpha_range(self):
        """Alpha should always be in [0.3, 0.7]."""
        rng = np.random.default_rng(42)
        state = NeurochemicalState()
        for _ in range(100):
            state.update(
                novelty=rng.random(),
                threat=rng.random(),
                focus=rng.random(),
                valence=rng.random() * 2 - 1,
            )
            alpha = affective_alpha(state)
            assert 0.29 <= alpha <= 0.71, f"Alpha out of range: {alpha}"


class TestAffectiveMessagePassing:
    """Test that affective modulation changes message passing behavior."""

    @pytest.fixture
    def triangle_graph(self):
        """A simple triangle graph (3 nodes, 3 edges)."""
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=np.float64)
        rng = np.random.default_rng(42)
        node_hvs = rng.random((3, 128))
        return node_hvs, adj

    def test_explore_converges_more_than_consolidate(self, triangle_graph):
        """Low alpha (explore/dopamine) → more neighbor influence → more convergence."""
        node_hvs, adj = triangle_graph

        # Explore: low alpha (neighbors dominate)
        h_explore = associative_message_passing(node_hvs, adj, L=3, alpha=0.3)
        # Consolidate: high alpha (self dominates)
        h_consolidate = associative_message_passing(node_hvs, adj, L=3, alpha=0.7)

        var_explore = np.var(h_explore, axis=0).mean()
        var_consolidate = np.var(h_consolidate, axis=0).mean()

        assert var_explore < var_consolidate, (
            f"Low alpha (explore) should cause more convergence: "
            f"var_explore={var_explore:.6f} should be < var_consolidate={var_consolidate:.6f}"
        )

    def test_cortisol_preserves_identity(self, triangle_graph):
        """High cortisol → high alpha → nodes stay closer to originals."""
        node_hvs, adj = triangle_graph

        # High alpha (cortisol/consolidate) preserves identity
        h_high_alpha = associative_message_passing(node_hvs, adj, L=3, alpha=0.7)
        h_low_alpha = associative_message_passing(node_hvs, adj, L=3, alpha=0.3)

        drift_high = np.linalg.norm(h_high_alpha - node_hvs)
        drift_low = np.linalg.norm(h_low_alpha - node_hvs)

        assert drift_high < drift_low, (
            f"High alpha should preserve identity: "
            f"drift_high={drift_high:.4f} should be < drift_low={drift_low:.4f}"
        )

    def test_vsgraph_integration(self):
        """Test affective message passing works with VSGraph block codes."""
        g = VSGraph(k=4, l=32, seed=42)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")

        adj = g.get_adjacency_matrix()
        node_hvs = np.array([g.get_vector(n) for n in g._node_list])
        # Flatten block codes for message passing
        flat_hvs = node_hvs.reshape(3, -1).astype(np.float64)

        state = NeurochemicalState()
        state.update(novelty=0.8, threat=0.0, focus=0.5, valence=0.3)

        result = affective_message_passing(flat_hvs, adj, state, L=2)
        assert result.shape == flat_hvs.shape
        assert np.all(np.isfinite(result))
