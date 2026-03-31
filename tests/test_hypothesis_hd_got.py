"""Hypothesis 2: Hyper-Dimensional Graph of Thoughts (HD-GoT).

Tests that competing logical hypotheses from MultiViewHMM can be structured
as nodes in a VSGraph, ranked by spike diffusion centrality, and aggregated
via OR-based message passing into a refined solution — all in O(k*l) VSA
operations, no token generation required.

This is geometric debate: hypotheses are nodes, agreement is edge weight,
centrality = logical viability, aggregation = consensus building.

All tests are self-contained — no modifications to existing code.
"""

import numpy as np
import pytest

from cubemind.ops import BlockCodes
from cubemind.experimental.vs_graph import (
    spike_diffusion,
    associative_message_passing,
    graph_readout,
)
from cubemind.reasoning.hmm_rule import HMMRule, HMMEnsemble, MultiViewHMM


# ── The HD-GoT pipeline under test ───────────────────────────────────

def build_hypothesis_graph(
    candidate_vectors: list[np.ndarray],
    bc: BlockCodes,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fully-connected graph from candidate hypothesis vectors.

    Edge weights = cosine similarity between candidates (how much they agree).
    Returns adjacency matrix and flattened node vectors.

    Args:
        candidate_vectors: List of (k, l) block-code hypothesis vectors.
        bc: BlockCodes instance for similarity computation.

    Returns:
        (adjacency, node_matrix): adjacency (N, N), node_matrix (N, k*l)
    """
    N = len(candidate_vectors)
    adj = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            sim = float(bc.similarity(candidate_vectors[i], candidate_vectors[j]))
            # Only connect if similarity is above threshold (agreeing hypotheses)
            if sim > 0.0:
                adj[i, j] = sim
                adj[j, i] = sim

    node_matrix = np.array([v.ravel() for v in candidate_vectors], dtype=np.float64)
    return adj, node_matrix


def hd_got_resolve(
    candidate_vectors: list[np.ndarray],
    bc: BlockCodes,
    diffusion_hops: int = 3,
    mp_layers: int = 2,
    top_k: int = 3,
) -> np.ndarray:
    """Resolve competing hypotheses via HD Graph-of-Thoughts.

    Pipeline:
      1. Build graph: hypotheses as nodes, similarity as edges
      2. Spike diffusion: rank hypotheses by centrality (logical viability)
      3. Message passing: aggregate neighbor information (consensus building)
      4. Select top-K by centrality rank
      5. Mean pool top-K → refined solution vector

    Args:
        candidate_vectors: List of (k, l) hypothesis vectors.
        bc: BlockCodes instance.
        diffusion_hops: Spike diffusion depth.
        mp_layers: Message passing iterations.
        top_k: Number of top hypotheses to aggregate.

    Returns:
        solution: (k, l) refined consensus vector.
    """
    N = len(candidate_vectors)
    if N == 0:
        raise ValueError("No candidates to resolve")
    if N == 1:
        return candidate_vectors[0].copy()

    # 1. Build hypothesis graph
    adj, node_matrix = build_hypothesis_graph(candidate_vectors, bc)

    # 2. Spike diffusion → rank by centrality
    # Binarize adjacency for spike diffusion (it expects binary)
    binary_adj = (adj > 0.1).astype(np.float64)
    ranks = spike_diffusion(binary_adj, K=diffusion_hops)

    # 3. Message passing → refine via consensus
    refined = associative_message_passing(node_matrix, binary_adj, L=mp_layers)

    # 4. Select top-K by rank
    k_select = min(top_k, N)
    top_indices = np.argsort(ranks)[-k_select:]

    # 5. Aggregate top-K → solution
    top_vecs = refined[top_indices]
    solution_flat = np.mean(top_vecs, axis=0)

    # Reshape back to (k, l)
    k, l = candidate_vectors[0].shape
    return solution_flat.reshape(k, l).astype(np.float32)


# ── Tests ─────────────────────────────────────────────────────────────

class TestHypothesisGraph:
    """Test graph construction from hypothesis vectors."""

    @pytest.fixture
    def bc(self):
        return BlockCodes(k=4, l=32)

    def test_self_loops_zero(self, bc):
        vecs = [bc.random_discrete(seed=i) for i in range(5)]
        adj, _ = build_hypothesis_graph(vecs, bc)
        assert np.all(np.diag(adj) == 0), "No self-loops"

    def test_symmetric(self, bc):
        vecs = [bc.random_discrete(seed=i) for i in range(5)]
        adj, _ = build_hypothesis_graph(vecs, bc)
        np.testing.assert_array_almost_equal(adj, adj.T, err_msg="Adjacency must be symmetric")

    def test_similar_vectors_high_edge_weight(self, bc):
        """Two identical hypotheses should have high edge weight."""
        v = bc.random_discrete(seed=42)
        vecs = [v.copy(), v.copy(), bc.random_discrete(seed=99)]
        adj, _ = build_hypothesis_graph(vecs, bc)
        assert adj[0, 1] > 0.9, f"Identical vectors should have high similarity, got {adj[0, 1]}"

    def test_node_matrix_shape(self, bc):
        vecs = [bc.random_discrete(seed=i) for i in range(4)]
        _, nodes = build_hypothesis_graph(vecs, bc)
        assert nodes.shape == (4, 4 * 32)


class TestHDGoTResolve:
    """Test the full HD-GoT resolution pipeline."""

    @pytest.fixture
    def bc(self):
        return BlockCodes(k=4, l=32)

    def test_single_candidate_passthrough(self, bc):
        v = bc.random_discrete(seed=42)
        result = hd_got_resolve([v], bc)
        np.testing.assert_array_equal(result, v)

    def test_output_shape(self, bc):
        vecs = [bc.random_discrete(seed=i) for i in range(5)]
        result = hd_got_resolve(vecs, bc)
        assert result.shape == (4, 32)

    def test_output_finite(self, bc):
        vecs = [bc.random_discrete(seed=i) for i in range(5)]
        result = hd_got_resolve(vecs, bc)
        assert np.all(np.isfinite(result))

    def test_majority_consensus_wins(self, bc):
        """If 4 out of 5 hypotheses agree, the result should be close to them."""
        agreed = bc.random_discrete(seed=42)
        outlier = bc.random_discrete(seed=999)

        # 4 copies of agreed + 1 outlier
        vecs = [agreed.copy() for _ in range(4)] + [outlier]
        result = hd_got_resolve(vecs, bc, top_k=3)

        sim_to_agreed = float(bc.similarity(result, agreed))
        sim_to_outlier = float(bc.similarity(result, outlier))

        assert sim_to_agreed > sim_to_outlier, (
            f"Consensus should be closer to majority: "
            f"sim_agreed={sim_to_agreed:.3f} vs sim_outlier={sim_to_outlier:.3f}"
        )

    def test_centrality_favors_connected(self, bc):
        """Hypotheses that agree with more others should rank higher."""
        # Create a cluster of 3 similar + 2 random
        base = bc.random_discrete(seed=42)
        rng = np.random.default_rng(42)
        cluster = [base + rng.normal(0, 0.01, base.shape).astype(np.float32)
                   for _ in range(3)]
        randoms = [bc.random_discrete(seed=i + 100) for i in range(2)]

        vecs = cluster + randoms
        adj, _ = build_hypothesis_graph(vecs, bc)
        binary_adj = (adj > 0.1).astype(np.float64)
        ranks = spike_diffusion(binary_adj, K=3)

        # Cluster members should have higher rank than randoms
        cluster_ranks = ranks[:3]
        random_ranks = ranks[3:]
        assert np.mean(cluster_ranks) > np.mean(random_ranks), (
            f"Cluster should rank higher: cluster={cluster_ranks}, random={random_ranks}"
        )

    def test_empty_raises(self, bc):
        with pytest.raises(ValueError):
            hd_got_resolve([], bc)


class TestHDGoTWithHMM:
    """Test HD-GoT with real MultiViewHMM candidates."""

    @pytest.fixture
    def bc(self):
        return BlockCodes(k=4, l=32)

    @pytest.fixture
    def codebook(self, bc):
        return bc.codebook_discrete(5, seed=42)

    def test_hmm_ensemble_candidates_resolve(self, bc, codebook):
        """HMMEnsemble predictions can be resolved via HD-GoT."""
        ensemble = HMMEnsemble(codebook, n_rules=5, seed=42)

        # Generate synthetic observations
        rng = np.random.default_rng(42)
        obs = [codebook[rng.integers(0, 5)] for _ in range(4)]

        # Get per-rule predictions as candidate hypotheses
        candidates = [rule.predict(obs) for rule in ensemble.rules]

        # Resolve via HD-GoT
        result = hd_got_resolve(candidates, bc, top_k=3)
        assert result.shape == (4, 32)
        assert np.all(np.isfinite(result))

    def test_multiview_candidates_resolve(self, bc, codebook):
        """MultiViewHMM view-specific predictions can be debated via HD-GoT."""
        mv = MultiViewHMM(codebook, bc, seed=42)

        # Synthetic panel sequence
        rng = np.random.default_rng(42)
        panels = [codebook[rng.integers(0, 5)] for _ in range(4)]

        # Get per-view predictions
        views = mv.make_views(panels)
        candidates = []
        for view_name, view_obs in views.items():
            if len(view_obs) >= 2:
                hmm = getattr(mv, f"hmm_{view_name}", None)
                if hmm is not None:
                    pred = hmm.predict(view_obs)
                    candidates.append(pred)

        if len(candidates) >= 2:
            result = hd_got_resolve(candidates, bc, top_k=2)
            assert result.shape == (4, 32)
            assert np.all(np.isfinite(result))
