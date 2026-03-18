"""VS-Graph: VSA-based graph classification via Spike Diffusion + Associative Message Passing.

450x faster than GNNs, competitive accuracy. Works with BlockCodes for block-code graphs.

Reference: Poursiami et al., "VS-Graph: Scalable and Efficient Graph Classification
Using Hyperdimensional Computing", 2025.
"""

from __future__ import annotations

import numpy as np

from cubemind.ops import BlockCodes
from cubemind.telemetry import metrics


# -- Spike Diffusion ----------------------------------------------------------


def spike_diffusion(adjacency: np.ndarray, K: int = 3) -> np.ndarray:
    """Rank nodes by multi-hop spike propagation.

    Initializes a unit spike per node and propagates for K hops via the
    adjacency matrix. Higher-centrality nodes accumulate larger values.

    Args:
        adjacency: (N, N) binary adjacency matrix.
        K: Number of diffusion hops.

    Returns:
        ranks: (N,) integer rank per node (0 = lowest spike, N-1 = highest).
    """
    adjacency = np.asarray(adjacency, dtype=np.float64)
    N = adjacency.shape[0]
    s = np.ones(N, dtype=np.float64)

    for _ in range(K):
        s = adjacency @ s

    ranks = np.argsort(np.argsort(s)).astype(np.int64)
    return ranks


# -- Node Encoding ------------------------------------------------------------


def encode_nodes(ranks: np.ndarray, D: int, seed: int = 42) -> np.ndarray:
    """Encode nodes as binary hypervectors based on their spike-diffusion rank.

    Args:
        ranks: (N,) integer rank per node.
        D: Hypervector dimensionality.
        seed: RNG seed for codebook generation.

    Returns:
        node_hvs: (N, D) binary {0, 1} array.
    """
    rng = np.random.default_rng(seed)
    unique_ranks = np.unique(ranks)
    codebook = {}
    for r in unique_ranks:
        codebook[int(r)] = rng.integers(0, 2, size=D).astype(np.float64)

    N = len(ranks)
    node_hvs = np.empty((N, D), dtype=np.float64)
    for i in range(N):
        node_hvs[i] = codebook[int(ranks[i])]
    return node_hvs


# -- Associative Message Passing ----------------------------------------------


def associative_message_passing(
    node_hvs: np.ndarray,
    adjacency: np.ndarray,
    L: int = 2,
    alpha: float = 0.5,
) -> np.ndarray:
    """Refine node hypervectors via OR-based message passing.

    OR aggregation is idempotent, preventing over-smoothing.

    Args:
        node_hvs: (N, D) binary or continuous node representations.
        adjacency: (N, N) binary adjacency matrix.
        L: Number of message passing layers.
        alpha: Blending coefficient (alpha * self + (1 - alpha) * message).

    Returns:
        refined_hvs: (N, D) refined node representations.
    """
    adjacency = np.asarray(adjacency, dtype=np.float64)
    h = node_hvs.copy().astype(np.float64)
    N, D = h.shape

    for _ in range(L):
        h_new = np.empty_like(h)
        for i in range(N):
            neighbors = np.where(adjacency[i] > 0)[0]
            if len(neighbors) == 0:
                h_new[i] = h[i]
            else:
                msg = np.max(h[neighbors], axis=0)
                h_new[i] = alpha * h[i] + (1.0 - alpha) * msg
        h = h_new

    return h


# -- Graph Readout ------------------------------------------------------------


def graph_readout(node_hvs: np.ndarray) -> np.ndarray:
    """Compute a single graph-level hypervector via mean pooling.

    Args:
        node_hvs: (N, D) node representations.

    Returns:
        graph_hv: (D,) graph-level embedding.
    """
    return np.mean(node_hvs, axis=0)


# -- VSGraph (block-code variant) ---------------------------------------------


class VSGraph:
    """Graph structure using block-code nodes and edges via BlockCodes.

    Supports add_node, add_edge, query_neighbors, and graph-level encoding
    for classification tasks.

    Args:
        k: Number of blocks.
        l: Block length.
        seed: RNG seed for codebook generation.
    """

    def __init__(self, k: int = 4, l: int = 32, seed: int = 42) -> None:
        self.k = k
        self.l = l
        self.seed = seed
        self._bc = BlockCodes(k=k, l=l)

        self._nodes: dict[str, np.ndarray] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._node_list: list[str] = []

    def add_node(self, node_id: str, vector: np.ndarray | None = None) -> None:
        """Add a node to the graph.

        Args:
            node_id: Unique string identifier.
            vector: Optional block-code vector (k, l). If None, generates one.
        """
        if vector is None:
            seed_val = hash(node_id) % (2**31)
            vector = self._bc.random_discrete(seed=seed_val)
        self._nodes[node_id] = vector
        if node_id not in self._adjacency:
            self._adjacency[node_id] = set()
        if node_id not in self._node_list:
            self._node_list.append(node_id)

    def add_edge(self, src: str, dst: str) -> None:
        """Add an undirected edge between two nodes.

        Creates nodes automatically if they don't exist.

        Args:
            src: Source node ID.
            dst: Destination node ID.
        """
        if src not in self._nodes:
            self.add_node(src)
        if dst not in self._nodes:
            self.add_node(dst)
        self._adjacency[src].add(dst)
        self._adjacency[dst].add(src)

    def query_neighbors(self, node_id: str) -> list[str]:
        """Get the neighbor IDs of a node.

        Args:
            node_id: The node to query.

        Returns:
            List of neighbor node IDs.
        """
        return sorted(self._adjacency.get(node_id, set()))

    def get_vector(self, node_id: str) -> np.ndarray:
        """Get the block-code vector for a node.

        Args:
            node_id: The node to look up.

        Returns:
            Block-code vector (k, l).
        """
        return self._nodes[node_id]

    def get_adjacency_matrix(self) -> np.ndarray:
        """Build the adjacency matrix from the current graph.

        Returns:
            (N, N) binary adjacency matrix.
        """
        n = len(self._node_list)
        adj = np.zeros((n, n), dtype=np.float64)
        node_idx = {nid: i for i, nid in enumerate(self._node_list)}
        for src, neighbors in self._adjacency.items():
            i = node_idx[src]
            for dst in neighbors:
                j = node_idx[dst]
                adj[i, j] = 1.0
        return adj

    def encode_graph(self) -> np.ndarray:
        """Encode the graph into a (k, l) block-code embedding.

        Uses spike diffusion + block-code message passing + readout.

        Returns:
            graph_hv: (k, l) graph-level block-code embedding.
        """
        adj = self.get_adjacency_matrix()
        n = adj.shape[0]

        if n == 0:
            return self._bc.random_discrete(seed=0)

        ranks = spike_diffusion(adj, K=3)

        # Encode nodes as block codes based on rank
        unique_ranks = np.unique(ranks)
        n_ranks = len(unique_ranks)
        codebook = self._bc.codebook_discrete(n_ranks, seed=self.seed)
        rank_to_idx = {int(r): i for i, r in enumerate(unique_ranks)}

        node_hvs = np.empty((n, self.k, self.l), dtype=np.float32)
        for i in range(n):
            node_hvs[i] = codebook[rank_to_idx[int(ranks[i])]]

        # Message passing using block-code bundling
        for _ in range(2):
            h_new = np.empty_like(node_hvs)
            for i in range(n):
                neighbors = np.where(adj[i] > 0)[0]
                if len(neighbors) == 0:
                    h_new[i] = node_hvs[i]
                else:
                    neighbor_vecs = [node_hvs[j] for j in neighbors]
                    msg = self._bc.bundle(neighbor_vecs, normalize=True)
                    h_new[i] = 0.5 * node_hvs[i] + 0.5 * msg
            node_hvs = h_new

        graph_hv = np.mean(node_hvs, axis=0).astype(np.float32)
        metrics.record("vs_graph.num_nodes", float(n))
        return graph_hv

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self._adjacency.values()) // 2
