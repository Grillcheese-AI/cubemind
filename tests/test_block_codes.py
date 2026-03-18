"""
Tests for cubemind.ops.block_codes.BlockCodes.

Validates:
  - Structural correctness (valid block codes after operations)
  - Theorem 1: magnitude preservation through binding chains
  - Theorem 7: positive semi-definiteness of the similarity kernel
  - Roundtrip unbind recovers the original vector exactly
  - PMF normalization and monotonicity
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes

# ── Fixtures ──────────────────────────────────────────────────────────────────

K = 16
L = 128


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def v1(bc: BlockCodes) -> np.ndarray:
    return bc.random_discrete(seed=0)


@pytest.fixture(scope="module")
def v2(bc: BlockCodes) -> np.ndarray:
    return bc.random_discrete(seed=1)


# ── Test: bind produces valid block code ──────────────────────────────────────


def test_bind_returns_valid_block_code(bc: BlockCodes, v1: np.ndarray, v2: np.ndarray):
    """After binding two one-hot block codes, each block sums to 1."""
    bound = bc.bind(v1, v2)
    assert bound.shape == (K, L), f"Expected shape ({K}, {L}), got {bound.shape}"
    block_sums = bound.sum(axis=-1)
    np.testing.assert_allclose(
        block_sums,
        np.ones(K, dtype=np.float32),
        atol=1e-4,
        err_msg="Each block of the bound vector must sum to 1",
    )


# ── Test: unbind roundtrip recovers original ─────────────────────────────────


def test_unbind_roundtrip(bc: BlockCodes, v1: np.ndarray, v2: np.ndarray):
    """bind(v1, v2) then unbind(result, v2) recovers v1 exactly."""
    bound = bc.bind(v1, v2)
    recovered = bc.unbind(bound, v2)
    # Discretize both to compare hot positions unambiguously
    rec_disc = bc.discretize(recovered)
    np.testing.assert_array_equal(
        rec_disc,
        v1,
        err_msg="Unbind after bind should recover the original one-hot vector",
    )


# ── Test: Theorem 1 — magnitude (L1) preserved through 1000 bindings ─────────


def test_theorem1_magnitude_preservation_1000_chains(bc: BlockCodes):
    """
    Theorem 1 (NVSA): For one-hot block codes, ||bind(a, b)||_1 = ||a||_1 = k.

    We verify this remains true after a chain of 1000 random bindings.
    Block circular convolution of one-hot vectors is a pure cyclic shift —
    no energy is created or destroyed. Each block L1-norm stays exactly 1.
    """
    rng = np.random.default_rng(42)
    current = bc.random_discrete(seed=rng.integers(0, 2**32))
    for i in range(1000):
        other = bc.random_discrete(seed=int(rng.integers(0, 2**32)))
        current = bc.bind(current, other)

    block_sums = current.sum(axis=-1)  # (k,)
    np.testing.assert_allclose(
        block_sums,
        np.ones(K, dtype=np.float32),
        atol=1e-3,
        err_msg=(
            "Theorem 1 violated: L1 norm per block must be 1 after 1000 chained bindings"
        ),
    )


# ── Test: Theorem 7 — kernel positive semi-definiteness ──────────────────────


def test_theorem7_kernel_positive_semidefinite(bc: BlockCodes):
    """
    Theorem 7 (NVSA): The block-code similarity function is a valid positive
    semi-definite kernel — the Gram matrix G[i,j] = similarity(v_i, v_j) has
    all non-negative eigenvalues.
    """
    n = 20
    codebook = bc.codebook_discrete(n, seed=7)

    # Build Gram matrix
    gram = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            gram[i, j] = bc.similarity(codebook[i], codebook[j])

    # Gram matrix should be symmetric
    np.testing.assert_allclose(gram, gram.T, atol=1e-6, err_msg="Gram matrix not symmetric")

    # All eigenvalues should be >= 0 (PSD)
    eigenvalues = np.linalg.eigvalsh(gram)
    min_eig = float(eigenvalues.min())
    assert min_eig >= -1e-6, (
        f"Theorem 7 violated: Gram matrix has negative eigenvalue {min_eig:.6f}. "
        "Similarity kernel must be positive semi-definite."
    )


# ── Test: similarity(a, a) == 1.0 ────────────────────────────────────────────


def test_similarity_self(bc: BlockCodes, v1: np.ndarray):
    """Similarity of a one-hot block code with itself should be exactly 1.0."""
    sim = bc.similarity(v1, v1)
    assert abs(sim - 1.0) < 1e-5, f"Expected similarity 1.0, got {sim}"


# ── Test: codebook_discrete shape ────────────────────────────────────────────


def test_codebook_discrete(bc: BlockCodes):
    """codebook_discrete(n) returns an array of shape (n, k, l)."""
    for n in [1, 5, 32]:
        cb = bc.codebook_discrete(n, seed=n)
        assert cb.shape == (n, K, L), f"Expected ({n}, {K}, {L}), got {cb.shape}"
        # Each block in every entry should sum to 1 (one-hot)
        block_sums = cb.sum(axis=-1)  # (n, k)
        np.testing.assert_allclose(
            block_sums,
            np.ones((n, K), dtype=np.float32),
            atol=1e-5,
            err_msg=f"Codebook entries are not valid one-hot block codes (n={n})",
        )


# ── Test: cosine_to_pmf sums to 1 and is monotonic ───────────────────────────


def test_cosine_to_pmf(bc: BlockCodes):
    """cosine_to_pmf output sums to 1 and higher similarity maps to higher probability."""
    # Build strictly increasing similarities
    sims = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    pmf = bc.cosine_to_pmf(sims, temperature=40.0)

    # Must be a valid probability distribution
    assert abs(pmf.sum() - 1.0) < 1e-5, f"PMF does not sum to 1: sum={pmf.sum()}"
    assert (pmf >= 0).all(), "PMF contains negative probabilities"
    assert pmf.shape == sims.shape, "PMF shape does not match similarities shape"

    # Monotonicity: higher similarity → higher probability
    for i in range(len(pmf) - 1):
        assert pmf[i] <= pmf[i + 1], (
            f"PMF not monotonically increasing at index {i}: "
            f"pmf[{i}]={pmf[i]:.4f} > pmf[{i+1}]={pmf[i+1]:.4f}"
        )
