"""
Tests for cubemind.execution.causal_codebook.CausalCodebook.

Validates:
  - codebook_dimensions (n_explicit, n_learned, n_axes)
  - encode_attributes_shape — output is (k, l) float32
  - encode_deterministic — same input = same output
  - encode_different_attrs_differ — different attrs produce different codes
  - save_load_roundtrip — save, load, encode produces identical result
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from cubemind.execution.causal_codebook import CausalCodebook
from cubemind.execution.attribute_extractor import ATTRIBUTE_NAMES

# ── Constants (small dims to avoid OOM) ──────────────────────────────────────

K = 4
L = 8
N_EXPLICIT = 3   # small subset of the 32 ATTRIBUTE_NAMES
N_LEARNED = K - N_EXPLICIT  # = 1
D_EMBED = 16     # small embedding dimension for tests


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_random_embeddings(n: int = 20, d: int = D_EMBED, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def make_attrs(values: list[float] | None = None) -> dict[str, float]:
    """Return a dict with n_explicit attributes."""
    names = ATTRIBUTE_NAMES[:N_EXPLICIT]
    if values is None:
        values = [0.1 * (i + 1) for i in range(N_EXPLICIT)]
    return dict(zip(names, values))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_codebook() -> CausalCodebook:
    """A codebook that has been fit on random embeddings."""
    cb = CausalCodebook(k=K, l=L, n_explicit=N_EXPLICIT)
    embs = make_random_embeddings(n=20, d=D_EMBED)
    cb.fit_pca(embs)
    return cb


# ── Test: codebook_dimensions ─────────────────────────────────────────────────


def test_codebook_dimensions_default_n_learned():
    """n_learned defaults to k - n_explicit."""
    cb = CausalCodebook(k=K, l=L, n_explicit=N_EXPLICIT)
    assert cb.n_explicit == N_EXPLICIT
    assert cb.n_learned == N_LEARNED
    assert cb.n_axes == N_EXPLICIT + N_LEARNED
    assert cb.n_axes == K


def test_codebook_dimensions_explicit_n_learned():
    """Explicit n_learned overrides the default."""
    cb = CausalCodebook(k=8, l=L, n_explicit=N_EXPLICIT, n_learned=2)
    assert cb.n_explicit == N_EXPLICIT
    assert cb.n_learned == 2
    assert cb.n_axes == N_EXPLICIT + 2


def test_codebook_attr_names_slice():
    """_attr_names is the first n_explicit entries of ATTRIBUTE_NAMES."""
    cb = CausalCodebook(k=K, l=L, n_explicit=N_EXPLICIT)
    assert cb._attr_names == ATTRIBUTE_NAMES[:N_EXPLICIT]


# ── Test: encode_attributes_shape ─────────────────────────────────────────────


def test_encode_attributes_shape(fitted_codebook: CausalCodebook):
    """encode returns (k, l) float32 array."""
    attrs = make_attrs()
    emb = np.random.default_rng(0).standard_normal(D_EMBED).astype(np.float32)
    code = fitted_codebook.encode(attrs, emb)
    assert code.shape == (K, L), f"Expected ({K}, {L}), got {code.shape}"
    assert code.dtype == np.float32, f"Expected float32, got {code.dtype}"


def test_encode_no_embedding_shape(fitted_codebook: CausalCodebook):
    """encode without embedding still returns (k, l) float32."""
    attrs = make_attrs()
    code = fitted_codebook.encode(attrs, embedding=None)
    assert code.shape == (K, L)
    assert code.dtype == np.float32


def test_encode_empty_attrs_shape(fitted_codebook: CausalCodebook):
    """encode with empty attrs dict uses defaults and returns (k, l) float32."""
    code = fitted_codebook.encode({}, embedding=None)
    assert code.shape == (K, L)
    assert code.dtype == np.float32


# ── Test: encode_deterministic ────────────────────────────────────────────────


def test_encode_deterministic(fitted_codebook: CausalCodebook):
    """Same attributes + same embedding always produces the same code."""
    attrs = make_attrs()
    emb = np.ones(D_EMBED, dtype=np.float32) * 0.5
    code1 = fitted_codebook.encode(attrs, emb)
    code2 = fitted_codebook.encode(attrs, emb)
    np.testing.assert_array_equal(code1, code2)


def test_encode_deterministic_no_embedding(fitted_codebook: CausalCodebook):
    """Same attributes with no embedding always produces the same code."""
    attrs = make_attrs([0.2, 0.5, 0.8])
    code1 = fitted_codebook.encode(attrs)
    code2 = fitted_codebook.encode(attrs)
    np.testing.assert_array_equal(code1, code2)


# ── Test: encode_different_attrs_differ ───────────────────────────────────────


def test_encode_different_attrs_differ(fitted_codebook: CausalCodebook):
    """Different attribute values produce different block-codes."""
    attrs_a = make_attrs([0.1, 0.2, 0.3])
    attrs_b = make_attrs([0.9, 0.8, 0.7])
    code_a = fitted_codebook.encode(attrs_a)
    code_b = fitted_codebook.encode(attrs_b)
    assert not np.array_equal(code_a, code_b), (
        "Different attributes must produce different codes"
    )


def test_encode_different_embeddings_differ(fitted_codebook: CausalCodebook):
    """Different embeddings produce different block-codes (when n_learned > 0)."""
    if fitted_codebook.n_learned == 0:
        pytest.skip("n_learned == 0, no embedding blocks to differentiate")
    attrs = make_attrs()
    emb_a = np.zeros(D_EMBED, dtype=np.float32)
    emb_b = np.ones(D_EMBED, dtype=np.float32)
    # Ensure the embeddings span the training range so they map to different bins
    # We fit a fresh codebook on embeddings that include these extremes
    cb = CausalCodebook(k=K, l=L, n_explicit=N_EXPLICIT)
    embs = np.vstack([
        make_random_embeddings(18, D_EMBED),
        emb_a[np.newaxis, :],
        emb_b[np.newaxis, :],
    ]).astype(np.float32)
    cb.fit_pca(embs)
    code_a = cb.encode(attrs, emb_a)
    code_b = cb.encode(attrs, emb_b)
    assert not np.array_equal(code_a, code_b), (
        "Different embeddings must produce different codes"
    )


# ── Test: output blocks are one-hot ──────────────────────────────────────────


def test_encode_blocks_one_hot(fitted_codebook: CausalCodebook):
    """Each block in the output must be a one-hot vector (sums to 1)."""
    attrs = make_attrs()
    emb = np.random.default_rng(7).standard_normal(D_EMBED).astype(np.float32)
    code = fitted_codebook.encode(attrs, emb)
    block_sums = code.sum(axis=-1)
    np.testing.assert_allclose(
        block_sums,
        np.ones(K, dtype=np.float32),
        atol=1e-5,
        err_msg="Each block must sum to 1.0 (one-hot)",
    )


# ── Test: save / load roundtrip ───────────────────────────────────────────────


def test_save_load_roundtrip(fitted_codebook: CausalCodebook):
    """save → load → encode produces byte-identical result."""
    attrs = make_attrs()
    emb = np.random.default_rng(99).standard_normal(D_EMBED).astype(np.float32)
    code_before = fitted_codebook.encode(attrs, emb)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "codebook.npz")
        fitted_codebook.save(path)
        loaded = CausalCodebook.load(path)

    code_after = loaded.encode(attrs, emb)
    np.testing.assert_array_equal(
        code_before, code_after,
        err_msg="save/load roundtrip must produce identical codes",
    )


def test_save_load_dimensions_preserved(fitted_codebook: CausalCodebook):
    """Loaded codebook preserves k, l, n_explicit, n_learned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "codebook_dims.npz")
        fitted_codebook.save(path)
        loaded = CausalCodebook.load(path)

    assert loaded.k == fitted_codebook.k
    assert loaded.l == fitted_codebook.l
    assert loaded.n_explicit == fitted_codebook.n_explicit
    assert loaded.n_learned == fitted_codebook.n_learned


# ── Test: fit_pca stores components ──────────────────────────────────────────


def test_fit_pca_stores_components():
    """fit_pca stores PCA components with correct shape."""
    cb = CausalCodebook(k=K, l=L, n_explicit=N_EXPLICIT)
    embs = make_random_embeddings(30, D_EMBED)
    cb.fit_pca(embs)
    # Components shape: (n_learned, d_embed)
    assert cb._pca_components is not None
    assert cb._pca_components.shape == (N_LEARNED, D_EMBED), (
        f"Expected ({N_LEARNED}, {D_EMBED}), got {cb._pca_components.shape}"
    )
    assert cb._pca_mean is not None
    assert cb._pca_mean.shape == (D_EMBED,)


def test_fit_pca_mean_is_float32():
    """fit_pca stores mean and components as float32."""
    cb = CausalCodebook(k=K, l=L, n_explicit=N_EXPLICIT)
    embs = make_random_embeddings(20, D_EMBED)
    cb.fit_pca(embs)
    assert cb._pca_mean.dtype == np.float32
    assert cb._pca_components.dtype == np.float32
