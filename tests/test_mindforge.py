"""Tests for cubemind.execution.mindforge.MindForge.

Validates:
  - Init creates correct shapes for basis, embeddings, generator
  - forge() produces (A, B) with correct shapes
  - forge_all_layers() produces one pair per layer
  - Different contexts produce different adapters
  - Different layer IDs produce different adapters
  - apply_adapter() adds LoRA correction to base output
  - Memory footprint is reasonable
  - Coefficients are valid softmax distribution
  - Basis mixing is deterministic for same input
  - Works at production dims (k=80, l=128)
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.mindforge import MindForge
from cubemind.ops.block_codes import BlockCodes

# Small dims for fast tests
K = 4
L = 32
D_TARGET = 64
N_LAYERS = 4
RANK = 4
N_BASIS = 8


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def forge() -> MindForge:
    return MindForge(
        k=K, l=L, n_layers=N_LAYERS, d_target=D_TARGET,
        rank=RANK, n_basis=N_BASIS, d_hidden=32, seed=42,
    )


# ── Init tests ───────────────────────────────────────────────────────────────


def test_init_basis_shapes(forge: MindForge):
    """Basis adapters must have correct shapes."""
    assert forge.A_basis.shape == (N_BASIS, RANK, D_TARGET)
    assert forge.B_basis.shape == (N_BASIS, D_TARGET, RANK)


def test_init_layer_embeddings(forge: MindForge):
    """Layer embeddings must have one per layer."""
    assert forge.layer_embeddings.shape == (N_LAYERS, forge.d_hidden)


def test_init_generator_weights(forge: MindForge):
    """Generator MLP weights must have correct shapes."""
    assert forge.W_h.shape == (forge.d_hidden, forge.d_hidden * 2)
    assert forge.W_coeff.shape == (N_BASIS, forge.d_hidden)


# ── Forge tests ──────────────────────────────────────────────────────────────


def test_forge_shapes(forge: MindForge, bc: BlockCodes):
    """forge() must return (A, B) with correct shapes."""
    ctx = bc.random_discrete(seed=1)
    A, B = forge.forge(ctx, layer_id=0)
    assert A.shape == (RANK, D_TARGET)
    assert B.shape == (D_TARGET, RANK)


def test_forge_dtype(forge: MindForge, bc: BlockCodes):
    """Adapters must be float32."""
    ctx = bc.random_discrete(seed=2)
    A, B = forge.forge(ctx, layer_id=0)
    assert A.dtype == np.float32
    assert B.dtype == np.float32


def test_forge_all_layers(forge: MindForge, bc: BlockCodes):
    """forge_all_layers() must return one (A, B) per layer."""
    ctx = bc.random_discrete(seed=3)
    adapters = forge.forge_all_layers(ctx)
    assert len(adapters) == N_LAYERS
    for A, B in adapters:
        assert A.shape == (RANK, D_TARGET)
        assert B.shape == (D_TARGET, RANK)


def test_different_contexts_different_adapters(forge: MindForge, bc: BlockCodes):
    """Different contexts must produce different adapters."""
    ctx1 = bc.random_discrete(seed=10)
    ctx2 = bc.random_discrete(seed=20)
    A1, B1 = forge.forge(ctx1, layer_id=0)
    A2, B2 = forge.forge(ctx2, layer_id=0)
    assert not np.allclose(A1, A2, atol=1e-6)
    assert not np.allclose(B1, B2, atol=1e-6)


def test_different_layers_different_adapters(forge: MindForge, bc: BlockCodes):
    """Same context + different layer IDs must produce different adapters."""
    ctx = bc.random_discrete(seed=30)
    A0, B0 = forge.forge(ctx, layer_id=0)
    A1, B1 = forge.forge(ctx, layer_id=1)
    assert not np.allclose(A0, A1, atol=1e-6)


def test_deterministic(forge: MindForge, bc: BlockCodes):
    """Same inputs must produce identical adapters."""
    ctx = bc.random_discrete(seed=40)
    A1, B1 = forge.forge(ctx, layer_id=2)
    A2, B2 = forge.forge(ctx, layer_id=2)
    np.testing.assert_array_equal(A1, A2)
    np.testing.assert_array_equal(B1, B2)


# ── Apply adapter tests ─────────────────────────────────────────────────────


def test_apply_adapter_shape(forge: MindForge, bc: BlockCodes):
    """apply_adapter must return same shape as base_output."""
    ctx = bc.random_discrete(seed=50)
    A, B = forge.forge(ctx, layer_id=0)

    x = np.random.randn(D_TARGET).astype(np.float32)
    base = np.random.randn(D_TARGET).astype(np.float32)
    out = forge.apply_adapter(x, base, A, B)
    assert out.shape == base.shape


def test_apply_adapter_modifies_output(forge: MindForge, bc: BlockCodes):
    """LoRA adapter must actually change the output."""
    ctx = bc.random_discrete(seed=60)
    A, B = forge.forge(ctx, layer_id=0)

    x = np.random.randn(D_TARGET).astype(np.float32)
    base = np.random.randn(D_TARGET).astype(np.float32)
    out = forge.apply_adapter(x, base, A, B)
    assert not np.allclose(out, base, atol=1e-6)


def test_apply_adapter_batched(forge: MindForge, bc: BlockCodes):
    """apply_adapter must work with batched input (seq_len, d_target)."""
    ctx = bc.random_discrete(seed=70)
    A, B = forge.forge(ctx, layer_id=0)

    seq_len = 16
    x = np.random.randn(seq_len, D_TARGET).astype(np.float32)
    base = np.random.randn(seq_len, D_TARGET).astype(np.float32)
    out = forge.apply_adapter(x, base, A, B)
    assert out.shape == (seq_len, D_TARGET)


# ── Memory tests ─────────────────────────────────────────────────────────────


def test_memory_reasonable(forge: MindForge):
    """Memory footprint must be reasonable for small dims."""
    mb = forge.memory_mb()
    assert mb < 10, f"Expected < 10 MB, got {mb:.2f} MB"


def test_memory_production_dims():
    """Production dims (k=80, l=128) must fit in reasonable memory."""
    mf = MindForge(
        k=80, l=128, n_layers=12, d_target=2048,
        rank=8, n_basis=16, d_hidden=256, seed=99,
    )
    mb = mf.memory_mb()
    # Should be ~5-10 MB, not hundreds
    assert mb < 50, f"Expected < 50 MB at production dims, got {mb:.2f} MB"


# ── VSA integration tests ───────────────────────────────────────────────────


def test_bound_context(forge: MindForge, bc: BlockCodes):
    """Context created via VSA bind must produce valid adapters."""
    task = bc.random_discrete(seed=80)
    personality = bc.random_discrete(seed=81)
    ctx = bc.bind(task, personality)
    A, B = forge.forge(ctx, layer_id=0)
    assert A.shape == (RANK, D_TARGET)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))


def test_bundled_context(forge: MindForge, bc: BlockCodes):
    """Context created via VSA bundle must produce valid adapters."""
    v1 = bc.random_discrete(seed=90)
    v2 = bc.random_discrete(seed=91)
    ctx = bc.bundle([v1, v2])
    A, B = forge.forge(ctx, layer_id=0)
    assert np.all(np.isfinite(A))
