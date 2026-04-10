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

import os

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


# ── Coefficient validity ────────────────────────────────────────────────────


def _extract_coeffs(forge: MindForge, ctx: np.ndarray, layer_id: int) -> np.ndarray:
    """Re-run the forge() generator path and return the softmax coefficients."""
    ctx_flat = forge.bc.to_flat(ctx)
    ctx_proj = ctx_flat @ forge.W_proj.T
    layer_emb = forge.layer_embeddings[layer_id]
    combined = np.concatenate([ctx_proj, layer_emb])
    from cubemind.execution.mindforge import gelu
    h = gelu(combined @ forge.W_h.T + forge.b_h)
    raw = h @ forge.W_coeff.T + forge.b_coeff
    raw_exp = np.exp(raw - np.max(raw))
    return (raw_exp / raw_exp.sum()).astype(np.float32)


def test_coeffs_form_probability_distribution(forge: MindForge, bc: BlockCodes):
    """Mixing coefficients must be a valid softmax: nonneg and sum to 1."""
    ctx = bc.random_discrete(seed=100)
    coeffs = _extract_coeffs(forge, ctx, layer_id=0)
    assert coeffs.shape == (N_BASIS,)
    assert np.all(coeffs >= 0.0)
    assert np.isclose(coeffs.sum(), 1.0, atol=1e-5)


def test_adapter_is_basis_convex_combination(forge: MindForge, bc: BlockCodes):
    """Forged A must equal Σᵢ coeffᵢ · A_basisᵢ (and same for B)."""
    ctx = bc.random_discrete(seed=101)
    A, B = forge.forge(ctx, layer_id=1)
    coeffs = _extract_coeffs(forge, ctx, layer_id=1)
    expected_A = np.tensordot(coeffs, forge.A_basis, axes=([0], [0]))
    expected_B = np.tensordot(coeffs, forge.B_basis, axes=([0], [0]))
    np.testing.assert_allclose(A, expected_A, atol=1e-5)
    np.testing.assert_allclose(B, expected_B, atol=1e-5)


# ── Rank constraint ─────────────────────────────────────────────────────────


def test_lora_product_is_low_rank(forge: MindForge, bc: BlockCodes):
    """The effective LoRA update B @ A must have rank <= forge.rank."""
    ctx = bc.random_discrete(seed=110)
    A, B = forge.forge(ctx, layer_id=0)
    delta_W = B @ A  # (d_target, d_target)
    assert delta_W.shape == (D_TARGET, D_TARGET)
    # Numerical rank — singular values above a small tolerance
    sv = np.linalg.svd(delta_W, compute_uv=False)
    tol = sv.max() * max(delta_W.shape) * np.finfo(np.float32).eps
    numerical_rank = int((sv > tol).sum())
    assert numerical_rank <= RANK, (
        f"LoRA update should have rank <= {RANK}, got {numerical_rank}"
    )


# ── Scale parameter ─────────────────────────────────────────────────────────


def test_scale_zero_passes_through_base(bc: BlockCodes):
    """scale=0 must make apply_adapter a no-op (output == base_output)."""
    forge_zero = MindForge(
        k=K, l=L, n_layers=N_LAYERS, d_target=D_TARGET,
        rank=RANK, n_basis=N_BASIS, d_hidden=32, scale=0.0, seed=42,
    )
    ctx = bc.random_discrete(seed=120)
    A, B = forge_zero.forge(ctx, layer_id=0)
    x = np.random.default_rng(0).standard_normal(D_TARGET).astype(np.float32)
    base = np.random.default_rng(1).standard_normal(D_TARGET).astype(np.float32)
    out = forge_zero.apply_adapter(x, base, A, B)
    np.testing.assert_array_equal(out, base.astype(np.float32))


def test_scale_linear_in_lora_correction(bc: BlockCodes):
    """Doubling scale must double the LoRA correction (base_output - out)."""
    f1 = MindForge(k=K, l=L, n_layers=N_LAYERS, d_target=D_TARGET,
                   rank=RANK, n_basis=N_BASIS, d_hidden=32, scale=1.0, seed=42)
    f2 = MindForge(k=K, l=L, n_layers=N_LAYERS, d_target=D_TARGET,
                   rank=RANK, n_basis=N_BASIS, d_hidden=32, scale=2.0, seed=42)
    ctx = bc.random_discrete(seed=130)
    A1, B1 = f1.forge(ctx, layer_id=0)
    A2, B2 = f2.forge(ctx, layer_id=0)
    # Same seed and shapes ⇒ identical adapters; only scale differs.
    np.testing.assert_array_equal(A1, A2)
    np.testing.assert_array_equal(B1, B2)
    x = np.random.default_rng(2).standard_normal(D_TARGET).astype(np.float32)
    base = np.zeros(D_TARGET, dtype=np.float32)
    out1 = f1.apply_adapter(x, base, A1, B1)
    out2 = f2.apply_adapter(x, base, A2, B2)
    np.testing.assert_allclose(out2, 2.0 * out1, atol=1e-5)


# ── Reproducibility and seeding ─────────────────────────────────────────────


def test_same_seed_produces_identical_state(bc: BlockCodes):
    """Two MindForge instances with the same seed must have identical params."""
    f1 = MindForge(k=K, l=L, n_layers=N_LAYERS, d_target=D_TARGET,
                   rank=RANK, n_basis=N_BASIS, d_hidden=32, seed=7)
    f2 = MindForge(k=K, l=L, n_layers=N_LAYERS, d_target=D_TARGET,
                   rank=RANK, n_basis=N_BASIS, d_hidden=32, seed=7)
    np.testing.assert_array_equal(f1.A_basis, f2.A_basis)
    np.testing.assert_array_equal(f1.B_basis, f2.B_basis)
    np.testing.assert_array_equal(f1.W_h, f2.W_h)
    np.testing.assert_array_equal(f1.W_coeff, f2.W_coeff)
    np.testing.assert_array_equal(f1.W_proj, f2.W_proj)
    np.testing.assert_array_equal(f1.layer_embeddings, f2.layer_embeddings)


def test_different_seeds_produce_different_state():
    """Different seeds must produce visibly different basis tensors."""
    f1 = MindForge(k=K, l=L, n_layers=N_LAYERS, d_target=D_TARGET,
                   rank=RANK, n_basis=N_BASIS, d_hidden=32, seed=1)
    f2 = MindForge(k=K, l=L, n_layers=N_LAYERS, d_target=D_TARGET,
                   rank=RANK, n_basis=N_BASIS, d_hidden=32, seed=2)
    assert not np.allclose(f1.A_basis, f2.A_basis)


# ── Numerical stability edges ───────────────────────────────────────────────


def test_zero_context_finite(forge: MindForge):
    """All-zero context must still produce finite adapters (no nan/inf)."""
    ctx = np.zeros((K, L), dtype=np.float32)
    A, B = forge.forge(ctx, layer_id=0)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))


def test_extreme_magnitude_context_finite(forge: MindForge, bc: BlockCodes):
    """Large-magnitude context must not overflow the softmax."""
    ctx = bc.random_discrete(seed=140).astype(np.float32) * 1e3
    A, B = forge.forge(ctx, layer_id=0)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))


def test_layer_id_out_of_range_raises(forge: MindForge, bc: BlockCodes):
    """Out-of-range layer_id should raise (numpy IndexError on layer_embeddings)."""
    ctx = bc.random_discrete(seed=150)
    with pytest.raises(IndexError):
        forge.forge(ctx, layer_id=N_LAYERS + 5)


# ── Memory accounting ──────────────────────────────────────────────────────


def test_memory_bytes_matches_actual_arrays(forge: MindForge):
    """memory_bytes() must match the sum of nbytes across the stored arrays.

    Catches drift between the analytical estimate and the real footprint.
    The estimate excludes b_coeff bias intentionally on the W_h side, so the
    test allows a tight ±0.5% slack.
    """
    actual = (
        forge.A_basis.nbytes
        + forge.B_basis.nbytes
        + forge.layer_embeddings.nbytes
        + forge.W_proj.nbytes
        + forge.W_h.nbytes
        + forge.b_h.nbytes
        + forge.W_coeff.nbytes
        + forge.b_coeff.nbytes
    )
    estimate = forge.memory_bytes()
    rel_err = abs(estimate - actual) / max(actual, 1)
    assert rel_err < 0.005, (
        f"memory_bytes() drift: estimate={estimate}, actual={actual}, "
        f"rel_err={rel_err:.3%}"
    )


# ── External LLM integration ────────────────────────────────────────────────
#
# These tests load a real GGUF model from data/external_llms and verify that
# MindForge can forge adapters whose shapes match the LLM's hidden dimension
# and layer count. This catches the most common production bug — wiring
# MindForge into an LLM with mismatched (d_target, n_layers) — without
# running any LLM inference.

EXTERNAL_LLM_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "external_llms"
)
HARRIER_GGUF = os.path.join(EXTERNAL_LLM_DIR, "harrier-oss-v1-0.6b.Q8_0.gguf")


def _load_harrier_meta() -> tuple[int, int]:
    """Load harrier 0.6B and return ``(n_embd, n_block)``.

    Uses ``llama_cpp.Llama`` with the smallest possible context to keep load
    fast (~0.75s on the dev box). Pulls block count from GGUF metadata so we
    avoid relying on llama_cpp's per-version layer-count APIs.
    """
    from llama_cpp import Llama
    llm = Llama(model_path=HARRIER_GGUF, n_ctx=128, n_threads=2, verbose=False)
    try:
        n_embd = int(llm.n_embd())
        # Qwen3 metadata key for layer count
        n_block = int(llm.metadata.get("qwen3.block_count", 0))
        if n_block == 0:
            # Fallback for non-Qwen architectures
            for key in ("llama.block_count", "general.block_count"):
                if key in llm.metadata:
                    n_block = int(llm.metadata[key])
                    break
        assert n_block > 0, f"could not find block_count in metadata: {list(llm.metadata.keys())}"
        return n_embd, n_block
    finally:
        del llm


@pytest.mark.skipif(
    not os.path.exists(HARRIER_GGUF),
    reason=f"GGUF not found at {HARRIER_GGUF}",
)
def test_external_llm_forge_matches_harrier_dims():
    """Forged adapters must match harrier-0.6B's (d_model, n_layers).

    Smoke test for the LLMInjector use case: if MindForge can't produce
    shape-compatible adapters for a real GGUF, every downstream injection
    will silently mis-broadcast.
    """
    n_embd, n_block = _load_harrier_meta()
    # Sanity: harrier 0.6B is 1024-d × 28 blocks per the GGUF metadata.
    assert n_embd == 1024
    assert n_block == 28

    mf = MindForge(
        k=8, l=64, n_layers=n_block, d_target=n_embd,
        rank=4, n_basis=8, d_hidden=128, seed=42,
    )
    bc_local = BlockCodes(k=8, l=64)
    ctx = bc_local.random_discrete(seed=200)

    # Forge once for layer 0 — must match harrier's (d_model)
    A0, B0 = mf.forge(ctx, layer_id=0)
    assert A0.shape == (4, n_embd)
    assert B0.shape == (n_embd, 4)

    # Forge for the LAST layer — must also work without indexing errors
    A_last, B_last = mf.forge(ctx, layer_id=n_block - 1)
    assert A_last.shape == (4, n_embd)
    assert not np.allclose(A0, A_last), "layer 0 and layer 27 adapters should differ"


@pytest.mark.skipif(
    not os.path.exists(HARRIER_GGUF),
    reason=f"GGUF not found at {HARRIER_GGUF}",
)
def test_external_llm_apply_adapter_on_realistic_hidden_state():
    """apply_adapter must run cleanly on a hidden state shaped like harrier's.

    We don't run llama_cpp inference — we just synthesize a random hidden
    state of the right shape and verify the LoRA pass produces finite output
    of the same shape. This is the integration contract LLMInjector relies on.
    """
    n_embd, n_block = _load_harrier_meta()
    mf = MindForge(
        k=8, l=64, n_layers=n_block, d_target=n_embd,
        rank=4, n_basis=8, d_hidden=128, seed=42,
    )
    bc_local = BlockCodes(k=8, l=64)
    ctx = bc_local.random_discrete(seed=201)
    adapters = mf.forge_all_layers(ctx)
    assert len(adapters) == n_block

    rng = np.random.default_rng(7)
    seq_len = 32
    for layer_id, (A, B) in enumerate(adapters):
        hidden = rng.standard_normal((seq_len, n_embd)).astype(np.float32) * 0.1
        out = mf.apply_adapter(hidden, hidden, A, B)
        assert out.shape == (seq_len, n_embd), f"layer {layer_id}: shape drift"
        assert np.all(np.isfinite(out)), f"layer {layer_id}: nan/inf in adapter output"
        # The LoRA correction must be nontrivial (not a pass-through)
        assert not np.allclose(out, hidden, atol=1e-6), (
            f"layer {layer_id}: apply_adapter produced pass-through (LoRA scale lost?)"
        )


@pytest.mark.skipif(
    not os.path.exists(HARRIER_GGUF),
    reason=f"GGUF not found at {HARRIER_GGUF}",
)
def test_external_llm_via_llm_injector():
    """End-to-end: LLMInjector(use_mindforge=True) wires MindForge to harrier dims.

    This is the actual production code path — the brain → injector → MindForge
    chain. We verify it constructs without errors and inject() runs cleanly
    on a synthetic hidden state.
    """
    from cubemind.brain.llm_injector import LLMInjector

    n_embd, n_block = _load_harrier_meta()
    injector = LLMInjector(
        n_layers=n_block, d_model=n_embd, d_brain=64,
        injection_strength=0.1, use_mindforge=True,
        k=8, l=64, seed=42,
    )
    # Sanity: MindForge must have actually loaded.
    assert injector._mindforge is not None

    bc_local = BlockCodes(k=8, l=64)
    brain_hv = bc_local.random_discrete(seed=210)
    brain_hidden = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    injector.update_brain_state(
        brain_hidden=brain_hidden, brain_hv=brain_hv,
        neurochemistry={"dopamine": 0.6, "cortisol": 0.3},
    )
    # Adapters should now be forged for every layer.
    assert injector._adapters is not None
    assert len(injector._adapters) == n_block

    # Inject at every layer and check shape + finiteness.
    seq_len = 16
    rng = np.random.default_rng(11)
    for layer_id in range(n_block):
        hidden = rng.standard_normal((seq_len, n_embd)).astype(np.float32) * 0.1
        out = injector.inject(hidden, layer_id=layer_id)
        assert out.shape == (seq_len, n_embd)
        assert np.all(np.isfinite(out))
