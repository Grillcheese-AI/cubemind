"""
Tests for cubemind.execution.world_encoder.WorldEncoder.

Validates:
  - Shape correctness (k, l) and dtype float32
  - Determinism (same input -> same output)
  - Diversity (different inputs -> different outputs)
  - Role binding (self-similarity > 0.99)
  - Narrative encoding shape
  - 128 action variants all unique
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.world_encoder import WorldEncoder
from cubemind.ops.block_codes import BlockCodes

# ── Constants (small dims to avoid OOM) ──────────────────────────────────────

K = 4
L = 32


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture(scope="module")
def enc() -> WorldEncoder:
    return WorldEncoder(k=K, l=L)


# ── Test: _hash_to_vec shape and dtype ───────────────────────────────────────


def test_hash_to_vec_shape_dtype(enc: WorldEncoder):
    """_hash_to_vec returns (k, l) float32."""
    vec = enc._hash_to_vec("hello world")
    assert vec.shape == (K, L), f"Expected ({K}, {L}), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


# ── Test: _hash_to_vec determinism ───────────────────────────────────────────


def test_hash_to_vec_determinism(enc: WorldEncoder):
    """Same text always produces the same vector."""
    v1 = enc._hash_to_vec("the cat sat on the mat")
    v2 = enc._hash_to_vec("the cat sat on the mat")
    np.testing.assert_array_equal(v1, v2)


# ── Test: _hash_to_vec diversity ─────────────────────────────────────────────


def test_hash_to_vec_diversity(enc: WorldEncoder):
    """Different texts produce different vectors."""
    v1 = enc._hash_to_vec("alpha")
    v2 = enc._hash_to_vec("beta")
    assert not np.array_equal(v1, v2), "Different texts must produce different vectors"


# ── Test: _hash_to_vec produces valid one-hot block code ─────────────────────


def test_hash_to_vec_valid_block_code(enc: WorldEncoder):
    """Each block should be one-hot (exactly one 1.0, rest 0.0)."""
    vec = enc._hash_to_vec("test input")
    block_sums = vec.sum(axis=-1)
    np.testing.assert_allclose(
        block_sums,
        np.ones(K, dtype=np.float32),
        atol=1e-5,
        err_msg="Each block must sum to 1 (one-hot)",
    )


# ── Test: _role_vec caching ──────────────────────────────────────────────────


def test_role_vec_cached(enc: WorldEncoder):
    """_role_vec returns identical objects for the same role name."""
    r1 = enc._role_vec("agent")
    r2 = enc._role_vec("agent")
    assert r1 is r2, "Role vectors must be cached (same object)"
    assert r1.shape == (K, L)
    assert r1.dtype == np.float32


# ── Test: encode_state shape and dtype ───────────────────────────────────────


def test_encode_state_shape_dtype(enc: WorldEncoder):
    """encode_state returns (k, l) float32."""
    state = enc.encode_state({"color": "red", "size": "large"})
    assert state.shape == (K, L), f"Expected ({K}, {L}), got {state.shape}"
    assert state.dtype == np.float32, f"Expected float32, got {state.dtype}"


# ── Test: encode_state determinism ───────────────────────────────────────────


def test_encode_state_determinism(enc: WorldEncoder):
    """Same attributes produce the same state vector."""
    attrs = {"mood": "happy", "location": "park"}
    s1 = enc.encode_state(attrs)
    s2 = enc.encode_state(attrs)
    np.testing.assert_allclose(s1, s2, atol=1e-14)


# ── Test: encode_state role binding recoverable ──────────────────────────────


def test_encode_state_role_binding(enc: WorldEncoder, bc: BlockCodes):
    """Binding a role-value pair and unbinding with the role recovers the value."""
    role_vec = enc._role_vec("color")
    value_vec = enc._hash_to_vec("red")
    bound = bc.bind(role_vec, value_vec)
    recovered = bc.unbind(bound, role_vec)
    sim = bc.similarity(bc.discretize(recovered), value_vec)
    assert sim > 0.99, f"Role unbinding similarity {sim} < 0.99"


# ── Test: encode_action shape and dtype ──────────────────────────────────────


def test_encode_action_shape_dtype(enc: WorldEncoder):
    """encode_action returns (k, l) float32."""
    vec = enc.encode_action("pick up the sword")
    assert vec.shape == (K, L), f"Expected ({K}, {L}), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


# ── Test: encode_action determinism ──────────────────────────────────────────


def test_encode_action_determinism(enc: WorldEncoder):
    """Same action text produces the same vector."""
    a1 = enc.encode_action("open door")
    a2 = enc.encode_action("open door")
    np.testing.assert_array_equal(a1, a2)


# ── Test: encode_action diversity ────────────────────────────────────────────


def test_encode_action_diversity(enc: WorldEncoder):
    """Different action texts produce different vectors."""
    a1 = enc.encode_action("go north")
    a2 = enc.encode_action("go south")
    assert not np.array_equal(a1, a2), "Different actions must differ"


# ── Test: encode_narrative shape ─────────────────────────────────────────────


def test_encode_narrative_shape_dtype(enc: WorldEncoder):
    """encode_narrative returns (k, l) float32."""
    text = "The knight entered the castle. A dragon appeared."
    vec = enc.encode_narrative(text)
    assert vec.shape == (K, L), f"Expected ({K}, {L}), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


# ── Test: encode_narrative determinism ───────────────────────────────────────


def test_encode_narrative_determinism(enc: WorldEncoder):
    """Same narrative produces the same vector."""
    text = "It was a dark and stormy night. The wind howled."
    n1 = enc.encode_narrative(text)
    n2 = enc.encode_narrative(text)
    np.testing.assert_allclose(n1, n2, atol=1e-14)


# ── Test: encode_narrative diversity ─────────────────────────────────────────


def test_encode_narrative_diversity(enc: WorldEncoder):
    """Different narratives produce different vectors."""
    n1 = enc.encode_narrative("The sun rose. Birds sang.")
    n2 = enc.encode_narrative("It rained all day. Puddles formed.")
    assert not np.array_equal(n1, n2), "Different narratives must differ"


# ── Test: generate_action_variants ───────────────────────────────────────────


def test_generate_action_variants_count_and_shape(enc: WorldEncoder):
    """generate_action_variants returns n unique (k, l) vectors."""
    base = enc.encode_action("swing sword")
    variants = enc.generate_action_variants(base, n=128)
    assert variants.shape == (128, K, L), (
        f"Expected (128, {K}, {L}), got {variants.shape}"
    )
    assert variants.dtype == np.float32


def test_generate_action_variants_all_unique(enc: WorldEncoder):
    """All 128 action variants must be pairwise distinct."""
    base = enc.encode_action("cast spell")
    variants = enc.generate_action_variants(base, n=128)
    flat = variants.reshape(128, -1)
    for i in range(128):
        for j in range(i + 1, 128):
            assert not np.array_equal(flat[i], flat[j]), (
                f"Variants {i} and {j} are identical"
            )


def test_generate_action_variants_determinism(enc: WorldEncoder):
    """Same base action produces the same set of variants."""
    base = enc.encode_action("dodge")
    v1 = enc.generate_action_variants(base, n=16)
    v2 = enc.generate_action_variants(base, n=16)
    np.testing.assert_allclose(v1, v2, atol=1e-14)
