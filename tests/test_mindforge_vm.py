"""Tests for MindForge integration with VSA-VM.

Tests the SDLS duality gate, CleanupMemory purification,
and HDR rule discovery pipeline through MindForge.
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes
from cubemind.execution.mindforge import MindForge

K, L = 8, 64


@pytest.fixture
def bc():
    return BlockCodes(k=K, l=L)


@pytest.fixture
def forge(bc):
    return MindForge(
        k=K, l=L, n_layers=4, d_target=64,
        rank=4, n_basis=8, d_hidden=32, seed=42,
    )


@pytest.fixture
def vm(bc):
    from cubemind.reasoning.vm import VSAVM
    return VSAVM(bc=bc, seed=42)


# ── SDLS DUALITY GATE ───────────────────────────────────────────────────


class TestSDLSDualityGate:
    """Test the Self-Dual Latent Space purification gate on MindForge."""

    def test_forge_has_sdls_purify(self, forge):
        """MindForge should have an sdls_purify method."""
        assert hasattr(forge, "sdls_purify")

    def test_clean_context_passes_gate(self, forge, bc):
        """A known clean context should pass through SDLS purification."""
        clean_ctx = bc.random_discrete(seed=42)
        # Register in cleanup memory
        forge.register_context("known_ctx", clean_ctx)

        purified = forge.sdls_purify(clean_ctx)
        sim = float(bc.similarity(purified, clean_ctx))
        assert sim > 0.9, f"Clean context should pass: sim={sim:.3f}"

    def test_noisy_context_gets_cleaned(self, forge, bc):
        """A noisy context should be snapped to nearest clean vector."""
        clean = bc.random_discrete(seed=42)
        forge.register_context("target", clean)

        # Add mild noise (enough to perturb but not destroy)
        rng = np.random.default_rng(99)
        noisy = bc.discretize(
            clean.astype(np.float32) + rng.normal(0, 0.05, clean.shape).astype(np.float32)
        )

        purified = forge.sdls_purify(noisy)
        sim = float(bc.similarity(purified, clean))
        assert sim > 0.9, f"Should snap to clean: sim={sim:.3f}"

    def test_unknown_context_returns_default(self, forge, bc):
        """A context too far from any stored vector returns a safe default."""
        unknown = bc.random_discrete(seed=9999)
        purified = forge.sdls_purify(unknown, threshold=0.95)
        # Should be the default (zero or identity context)
        assert purified is not None
        assert purified.shape == (K, L)

    def test_duality_check_both_directions(self, forge, bc):
        """Self-duality: unbind(bind(role, val), role) ≈ val AND vice versa."""
        role = bc.random_discrete(seed=10)
        value = bc.random_discrete(seed=20)
        context = bc.bind(role, value)

        score = forge.verify_duality(context, role, value)
        assert score > 0.8, f"Duality score should be high: {score:.3f}"

    def test_duality_check_fails_for_random(self, forge, bc):
        """Random vectors should fail the duality check."""
        random_ctx = bc.random_discrete(seed=100)
        random_role = bc.random_discrete(seed=200)
        random_val = bc.random_discrete(seed=300)

        score = forge.verify_duality(random_ctx, random_role, random_val)
        assert score < 0.5, f"Random should fail duality: {score:.3f}"

    def test_sdls_modulates_softmax_temperature(self, forge, bc):
        """Low duality score → high temperature (generic adapter).
        High duality score → low temperature (specialized adapter)."""
        clean = bc.random_discrete(seed=42)
        forge.register_context("known", clean)

        # Forge with clean context — should use sharp coefficients
        A_clean, B_clean = forge.forge_with_sdls(clean, layer_id=0)
        assert A_clean.shape == (4, 64)

        # Forge with noisy context — should produce different (more generic) adapter
        noisy = bc.random_discrete(seed=9999)
        A_noisy, B_noisy = forge.forge_with_sdls(noisy, layer_id=0)
        assert A_noisy.shape == (4, 64)

        # They should differ
        assert not np.allclose(A_clean, A_noisy, atol=1e-3)


# ── VM + MINDFORGE INTEGRATION ──────────────────────────────────────────


class TestVMMindForge:
    """Test MindForge wired into the VM as the JIT compiler."""

    def test_vm_forge_opcode(self, vm, bc):
        """FORGE opcode generates adapters via MindForge."""
        forge = MindForge(
            k=K, l=L, n_layers=4, d_target=64,
            rank=4, n_basis=8, d_hidden=32, seed=42,
        )
        vm.forge = forge

        vm.execute("CREATE", "ctx", "context")
        vm.execute("ASSIGN", "ctx", 42)

        result = vm.execute("FORGE", "ctx", 0)  # forge for layer 0
        assert result is not None
        A, B = result
        assert A.shape == (4, 64)
        assert B.shape == (64, 4)

    def test_vm_forge_all_layers(self, vm, bc):
        """FORGE_ALL generates adapters for every layer."""
        forge = MindForge(
            k=K, l=L, n_layers=4, d_target=64,
            rank=4, n_basis=8, d_hidden=32, seed=42,
        )
        vm.forge = forge

        vm.execute("CREATE", "ctx", "context")
        vm.execute("ASSIGN", "ctx", 7)

        result = vm.execute("FORGE_ALL", "ctx")
        assert isinstance(result, list)
        assert len(result) == 4
        for A, B in result:
            assert A.shape == (4, 64)

    def test_forge_different_contexts_different_adapters(self, vm, bc):
        """Different contexts should produce different adapters."""
        forge = MindForge(
            k=K, l=L, n_layers=4, d_target=64,
            rank=4, n_basis=8, d_hidden=32, seed=42,
        )
        vm.forge = forge

        vm.execute("CREATE", "ctx1", "context")
        vm.execute("ASSIGN", "ctx1", 1)
        A1, _ = vm.execute("FORGE", "ctx1", 0)

        vm.execute("CREATE", "ctx2", "context")
        vm.execute("ASSIGN", "ctx2", 99)
        A2, _ = vm.execute("FORGE", "ctx2", 0)

        assert not np.allclose(A1, A2, atol=1e-3)
