"""HE-MoE Staging: Stress tests with hypothesis property-based testing.

Finds edge cases: NaN, Inf, overflow, zero vectors, huge values,
domain shifts, memory leaks, long runs.

Run: uv run pytest sandbox/he_moe/staging/stress_test.py -v
"""

from __future__ import annotations

import sys
import time

import numpy as np
from hypothesis import given, settings, strategies as st

_sandbox = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent)
_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent)
for p in [_sandbox, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from he_moe.experiment import HEMoE


# ═══════════════════════════════════════════════════════════════════════════════
# Property-based tests (hypothesis finds edge cases)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    """Hypothesis-driven: find inputs that break HE-MoE."""

    @given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False,
                              allow_infinity=False),
                    min_size=8, max_size=8))
    @settings(max_examples=200, deadline=5000)
    def test_forward_never_nan(self, values):
        m = HEMoE(d_input=8, d_output=8, initial_experts=4, seed=42)
        x = np.array(values, dtype=np.float32)
        y = m.forward(x)
        assert np.all(np.isfinite(y)), f"NaN/Inf in output for input {values}"

    @given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False,
                              allow_infinity=False),
                    min_size=8, max_size=8),
           st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False,
                              allow_infinity=False),
                    min_size=8, max_size=8))
    @settings(max_examples=200, deadline=5000)
    def test_train_step_never_nan(self, x_vals, t_vals):
        m = HEMoE(d_input=8, d_output=8, initial_experts=4, seed=42)
        x = np.array(x_vals, dtype=np.float32)
        t = np.array(t_vals, dtype=np.float32)
        result = m.train_step(x, t)
        assert np.isfinite(result["loss"]), f"Loss NaN for x={x_vals[:3]}"

    @given(st.integers(min_value=1, max_value=8),
           st.integers(min_value=1, max_value=4))
    @settings(max_examples=50, deadline=5000)
    def test_various_expert_counts(self, n_experts, top_k):
        top_k = min(top_k, n_experts)
        m = HEMoE(d_input=8, d_output=8, initial_experts=n_experts,
                    top_k=top_k, seed=42)
        x = np.random.randn(8).astype(np.float32)
        y = m.forward(x)
        assert y.shape == (8,)


# ═══════════════════════════════════════════════════════════════════════════════
# Adversarial inputs
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdversarial:
    """Hand-crafted adversarial inputs."""

    def test_zero_input(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        y = m.forward(np.zeros(8, dtype=np.float32))
        assert np.all(np.isfinite(y))

    def test_huge_input(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        y = m.forward(np.ones(8, dtype=np.float32) * 1e6)
        assert np.all(np.isfinite(y))

    def test_tiny_input(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        y = m.forward(np.ones(8, dtype=np.float32) * 1e-10)
        assert np.all(np.isfinite(y))

    def test_alternating_sign(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        x = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32) * 50
        y = m.forward(x)
        assert np.all(np.isfinite(y))

    def test_all_same_value(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        y = m.forward(np.ones(8, dtype=np.float32) * 3.14)
        assert np.all(np.isfinite(y))

    def test_one_hot(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        x = np.zeros(8, dtype=np.float32)
        x[0] = 100.0
        y = m.forward(x)
        assert np.all(np.isfinite(y))


# ═══════════════════════════════════════════════════════════════════════════════
# Long run stability
# ═══════════════════════════════════════════════════════════════════════════════


class TestLongRun:
    """Extended training runs to catch drift and overflow."""

    def test_10k_steps_stable(self):
        m = HEMoE(d_input=16, d_output=16, initial_experts=4,
                    max_experts=12, spawn_threshold=0.3,
                    charge_flip_threshold=0.3, eta_force=0.005,
                    eta_oja=0.005, eta_consol=0.001, seed=42)
        rng = np.random.default_rng(42)
        nan_count = 0
        for step in range(10000):
            r = m.train_step(rng.standard_normal(16).astype(np.float32),
                             rng.standard_normal(16).astype(np.float32))
            if not np.isfinite(r["loss"]):
                nan_count += 1
        assert nan_count == 0, f"NaN in {nan_count} of 10000 steps"
        assert m.n_experts <= 12

    def test_domain_shift(self):
        """Train on domain A, switch to B, check no collapse."""
        m = HEMoE(d_input=8, d_output=8, initial_experts=4,
                    spawn_threshold=0.2, seed=42)
        rng = np.random.default_rng(42)

        # Domain A: small values
        for _ in range(500):
            m.train_step(rng.standard_normal(8).astype(np.float32) * 0.1,
                         rng.standard_normal(8).astype(np.float32) * 0.1)

        # Domain B: large values (shift)
        losses_b = []
        for _ in range(500):
            r = m.train_step(rng.standard_normal(8).astype(np.float32) * 10,
                             rng.standard_normal(8).astype(np.float32) * 10)
            losses_b.append(r["loss"])

        # Should not explode
        assert all(np.isfinite(l) for l in losses_b)

    def test_memory_doesnt_grow_unbounded(self):
        """Capsule count should stay within max_capsules."""
        m = HEMoE(d_input=8, d_output=8, seed=42)
        m.max_capsules = 100
        rng = np.random.default_rng(42)

        for _ in range(5000):
            m.train_step(rng.standard_normal(8).astype(np.float32) * 3,
                         rng.standard_normal(8).astype(np.float32) * 3)

        assert len(m.capsules) <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# Performance benchmark
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformance:
    """Basic performance measurements."""

    def test_forward_latency(self):
        m = HEMoE(d_input=32, d_output=32, initial_experts=8, top_k=2, seed=42)
        x = np.random.randn(32).astype(np.float32)

        # Warmup
        for _ in range(10):
            m.forward(x)

        t0 = time.perf_counter()
        for _ in range(1000):
            m.forward(x)
        elapsed = (time.perf_counter() - t0) * 1000  # ms

        avg_ms = elapsed / 1000
        print(f"\n  Forward: {avg_ms:.3f} ms/call ({1000/elapsed*1000:.0f} calls/sec)")
        assert avg_ms < 10, f"Too slow: {avg_ms:.1f} ms"

    def test_train_step_latency(self):
        m = HEMoE(d_input=32, d_output=32, initial_experts=8, seed=42)
        rng = np.random.default_rng(42)

        t0 = time.perf_counter()
        for _ in range(500):
            m.train_step(rng.standard_normal(32).astype(np.float32),
                         rng.standard_normal(32).astype(np.float32))
        elapsed = (time.perf_counter() - t0) * 1000

        avg_ms = elapsed / 500
        print(f"\n  Train step: {avg_ms:.3f} ms/step ({500/elapsed*1000:.0f} steps/sec)")
        assert avg_ms < 50, f"Too slow: {avg_ms:.1f} ms"
