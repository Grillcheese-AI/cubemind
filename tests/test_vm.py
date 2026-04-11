"""Tests for cubemind.reasoning.vm — VSA Virtual Machine.

Tests the ~15 universal primitives that all reasoning reduces to.
Each primitive operates on block-code vectors via the VSA algebra.
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.ops.block_codes import BlockCodes

K, L = 4, 32


@pytest.fixture
def bc():
    return BlockCodes(k=K, l=L)


@pytest.fixture
def vm(bc):
    from cubemind.reasoning.vm import VSAVM
    return VSAVM(bc=bc, seed=42)


# ── Register Operations ─────────────────────────────────────────────────


class TestRegisters:

    def test_vm_starts_with_empty_registers(self, vm):
        assert len(vm.registers) == 0

    def test_create_register(self, vm):
        vm.execute("CREATE", "apple", "fruit")
        assert "apple" in vm.registers
        assert vm.registers["apple"].shape == (K, L)

    def test_create_multiple_registers(self, vm):
        vm.execute("CREATE", "apple", "fruit")
        vm.execute("CREATE", "basket", "container")
        assert "apple" in vm.registers
        assert "basket" in vm.registers

    def test_destroy_register(self, vm):
        vm.execute("CREATE", "temp", "thing")
        vm.execute("DESTROY", "temp")
        assert "temp" not in vm.registers

    def test_destroy_nonexistent_is_noop(self, vm):
        vm.execute("DESTROY", "nonexistent")  # should not raise


# ── ASSIGN ───────────────────────────────────────────────────────────────


class TestAssign:

    def test_assign_creates_binding(self, vm, bc):
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 5)
        # The register should now encode the value 5
        assert vm.registers["x"].shape == (K, L)

    def test_assign_different_values_produce_different_vectors(self, vm):
        vm.execute("CREATE", "a", "number")
        vm.execute("ASSIGN", "a", 3)
        vec_3 = vm.registers["a"].copy()

        vm.execute("ASSIGN", "a", 7)
        vec_7 = vm.registers["a"].copy()

        # Different values should produce different block-codes
        assert not np.array_equal(vec_3, vec_7)

    def test_assign_same_value_is_deterministic(self, vm):
        vm.execute("CREATE", "a", "number")
        vm.execute("ASSIGN", "a", 42)
        vec1 = vm.registers["a"].copy()

        vm.execute("ASSIGN", "a", 42)
        vec2 = vm.registers["a"].copy()

        np.testing.assert_array_equal(vec1, vec2)


# ── ADD / SUB ────────────────────────────────────────────────────────────


class TestArithmetic:

    def test_add_changes_register(self, vm):
        vm.execute("CREATE", "count", "number")
        vm.execute("ASSIGN", "count", 5)
        before = vm.registers["count"].copy()

        vm.execute("ADD", "count", 3)
        after = vm.registers["count"]

        assert not np.array_equal(before, after)

    def test_sub_changes_register(self, vm):
        vm.execute("CREATE", "count", "number")
        vm.execute("ASSIGN", "count", 10)
        before = vm.registers["count"].copy()

        vm.execute("SUB", "count", 4)
        after = vm.registers["count"]

        assert not np.array_equal(before, after)

    def test_add_then_query_recovers_result(self, vm):
        """ASSIGN 5, ADD 3 → QUERY should recover ~8."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 5)
        vm.execute("ADD", "x", 3)
        result = vm.execute("QUERY", "x")
        assert result == 8

    def test_sub_then_query_recovers_result(self, vm):
        """ASSIGN 10, SUB 4 → QUERY should recover ~6."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 10)
        vm.execute("SUB", "x", 4)
        result = vm.execute("QUERY", "x")
        assert result == 6


# ── TRANSFER ─────────────────────────────────────────────────────────────


class TestTransfer:

    def test_transfer_moves_quantity(self, vm):
        """'John gives 3 apples to Mary' → TRANSFER(john, mary, 3)"""
        vm.execute("CREATE", "john", "person")
        vm.execute("CREATE", "mary", "person")
        vm.execute("ASSIGN", "john", 5)
        vm.execute("ASSIGN", "mary", 2)

        vm.execute("TRANSFER", "john", "mary", 3)

        assert vm.execute("QUERY", "john") == 2
        assert vm.execute("QUERY", "mary") == 5

    def test_transfer_zero_is_noop(self, vm):
        vm.execute("CREATE", "a", "person")
        vm.execute("ASSIGN", "a", 5)
        before = vm.registers["a"].copy()

        vm.execute("TRANSFER", "a", "a", 0)
        np.testing.assert_array_equal(vm.registers["a"], before)


# ── COMPARE ──────────────────────────────────────────────────────────────


class TestCompare:

    def test_compare_equal_values(self, vm):
        vm.execute("CREATE", "a", "number")
        vm.execute("CREATE", "b", "number")
        vm.execute("ASSIGN", "a", 5)
        vm.execute("ASSIGN", "b", 5)

        result = vm.execute("COMPARE", "a", "b")
        assert result == "equal"

    def test_compare_different_values(self, vm):
        vm.execute("CREATE", "a", "number")
        vm.execute("CREATE", "b", "number")
        vm.execute("ASSIGN", "a", 3)
        vm.execute("ASSIGN", "b", 7)

        result = vm.execute("COMPARE", "a", "b")
        assert result in ("less", "greater", "not_equal")


# ── STORE / RECALL ───────────────────────────────────────────────────────


class TestMemory:

    def test_store_and_recall(self, vm, bc):
        """Store a rule, then recall it by similar context."""
        vm.execute("CREATE", "pattern", "rule")
        vm.execute("ASSIGN", "pattern", 42)

        vm.execute("STORE", "pattern", "addition_rule")

        result = vm.execute("RECALL", "pattern")
        assert result is not None

    def test_recall_nonexistent_returns_none(self, vm):
        vm.execute("CREATE", "unknown", "rule")
        vm.execute("ASSIGN", "unknown", 999)
        result = vm.execute("RECALL", "unknown")
        # No rules stored yet, should return None or empty
        assert result is None or result == []


# ── PROGRAM EXECUTION ────────────────────────────────────────────────────


class TestProgram:

    def test_run_simple_program(self, vm):
        """'John has 5 apples. He gives 2 to Mary. How many does John have?'"""
        program = [
            ("CREATE", "john", "person"),
            ("CREATE", "mary", "person"),
            ("ASSIGN", "john", 5),
            ("ASSIGN", "mary", 0),
            ("TRANSFER", "john", "mary", 2),
            ("QUERY", "john"),
        ]
        result = vm.run(program)
        assert result == 3

    def test_run_accumulation_program(self, vm):
        """'Start with 10. Add 5. Subtract 3. What is the result?'"""
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 10),
            ("ADD", "x", 5),
            ("SUB", "x", 3),
            ("QUERY", "x"),
        ]
        result = vm.run(program)
        assert result == 12

    def test_run_multi_entity_program(self, vm):
        """'A has 8, B has 3. A gives 4 to B. How many does B have?'"""
        program = [
            ("CREATE", "a", "person"),
            ("CREATE", "b", "person"),
            ("ASSIGN", "a", 8),
            ("ASSIGN", "b", 3),
            ("TRANSFER", "a", "b", 4),
            ("QUERY", "b"),
        ]
        result = vm.run(program)
        assert result == 7

    def test_step_count_advances(self, vm):
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 1),
            ("ADD", "x", 1),
        ]
        vm.run(program)
        assert vm.step_count == 3


# ── TRACE / RULE LEARNING ───────────────────────────────────────────────


class TestRuleLearning:

    def test_trace_recording(self, vm):
        """VM should record instruction trace for rule learning."""
        vm.trace_enabled = True
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 5),
            ("ADD", "x", 3),
        ]
        vm.run(program)
        assert len(vm.trace) == 3
        assert vm.trace[0][0] == "CREATE"
        assert vm.trace[2][0] == "ADD"

    def test_store_trace_as_rule(self, vm):
        """After solving a problem, store the trace as a reusable rule."""
        vm.trace_enabled = True
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 5),
            ("ADD", "x", 3),
            ("QUERY", "x"),
        ]
        vm.run(program)

        # Store the trace as a named rule
        vm.store_rule("addition_pattern")
        assert "addition_pattern" in vm.rules


# ── HYPERSEED VALUE ENCODING ──────────────────────────────────────────────


class TestHyperSeed:
    """Test HyperSeed-based value encoding.

    HyperSeed generates value vectors by iterative binding from a base:
        v[n] = bind(v[n-1], increment_vec)

    This gives:
    - Nearby numbers produce similar vectors (similarity gradient)
    - Arithmetic in VSA space: v[a+b] = bind(v[a], v[b])
    - unbind(v[a+b], v[a]) ≈ v[b]
    """

    def test_hyperseed_exists(self, vm):
        """VM should have a HyperSeed encoder."""
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(vm.bc, seed=42)
        assert hs is not None

    def test_encode_returns_block_code(self, vm):
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(vm.bc, seed=42)
        v = hs.encode(5)
        assert v.shape == (K, L)

    def test_same_value_same_vector(self, vm):
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(vm.bc, seed=42)
        v1 = hs.encode(5)
        v2 = hs.encode(5)
        np.testing.assert_array_equal(v1, v2)

    def test_different_values_different_vectors(self, vm):
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(vm.bc, seed=42)
        v3 = hs.encode(3)
        v7 = hs.encode(7)
        assert not np.array_equal(v3, v7)

    def test_nearby_values_more_similar_than_distant(self):
        """v[5] should be more similar to v[6] than to v[100].

        Uses k=16, l=64 because the similarity gradient needs enough
        blocks for partial shifting to create a measurable difference.
        """
        from cubemind.reasoning.vm import HyperSeed
        bc_large = BlockCodes(k=16, l=64)
        hs = HyperSeed(bc_large, seed=42)
        v5 = hs.encode(5)
        v6 = hs.encode(6)
        v100 = hs.encode(100)

        sim_near = float(bc_large.similarity(v5, v6))
        sim_far = float(bc_large.similarity(v5, v100))
        assert sim_near > sim_far, (
            f"Nearby should be more similar: sim(5,6)={sim_near:.3f} vs sim(5,100)={sim_far:.3f}"
        )

    def test_addition_in_vsa_space(self, vm, bc):
        """v[a+b] ≈ bind(v[a], v[b]) — arithmetic works via binding."""
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(bc, seed=42)

        v3 = hs.encode(3)
        v5 = hs.encode(5)
        v8 = hs.encode(8)

        # bind(v[3], v[5]) should be similar to v[8]
        v_sum = bc.bind(v3, v5)
        sim = float(bc.similarity(v_sum, v8))
        assert sim > 0.5, f"bind(v[3], v[5]) should ≈ v[8]: sim={sim:.3f}"

    def test_subtraction_via_unbind(self, vm, bc):
        """unbind(v[a+b], v[a]) ≈ v[b] — subtraction works via unbinding."""
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(bc, seed=42)

        v3 = hs.encode(3)
        v8 = hs.encode(8)
        v5 = hs.encode(5)

        # unbind(v[8], v[3]) should be similar to v[5]
        v_diff = bc.unbind(v8, v3)
        sim = float(bc.similarity(v_diff, v5))
        assert sim > 0.5, f"unbind(v[8], v[3]) should ≈ v[5]: sim={sim:.3f}"

    def test_increment_vector_is_consistent(self, vm, bc):
        """The delta between consecutive values should be consistent."""
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(bc, seed=42)

        # delta(n, n+1) should be similar for different n
        d01 = bc.unbind(hs.encode(1), hs.encode(0))
        d23 = bc.unbind(hs.encode(3), hs.encode(2))
        d78 = bc.unbind(hs.encode(8), hs.encode(7))

        sim_01_23 = float(bc.similarity(d01, d23))
        sim_01_78 = float(bc.similarity(d01, d78))
        assert sim_01_23 > 0.8, f"Increment should be consistent: sim={sim_01_23:.3f}"
        assert sim_01_78 > 0.8, f"Increment should be consistent: sim={sim_01_78:.3f}"

    def test_zero_is_identity(self, vm, bc):
        """v[0] should act as identity: bind(v[n], v[0]) ≈ v[n]."""
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(bc, seed=42)

        v0 = hs.encode(0)
        v5 = hs.encode(5)
        result = bc.bind(v5, v0)
        sim = float(bc.similarity(result, v5))
        assert sim > 0.8, f"v[0] should be identity-like: sim={sim:.3f}"

    def test_negative_values(self, vm, bc):
        """Negative values should work: v[-3] = inverse of v[3]."""
        from cubemind.reasoning.vm import HyperSeed
        hs = HyperSeed(bc, seed=42)

        v3 = hs.encode(3)
        v_neg3 = hs.encode(-3)

        # bind(v[3], v[-3]) should ≈ v[0] (identity)
        v0 = hs.encode(0)
        result = bc.bind(v3, v_neg3)
        sim = float(bc.similarity(result, v0))
        assert sim > 0.5, f"v[3] + v[-3] should ≈ v[0]: sim={sim:.3f}"

    def test_vm_uses_hyperseed_for_values(self, vm):
        """After enabling HyperSeed, the VM should use it for ASSIGN/ADD/SUB."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 5)
        vm.execute("ADD", "x", 3)
        result = vm.execute("QUERY", "x")
        assert result == 8  # should still work


# ── UNIVERSAL ROLES ──────────────────────────────────────────────────────


class TestUniversalRoles:
    """Test that the VM provides universal role vectors for structured binding."""

    def test_roles_exist(self, vm):
        """VM should expose universal role vectors."""
        from cubemind.reasoning.vm import ROLES
        expected = ["AGENT", "ACTION", "OBJECT", "QUANTITY", "SOURCE",
                    "DESTINATION", "CONTEXT", "STATE"]
        for role in expected:
            assert role in ROLES
            assert ROLES[role].shape == (K, L)

    def test_roles_are_dissimilar(self, vm, bc):
        """Universal roles should be near-orthogonal (low similarity)."""
        from cubemind.reasoning.vm import ROLES
        names = list(ROLES.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = float(bc.similarity(ROLES[names[i]], ROLES[names[j]]))
                assert abs(sim) < 0.3, (
                    f"Roles {names[i]} and {names[j]} too similar: {sim:.3f}"
                )

    def test_bind_agent_object_quantity(self, vm, bc):
        """BIND_ROLE creates structured representation from role-filler pairs."""
        vm.execute("CREATE", "john", "person")
        vm.execute("BIND_ROLE", "john", "AGENT", "john_entity")
        vm.execute("BIND_ROLE", "john", "OBJECT", "apple")
        vm.execute("BIND_ROLE", "john", "QUANTITY", 5)

        # Should be able to unbind and recover
        result = vm.execute("UNBIND_ROLE", "john", "QUANTITY")
        assert result == 5

    def test_unbind_agent(self, vm, bc):
        """UNBIND_ROLE recovers the filler from a role-bound register."""
        vm.execute("CREATE", "event", "action")
        vm.execute("BIND_ROLE", "event", "AGENT", "mary")
        vm.execute("BIND_ROLE", "event", "ACTION", "gives")

        agent = vm.execute("UNBIND_ROLE", "event", "AGENT")
        assert agent == "mary"

    def test_structured_state_bundle(self, vm, bc):
        """Multiple role bindings bundle into a single structured state."""
        vm.execute("CREATE", "state", "snapshot")
        vm.execute("BIND_ROLE", "state", "AGENT", "john")
        vm.execute("BIND_ROLE", "state", "OBJECT", "apple")
        vm.execute("BIND_ROLE", "state", "QUANTITY", 5)
        vm.execute("BIND_ROLE", "state", "ACTION", "has")

        # All fillers recoverable from the same register
        assert vm.execute("UNBIND_ROLE", "state", "AGENT") == "john"
        assert vm.execute("UNBIND_ROLE", "state", "OBJECT") == "apple"
        assert vm.execute("UNBIND_ROLE", "state", "QUANTITY") == 5
        assert vm.execute("UNBIND_ROLE", "state", "ACTION") == "has"


# ── PATTERN DISCOVERY ────────────────────────────────────────────────────


class TestPatternDiscovery:
    """Test that the VM discovers patterns in sequences of block-codes
    without knowing what the attributes mean."""

    def test_diff_detects_change(self, vm, bc):
        """DIFF(a, b) returns a delta vector representing what changed."""
        v0 = bc.random_discrete(seed=10)
        v1 = bc.random_discrete(seed=20)

        delta = vm.execute("DIFF", v0, v1)
        assert delta.shape == (K, L)
        # Delta should not be zero (vectors are different)
        assert not np.allclose(delta, 0)

    def test_diff_same_vectors_is_identity(self, vm, bc):
        """DIFF of identical vectors should be close to identity/zero-change."""
        v0 = bc.random_discrete(seed=10)
        delta = vm.execute("DIFF", v0, v0)
        # Binding a vector with itself → identity-like (high self-sim)
        assert delta.shape == (K, L)

    def test_detect_constant_pattern(self, vm, bc):
        """DETECT_PATTERN on identical vectors → 'constant'."""
        v = bc.random_discrete(seed=42)
        sequence = [v, v, v]
        pattern = vm.execute("DETECT_PATTERN", sequence)
        assert pattern["type"] == "constant"

    def test_detect_progression_pattern(self, vm, bc):
        """DETECT_PATTERN on vectors with consistent deltas → 'progression'."""
        # Create a progression: v0, v0⊕δ, v0⊕δ⊕δ
        v0 = bc.random_discrete(seed=10)
        delta = bc.random_discrete(seed=99)
        v1 = bc.bind(v0, delta)
        v2 = bc.bind(v1, delta)
        sequence = [v0, v1, v2]

        pattern = vm.execute("DETECT_PATTERN", sequence)
        assert pattern["type"] == "progression"
        assert pattern["delta"].shape == (K, L)

    def test_predict_next_constant(self, vm, bc):
        """PREDICT on a constant sequence returns the same vector."""
        v = bc.random_discrete(seed=42)
        sequence = [v, v, v]
        predicted = vm.execute("PREDICT", sequence)
        sim = float(bc.similarity(predicted, v))
        assert sim > 0.9, f"Constant prediction should match: sim={sim:.3f}"

    def test_predict_next_progression(self, vm, bc):
        """PREDICT on a progression applies the delta one more time."""
        v0 = bc.random_discrete(seed=10)
        delta = bc.random_discrete(seed=99)
        v1 = bc.bind(v0, delta)
        v2 = bc.bind(v1, delta)
        expected = bc.bind(v2, delta)  # v3 = v2 ⊕ δ

        sequence = [v0, v1, v2]
        predicted = vm.execute("PREDICT", sequence)

        sim = float(bc.similarity(predicted, expected))
        assert sim > 0.9, f"Progression prediction should match: sim={sim:.3f}"

    def test_match_selects_best_candidate(self, vm, bc):
        """MATCH finds the candidate most similar to the target."""
        target = bc.random_discrete(seed=42)
        candidates = [
            bc.random_discrete(seed=100),
            bc.random_discrete(seed=200),
            target.copy(),  # this one should win
            bc.random_discrete(seed=300),
        ]
        best_idx = vm.execute("MATCH", target, candidates)
        assert best_idx == 2

    def test_full_raven_style_solve(self, vm, bc):
        """End-to-end: detect pattern in 3 panels, predict 4th, match answer.

        This is the RAVEN solving pipeline — without knowing what the
        'attributes' are. The VM just sees block-codes and finds structure.
        """
        # Create a progression pattern
        v0 = bc.random_discrete(seed=10)
        delta = bc.random_discrete(seed=50)
        v1 = bc.bind(v0, delta)
        v2 = bc.bind(v1, delta)
        v3_expected = bc.bind(v2, delta)

        # Create answer candidates (one correct, rest random)
        candidates = [
            bc.random_discrete(seed=200),
            bc.random_discrete(seed=300),
            v3_expected.copy(),
            bc.random_discrete(seed=400),
            bc.random_discrete(seed=500),
        ]

        # The VM program to solve it
        program = [
            ("DETECT_PATTERN", [v0, v1, v2]),
            ("PREDICT", [v0, v1, v2]),
            ("MATCH", None, candidates),  # None = use last predicted
        ]
        result = vm.run(program)
        assert result == 2  # index of correct answer
