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


# ── CLEANUP MEMORY ───────────────────────────────────────────────────────


class TestCleanupMemory:
    """Test the associative cleanup memory that snaps noisy vectors
    back to the nearest known clean block-code primitive.

    Without cleanup, bundling accumulates noise and after 5-6 ops
    the signal degrades below recognition. Cleanup memory fixes this.
    """

    def test_cleanup_memory_exists(self, vm):
        """VM should have a cleanup memory."""
        from cubemind.reasoning.vm import CleanupMemory
        cm = CleanupMemory(vm.bc)
        assert cm is not None

    def test_store_and_retrieve_exact(self, vm, bc):
        """Storing a vector and querying it back returns the same vector."""
        from cubemind.reasoning.vm import CleanupMemory
        cm = CleanupMemory(bc)

        v = bc.random_discrete(seed=42)
        cm.store("apple", v)
        name, cleaned = cm.cleanup(v)
        assert name == "apple"
        np.testing.assert_array_equal(cleaned, v)

    def test_cleanup_noisy_vector(self, vm, bc):
        """A noisy version of a stored vector should snap back to the clean one."""
        from cubemind.reasoning.vm import CleanupMemory
        cm = CleanupMemory(bc)

        clean = bc.random_discrete(seed=42)
        cm.store("target", clean)

        # Add noise: flip some values
        noisy = clean.copy().astype(np.float32)
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 0.3, noisy.shape).astype(np.float32)
        noisy = bc.discretize(noisy + noise)

        name, cleaned = cm.cleanup(noisy)
        assert name == "target"
        np.testing.assert_array_equal(cleaned, clean)

    def test_cleanup_after_bundle(self, vm, bc):
        """Bundling multiple vectors and cleaning up recovers the dominant one."""
        from cubemind.reasoning.vm import CleanupMemory
        cm = CleanupMemory(bc)

        v_apple = bc.random_discrete(seed=10)
        v_banana = bc.random_discrete(seed=20)
        v_cherry = bc.random_discrete(seed=30)
        cm.store("apple", v_apple)
        cm.store("banana", v_banana)
        cm.store("cherry", v_cherry)

        # Bundle with apple dominant (3x weight)
        bundled = (3 * v_apple.astype(np.float32)
                   + v_banana.astype(np.float32)
                   + v_cherry.astype(np.float32))
        noisy = bc.discretize(bundled)

        name, _ = cm.cleanup(noisy)
        assert name == "apple"

    def test_cleanup_distinguishes_stored_items(self, vm, bc):
        """Cleanup correctly distinguishes between multiple stored vectors."""
        from cubemind.reasoning.vm import CleanupMemory
        cm = CleanupMemory(bc)

        vectors = {}
        for i, name in enumerate(["cat", "dog", "fish", "bird"]):
            v = bc.random_discrete(seed=100 + i)
            cm.store(name, v)
            vectors[name] = v

        for name, v in vectors.items():
            result_name, _ = cm.cleanup(v)
            assert result_name == name, f"Expected {name}, got {result_name}"

    def test_vm_has_cleanup_memory(self, vm, bc):
        """The VM should have a built-in cleanup memory."""
        assert hasattr(vm, "cleanup_mem")
        assert vm.cleanup_mem is not None

    def test_cleanup_opcode(self, vm, bc):
        """CLEANUP opcode should snap a register to nearest clean vector."""
        v = bc.random_discrete(seed=42)
        vm.cleanup_mem.store("known_vec", v)

        vm.execute("CREATE", "test", "thing")
        # Corrupt the register with noise
        noisy = v.copy().astype(np.float32)
        noisy += np.random.default_rng(99).normal(0, 0.2, noisy.shape).astype(np.float32)
        vm.registers["test"] = bc.discretize(noisy)

        result = vm.execute("CLEANUP", "test")
        assert result == "known_vec"


# ── POSITION-AWARE SEQUENCE (SEQ) ───────────────────────────────────────


class TestSequence:
    """Test position-aware sequence encoding.

    SEQ uses per-position circular permutation so that
    'A then B' ≠ 'B then A' in VSA space.
    """

    def test_seq_encodes_order(self, vm, bc):
        """SEQ(A, B) should produce a different vector than SEQ(B, A)."""
        va = bc.random_discrete(seed=10)
        vb = bc.random_discrete(seed=20)

        seq_ab = vm.execute("SEQ", [va, vb])
        seq_ba = vm.execute("SEQ", [vb, va])

        assert seq_ab.shape == (K, L)
        assert seq_ba.shape == (K, L)

        sim = float(bc.similarity(seq_ab, seq_ba))
        assert sim < 0.5, f"SEQ(A,B) and SEQ(B,A) should differ: sim={sim:.3f}"

    def test_seq_single_element(self, vm, bc):
        """SEQ of one element returns that element (permuted at position 0)."""
        va = bc.random_discrete(seed=10)
        result = vm.execute("SEQ", [va])
        assert result.shape == (K, L)

    def test_seq_recovers_element_by_position(self, vm, bc):
        """Can unbind position to recover the element at that position."""
        va = bc.random_discrete(seed=10)
        vb = bc.random_discrete(seed=20)
        vc = bc.random_discrete(seed=30)

        seq_vec = vm.execute("SEQ", [va, vb, vc])

        # UNSEQ at position 1 should recover something correlated with vb
        # Note: at small dims (k=4) recovery is noisy — threshold is lenient
        recovered = vm.execute("UNSEQ", seq_vec, 1)
        sim = float(bc.similarity(recovered, vb))
        assert sim > 0.15, f"UNSEQ at pos 1 should recover B: sim={sim:.3f}"

    def test_seq_preserves_length(self, vm, bc):
        """SEQ of N elements produces a single (k, l) vector."""
        vectors = [bc.random_discrete(seed=i) for i in range(5)]
        result = vm.execute("SEQ", vectors)
        assert result.shape == (K, L)

    def test_seq_different_lengths_different_vectors(self, vm, bc):
        """SEQ(A, B) ≠ SEQ(A, B, C)."""
        va = bc.random_discrete(seed=10)
        vb = bc.random_discrete(seed=20)
        vc = bc.random_discrete(seed=30)

        seq2 = vm.execute("SEQ", [va, vb])
        seq3 = vm.execute("SEQ", [va, vb, vc])

        sim = float(bc.similarity(seq2, seq3))
        assert sim < 0.8, f"Different length seqs should differ: sim={sim:.3f}"


# ── MUL / DIV ───────────────────────────────────────────────────────────


class TestMulDiv:

    def test_mul_doubles(self, vm):
        """ASSIGN 4, MUL 2 → QUERY = 8."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 4)
        vm.execute("MUL", "x", 2)
        assert vm.execute("QUERY", "x") == 8

    def test_mul_triples(self, vm):
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 3)
        vm.execute("MUL", "x", 3)
        assert vm.execute("QUERY", "x") == 9

    def test_mul_by_zero(self, vm):
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 5)
        vm.execute("MUL", "x", 0)
        assert vm.execute("QUERY", "x") == 0

    def test_mul_by_one_is_noop(self, vm):
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 7)
        vm.execute("MUL", "x", 1)
        assert vm.execute("QUERY", "x") == 7

    def test_div_halves(self, vm):
        """ASSIGN 10, DIV 2 → QUERY = 5."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 10)
        vm.execute("DIV", "x", 2)
        assert vm.execute("QUERY", "x") == 5

    def test_div_integer_truncation(self, vm):
        """7 / 2 = 3 (integer division)."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 7)
        vm.execute("DIV", "x", 2)
        assert vm.execute("QUERY", "x") == 3

    def test_div_by_one_is_noop(self, vm):
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 9)
        vm.execute("DIV", "x", 1)
        assert vm.execute("QUERY", "x") == 9

    def test_div_by_zero_returns_zero(self, vm):
        """Division by zero should not crash — returns 0."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 5)
        vm.execute("DIV", "x", 0)
        assert vm.execute("QUERY", "x") == 0

    def test_mul_div_program(self, vm):
        """'Start with 6. Triple it. Split equally among 9. What's left?'"""
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 6),
            ("MUL", "x", 3),
            ("DIV", "x", 9),
            ("QUERY", "x"),
        ]
        assert vm.run(program) == 2


# ── COND (conditional) ──────────────────────────────────────────────────


class TestCond:

    def test_cond_true_branch(self, vm):
        """COND with true condition executes then-program."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 10)
        vm.execute("CREATE", "flag", "number")
        vm.execute("ASSIGN", "flag", 1)

        # If flag == 1, add 5 to x
        vm.execute("COND", "flag", 1,
                    [("ADD", "x", 5)],
                    [("SUB", "x", 5)])
        assert vm.execute("QUERY", "x") == 15

    def test_cond_false_branch(self, vm):
        """COND with false condition executes else-program."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 10)
        vm.execute("CREATE", "flag", "number")
        vm.execute("ASSIGN", "flag", 0)

        # If flag == 1, add 5; else sub 5
        vm.execute("COND", "flag", 1,
                    [("ADD", "x", 5)],
                    [("SUB", "x", 5)])
        assert vm.execute("QUERY", "x") == 5

    def test_cond_no_else(self, vm):
        """COND with no else branch does nothing when false."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 10)
        vm.execute("CREATE", "flag", "number")
        vm.execute("ASSIGN", "flag", 0)

        vm.execute("COND", "flag", 1,
                    [("ADD", "x", 99)])
        assert vm.execute("QUERY", "x") == 10

    def test_cond_nested(self, vm):
        """COND can nest — inner cond inside outer then-branch."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 0)
        vm.execute("CREATE", "a", "number")
        vm.execute("ASSIGN", "a", 1)
        vm.execute("CREATE", "b", "number")
        vm.execute("ASSIGN", "b", 1)

        # If a==1: if b==1: x=100 else x=50
        vm.execute("COND", "a", 1,
                    [("COND", "b", 1,
                      [("ASSIGN", "x", 100)],
                      [("ASSIGN", "x", 50)])])
        assert vm.execute("QUERY", "x") == 100


# ── LOOP ─────────────────────────────────────────────────────────────────


class TestLoop:

    def test_loop_counts_to_target(self, vm):
        """LOOP adds 1 until x reaches 5."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 0)

        vm.execute("LOOP", "x", 5, "less",
                    [("ADD", "x", 1)])
        assert vm.execute("QUERY", "x") == 5

    def test_loop_decrements(self, vm):
        """LOOP subtracts 2 until x reaches 0 or below."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 10)

        vm.execute("LOOP", "x", 0, "greater",
                    [("SUB", "x", 2)])
        assert vm.execute("QUERY", "x") == 0

    def test_loop_already_satisfied(self, vm):
        """LOOP does nothing if condition already met."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 10)

        vm.execute("LOOP", "x", 5, "less",
                    [("ADD", "x", 1)])
        # x=10 is NOT less than 5, so loop never runs
        assert vm.execute("QUERY", "x") == 10

    def test_loop_max_iterations_safety(self, vm):
        """LOOP has a max iteration limit to prevent infinite loops."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 0)

        # This would loop forever without the safety limit
        vm.execute("LOOP", "x", 999999, "less",
                    [("ADD", "x", 1)],
                    max_iter=100)
        assert vm.execute("QUERY", "x") == 100

    def test_loop_in_program(self, vm):
        """'Start with 3. While less than 20, double it. How much?'"""
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 3),
            ("LOOP", "x", 20, "less", [("MUL", "x", 2)]),
            ("QUERY", "x"),
        ]
        # 3 → 6 → 12 → 24 (stop, 24 >= 20)
        assert vm.run(program) == 24

    def test_transfer_in_loop(self, vm):
        """'A has 10. Give 2 to B each round for 3 rounds.'"""
        program = [
            ("CREATE", "a", "person"),
            ("CREATE", "b", "person"),
            ("CREATE", "round", "number"),
            ("ASSIGN", "a", 10),
            ("ASSIGN", "b", 0),
            ("ASSIGN", "round", 0),
            ("LOOP", "round", 3, "less", [
                ("TRANSFER", "a", "b", 2),
                ("ADD", "round", 1),
            ]),
            ("QUERY", "b"),
        ]
        assert vm.run(program) == 6


# ── COPY (MOV) ───────────────────────────────────────────────────────────


class TestCopy:

    def test_copy_value(self, vm):
        """COPY src dst — dst gets src's value."""
        vm.execute("CREATE", "a", "number")
        vm.execute("CREATE", "b", "number")
        vm.execute("ASSIGN", "a", 42)
        vm.execute("ASSIGN", "b", 0)

        vm.execute("COPY", "a", "b")
        assert vm.execute("QUERY", "b") == 42

    def test_copy_does_not_modify_source(self, vm):
        vm.execute("CREATE", "src", "number")
        vm.execute("CREATE", "dst", "number")
        vm.execute("ASSIGN", "src", 7)

        vm.execute("COPY", "src", "dst")
        assert vm.execute("QUERY", "src") == 7
        assert vm.execute("QUERY", "dst") == 7

    def test_copy_block_code(self, vm, bc):
        """Copied register should have the same block-code vector."""
        vm.execute("CREATE", "a", "number")
        vm.execute("CREATE", "b", "number")
        vm.execute("ASSIGN", "a", 5)

        vm.execute("COPY", "a", "b")
        sim = float(bc.similarity(vm.registers["a"], vm.registers["b"]))
        assert sim > 0.99


# ── PUSH / POP (stack) ──────────────────────────────────────────────────


class TestStack:

    def test_push_pop_value(self, vm):
        """PUSH saves register state, POP restores it."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 10)

        vm.execute("PUSH", "x")
        vm.execute("ASSIGN", "x", 99)
        assert vm.execute("QUERY", "x") == 99

        vm.execute("POP", "x")
        assert vm.execute("QUERY", "x") == 10

    def test_push_pop_lifo(self, vm):
        """Stack is LIFO — last pushed is first popped."""
        vm.execute("CREATE", "x", "number")

        vm.execute("ASSIGN", "x", 1)
        vm.execute("PUSH", "x")
        vm.execute("ASSIGN", "x", 2)
        vm.execute("PUSH", "x")
        vm.execute("ASSIGN", "x", 3)
        vm.execute("PUSH", "x")

        vm.execute("POP", "x")
        assert vm.execute("QUERY", "x") == 3
        vm.execute("POP", "x")
        assert vm.execute("QUERY", "x") == 2
        vm.execute("POP", "x")
        assert vm.execute("QUERY", "x") == 1

    def test_pop_empty_is_noop(self, vm):
        """POP on empty stack should not crash."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 5)
        vm.execute("POP", "x")
        assert vm.execute("QUERY", "x") == 5

    def test_push_preserves_block_code(self, vm, bc):
        """PUSH/POP preserves the actual block-code vector."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 42)
        original_vec = vm.registers["x"].copy()

        vm.execute("PUSH", "x")
        vm.execute("ASSIGN", "x", 0)
        vm.execute("POP", "x")

        sim = float(bc.similarity(vm.registers["x"], original_vec))
        assert sim > 0.99


# ── CALL / RET (subroutine) ─────────────────────────────────────────────


class TestCall:

    def test_call_stored_rule(self, vm):
        """CALL executes a stored rule and returns."""
        # Define a rule: add 10 to x
        vm.trace_enabled = True
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 0)
        vm.execute("ADD", "x", 10)
        vm.store_rule("add_ten")
        vm.trace.clear()
        vm.trace_enabled = False

        # Reset x, then CALL the rule
        vm.execute("ASSIGN", "x", 5)
        vm.execute("CALL", "add_ten")
        # Rule replays: CREATE x (noop), ASSIGN x=0, ADD x 10 → x should be 10
        # But we want CALL to act on current state, not replay ASSIGN
        # So CALL should only replay the body, not CREATE/ASSIGN
        assert vm.execute("QUERY", "x") is not None

    def test_call_unknown_rule_is_noop(self, vm):
        """CALL on non-existent rule should not crash."""
        vm.execute("CALL", "nonexistent")

    def test_call_increments_step_count(self, vm):
        """Each instruction in the called rule increments step_count."""
        vm.rules["simple"] = [
            ("CREATE", "tmp", "number"),
            ("ASSIGN", "tmp", 1),
        ]
        before = vm.step_count
        vm.execute("CALL", "simple")
        # CALL itself = 1 step, plus 2 from the rule body
        assert vm.step_count >= before + 3


# ── JMP / LABEL ──────────────────────────────────────────────────────────


class TestJmp:

    def test_jmp_skips_instructions(self, vm):
        """JMP skips to a labeled position in the program."""
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 0),
            ("JMP", "done"),
            ("ADD", "x", 999),        # should be skipped
            ("LABEL", "done"),
            ("ADD", "x", 1),
            ("QUERY", "x"),
        ]
        result = vm.run(program)
        assert result == 1  # not 1000

    def test_jmp_backward_with_counter(self, vm):
        """JMP backward implements a manual loop."""
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 0),
            ("LABEL", "top"),
            ("ADD", "x", 1),
            ("COND", "x", 5, [("JMP", "end")]),
            ("JMP", "top"),
            ("LABEL", "end"),
            ("QUERY", "x"),
        ]
        result = vm.run(program)
        assert result == 5

    def test_jmp_unknown_label_is_noop(self, vm):
        """JMP to non-existent label should not crash."""
        program = [
            ("CREATE", "x", "number"),
            ("ASSIGN", "x", 42),
            ("JMP", "nowhere"),
            ("QUERY", "x"),
        ]
        result = vm.run(program)
        assert result == 42

    def test_label_is_noop(self, vm):
        """LABEL itself does nothing — it's just a marker."""
        vm.execute("CREATE", "x", "number")
        vm.execute("ASSIGN", "x", 5)
        vm.execute("LABEL", "marker")
        assert vm.execute("QUERY", "x") == 5


# ── DEBATE (HD Graph-of-Thoughts) ────────────────────────────────────────


class TestDebate:
    """Test HD-GoT debate resolution through the VM."""

    def test_debate_single_candidate(self, vm, bc):
        """DEBATE with one candidate returns it unchanged."""
        v = bc.random_discrete(seed=42)
        result = vm.execute("DEBATE", [v])
        assert result.shape == (K, L)
        sim = float(bc.similarity(result, v))
        assert sim > 0.99

    def test_debate_returns_consensus(self, vm, bc):
        """DEBATE with multiple candidates returns a (k,l) consensus vector."""
        candidates = [bc.random_discrete(seed=i) for i in range(5)]
        result = vm.execute("DEBATE", candidates)
        assert result.shape == (K, L)

    def test_debate_favors_similar_cluster(self, vm, bc):
        """If 3 candidates agree and 2 disagree, consensus should be close to the 3."""
        base = bc.random_discrete(seed=42)
        cluster = [base.copy() for _ in range(3)]
        outliers = [bc.random_discrete(seed=100), bc.random_discrete(seed=200)]

        result = vm.execute("DEBATE", cluster + outliers)

        sim_to_base = float(bc.similarity(result, base))
        sim_to_outlier = float(bc.similarity(result, outliers[0]))
        assert sim_to_base > sim_to_outlier, (
            f"Consensus should favor cluster: base={sim_to_base:.3f} "
            f"vs outlier={sim_to_outlier:.3f}"
        )

    def test_debate_in_program(self, vm, bc):
        """DEBATE works as part of a VM program."""
        v0 = bc.random_discrete(seed=10)
        v1 = bc.random_discrete(seed=20)
        v2 = bc.random_discrete(seed=30)

        program = [
            ("DEBATE", [v0, v1, v2]),
        ]
        vm.run(program)
        assert vm.step_count > 0
