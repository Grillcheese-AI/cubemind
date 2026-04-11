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
