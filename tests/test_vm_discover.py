"""Tests for VSA-VM rule discovery — the brain of the system.

DISCOVER takes (input, output) example pairs and induces the rule
that transforms input → output, WITHOUT being told which opcodes to use.

This is the HDR (Hypervector Discover Rule) algorithm:
1. Unbind output from input → get the "transformation vector"
2. Match against known operation signatures → identify the opcode
3. If new, store as a discovered rule for future reuse
4. Build a program from the discovered sequence of operations

The key test: give it examples, it figures out what happened,
and can replay it on new data.
"""

from __future__ import annotations

import pytest

from cubemind.ops.block_codes import BlockCodes

K, L = 8, 64


@pytest.fixture
def bc():
    return BlockCodes(k=K, l=L)


@pytest.fixture
def vm(bc):
    from cubemind.reasoning.vm import VSAVM
    return VSAVM(bc=bc, seed=42)


# ── DISCOVER: single operation induction ─────────────────────────────────


class TestDiscoverSingleOp:
    """Discover what operation was applied from a single (input, output) pair."""

    def test_discover_constant(self, vm, bc):
        """input == output → discovered rule is 'constant' (no change)."""
        v = bc.random_discrete(seed=42)
        result = vm.execute("DISCOVER", v, v)
        assert result["rule_type"] == "constant"

    def test_discover_binding(self, vm, bc):
        """output = bind(input, delta) → discovers the delta vector."""
        inp = bc.random_discrete(seed=10)
        delta = bc.random_discrete(seed=50)
        out = bc.bind(inp, delta)

        result = vm.execute("DISCOVER", inp, out)
        assert result["rule_type"] == "bind"
        # The discovered delta should be similar to the actual delta
        sim = float(bc.similarity(result["delta"], delta))
        assert sim > 0.8, f"Should discover delta: sim={sim:.3f}"

    def test_discovered_delta_is_reusable(self, vm, bc):
        """The discovered delta can be applied to a new input to predict output."""
        inp = bc.random_discrete(seed=10)
        delta = bc.random_discrete(seed=50)
        out = bc.bind(inp, delta)

        result = vm.execute("DISCOVER", inp, out)

        # Apply discovered delta to a DIFFERENT input
        new_inp = bc.random_discrete(seed=99)
        new_out_expected = bc.bind(new_inp, delta)
        new_out_predicted = bc.bind(new_inp, result["delta"])

        sim = float(bc.similarity(new_out_predicted, new_out_expected))
        assert sim > 0.8, f"Delta should transfer: sim={sim:.3f}"


# ── DISCOVER: sequence induction ─────────────────────────────────────────


class TestDiscoverSequence:
    """Discover rules from a sequence of (input, output) example pairs."""

    def test_discover_consistent_rule(self, vm, bc):
        """All pairs share the same transformation → one rule."""
        delta = bc.random_discrete(seed=50)
        pairs = []
        for i in range(5):
            inp = bc.random_discrete(seed=i + 100)
            out = bc.bind(inp, delta)
            pairs.append((inp, out))

        result = vm.execute("DISCOVER_SEQUENCE", pairs)
        assert result["n_rules"] == 1
        assert result["rules"][0]["rule_type"] == "bind"

        # The discovered delta should match the original
        sim = float(bc.similarity(result["rules"][0]["delta"], delta))
        assert sim > 0.8

    def test_discover_mixed_rules(self, vm, bc):
        """Pairs with two different transformations → two rules."""
        delta_a = bc.random_discrete(seed=50)
        delta_b = bc.random_discrete(seed=60)

        pairs = []
        # 3 pairs with delta_a
        for i in range(3):
            inp = bc.random_discrete(seed=i + 100)
            pairs.append((inp, bc.bind(inp, delta_a)))
        # 2 pairs with delta_b
        for i in range(2):
            inp = bc.random_discrete(seed=i + 200)
            pairs.append((inp, bc.bind(inp, delta_b)))

        result = vm.execute("DISCOVER_SEQUENCE", pairs)
        assert result["n_rules"] >= 2

    def test_discover_constant_sequence(self, vm, bc):
        """All outputs same as inputs → constant rule."""
        pairs = [(bc.random_discrete(seed=i), bc.random_discrete(seed=i)) for i in range(5)]
        result = vm.execute("DISCOVER_SEQUENCE", pairs)
        assert result["n_rules"] == 1
        assert result["rules"][0]["rule_type"] == "constant"


# ── DISCOVER: RAVEN-style matrix ─────────────────────────────────────────


class TestDiscoverRavenStyle:
    """Discover the rule governing a RAVEN-style matrix (3 rows of 3 panels).

    Each row applies the same rule. The VM must:
    1. Discover the rule from rows 1 and 2
    2. Apply it to row 3 to predict the missing panel
    3. Match against candidates
    """

    def test_discover_progression_from_rows(self, vm, bc):
        """Two rows with same delta → discover the rule."""
        delta = bc.random_discrete(seed=42)

        # Row 1: a, a⊗δ, a⊗δ⊗δ
        a = bc.random_discrete(seed=10)
        row1 = [a, bc.bind(a, delta), bc.bind(bc.bind(a, delta), delta)]

        # Row 2: b, b⊗δ, b⊗δ⊗δ
        b = bc.random_discrete(seed=20)
        row2 = [b, bc.bind(b, delta), bc.bind(bc.bind(b, delta), delta)]

        # Discover rule from rows
        pairs = [
            (row1[0], row1[1]),  # a → a⊗δ
            (row1[1], row1[2]),  # a⊗δ → a⊗δ⊗δ
            (row2[0], row2[1]),  # b → b⊗δ
            (row2[1], row2[2]),  # b⊗δ → b⊗δ⊗δ
        ]
        result = vm.execute("DISCOVER_SEQUENCE", pairs)
        assert result["n_rules"] >= 1
        assert result["rules"][0]["rule_type"] == "bind"

    def test_full_raven_solve_no_hardcoding(self, vm, bc):
        """Complete RAVEN solve: discover rule → predict → match.

        NO hardcoded opcodes — the VM discovers the transformation rule
        from example rows and applies it to solve row 3.
        """
        delta = bc.random_discrete(seed=42)

        # Row 1
        a = bc.random_discrete(seed=10)
        row1 = [a, bc.bind(a, delta), bc.bind(bc.bind(a, delta), delta)]

        # Row 2
        b = bc.random_discrete(seed=20)
        row2 = [b, bc.bind(b, delta), bc.bind(bc.bind(b, delta), delta)]

        # Row 3 (incomplete — panel 3 is the answer)
        c = bc.random_discrete(seed=30)
        c1 = bc.bind(c, delta)
        c2_correct = bc.bind(c1, delta)  # the answer

        # Candidates
        candidates = [
            bc.random_discrete(seed=200),
            bc.random_discrete(seed=300),
            c2_correct.copy(),  # correct at index 2
            bc.random_discrete(seed=400),
        ]

        # Step 1: DISCOVER the rule from rows 1 and 2
        pairs = [
            (row1[0], row1[1]),
            (row1[1], row1[2]),
            (row2[0], row2[1]),
            (row2[1], row2[2]),
        ]
        discovery = vm.execute("DISCOVER_SEQUENCE", pairs)
        discovered_delta = discovery["rules"][0]["delta"]

        # Step 2: APPLY the discovered rule to row 3
        predicted = bc.bind(c1, discovered_delta)

        # Step 3: MATCH against candidates
        best_idx = vm.execute("MATCH", predicted, candidates)

        assert best_idx == 2, f"Should pick correct answer (idx 2), got {best_idx}"


# ── DISCOVER → STORE → RECALL → APPLY (self-programming) ────────────────


class TestSelfProgramming:
    """The VM discovers a rule, stores it, and reuses it on new data."""

    def test_discover_store_recall_apply(self, vm, bc):
        """Full self-programming cycle:
        1. See examples → DISCOVER rule
        2. STORE the rule
        3. See new input → RECALL matching rule
        4. APPLY the rule to predict output
        """
        delta = bc.random_discrete(seed=42)

        # Phase 1: Show examples, discover rule
        pairs = []
        for i in range(3):
            inp = bc.random_discrete(seed=i + 100)
            out = bc.bind(inp, delta)
            pairs.append((inp, out))

        discovery = vm.execute("DISCOVER_SEQUENCE", pairs)
        rule = discovery["rules"][0]

        # Phase 2: Store the discovered rule
        vm.execute("CREATE", "rule_delta", "rule")
        vm.registers["rule_delta"] = rule["delta"]
        vm.execute("REMEMBER", "rule_delta")

        # Phase 3: New input — apply the stored rule
        new_inp = bc.random_discrete(seed=999)
        expected_out = bc.bind(new_inp, delta)

        # Apply discovered delta
        predicted_out = bc.bind(new_inp, rule["delta"])

        sim = float(bc.similarity(predicted_out, expected_out))
        assert sim > 0.8, f"Self-programmed prediction should work: sim={sim:.3f}"

    def test_discover_two_problems_different_rules(self, vm, bc):
        """Two different problem types → two different stored rules."""
        delta_add = bc.random_discrete(seed=42)
        delta_rotate = bc.random_discrete(seed=99)

        # Problem type 1: "addition"
        pairs_add = [(bc.random_discrete(seed=i), bc.bind(bc.random_discrete(seed=i), delta_add))
                     for i in range(100, 103)]
        disc1 = vm.execute("DISCOVER_SEQUENCE", pairs_add)

        # Problem type 2: "rotation"
        pairs_rot = [(bc.random_discrete(seed=i), bc.bind(bc.random_discrete(seed=i), delta_rotate))
                     for i in range(200, 203)]
        disc2 = vm.execute("DISCOVER_SEQUENCE", pairs_rot)

        # The two discovered rules should be different
        sim = float(bc.similarity(disc1["rules"][0]["delta"], disc2["rules"][0]["delta"]))
        assert sim < 0.5, f"Different problems should produce different rules: sim={sim:.3f}"
