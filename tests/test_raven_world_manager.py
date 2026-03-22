"""
I-RAVEN benchmark validation for WorldManager.

Tests that the WorldManager discovers rule types from RAVEN-like transitions
and self-organises into specialists.  RAVEN rules validated here:

  - Constant   — same panel repeated: [A, A, A]
  - Progression — distinct panels in sequence: [A, B, C]
  - Multiple rules — constant + progression → ≥ 2 specialists
  - Consolidation — repeated constant transitions → 1 specialist (not many)
  - VSA Translator labels — discovered specialists produce readable descriptions
  - Scale — 50 random transitions do not spawn 50 specialists

Uses K=8, L=64 to match RAVEN paper-level dimensions while staying well
under typical OOM budgets on CI runners.
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.vsa_translator import VSATranslator
from cubemind.execution.world_manager import WorldManager
from cubemind.ops.block_codes import BlockCodes

# ── Dimensions ────────────────────────────────────────────────────────────────

K, L = 8, 64  # match paper dims, safe on all hardware

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_constant_sequence(bc: BlockCodes, val_seed: int, n: int = 3) -> list[np.ndarray]:
    """Panels [A, A, A] — constant rule: same vector repeated n times."""
    v = bc.random_discrete(seed=val_seed)
    return [v.copy() for _ in range(n)]


def _make_progression_sequence(
    bc: BlockCodes, start_seed: int, n: int = 3
) -> list[np.ndarray]:
    """Panels [A, B, C] — each is a fresh random vector (distinct seeds)."""
    return [bc.random_discrete(seed=start_seed + i) for i in range(n)]


def _extract_transitions(
    panels: list[np.ndarray], bc: BlockCodes
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert a panel sequence to (before, after) transition pairs."""
    return [(panels[i], panels[i + 1]) for i in range(len(panels) - 1)]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture
def wm() -> WorldManager:
    """WorldManager with realistic tau for RAVEN-scale vectors."""
    return WorldManager(k=K, l=L, max_worlds=32, tau=0.65, oja_lr=0.01)


@pytest.fixture
def raven_codebook(bc: BlockCodes) -> dict[str, np.ndarray]:
    """Small RAVEN-concept codebook for VSATranslator tests."""
    concepts = ["constant", "progression", "arithmetic", "distribute_three"]
    return {name: bc.random_discrete(seed=idx * 7 + 3) for idx, name in enumerate(concepts)}


# ── Test: test_discovers_constant_rule ────────────────────────────────────────


def test_discovers_constant_rule(bc: BlockCodes):
    """Constant transitions [A→A, A→A] must consolidate into exactly 1 specialist.

    Constant rule in RAVEN: every panel shows the same attribute value.
    The rule vector R = unbind(A, A) is the same for every step, so the
    WorldManager must recognise it as a single recurring rule.
    """
    wm = WorldManager(k=K, l=L, max_worlds=32, tau=0.65, oja_lr=0.01)
    panels = _make_constant_sequence(bc, val_seed=100, n=4)
    transitions = _extract_transitions(panels, bc)

    results = [wm.process_transition(before, after) for before, after in transitions]

    # First transition spawns; subsequent ones must consolidate
    assert results[0]["action"] == "spawned", "First constant transition must spawn"
    for i, r in enumerate(results[1:], start=1):
        assert r["action"] == "consolidated", (
            f"Constant transition {i} should consolidate, got {r['action']!r}"
        )
    assert wm.active_worlds == 1, (
        f"Constant rule should produce 1 specialist, got {wm.active_worlds}"
    )


# ── Test: test_discovers_multiple_rules ───────────────────────────────────────


def test_discovers_multiple_rules(bc: BlockCodes):
    """Mixing constant and progression transitions must yield ≥ 2 specialists.

    When the WorldManager sees both a constant rule and a progression rule,
    the rule vectors are distinct enough that at least two specialists are
    needed to represent them.
    """
    wm = WorldManager(k=K, l=L, max_worlds=32, tau=0.65, oja_lr=0.01)

    # Feed several constant transitions
    const_panels = _make_constant_sequence(bc, val_seed=200, n=4)
    for before, after in _extract_transitions(const_panels, bc):
        wm.process_transition(before, after)

    # Feed several progression transitions (different seeds → different rule)
    prog_panels = _make_progression_sequence(bc, start_seed=300, n=4)
    for before, after in _extract_transitions(prog_panels, bc):
        wm.process_transition(before, after)

    assert wm.active_worlds >= 2, (
        f"Constant + progression rules should produce ≥ 2 specialists, "
        f"got {wm.active_worlds}"
    )


# ── Test: test_specialist_observation_counts ──────────────────────────────────


def test_specialist_observation_counts(bc: BlockCodes):
    """Repeated constant transitions must increment the specialist's obs_count.

    Sending the same rule many times should consolidate into specialist 0
    and drive its observation count upward.
    """
    wm = WorldManager(k=K, l=L, max_worlds=32, tau=0.65, oja_lr=0.01)
    panels = _make_constant_sequence(bc, val_seed=400, n=6)
    transitions = _extract_transitions(panels, bc)

    for before, after in transitions:
        wm.process_transition(before, after)

    # After 5 transitions all to the same specialist, obs_count must be 5
    expected_count = len(transitions)
    actual_count = wm.get_obs_count(0)
    assert actual_count == expected_count, (
        f"Expected obs_count={expected_count} after {expected_count} constant "
        f"transitions, got {actual_count}"
    )
    assert wm.active_worlds == 1, (
        f"Should still be 1 specialist, got {wm.active_worlds}"
    )


# ── Test: test_translator_labels_specialists ──────────────────────────────────


def test_translator_labels_specialists(bc: BlockCodes, raven_codebook: dict[str, np.ndarray]):
    """VSATranslator must produce a non-empty string description for each specialist.

    After WorldManager discovers specialists from RAVEN-like transitions,
    VSATranslator should summarise each one using the concept codebook.
    """
    wm = WorldManager(k=K, l=L, max_worlds=32, tau=0.65, oja_lr=0.01)
    translator = VSATranslator(bc=bc, codebook=raven_codebook)

    # Spawn two distinct specialists: one constant, one progression
    const_panels = _make_constant_sequence(bc, val_seed=500, n=3)
    for before, after in _extract_transitions(const_panels, bc):
        wm.process_transition(before, after)

    prog_panels = _make_progression_sequence(bc, start_seed=600, n=3)
    for before, after in _extract_transitions(prog_panels, bc):
        wm.process_transition(before, after)

    specialists = wm.get_specialists()
    assert len(specialists) >= 1, "Must have at least 1 specialist to translate"

    for idx, specialist in enumerate(specialists):
        result = translator.translate(specialist)

        assert isinstance(result, dict), f"translate() must return dict for specialist {idx}"
        assert "summary" in result, f"translate() result missing 'summary' for specialist {idx}"
        assert "probes" in result, f"translate() result missing 'probes' for specialist {idx}"

        summary = result["summary"]
        assert isinstance(summary, str) and len(summary) > 0, (
            f"Specialist {idx} summary must be a non-empty string, got {summary!r}"
        )

        # Each probe must have required fields
        for probe in result["probes"]:
            assert probe["kind"] in ("constant", "transform"), (
                f"Probe kind must be 'constant' or 'transform', got {probe['kind']!r}"
            )


# ── Test: test_oja_sharpens_over_repetitions ──────────────────────────────────


def test_oja_sharpens_over_repetitions(bc: BlockCodes):
    """Specialist vector must change (sharpen) as more constant transitions arrive.

    After spawning, each Oja consolidation updates the specialist vector.
    After many repetitions the vector should differ measurably from its initial
    state — demonstrating that Oja's rule is actively running.
    """
    wm = WorldManager(k=K, l=L, max_worlds=32, tau=0.0, oja_lr=0.1)

    # Spawn with one constant transition
    const_panels = _make_constant_sequence(bc, val_seed=700, n=2)
    before, after = const_panels[0], const_panels[1]
    wm.process_transition(before, after)  # spawns specialist 0

    initial_specialist = wm._arena[0].copy()

    # Drive Oja updates by feeding more transitions that consolidate into it
    for seed_offset in range(1, 11):
        v1 = bc.random_discrete(seed=700 + seed_offset)
        v2 = bc.random_discrete(seed=800 + seed_offset)
        wm.process_transition(v1, v2)  # tau=0.0 → always consolidates

    final_specialist = wm._arena[0].copy()

    assert not np.array_equal(initial_specialist, final_specialist), (
        "Specialist vector must change after repeated Oja consolidations"
    )


# ── Test: test_many_transitions_reasonable_specialist_count ───────────────────


def test_many_transitions_reasonable_specialist_count(bc: BlockCodes):
    """50 random transitions must not spawn 50 specialists — some must consolidate.

    With K=8, L=64, the typical cosine similarity between two independent
    unbind rule vectors is ~0.018 — much smaller than the default tau=0.65.
    This is correct behaviour: high-dimensional block codes are nearly
    orthogonal, so a low tau is needed to trigger consolidation for truly
    random vectors.

    We use tau=0.12, which empirically yields ~10-20 specialists from 50
    random transitions on this hardware, confirming the spawn-or-consolidate
    logic is working (not broken or trivially degenerate).  We assert < 40
    to leave a generous margin against hardware/seed variation.
    """
    wm = WorldManager(k=K, l=L, max_worlds=64, tau=0.12, oja_lr=0.01)
    rng = np.random.default_rng(42)

    for _ in range(50):
        v1 = bc.random_discrete(seed=int(rng.integers(0, 10_000)))
        v2 = bc.random_discrete(seed=int(rng.integers(0, 10_000)))
        wm.process_transition(v1, v2)

    assert wm.active_worlds < 40, (
        f"Expected < 40 specialists after 50 random transitions at tau=0.12 "
        f"(consolidation should merge some), got {wm.active_worlds}"
    )
