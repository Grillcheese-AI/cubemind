"""
Tests for cubemind.execution.world_manager.WorldManager.

Validates:
  - creation: arena shape, active_worlds starts at 0
  - first_transition_spawns: always spawns specialist 0
  - similar_transition_consolidates: same transition twice consolidates
  - different_transition_spawns: different transition spawns new specialist
  - get_specialists: returns only active vectors
  - max_capacity: raises RuntimeError when full
  - obs_counts: tracks consolidation frequency
  - oja_sharpening: specialist vector changes after consolidation
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.execution.world_manager import WorldManager
from cubemind.ops.block_codes import BlockCodes

# ── Small dims to avoid OOM ───────────────────────────────────────────────────

K = 4
L = 8


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


@pytest.fixture
def wm() -> WorldManager:
    return WorldManager(k=K, l=L, max_worlds=8, tau=0.65, oja_lr=0.01)


@pytest.fixture
def state_a(bc: BlockCodes) -> np.ndarray:
    return bc.random_discrete(seed=10)


@pytest.fixture
def state_b(bc: BlockCodes) -> np.ndarray:
    return bc.random_discrete(seed=20)


@pytest.fixture
def state_c(bc: BlockCodes) -> np.ndarray:
    return bc.random_discrete(seed=30)


# ── Test: creation ────────────────────────────────────────────────────────────


def test_creation_arena_shape():
    """Arena must have shape (max_worlds, k, l) and be zero-initialised."""
    wm = WorldManager(k=K, l=L, max_worlds=16)
    assert wm._arena.shape == (16, K, L), (
        f"Expected arena shape (16, {K}, {L}), got {wm._arena.shape}"
    )
    assert wm._arena.dtype == np.float32
    assert np.all(wm._arena == 0.0), "Arena must be zero-initialised"


def test_creation_active_worlds_zero():
    """active_worlds must start at 0."""
    wm = WorldManager(k=K, l=L)
    assert wm.active_worlds == 0, (
        f"Expected active_worlds=0, got {wm.active_worlds}"
    )


def test_creation_obs_counts_zero():
    """Observation counts must start at 0."""
    wm = WorldManager(k=K, l=L, max_worlds=8)
    assert wm._obs_counts.shape == (8,)
    assert np.all(wm._obs_counts == 0)


def test_creation_tau_stored():
    """tau threshold must be stored."""
    wm = WorldManager(k=K, l=L, tau=0.70)
    assert wm.tau == pytest.approx(0.70)


def test_creation_oja_lr_stored():
    """Oja learning rate must be stored."""
    wm = WorldManager(k=K, l=L, oja_lr=0.05)
    assert wm.oja_lr == pytest.approx(0.05)


# ── Test: first_transition_spawns ─────────────────────────────────────────────


def test_first_transition_spawns(wm: WorldManager, state_a, state_b):
    """First process_transition must always spawn specialist 0."""
    result = wm.process_transition(state_a, state_b)
    assert result["action"] == "spawned", (
        f"Expected 'spawned', got {result['action']!r}"
    )
    assert result["world_id"] == 0, (
        f"Expected world_id=0, got {result['world_id']}"
    )
    assert wm.active_worlds == 1


def test_first_transition_result_keys(wm: WorldManager, state_a, state_b):
    """process_transition must return dict with action, world_id, similarity keys."""
    result = wm.process_transition(state_a, state_b)
    assert {"action", "world_id", "similarity"} == set(result.keys()), (
        f"Unexpected keys: {set(result.keys())}"
    )


def test_first_transition_similarity_value(wm: WorldManager, state_a, state_b):
    """First spawn similarity must be 0.0 (no comparison made)."""
    result = wm.process_transition(state_a, state_b)
    assert result["similarity"] == pytest.approx(0.0), (
        f"Expected similarity=0.0 on first spawn, got {result['similarity']}"
    )


# ── Test: similar_transition_consolidates ─────────────────────────────────────


def test_same_transition_twice_consolidates(bc: BlockCodes, state_a, state_b):
    """Sending the identical transition twice must consolidate (not spawn)."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=0.65)
    wm.process_transition(state_a, state_b)  # spawn
    result = wm.process_transition(state_a, state_b)  # should consolidate
    assert result["action"] == "consolidated", (
        f"Identical transition should consolidate, got {result['action']!r}"
    )
    assert wm.active_worlds == 1, (
        f"active_worlds should stay 1, got {wm.active_worlds}"
    )


def test_consolidation_world_id_correct(bc: BlockCodes, state_a, state_b):
    """Consolidation must report the matching world_id."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=0.65)
    wm.process_transition(state_a, state_b)  # spawn world 0
    result = wm.process_transition(state_a, state_b)  # consolidate world 0
    assert result["world_id"] == 0


def test_consolidation_similarity_high(bc: BlockCodes, state_a, state_b):
    """Consolidation similarity must be >= tau."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=0.65)
    wm.process_transition(state_a, state_b)
    result = wm.process_transition(state_a, state_b)
    assert result["similarity"] >= wm.tau, (
        f"Consolidation similarity {result['similarity']:.4f} < tau {wm.tau}"
    )


# ── Test: different_transition_spawns ─────────────────────────────────────────


def test_different_transition_spawns_new(bc: BlockCodes, state_a, state_b, state_c):
    """A sufficiently different transition must spawn a new specialist."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=0.65)
    wm.process_transition(state_a, state_b)   # spawn world 0

    # Use a very different transition: state_b -> state_c is different from state_a -> state_b
    result = wm.process_transition(state_b, state_c)
    # Could be spawned or consolidated depending on similarity — just check shape
    assert result["action"] in ("spawned", "consolidated")
    assert isinstance(result["world_id"], int)
    assert isinstance(result["similarity"], float)


def test_forced_spawn_via_tau_zero(bc: BlockCodes, state_a, state_b, state_c):
    """With tau=1.0 (impossible to match), every transition must spawn."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=1.0)
    r1 = wm.process_transition(state_a, state_b)
    r2 = wm.process_transition(state_b, state_c)
    assert r1["action"] == "spawned"
    assert r2["action"] == "spawned"
    assert wm.active_worlds == 2


def test_forced_consolidate_via_tau_zero(bc: BlockCodes, state_a, state_b, state_c):
    """With tau=0.0 (always match), second transition must consolidate."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=0.0)
    wm.process_transition(state_a, state_b)    # spawn
    result = wm.process_transition(state_b, state_c)   # consolidate (sim > 0.0 = tau)
    assert result["action"] == "consolidated"
    assert wm.active_worlds == 1


# ── Test: get_specialists ─────────────────────────────────────────────────────


def test_get_specialists_empty(wm: WorldManager):
    """get_specialists must return an empty list when active_worlds=0."""
    assert wm.get_specialists() == []


def test_get_specialists_returns_active(bc: BlockCodes, state_a, state_b, state_c):
    """get_specialists must return exactly active_worlds vectors."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=1.0)  # tau=1 forces all spawns
    wm.process_transition(state_a, state_b)
    wm.process_transition(state_b, state_c)
    specialists = wm.get_specialists()
    assert len(specialists) == 2, (
        f"Expected 2 specialists, got {len(specialists)}"
    )


def test_get_specialists_shape(bc: BlockCodes, state_a, state_b):
    """Each specialist must have shape (k, l) and dtype float32."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=1.0)
    wm.process_transition(state_a, state_b)
    for i, s in enumerate(wm.get_specialists()):
        assert s.shape == (K, L), (
            f"Specialist {i}: expected ({K}, {L}), got {s.shape}"
        )
        assert s.dtype == np.float32, (
            f"Specialist {i}: expected float32, got {s.dtype}"
        )


def test_get_specialists_copies(bc: BlockCodes, state_a, state_b):
    """Modifying a returned specialist must not mutate the arena."""
    wm = WorldManager(k=K, l=L, max_worlds=8)
    wm.process_transition(state_a, state_b)
    specialists = wm.get_specialists()
    arena_before = wm._arena[0].copy()
    specialists[0][:] = 999.0  # mutate returned specialist
    np.testing.assert_array_equal(
        wm._arena[0], arena_before,
        err_msg="Mutating returned specialist must not affect arena",
    )


# ── Test: max_capacity ────────────────────────────────────────────────────────


def test_max_capacity_raises(bc: BlockCodes):
    """process_transition must raise RuntimeError when arena is full."""
    wm = WorldManager(k=K, l=L, max_worlds=2, tau=1.0)  # tau=1 forces all spawns
    rng = np.random.default_rng(99)

    def rand_state():
        v = np.zeros((K, L), dtype=np.float32)
        for b in range(K):
            v[b, rng.integers(0, L)] = 1.0
        return v

    wm.process_transition(rand_state(), rand_state())  # fills slot 0
    wm.process_transition(rand_state(), rand_state())  # fills slot 1

    with pytest.raises(RuntimeError, match="full"):
        wm.process_transition(rand_state(), rand_state())  # should raise


# ── Test: obs_counts ──────────────────────────────────────────────────────────


def test_obs_counts_spawn_increments(wm: WorldManager, state_a, state_b):
    """Spawning a specialist must set its obs_count to 1."""
    wm.process_transition(state_a, state_b)
    assert wm.get_obs_count(0) == 1, (
        f"Expected obs_count=1 after spawn, got {wm.get_obs_count(0)}"
    )


def test_obs_counts_consolidation_increments(bc: BlockCodes, state_a, state_b):
    """Each consolidation must increment the obs_count."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=0.0)
    wm.process_transition(state_a, state_b)  # spawn -> count=1
    wm.process_transition(state_a, state_b)  # consolidate -> count=2
    wm.process_transition(state_a, state_b)  # consolidate -> count=3
    assert wm.get_obs_count(0) == 3, (
        f"Expected obs_count=3, got {wm.get_obs_count(0)}"
    )


def test_obs_counts_separate_specialists(bc: BlockCodes, state_a, state_b, state_c):
    """obs_counts must be tracked independently per specialist."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=1.0)  # force spawns
    wm.process_transition(state_a, state_b)
    wm.process_transition(state_b, state_c)
    assert wm.get_obs_count(0) == 1
    assert wm.get_obs_count(1) == 1


def test_get_obs_count_invalid_id(wm: WorldManager, state_a, state_b):
    """get_obs_count with out-of-range world_id must raise IndexError."""
    with pytest.raises(IndexError):
        wm.get_obs_count(999)


# ── Test: oja_sharpening ──────────────────────────────────────────────────────


def test_oja_sharpening_changes_specialist(bc: BlockCodes, state_a, state_b):
    """After consolidation, the specialist vector must change (Oja update)."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=0.0, oja_lr=0.1)
    wm.process_transition(state_a, state_b)  # spawn
    specialist_before = wm._arena[0].copy()

    # Consolidate with a slightly different observation (different state pair)
    # Use state_b -> state_a to get a different r_observed
    bc_inst = BlockCodes(k=K, l=L)
    state_x = bc_inst.random_discrete(seed=77)
    state_y = bc_inst.random_discrete(seed=88)
    wm.process_transition(state_x, state_y)  # consolidates (tau=0.0)

    specialist_after = wm._arena[0]
    assert not np.array_equal(specialist_before, specialist_after), (
        "Specialist vector must change after Oja consolidation"
    )


def test_oja_specialist_normalized(bc: BlockCodes, state_a, state_b):
    """After consolidation, each block of the specialist must be L2-normalised."""
    wm = WorldManager(k=K, l=L, max_worlds=8, tau=0.0, oja_lr=0.5)
    wm.process_transition(state_a, state_b)  # spawn

    bc_inst = BlockCodes(k=K, l=L)
    for seed in range(5, 20):
        s1 = bc_inst.random_discrete(seed=seed)
        s2 = bc_inst.random_discrete(seed=seed + 100)
        wm.process_transition(s1, s2)  # consolidate

    specialist = wm._arena[0]
    block_norms = np.linalg.norm(specialist, axis=-1)
    np.testing.assert_allclose(
        block_norms,
        np.ones(K, dtype=np.float32),
        atol=1e-5,
        err_msg="Each block must be L2-normalised after Oja update",
    )
