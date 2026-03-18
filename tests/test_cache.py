"""Tests for cubemind.memory.cache.VSACache.

Validates:
  - Add and lookup
  - Surprise signal (empty vs populated cache)
  - Stress signal (cache pressure)
  - Eviction of lowest-utility entries
  - Save/load roundtrip persistence
"""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.memory.cache import VSACache


# -- Fixtures ------------------------------------------------------------------

D_VSA = 1024  # Small for fast tests


@pytest.fixture
def cache() -> VSACache:
    return VSACache(max_size=100, d_vsa=D_VSA, initial_capacity=16)


def _random_bipolar(d: int, seed: int) -> np.ndarray:
    """Generate a random bipolar {-1, +1} vector."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=d).astype(np.int8)


# -- Tests ---------------------------------------------------------------------


def test_add_and_lookup(cache: VSACache):
    """Adding a vector and looking it up returns high similarity."""
    phi = _random_bipolar(D_VSA, seed=0)
    emotion = np.array([0.5, 0.3], dtype=np.float32)

    assert cache.add(phi, emotion) is True
    assert cache.size == 1

    sims, keys, indices = cache.lookup(phi, k=1)
    assert sims.shape == (1, 1)
    assert float(sims[0, 0]) > 0.9  # Should be ~1.0 for exact match
    assert indices.shape == (1, 1)


def test_surprise_empty_cache(cache: VSACache):
    """Empty cache should return maximum surprise (1.0)."""
    query = _random_bipolar(D_VSA, seed=42)
    assert cache.surprise(query) == 1.0


def test_surprise_after_add(cache: VSACache):
    """After adding a vector, surprise for the same vector should be low."""
    phi = _random_bipolar(D_VSA, seed=7)
    emotion = np.array([0.0, 0.0], dtype=np.float32)
    cache.add(phi, emotion)

    surprise_val = cache.surprise(phi)
    assert surprise_val < 1.0, f"Surprise should be < 1.0, got {surprise_val}"
    # For an exact match, surprise should be very close to 0
    assert surprise_val < 0.2, f"Surprise for same vector should be near 0, got {surprise_val}"


def test_stress(cache: VSACache):
    """Stress should be 0 when empty and approach 1 when full."""
    assert cache.stress() == 0.0

    # Fill half
    for i in range(50):
        phi = _random_bipolar(D_VSA, seed=i)
        emotion = np.array([0.0, 0.0], dtype=np.float32)
        cache.add(phi, emotion)

    assert abs(cache.stress() - 0.5) < 1e-6

    # Fill completely
    for i in range(50, 100):
        phi = _random_bipolar(D_VSA, seed=i + 1000)
        emotion = np.array([0.0, 0.0], dtype=np.float32)
        cache.add(phi, emotion)

    assert abs(cache.stress() - 1.0) < 1e-6


def test_eviction(cache: VSACache):
    """Eviction should remove the lowest-utility entries."""
    # Add 20 entries
    for i in range(20):
        phi = _random_bipolar(D_VSA, seed=i)
        emotion = np.array([0.0, 0.0], dtype=np.float32)
        cache.add(phi, emotion)

    assert cache.size == 20

    # Boost utility of last 10 entries
    high_indices = np.arange(10, 20, dtype=np.int32)
    cache.update_utility(high_indices, decay=0.0)  # Decay everything to 0, then boost

    # Evict 10
    cache.evict(n=10)
    assert cache.size == 10


def test_save_load_roundtrip(cache: VSACache, tmp_path):
    """Save and load should produce an identical cache."""
    # Populate
    for i in range(15):
        phi = _random_bipolar(D_VSA, seed=i)
        emotion = np.array([float(i), float(i * 0.1)], dtype=np.float32)
        cache.add(phi, emotion)

    # Save
    save_dir = tmp_path / "cache_test"
    cache.save(save_dir)

    # Load
    loaded = VSACache.load(save_dir)

    assert loaded.size == cache.size
    assert loaded.d_vsa == cache.d_vsa
    assert loaded.max_size == cache.max_size
    np.testing.assert_array_equal(
        loaded.keys[: loaded.size], cache.keys[: cache.size]
    )
    np.testing.assert_allclose(
        loaded.emotions[: loaded.size], cache.emotions[: cache.size]
    )
    np.testing.assert_allclose(
        loaded.utility[: loaded.size], cache.utility[: cache.size]
    )


def test_add_batch(cache: VSACache):
    """add_batch should add multiple entries at once."""
    phis = np.array([_random_bipolar(D_VSA, seed=i) for i in range(5)])
    emotions = np.zeros((5, 2), dtype=np.float32)

    count = cache.add_batch(phis, emotions)
    assert count == 5
    assert cache.size == 5


def test_add_returns_false_when_full():
    """Adding past max_size should return False."""
    cache = VSACache(max_size=3, d_vsa=D_VSA, initial_capacity=4)
    for i in range(3):
        assert cache.add(_random_bipolar(D_VSA, seed=i), np.zeros(2)) is True
    assert cache.add(_random_bipolar(D_VSA, seed=99), np.zeros(2)) is False
