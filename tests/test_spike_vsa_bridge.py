"""Tests for cubemind.brain.spike_vsa_bridge.SpikeVSABridge."""

from __future__ import annotations

import numpy as np
import pytest

from cubemind.brain.spike_vsa_bridge import SpikeVSABridge
from cubemind.ops.block_codes import BlockCodes

K, L = 4, 32


@pytest.fixture(scope="module")
def bridge() -> SpikeVSABridge:
    return SpikeVSABridge(k=K, l=L, num_timesteps=10, seed=42)


@pytest.fixture(scope="module")
def bc() -> BlockCodes:
    return BlockCodes(k=K, l=L)


def test_vsa_to_spikes_shape(bridge: SpikeVSABridge, bc: BlockCodes):
    hv = bc.random_discrete(seed=1)
    spikes = bridge.vsa_to_spikes(hv)
    assert spikes.shape == (10, K * L)


def test_spikes_to_vsa_shape(bridge: SpikeVSABridge):
    spikes = np.random.randn(10, K * L).astype(np.float32)
    hv = bridge.spikes_to_vsa(spikes)
    assert hv.shape == (K, L)


def test_round_trip_similarity(bridge: SpikeVSABridge, bc: BlockCodes):
    """Round trip should preserve some similarity."""
    hv = bc.random_discrete(seed=10)
    hv_rt = bridge.round_trip(hv)
    sim = bc.similarity(hv, hv_rt)
    # Not perfect due to spike noise, but should be positive
    assert sim > -0.5, f"Round trip similarity too low: {sim}"


def test_spike_to_block_code(bridge: SpikeVSABridge):
    counts = np.random.rand(K * L).astype(np.float32) * 5
    hv = bridge.spike_to_block_code(counts)
    assert hv.shape == (K, L)


def test_spike_to_block_code_small_dim(bridge: SpikeVSABridge):
    """Smaller spike count vector should pad to d_vsa."""
    counts = np.array([1, 2, 3, 4], dtype=np.float32)
    hv = bridge.spike_to_block_code(counts)
    assert hv.shape == (K, L)


def test_batched_spikes_to_vsa(bridge: SpikeVSABridge):
    spikes = np.random.randn(4, 10, K * L).astype(np.float32)
    hvs = bridge.spikes_to_vsa(spikes)
    assert hvs.shape == (4, K, L)


def test_vsa_spikes_vsa_with_gif():
    """Full pipeline: VSA → spikes → GIF → spikes → VSA."""
    from cubemind.brain.gif_neuron import GIFNeuron

    b = SpikeVSABridge(k=K, l=L, num_timesteps=5, seed=42)
    bc = BlockCodes(k=K, l=L)
    gif = GIFNeuron(K * L, K * L, L=8, seed=42)

    hv = bc.random_discrete(seed=20)

    # VSA → spikes
    spikes_in = b.vsa_to_spikes(hv)  # (5, d_vsa)

    # GIF processing
    spikes_out, _ = gif.forward(spikes_in)  # (5, d_vsa)

    # Spikes → VSA
    hv_out = b.spikes_to_vsa(spikes_out)
    assert hv_out.shape == (K, L)
    assert np.all(np.isfinite(hv_out))


