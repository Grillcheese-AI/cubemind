"""Tests for SNN perceptual layer: LIF neurons, neurochemistry, temporal encoding."""

import numpy as np

from cubemind.perception.snn import (
    IFNeuronLayer,
    LIFNeuronLayer,
    NeurochemicalState,
    SNNEncoder,
    _cyclic_shift_uint32,
)


# ── LIF Neuron tests ─────────────────────────────────────────────────────

class TestLIFNeuronLayer:
    def test_output_shape(self):
        lif = LIFNeuronLayer(n_neurons=64, tau=2.0)
        x = np.ones(64, dtype=np.float32)
        spikes = lif.step(x)
        assert spikes.shape == (64,)
        assert spikes.dtype == np.float32

    def test_spikes_are_binary(self):
        lif = LIFNeuronLayer(n_neurons=32, tau=2.0, v_threshold=0.5)
        x = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        for _ in range(10):
            spikes = lif.step(x)
            assert set(np.unique(spikes)).issubset({0.0, 1.0})

    def test_constant_input_eventually_fires(self):
        lif = LIFNeuronLayer(n_neurons=1, tau=2.0, v_threshold=1.0)
        x = np.array([0.6], dtype=np.float32)
        fired = False
        for _ in range(20):
            spikes = lif.step(x)
            if spikes[0] > 0:
                fired = True
                break
        assert fired, "Constant input should eventually cause a spike"

    def test_reset_clears_membrane(self):
        lif = LIFNeuronLayer(n_neurons=8, tau=2.0, v_threshold=5.0)
        # Sub-threshold input: membrane charges but doesn't spike/reset
        lif.step(np.ones(8, dtype=np.float32))
        assert not np.allclose(lif.v, 0.0)
        lif.reset()
        assert np.allclose(lif.v, 0.0)

    def test_zero_input_no_spikes(self):
        lif = LIFNeuronLayer(n_neurons=16, tau=2.0)
        spikes = lif.step(np.zeros(16, dtype=np.float32))
        assert np.all(spikes == 0.0)

    def test_reset_after_spike(self):
        """After a spike, membrane should reset to v_reset."""
        lif = LIFNeuronLayer(n_neurons=1, tau=2.0, v_threshold=0.5, v_reset=0.0)
        # Strong input to force spike
        lif.step(np.array([2.0], dtype=np.float32))
        assert lif.v[0] == 0.0  # Reset after spike


# ── IF Neuron tests ──────────────────────────────────────────────────────

class TestIFNeuronLayer:
    def test_output_shape(self):
        n = IFNeuronLayer(n_neurons=32)
        spikes = n.step(np.ones(32, dtype=np.float32))
        assert spikes.shape == (32,)

    def test_accumulates_without_leak(self):
        """IF neuron should accumulate charge without any decay."""
        n = IFNeuronLayer(n_neurons=1, v_threshold=3.0)
        # Each step adds 0.5 — should fire after 6 steps (3.0 / 0.5)
        x = np.array([0.5], dtype=np.float32)
        fired_at = None
        for t in range(10):
            spikes = n.step(x)
            if spikes[0] > 0:
                fired_at = t
                break
        assert fired_at == 5, f"IF should fire at step 5, fired at {fired_at}"

    def test_more_sensitive_than_lif(self):
        """IF fires earlier than LIF on weak constant input (no leak eats charge)."""
        x = np.array([0.3], dtype=np.float32)

        if_neuron = IFNeuronLayer(n_neurons=1, v_threshold=1.0)
        lif_neuron = LIFNeuronLayer(n_neurons=1, tau=2.0, v_threshold=1.0)

        if_fired, lif_fired = None, None
        for t in range(50):
            if if_fired is None and if_neuron.step(x)[0] > 0:
                if_fired = t
            if lif_fired is None and lif_neuron.step(x)[0] > 0:
                lif_fired = t
            if if_fired is not None and lif_fired is not None:
                break

        assert if_fired is not None, "IF should fire on weak input"
        # LIF may never fire if leak > input, or fire later
        if lif_fired is not None:
            assert if_fired <= lif_fired, f"IF ({if_fired}) should fire before LIF ({lif_fired})"

    def test_reset(self):
        n = IFNeuronLayer(n_neurons=4, v_threshold=5.0)
        n.step(np.ones(4, dtype=np.float32))
        assert not np.allclose(n.v, 0.0)
        n.reset()
        assert np.allclose(n.v, 0.0)


# ── Neurochemical State tests ─────────────────────────────────────────────

class TestNeurochemicalState:
    def test_initial_resting_state(self):
        nc = NeurochemicalState()
        assert nc.cortisol == 0.2
        assert nc.dopamine == 0.4
        assert nc.serotonin == 0.5
        assert nc.oxytocin == 0.3

    def test_novelty_increases_dopamine(self):
        nc = NeurochemicalState()
        baseline = nc.dopamine
        nc.update(novelty=1.0, valence=0.5)
        assert nc.dopamine > baseline

    def test_threat_increases_cortisol(self):
        nc = NeurochemicalState()
        baseline = nc.cortisol
        nc.update(threat=1.0)
        assert nc.cortisol > baseline

    def test_focus_increases_oxytocin(self):
        nc = NeurochemicalState()
        baseline = nc.oxytocin
        nc.update(focus=1.0, valence=0.5)
        assert nc.oxytocin > baseline

    def test_stress_dampens_dopamine(self):
        """High cortisol should suppress dopamine (non-linear coupling)."""
        nc = NeurochemicalState()
        # Spike cortisol via repeated threats
        for _ in range(20):
            nc.update(threat=1.0)
        # Cortisol should be high, dopamine dampened
        assert nc.cortisol > 0.5
        assert nc.dopamine < 0.4  # Below resting level

    def test_modulate_threshold(self):
        nc = NeurochemicalState()
        base = 1.0
        # High dopamine → lower threshold
        nc.dopamine = 0.9
        nc.cortisol = 0.2
        assert nc.modulate_threshold(base) < base
        # High cortisol → higher threshold (defensive)
        nc.dopamine = 0.4
        nc.cortisol = 0.8
        assert nc.modulate_threshold(base) > base

    def test_modulate_tau(self):
        nc = NeurochemicalState()
        base_tau = 2.0
        # High serotonin → higher tau (more stable)
        nc.serotonin = 0.9
        nc.cortisol = 0.2
        assert nc.modulate_tau(base_tau) > base_tau
        # High cortisol → lower tau (faster response)
        nc.serotonin = 0.5
        nc.cortisol = 0.8
        assert nc.modulate_tau(base_tau) < base_tau

    def test_clamped_to_unit(self):
        nc = NeurochemicalState()
        for _ in range(100):
            nc.update(novelty=1.0, threat=1.0, focus=1.0, valence=1.0)
        assert 0.0 <= nc.cortisol <= 1.0
        assert 0.0 <= nc.dopamine <= 1.0
        assert 0.0 <= nc.serotonin <= 1.0
        assert 0.0 <= nc.oxytocin <= 1.0

    def test_dominant_emotion(self):
        nc = NeurochemicalState()
        nc.update(threat=1.0)
        nc.update(threat=1.0)
        nc.update(threat=1.0)
        # Repeated threat should push toward anxious
        assert nc.dominant_emotion in ("anxious", "neutral")

    def test_joy_from_positive_valence(self):
        nc = NeurochemicalState()
        for _ in range(10):
            nc.update(valence=0.8, novelty=0.3)
        assert nc.dominant_emotion in ("joy", "warm", "curious")

    def test_to_dict(self):
        nc = NeurochemicalState()
        d = nc.to_dict()
        assert "cortisol" in d
        assert "dopamine" in d
        assert "emotion" in d

    def test_arousal_in_range(self):
        nc = NeurochemicalState()
        assert 0.0 <= nc.arousal <= 1.0


# ── Cyclic shift tests ───────────────────────────────────────────────────

class TestCyclicShift:
    def test_shift_zero_is_identity(self):
        packed = np.array([0xDEADBEEF, 0xCAFEBABE], dtype=np.uint32)
        shifted = _cyclic_shift_uint32(packed, shift=0)
        assert np.array_equal(shifted, packed)

    def test_full_rotation_is_identity(self):
        packed = np.array([0xDEADBEEF, 0xCAFEBABE], dtype=np.uint32)
        shifted = _cyclic_shift_uint32(packed, shift=64)  # Full rotation
        assert np.array_equal(shifted, packed)

    def test_shift_changes_value(self):
        packed = np.array([0x00000001], dtype=np.uint32)
        shifted = _cyclic_shift_uint32(packed, shift=1)
        assert not np.array_equal(shifted, packed)

    def test_does_not_modify_input(self):
        packed = np.array([0xDEADBEEF], dtype=np.uint32)
        original = packed.copy()
        _cyclic_shift_uint32(packed, shift=5)
        assert np.array_equal(packed, original)


# ── SNN Encoder tests ────────────────────────────────────────────────────

class TestSNNEncoder:
    def test_step_output_shape(self):
        enc = SNNEncoder(d_input=32, n_neurons=64, d_vsa=256)
        x = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        spikes = enc.step(x)
        assert spikes.shape == (64,)

    def test_encode_stream_shape(self):
        enc = SNNEncoder(d_input=32, n_neurons=64, d_vsa=256)
        stream = np.random.default_rng(42).standard_normal((10, 32)).astype(np.float32)
        packed = enc.encode_stream(stream)
        expected_words = int(np.ceil(256 / 32))
        assert packed.shape == (expected_words,)
        assert packed.dtype == np.uint32

    def test_different_streams_different_encodings(self):
        enc = SNNEncoder(d_input=32, n_neurons=64, d_vsa=256)
        rng = np.random.default_rng(42)
        s1 = rng.standard_normal((20, 32)).astype(np.float32)
        s2 = rng.standard_normal((20, 32)).astype(np.float32)
        v1 = enc.encode_stream(s1)
        v2 = enc.encode_stream(s2)
        assert not np.array_equal(v1, v2)

    def test_encode_stream_with_spikes_returns_history(self):
        enc = SNNEncoder(d_input=32, n_neurons=64, d_vsa=256)
        stream = np.random.default_rng(42).standard_normal((5, 32)).astype(np.float32)
        packed, spike_hist, arousal_hist = enc.encode_stream_with_spikes(stream)
        assert len(spike_hist) == 5
        assert len(arousal_hist) == 5
        assert spike_hist[0].shape == (64,)
        assert 0.0 <= arousal_hist[-1] <= 1.0

    def test_neurochemistry_updates_during_stream(self):
        enc = SNNEncoder(d_input=32, n_neurons=64, d_vsa=256)
        initial_dopamine = enc.neurochemistry.dopamine
        stream = np.random.default_rng(42).standard_normal((30, 32)).astype(np.float32) * 5
        enc.encode_stream(stream)
        # After strong input, neurochemistry should have shifted
        assert enc.neurochemistry.dopamine != initial_dopamine

    def test_reset_restores_state(self):
        enc = SNNEncoder(d_input=16, n_neurons=32, d_vsa=128)
        stream = np.ones((5, 16), dtype=np.float32)
        enc.encode_stream(stream)
        enc.reset()
        # After reset, same stream should produce same encoding
        v1 = enc.encode_stream(stream)
        enc.reset()
        v2 = enc.encode_stream(stream)
        assert np.array_equal(v1, v2)

    def test_if_neuron_type(self):
        """SNNEncoder with neuron_type='if' should work and be more sensitive."""
        enc_if = SNNEncoder(d_input=16, n_neurons=32, d_vsa=128, neuron_type="if")
        enc_lif = SNNEncoder(d_input=16, n_neurons=32, d_vsa=128, neuron_type="lif")

        # Weak input — IF should produce more spikes than LIF
        stream = np.ones((10, 16), dtype=np.float32) * 0.2
        enc_if.reset()
        enc_lif.reset()

        if_spikes = sum(float(np.sum(enc_if.step(stream[t]))) for t in range(10))
        enc_lif.reset()
        lif_spikes = sum(float(np.sum(enc_lif.step(stream[t]))) for t in range(10))

        assert if_spikes >= lif_spikes, (
            f"IF ({if_spikes}) should fire at least as much as LIF ({lif_spikes}) on weak input"
        )

    def test_stdp_potentiates_active_weights(self):
        """STDP should increase weights for inputs that contributed to spikes."""
        enc = SNNEncoder(d_input=8, n_neurons=4, d_vsa=64, v_threshold=0.5)
        enc.stdp_enabled = True

        # Record initial weights
        w_before = enc.W_in.copy()

        # Strong input on channels 0-3, zero on 4-7
        x = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Run several steps to trigger spikes + STDP
        for _ in range(10):
            enc.step(x)

        w_after = enc.W_in.copy()

        # Weights for active inputs (cols 0-3) should have changed
        active_delta = np.abs(w_after[:, :4] - w_before[:, :4]).sum()
        inactive_delta = np.abs(w_after[:, 4:] - w_before[:, 4:]).sum()

        assert active_delta > inactive_delta, (
            f"Active inputs should see more weight change: active={active_delta:.4f}, "
            f"inactive={inactive_delta:.4f}"
        )

    def test_stdp_disabled_no_weight_change(self):
        """With STDP disabled, weights should not change."""
        enc = SNNEncoder(d_input=8, n_neurons=4, d_vsa=64, v_threshold=0.5)
        enc.stdp_enabled = False

        w_before = enc.W_in.copy()
        x = np.ones(8, dtype=np.float32)
        for _ in range(10):
            enc.step(x)

        assert np.allclose(enc.W_in, w_before), "Weights should not change with STDP disabled"

    def test_stdp_weights_clipped(self):
        """STDP weights should stay within clip bounds."""
        enc = SNNEncoder(d_input=4, n_neurons=4, d_vsa=32, v_threshold=0.3)
        enc.stdp_enabled = True
        enc.stdp_weight_clip = 2.0

        x = np.ones(4, dtype=np.float32) * 5.0
        for _ in range(100):
            enc.step(x)

        assert np.all(enc.W_in <= 2.0), f"Max weight: {enc.W_in.max()}"
        assert np.all(enc.W_in >= -2.0), f"Min weight: {enc.W_in.min()}"

    def test_static_input_produces_sparse_spikes(self):
        """Constant input should produce spikes only at the beginning, then go quiet."""
        enc = SNNEncoder(d_input=16, n_neurons=32, d_vsa=128, tau=2.0)
        # Identical frames = no change = SNN should quiet down
        frame = np.ones(16, dtype=np.float32) * 0.3
        enc.reset()
        spike_rates = []
        for _ in range(20):
            spikes = enc.step(frame)
            spike_rates.append(float(np.mean(spikes)))
        # Later timesteps should have similar or lower spike rate
        # (constant input with leak means membrane stabilizes)
        assert spike_rates[-1] <= spike_rates[0] + 0.1
