"""Test 5-hormone neurochemical ODE model.

Tests biological plausibility of couplings, receptor dynamics,
refractory rebound, and emotion classification.
"""

import numpy as np
import pytest

from cubemind.brain.neurochemistry import Neurochemistry


class TestBasicDynamics:

    def test_resting_state_stable(self):
        nc = Neurochemistry()
        for _ in range(100):
            nc.update()  # No input
        # Should settle near resting levels
        assert 0.05 <= nc.cortisol < 0.20  # Slow integrator, drifts to floor
        assert 0.15 < nc.dopamine < 0.5
        assert 0.1 < nc.serotonin < 0.7
        assert 0.05 < nc.oxytocin < 0.5
        assert 0.05 < nc.noradrenaline < 0.3

    def test_all_in_range(self):
        nc = Neurochemistry()
        rng = np.random.default_rng(42)
        for _ in range(200):
            nc.update(novelty=rng.random(), threat=rng.random(),
                      focus=rng.random(), valence=rng.random() * 2 - 1,
                      social=rng.random())
        for attr in ("cortisol", "dopamine", "serotonin", "oxytocin", "noradrenaline"):
            v = getattr(nc, attr)
            assert 0 <= v <= 1, f"{attr}={v} out of range"


class TestCouplings:

    def test_cortisol_suppresses_dopamine(self):
        """Sustained threat → cortisol slowly rises → eventually suppresses DA."""
        nc = Neurochemistry()
        # Need MANY steps because cortisol is a slow EMA (minutes timescale)
        for _ in range(200):
            nc.update(threat=0.9, novelty=0.0)
        assert nc.cortisol > 0.3, f"Sustained threat should raise C: {nc.cortisol:.3f}"
        assert nc.dopamine < 0.40, f"High C should suppress DA: {nc.dopamine:.3f}"

    def test_cortisol_suppresses_serotonin(self):
        nc = Neurochemistry()
        initial_5ht = nc.serotonin
        for _ in range(20):
            nc.update(threat=0.9)
        assert nc.serotonin < initial_5ht

    def test_serotonin_boosts_oxytocin(self):
        nc = Neurochemistry()
        for _ in range(20):
            nc.update(valence=0.8, social=0.5)  # Positive → 5-HT up
        assert nc.serotonin > 0.5
        assert nc.oxytocin > 0.3

    def test_oxytocin_dopamine_bidirectional(self):
        """OT boosts DA and DA modulates OT (MDPI 2025)."""
        nc = Neurochemistry()
        for _ in range(20):
            nc.update(social=0.9, valence=0.5)
        # High social → high OT → should boost DA via coupling
        assert nc.oxytocin > 0.4
        # DA should be maintained (not just from novelty, also from OT coupling)
        assert nc.dopamine > 0.15

    def test_noradrenaline_independent_from_cortisol(self):
        """NE should respond to novelty even when cortisol is low."""
        nc = Neurochemistry()
        for _ in range(20):
            nc.update(novelty=0.8, threat=0.0)  # Novel but not threatening
        assert nc.noradrenaline > 0.3
        assert nc.cortisol < 0.3


class TestReceptorDynamics:

    def test_da_sensitivity_changes_with_level(self):
        """DA sensitivity should diverge from 1.0 over time based on DA level."""
        nc = Neurochemistry()
        # Sustained high novelty → high DA → downregulation
        for _ in range(50):
            nc.update(novelty=0.9, valence=0.8)
        assert nc._da_sensitivity != 1.0, "Sensitivity should change"

    def test_refractory_rebound(self):
        """After low DA phase, sudden novelty should produce DA recovery."""
        nc = Neurochemistry()
        # Phase 1: Low stimulation (DA drifts toward tonic)
        for _ in range(50):
            nc.update(novelty=0.0, valence=-0.3)
        da_low = nc.dopamine

        # Phase 2: Sudden novelty burst
        for _ in range(15):
            nc.update(novelty=0.9, valence=0.5)
        da_recovered = nc.dopamine

        assert da_recovered > da_low, (
            f"Novelty should recover DA: {da_low:.3f} → {da_recovered:.3f}")

    def test_oxytocin_never_saturates(self):
        nc = Neurochemistry()
        for _ in range(100):
            nc.update(social=1.0, valence=1.0)
        assert nc.oxytocin < 0.96, f"OT should not saturate: {nc.oxytocin:.3f}"


class TestNoradrenaline:

    def test_ne_rises_on_novelty(self):
        nc = Neurochemistry()
        initial = nc.noradrenaline
        for _ in range(10):
            nc.update(novelty=0.8)
        assert nc.noradrenaline > initial

    def test_ne_rises_on_threat(self):
        nc = Neurochemistry()
        initial = nc.noradrenaline
        for _ in range(10):
            nc.update(threat=0.8)
        assert nc.noradrenaline > initial

    def test_ne_rises_on_focus(self):
        nc = Neurochemistry()
        initial = nc.noradrenaline
        for _ in range(10):
            nc.update(focus=0.8)
        assert nc.noradrenaline > initial

    def test_serotonin_dampens_ne(self):
        """High 5-HT should reduce NE (calm dampens alertness)."""
        nc = Neurochemistry()
        nc.serotonin = 0.9
        nc.noradrenaline = 0.5
        for _ in range(20):
            nc.update(valence=0.5)  # Keep 5-HT high
        assert nc.noradrenaline < 0.5


class TestEmotionClassification:

    def test_threat_produces_alert_or_vigilant(self):
        """Threat → NE spikes fast. Emotion depends on 5-HT baseline."""
        nc = Neurochemistry()
        nc.serotonin = 0.2  # Start with low 5-HT to avoid "calm"
        for _ in range(15):
            nc.update(threat=0.9)
        assert nc.dominant_emotion in ("alert", "anxious", "curious")

    def test_novelty_produces_curious(self):
        nc = Neurochemistry()
        for _ in range(15):
            nc.update(novelty=0.8, valence=0.3)
        assert nc.dominant_emotion in ("curious", "alert", "joy")

    def test_social_produces_warm(self):
        nc = Neurochemistry()
        for _ in range(15):
            nc.update(social=0.8, valence=0.5)
        assert nc.dominant_emotion in ("warm", "joy", "calm")

    def test_calm_from_serotonin(self):
        nc = Neurochemistry()
        for _ in range(30):
            nc.update(valence=0.3, threat=0.0, novelty=0.0)
        assert nc.dominant_emotion in ("calm", "warm", "neutral")

    def test_negative_valence_not_joyful(self):
        """Sustained negative valence should NOT produce joy."""
        nc = Neurochemistry()
        for _ in range(30):
            nc.update(valence=-0.8, novelty=0.0, threat=0.0)
        assert nc.dominant_emotion != "joy"


class TestHartmannWeight:

    def test_weight_high_when_happy(self):
        nc = Neurochemistry()
        for _ in range(15):
            nc.update(novelty=0.7, valence=0.8)
        assert nc.weight > 0.5, f"Happy should feel light: {nc.weight:.3f}"

    def test_weight_low_when_stressed(self):
        """Sustained threat → cortisol slowly rises → weight drops."""
        nc = Neurochemistry()
        for _ in range(200):  # Cortisol needs time (slow EMA)
            nc.update(threat=0.9, valence=-0.5)
        assert nc.weight < 0.6, f"Stressed should feel heavier: {nc.weight:.3f}"


class TestSNNModulation:

    def test_threshold_modulation(self):
        nc = Neurochemistry()
        base = 0.1
        # High DA → lower threshold
        nc.dopamine = 0.8
        nc.cortisol = 0.1
        nc.noradrenaline = 0.3
        t = nc.modulate_threshold(base)
        assert t < base

    def test_tau_modulation(self):
        nc = Neurochemistry()
        base = 15.0
        # High 5-HT → higher tau (more stable)
        nc.serotonin = 0.9
        nc.cortisol = 0.1
        t = nc.modulate_tau(base)
        assert t > base
