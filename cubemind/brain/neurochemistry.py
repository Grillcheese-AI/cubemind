"""Biologically-grounded Neurochemical ODE Model.

5-signal system calibrated to real neuronal firing rates:

  Signal       | Neuron type        | Tonic    | Burst    | Timescale
  -------------|--------------------|---------:|---------:|----------
  Dopamine     | VTA/SNc DA         | 1-3 Hz   | 10-25 Hz | Fast (seconds)
  Serotonin    | DRN 5-HT           | 0.1-3 Hz | 17 Hz    | Medium (tens of s)
  Cortisol     | HPA axis (CRH/PVN) | slow EMA | —        | Very slow (minutes)
  Oxytocin     | PVN magno/parvo    | 1-3 Hz   | 80 Hz    | Medium, with decay
  Noradrenaline| LC                 | ~1 Hz    | 10-25 Hz | Fast (seconds)

KEY DESIGN: Cortisol is a SLOW exponential moving average of arousal,
NOT a per-frame signal. The per-frame stress response is noradrenaline
(LC phasic bursts). Cortisol integrates over minutes — it's the HPA
hormonal cascade (CRH → ACTH → cortisol) with 15-30 min peak latency.

ODE: dH/dt = alpha * drive * receptor_sensitivity - beta * (H - resting)
Receptor sensitivity: sigmoid saturation, not linear.
Refractory rebound: after sustained depletion, sensitivity upregulates.

References:
  - Springer J Comp Neurosci (Feb 2026): modular neuromodulator framework
  - MDPI Int J Mol Sci (2025): OT-DA bidirectional signaling in striatum
  - eLife (Jan 2026): subregion-specific DA dynamics, tonic level
  - Bhatt et al. Nature Comms (2026): NE spreads hippocampal association
  - Bhagat et al. Nature Neurosci (2026): ACh demixes DA for learning vs action
  - Hartmann et al. (2021): valence-as-weight (DA/C ratio)
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: float, center: float = 0.5, steepness: float = 10.0) -> float:
    """Receptor saturation curve. Prevents linear blow-up at extremes."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


class Neurochemistry:
    """5-hormone neurochemical ODE with receptor dynamics.

    Designed for real-time modulation of SNN and cognitive routing.
    All values in [0, 1]. Update rate: once per perception frame.

    Args:
        dt: Integration timestep (smaller = smoother, slower).
    """

    def __init__(self, dt: float = 0.8) -> None:
        # ── Resting concentrations (normalized from biological firing rates) ──
        # DA: tonic 1-3 Hz out of ~10 Hz max → 0.2-0.3 normalized
        # 5-HT: tonic 0.1-3 Hz out of ~17 Hz → 0.1-0.2, but slow so accumulates
        # NE: tonic ~1 Hz out of ~25 Hz → 0.04-0.1
        # OT: tonic 1-3 Hz baseline, bursts to 80 Hz → 0.05-0.1 resting
        # C: slow HPA integrator, NOT a firing rate → starts at ~0.15
        self._resting = {
            "DA": 0.30, "5HT": 0.45, "NE": 0.15, "OT": 0.20, "C": 0.15,
        }

        self.dopamine: float = self._resting["DA"]
        self.serotonin: float = self._resting["5HT"]
        self.noradrenaline: float = self._resting["NE"]
        self.oxytocin: float = self._resting["OT"]
        self.cortisol: float = self._resting["C"]

        # ── Receptor sensitivity (upregulates during depletion) ──────
        self._da_sensitivity: float = 1.0
        self._ne_sensitivity: float = 1.0

        # ── ODE parameters: alpha (drive gain), beta (decay toward resting) ──
        # Fast signals (DA, NE): high alpha, high beta (responsive, quick decay)
        # Medium signals (5-HT, OT): moderate alpha/beta
        # Slow signal (C): very low alpha/beta (minutes timescale)
        self._alpha = {
            "DA": 0.35, "5HT": 0.20, "NE": 0.40, "OT": 0.22, "C": 0.02,
        }
        self._beta = {
            "DA": 0.25, "5HT": 0.10, "NE": 0.30, "OT": 0.18, "C": 0.01,
        }

        # ── Coupling strengths (from literature) ─────────────────────
        self._couplings = {
            "C_suppresses_DA": -0.10,   # HPA → mesolimbic suppression
            "C_suppresses_5HT": -0.06,  # Cortisol dampens serotonin
            "C_suppresses_OT": -0.04,   # Cortisol blocks social bonding
            "5HT_boosts_OT": 0.05,      # Serotonin facilitates OT release
            "OT_boosts_DA": 0.06,       # OT modulates DA release (MDPI 2025)
            "DA_modulates_OT": 0.03,    # DA affects OT receptor sensitivity
            "NE_boosts_DA": 0.05,       # NE → DA co-release in VTA
            "NE_suppresses_5HT": -0.03, # NE dampens 5-HT (arousal > calm)
            "5HT_dampens_NE": -0.04,    # 5-HT calms arousal
        }

        self._dt = dt

        # ── Affect state (derived) ───────────────────────────────────
        self.valence: float = 0.0
        self.affect_arousal: float = 0.0
        self.stress: float = 0.0
        self.dominant_emotion: str = "neutral"

    # ── Core ODE update ──────────────────────────────────────────────

    def update(
        self,
        novelty: float = 0.0,
        threat: float = 0.0,
        focus: float = 0.0,
        valence: float = 0.0,
        social: float = 0.0,
    ) -> None:
        """Update all hormone levels via coupled ODE dynamics.

        Args:
            novelty:  [0, 1] — new/surprising → DA + NE.
            threat:   [0, 1] — danger/urgency → C + NE.
            focus:    [0, 1] — attention demand → NE.
            valence:  [-1, 1] — positive/negative affect → DA, 5-HT, OT.
            social:   [0, 1] — social/empathic context → OT.
        """
        joy = max(0.0, valence)
        neg = max(0.0, -valence)

        # ── Compute drive signals ────────────────────────────────────

        # NE (fast): phasic bursts on novelty/threat/focus (LC neurons)
        drive_NE = (novelty * 0.5 + threat * 0.4
                    + focus * 0.4) * self._ne_sensitivity

        # DA (fast): phasic bursts on reward prediction error (VTA)
        # Tonic 0.2-0.3, burst to 0.8 on novelty
        drive_DA = (novelty * 0.5 + joy * 0.4
                    + 0.15) * self._da_sensitivity

        # 5-HT (medium): slow clock-like, dips on aversive
        # Tonic 0.4-0.5, suppressed by threat (DRN GABA inhibition)
        drive_5HT = 0.35 + 0.25 * joy - 0.2 * threat - 0.1 * neg

        # OT (medium): bursts on social peaks, decays to baseline
        drive_OT = social * 0.5 + joy * 0.2 + 0.05

        # C (SLOW): NOT driven per-frame. EMA of arousal over minutes.
        # The HPA cascade is too slow for per-frame updates.
        # Cortisol integrates the sustained arousal level.
        # drive_C is computed AFTER NE update (cortisol tracks NE over time).

        drives = {
            "DA": float(np.clip(drive_DA, 0, 1)),
            "5HT": float(np.clip(drive_5HT, 0, 1)),
            "NE": float(np.clip(drive_NE, 0, 1)),
            "OT": float(np.clip(drive_OT, 0, 1)),
        }

        # ── ODE step for FAST signals: dH/dt = alpha*drive - beta*(H - resting)
        values = {
            "DA": self.dopamine,
            "5HT": self.serotonin,
            "NE": self.noradrenaline,
            "OT": self.oxytocin,
        }

        new = {}
        for h in ("DA", "NE", "5HT", "OT"):
            rest = self._resting[h]
            delta = self._alpha[h] * drives[h] - self._beta[h] * (values[h] - rest)
            new[h] = values[h] + self._dt * delta

        # ── CORTISOL: slow EMA of arousal (NE + threat) ─────────────
        # Timescale: minutes. EMA factor ~0.01 means ~100 frames to respond.
        # This is the HPA cascade: CRH → ACTH → cortisol (15-30 min peak)
        instantaneous_stress = float(np.clip(
            new["NE"] * 0.5 + threat * 0.3 + neg * 0.2, 0, 1))
        ema_rate = 0.015  # Very slow integration
        new["C"] = self.cortisol + ema_rate * (instantaneous_stress - self.cortisol)

        # ── Non-linear couplings (receptor-saturated) ────────────────
        # Use sigmoid to prevent runaway: coupling * sigmoid(source)
        c_sat = _sigmoid(new["C"], center=0.4, steepness=8)
        da_sat = _sigmoid(new["DA"], center=0.4, steepness=8)
        sht_sat = _sigmoid(new["5HT"], center=0.5, steepness=8)
        ot_sat = _sigmoid(new["OT"], center=0.4, steepness=8)
        ne_sat = _sigmoid(new["NE"], center=0.3, steepness=8)

        # Cortisol suppresses DA, 5-HT, OT
        new["DA"] += self._couplings["C_suppresses_DA"] * c_sat
        new["5HT"] += self._couplings["C_suppresses_5HT"] * c_sat
        new["OT"] += self._couplings["C_suppresses_OT"] * c_sat

        # Serotonin ↔ Oxytocin ↔ Dopamine (bidirectional triangle)
        new["OT"] += self._couplings["5HT_boosts_OT"] * sht_sat
        new["DA"] += self._couplings["OT_boosts_DA"] * ot_sat
        new["OT"] += self._couplings["DA_modulates_OT"] * da_sat

        # Noradrenaline interactions
        new["DA"] += self._couplings["NE_boosts_DA"] * ne_sat
        new["5HT"] += self._couplings["NE_suppresses_5HT"] * ne_sat
        new["NE"] += self._couplings["5HT_dampens_NE"] * sht_sat

        # ── Clamp with biological constraints ────────────────────────
        # DA: tonic floor 0.2 (VTA pacemaker never stops), ceiling 0.85
        self.dopamine = float(np.clip(new["DA"], 0.15, 0.85))
        # 5-HT: slow, never zero (DRN pacemaker), micro-dips allowed
        self.serotonin = float(np.clip(new["5HT"], 0.1, 0.90))
        # NE: fast, tonic floor (LC never fully silent in wakefulness)
        self.noradrenaline = float(np.clip(new["NE"], 0.05, 0.90))
        # OT: decays toward baseline between bursts (PVN burst/decay pattern)
        ot_decay = 0.05 * (new["OT"] - self._resting["OT"])
        self.oxytocin = float(np.clip(new["OT"] - ot_decay, 0.05, 0.85))
        # C: slow integrator, mild range
        self.cortisol = float(np.clip(new["C"], 0.05, 0.80))

        # ── Receptor sensitivity: refractory rebound ─────────────────
        # Sustained low DA → receptors upregulate (become more sensitive)
        # Sustained high DA → receptors downregulate (tolerance)
        if self.dopamine < 0.2:
            self._da_sensitivity = min(1.5, self._da_sensitivity + 0.02)
        elif self.dopamine > 0.6:
            self._da_sensitivity = max(0.5, self._da_sensitivity - 0.01)
        else:
            self._da_sensitivity += 0.005 * (1.0 - self._da_sensitivity)  # Return to 1.0

        if self.noradrenaline < 0.15:
            self._ne_sensitivity = min(1.5, self._ne_sensitivity + 0.02)
        elif self.noradrenaline > 0.6:
            self._ne_sensitivity = max(0.5, self._ne_sensitivity - 0.01)
        else:
            self._ne_sensitivity += 0.005 * (1.0 - self._ne_sensitivity)

        # ── Derived affect state ─────────────────────────────────────
        self.valence = float(np.clip(valence, -1, 1))
        self.affect_arousal = float(np.clip(
            novelty + threat + self.noradrenaline * 0.3, 0, 1))
        self.stress = float(self.cortisol)

        # ── Emotion classification ───────────────────────────────────
        self.dominant_emotion = self._classify_emotion(novelty, valence)

    def _classify_emotion(self, novelty: float, valence: float) -> str:
        """Determine dominant emotion from hormone landscape."""
        scores = {
            "anxious": self.cortisol * 0.6 + (1 - self.serotonin) * 0.2
                       + self.noradrenaline * 0.2,
            "joy": self.dopamine * 0.4 + max(0, valence) * 0.3
                   + self.serotonin * 0.3,
            "curious": self.dopamine * 0.3 + novelty * 0.4
                       + self.noradrenaline * 0.3,
            "warm": self.oxytocin * 0.5 + self.serotonin * 0.3
                    + max(0, valence) * 0.2,
            "sad": (1 - self.dopamine) * 0.3 + max(0, -valence) * 0.4
                   + (1 - self.serotonin) * 0.3,
            "alert": self.noradrenaline * 0.5 + self.cortisol * 0.2
                     + novelty * 0.3,
            "calm": self.serotonin * 0.4 + (1 - self.cortisol) * 0.3
                    + (1 - self.noradrenaline) * 0.3,
            "neutral": 0.3,
        }
        return max(scores, key=scores.get)

    # ── SNN modulation (same interface as NeurochemicalState) ────────

    def modulate_threshold(self, base_threshold: float) -> float:
        """DA lowers threshold (excited), C raises it (defensive), NE lowers (alert)."""
        return base_threshold * (
            1.0
            - 0.25 * (self.dopamine - 0.4)
            - 0.15 * (self.noradrenaline - 0.25)
            + 0.15 * self.cortisol
        )

    def modulate_tau(self, base_tau: float) -> float:
        """5-HT stabilizes (higher tau), C speeds up, NE speeds up."""
        return base_tau * (
            1.0
            + 0.25 * (self.serotonin - 0.5)
            - 0.3 * (self.cortisol - 0.2)
            - 0.15 * (self.noradrenaline - 0.25)
        )

    # ── Properties ───────────────────────────────────────────────────

    @property
    def arousal(self) -> float:
        """Overall arousal: NE + C + DA - 5-HT."""
        return float(np.clip(
            0.3 * self.noradrenaline + 0.25 * self.cortisol
            + 0.25 * self.dopamine + 0.2 * (1 - self.serotonin),
            0, 1))

    @property
    def weight(self) -> float:
        """Hartmann valence-as-weight: DA/(DA+C). Light=positive, Heavy=negative."""
        return self.dopamine / (self.dopamine + self.cortisol + 1e-8)

    @property
    def da_sensitivity(self) -> float:
        """Current dopamine receptor sensitivity (refractory state)."""
        return self._da_sensitivity

    def to_dict(self) -> dict[str, float]:
        return {
            "cortisol": self.cortisol,
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "oxytocin": self.oxytocin,
            "noradrenaline": self.noradrenaline,
            "valence": self.valence,
            "arousal": self.arousal,
            "stress": self.stress,
            "weight": self.weight,
            "da_sensitivity": self._da_sensitivity,
            "emotion": self.dominant_emotion,
        }
