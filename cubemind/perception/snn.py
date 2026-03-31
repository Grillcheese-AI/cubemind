"""SNN Perceptual Layer — spiking neural network for temporal stream encoding.

First perceptual network in the CubeMind cognitive pipeline:
  Temporal stream → SNN LIF spikes → VSA projection → binary temporal memory

The SNN processes temporal data (video frames, audio, sensor streams) through
Leaky Integrate-and-Fire neurons. Because LIF neurons only fire when input
*changes*, static inputs produce zero spikes — extreme sparsity for free.

The spike output feeds into the VSA bridge:
  spikes (binary) → linear projection → binarize & pack → cyclic shift + XOR
  → single packed binary temporal memory vector

Uses grilly.nn.snn_neurons.LIFNode when available (GPU-accelerated via Vulkan
compute shaders). Falls back to a numpy CPU implementation.

Neurochemical modulation (dopamine, adrenaline, acetylcholine) is implemented
as dynamic threshold and leak adjustments — the "emotional states" from the
CubeMind identity prompt.
"""

from __future__ import annotations

import math

import numpy as np

from cubemind.ops.vsa_bridge import LSHProjector, binarize_and_pack

# Try grilly SNN
_GRILLY_SNN = False
_LIFNode = None
try:
    from grilly.nn.snn_neurons import LIFNode as _GrillyLIF
    _LIFNode = _GrillyLIF
    _GRILLY_SNN = True
except ImportError:
    pass


# ── CPU LIF Neuron (fallback) ─────────────────────────────────────────────

class IFNeuronLayer:
    """Integrate-and-Fire neuron layer (no leak).

    H[t] = V[t-1] + X[t]
    spike = 1 if H[t] >= threshold else 0
    V[t] = reset if spike else H[t]

    No leak means charge accumulates indefinitely — more sensitive to
    weak persistent signals than LIF. Good for detecting slow trends
    or accumulating evidence over long time windows.

    Args:
        n_neurons:   Number of neurons in the layer.
        v_threshold: Spike threshold voltage.
        v_reset:     Reset voltage after spike.
    """

    def __init__(
        self, n_neurons: int, v_threshold: float = 1.0, v_reset: float = 0.0,
    ) -> None:
        self.n_neurons = n_neurons
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v = np.full(n_neurons, v_reset, dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        self.v = self.v + x
        spikes = (self.v >= self.v_threshold).astype(np.float32)
        self.v = np.where(spikes > 0, self.v_reset, self.v)
        return spikes

    def reset(self) -> None:
        self.v = np.full(self.n_neurons, self.v_reset, dtype=np.float32)


class LIFNeuronLayer:
    """Leaky Integrate-and-Fire neuron layer (numpy fallback).

    H[t] = V[t-1] * decay + X[t]
    spike = 1 if H[t] >= threshold else 0
    V[t] = reset if spike else H[t]

    The leak makes LIF less sensitive than IF — sub-threshold input
    decays away, so neurons need strong/fast input to reach threshold.
    Better for detecting transient changes (edges, onsets).

    Args:
        n_neurons:   Number of neurons in the layer.
        tau:         Membrane time constant (higher = slower leak).
        v_threshold: Spike threshold voltage.
        v_reset:     Reset voltage after spike.
    """

    def __init__(
        self, n_neurons: int, tau: float = 2.0,
        v_threshold: float = 1.0, v_reset: float = 0.0,
    ) -> None:
        self.n_neurons = n_neurons
        self.tau = tau
        self.decay = 1.0 - 1.0 / tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v = np.full(n_neurons, v_reset, dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Single timestep: input → membrane update → spike output.

        Args:
            x: (n_neurons,) float32 input current.

        Returns:
            (n_neurons,) float32 binary spikes {0, 1}.
        """
        # Charge (with leak)
        self.v = self.decay * (self.v - self.v_reset) + self.v_reset + x

        # Fire
        spikes = (self.v >= self.v_threshold).astype(np.float32)

        # Reset
        self.v = np.where(spikes > 0, self.v_reset, self.v)

        return spikes

    def reset(self) -> None:
        """Reset membrane potentials. Call between sequences."""
        self.v = np.full(self.n_neurons, self.v_reset, dtype=np.float32)


# ── Neurochemical Modulator ───────────────────────────────────────────────

class NeurochemicalState:
    """Emotional state via neurochemical-analog ODE dynamics.

    Ported from grillcheese.brain.endocrine — 4-hormone system with
    non-linear couplings, modulating SNN dynamics in real-time.

    Hormones:
      - cortisol:  Stress hormone. Rises under threat/uncertainty.
                   Dampens dopamine and serotonin (stress kills reward/mood).
      - dopamine:  Reward/novelty signal. Lowers spike threshold → more firing.
                   Driven by positive valence, suppressed by stress.
      - serotonin: Mood/stability signal. Modulates leak rate (tau).
                   High serotonin → slower leak → more stable membrane.
      - oxytocin:  Social/trust signal. Affects attention focus.
                   Driven by empathy/positive interaction.

    ODE dynamics: dH/dt = alpha * drive - beta * H
    Non-linear couplings: cortisol suppresses dopamine/serotonin.

    Values in [0, 1]. Resting state calibrated from grillcheese defaults.
    """

    def __init__(self) -> None:
        self.cortisol: float = 0.2
        self.dopamine: float = 0.4
        self.serotonin: float = 0.5
        self.oxytocin: float = 0.3

        # ODE parameters (from grillcheese.brain.endocrine)
        self._alpha = np.array([0.42, 0.36, 0.30, 0.28], dtype=np.float32)
        self._beta = np.array([0.24, 0.19, 0.16, 0.14], dtype=np.float32)
        self._dt = 1.0

        # Affect state (from grillcheese.brain.amygdala)
        self.valence: float = 0.0   # [-1, 1] negative to positive
        self.affect_arousal: float = 0.0  # [0, 1]
        self.stress: float = 0.0    # [0, 1]
        self.dominant_emotion: str = "neutral"

    def modulate_threshold(self, base_threshold: float) -> float:
        """Dopamine lowers threshold (excited), cortisol raises it (defensive)."""
        return base_threshold * (1.0 - 0.3 * (self.dopamine - 0.4) + 0.15 * self.cortisol)

    def modulate_tau(self, base_tau: float) -> float:
        """Serotonin stabilizes (higher tau), cortisol speeds up (lower tau)."""
        return base_tau * (1.0 + 0.3 * (self.serotonin - 0.5) - 0.4 * (self.cortisol - 0.2))

    def update(
        self,
        novelty: float = 0.0,
        threat: float = 0.0,
        focus: float = 0.0,
        valence: float = 0.0,
    ) -> None:
        """Update hormone levels via ODE dynamics.

        Args:
            novelty:  [0, 1] — new/surprising input → dopamine.
            threat:   [0, 1] — urgent/dangerous input → cortisol.
            focus:    [0, 1] — attention demand → oxytocin.
            valence:  [-1, 1] — positive/negative affect.
        """
        # Compute drive signals (from grillcheese endocrine)
        joy = max(0.0, valence)
        stress_signal = max(0.0, threat)

        drive = np.array([
            stress_signal + 0.5 * novelty + 0.25 * threat,           # cortisol
            max(0.0, valence) + 0.35 * joy - 0.20 * self.stress + novelty * 0.3,  # dopamine
            0.5 + 0.5 * valence - 0.25 * self.stress + 0.10 * joy,  # serotonin
            0.12 + 0.35 * joy + 0.22 * max(0.0, valence) + 0.20 * focus,  # oxytocin
        ], dtype=np.float32)
        drive = np.clip(drive, 0.0, 1.0)

        current = np.array([
            self.cortisol, self.dopamine, self.serotonin, self.oxytocin,
        ], dtype=np.float32)

        # ODE step
        delta = self._alpha * drive - self._beta * current
        values = np.clip(current + self._dt * delta, 0.0, 1.0)

        # Non-linear couplings (from grillcheese): stress dampens reward/mood
        values[1] = float(np.clip(values[1] * (1.0 - 0.12 * values[0]), 0.0, 1.0))
        values[2] = float(np.clip(values[2] * (1.0 - 0.08 * values[0]), 0.0, 1.0))
        values[3] = float(np.clip(values[3] + 0.06 * values[2] - 0.05 * values[0], 0.0, 1.0))

        self.cortisol = float(values[0])
        self.dopamine = float(values[1])
        self.serotonin = float(values[2])
        self.oxytocin = float(values[3])

        # Update affect state
        self.valence = float(np.clip(valence, -1.0, 1.0))
        self.affect_arousal = float(np.clip(novelty + threat, 0.0, 1.0))
        self.stress = float(np.clip(self.cortisol, 0.0, 1.0))

        # Dominant emotion heuristic
        if self.stress > 0.6:
            self.dominant_emotion = "anxious"
        elif self.dopamine > 0.6 and valence > 0.3:
            self.dominant_emotion = "joy"
        elif valence < -0.3:
            self.dominant_emotion = "sad"
        elif self.oxytocin > 0.5 and valence > 0:
            self.dominant_emotion = "warm"
        elif novelty > 0.5:
            self.dominant_emotion = "curious"
        else:
            self.dominant_emotion = "neutral"

    @property
    def arousal(self) -> float:
        """Overall arousal level."""
        return float(np.clip(
            0.3 * self.cortisol + 0.3 * self.dopamine + 0.2 * self.affect_arousal + 0.2 * (1.0 - self.serotonin),
            0.0, 1.0,
        ))

    def to_dict(self) -> dict[str, float]:
        return {
            "cortisol": self.cortisol,
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "oxytocin": self.oxytocin,
            "valence": self.valence,
            "arousal": self.arousal,
            "stress": self.stress,
            "emotion": self.dominant_emotion,
        }


# ── SNN Encoder ───────────────────────────────────────────────────────────

class SNNEncoder:
    """Spiking Neural Network encoder for temporal streams.

    Pipeline per timestep:
      input (d_input,) → Linear → spiking neurons (n_neurons,) → spikes {0, 1}

    Full temporal encoding:
      For each frame in stream:
        1. Project input through weight matrix
        2. Spiking neurons integrate and fire
        3. Project spikes to VSA space
        4. Binarize and pack
        5. Cyclic shift accumulator + XOR bind

    Result: single packed binary vector encoding the entire temporal sequence.

    Neuron types:
      - "lif": Leaky Integrate-and-Fire. Leak makes it less sensitive — good
               for detecting transient changes (edges, onsets, fast events).
      - "if":  Integrate-and-Fire (no leak). Accumulates charge indefinitely —
               more sensitive to weak persistent signals, good for slow trends
               and accumulating evidence over long time windows.

    Args:
        d_input:     Input dimension per timestep (e.g., 256 for a frame).
        n_neurons:   Number of spiking neurons.
        d_vsa:       VSA dimension for binary output.
        neuron_type: "lif" (default) or "if".
        tau:         LIF membrane time constant (ignored for IF).
        v_threshold: Spike threshold.
        seed:        Random seed.
    """

    def __init__(
        self,
        d_input: int = 256,
        n_neurons: int = 1024,
        d_vsa: int = 10240,
        neuron_type: str = "lif",
        tau: float = 2.0,
        v_threshold: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.d_input = d_input
        self.n_neurons = n_neurons
        self.d_vsa = d_vsa
        self.neuron_type = neuron_type
        self.words_per_vec = int(math.ceil(d_vsa / 32))

        rng = np.random.default_rng(seed)

        # Input projection: d_input → n_neurons
        std = 1.0 / math.sqrt(d_input)
        self.W_in = rng.normal(0, std, (n_neurons, d_input)).astype(np.float32)

        # Spiking layer (grilly GPU or numpy fallback)
        self._use_grilly = _GRILLY_SNN
        if _GRILLY_SNN:
            if neuron_type == "if":
                from grilly.nn.snn_neurons import IFNode as _GrillyIF
                self._lif = _GrillyIF(v_threshold=v_threshold, step_mode='s')
            else:
                self._lif = _LIFNode(tau=tau, v_threshold=v_threshold, step_mode='s')
        else:
            if neuron_type == "if":
                self._lif = IFNeuronLayer(n_neurons, v_threshold=v_threshold)
            else:
                self._lif = LIFNeuronLayer(n_neurons, tau=tau, v_threshold=v_threshold)

        # Spike → VSA projection
        self.lsh = LSHProjector(d_input=n_neurons, d_output=d_vsa, seed=seed + 1)

        # Neurochemical emotional state
        self.neurochemistry = NeurochemicalState()

        # Base parameters (before modulation)
        self._base_tau = tau
        self._base_threshold = v_threshold

        # ── Photonic-inspired STDP (Feldmann et al., Nature 2019) ──────
        # Self-learning via simplified spike-timing-dependent plasticity:
        #   - Inputs that contributed to a spike → potentiate (weight up)
        #   - Inputs that didn't contribute → depress (weight down)
        # Mimics the PCM feedback waveguide: output spike overlaps with
        # input pulses at synaptic PCM cells, amorphizing contributing
        # synapses (high weight) and crystallizing non-contributing ones.
        self.stdp_enabled = True
        self.stdp_lr_potentiate = 0.005   # Learning rate for potentiation
        self.stdp_lr_depress = 0.002      # Learning rate for depression
        self.stdp_weight_clip = 3.0       # Max absolute weight value
        self._last_input: np.ndarray | None = None

    def reset(self) -> None:
        """Reset SNN state between sequences."""
        if self._use_grilly:
            self._lif.reset()
        else:
            self._lif.reset()

    def step(self, x: np.ndarray) -> np.ndarray:
        """Single timestep: input → spikes, with optional STDP learning.

        If stdp_enabled, synaptic weights are updated after each step
        using the photonic STDP rule (Feldmann et al., Nature 2019):
          - Neurons that fired: potentiate weights from active inputs
          - Neurons that didn't fire: depress weights from active inputs

        Args:
            x: (d_input,) float32 input vector.

        Returns:
            (n_neurons,) float32 binary spikes.
        """
        x_flat = x.ravel().astype(np.float32)

        # Apply neurochemical modulation
        if not self._use_grilly:
            self._lif.v_threshold = self.neurochemistry.modulate_threshold(self._base_threshold)
            tau_mod = self.neurochemistry.modulate_tau(self._base_tau)
            if hasattr(self._lif, 'decay'):
                self._lif.decay = 1.0 - 1.0 / max(tau_mod, 1.0)

        # Project input through synaptic weights
        current = self.W_in @ x_flat

        # Spiking neuron step
        if self._use_grilly:
            spikes = self._lif(current)
            if not isinstance(spikes, np.ndarray):
                spikes = np.asarray(spikes, dtype=np.float32)
        else:
            spikes = self._lif.step(current)

        # Photonic STDP: self-learning weight update
        if self.stdp_enabled and np.any(spikes > 0):
            self._stdp_update(x_flat, spikes)

        self._last_input = x_flat
        return spikes

    def _stdp_update(self, x: np.ndarray, spikes: np.ndarray) -> None:
        """Photonic-inspired STDP weight update.

        Mimics the PCM feedback waveguide mechanism:
        - Active inputs (|x| > 0) that fed into a firing neuron:
          potentiate (amorphize → high transmission weight)
        - Active inputs that fed into a non-firing neuron:
          depress (crystallize → low transmission weight)

        This is a simplified clocked STDP — no timing delay, just
        co-activity within the same timestep.

        The dopamine level modulates learning rate (reward-gated STDP):
        high dopamine → faster learning (novel/rewarding patterns).
        """
        # Which inputs were active (had signal)
        active_inputs = (np.abs(x) > 1e-6).astype(np.float32)

        # Dopamine-modulated learning rates
        dopa_mod = 0.5 + self.neurochemistry.dopamine
        lr_pot = self.stdp_lr_potentiate * dopa_mod
        lr_dep = self.stdp_lr_depress * dopa_mod

        # Vectorized broadcasting: no np.outer, no temporary matrices
        fired = spikes > 0
        not_fired = ~fired

        # Potentiate rows that fired (broadcast active_inputs across selected rows)
        if np.any(fired):
            self.W_in[fired, :] += lr_pot * active_inputs

        # Depress rows that didn't fire
        if np.any(not_fired):
            self.W_in[not_fired, :] -= lr_dep * active_inputs

        # In-place decay and clip (PCM crystallization)
        self.W_in *= 0.995
        np.clip(self.W_in, -self.stdp_weight_clip, self.stdp_weight_clip, out=self.W_in)

    def encode_stream(self, stream: np.ndarray) -> np.ndarray:
        """Encode a temporal stream into a single packed binary VSA vector.

        Uses cyclic shift + XOR temporal binding:
          accumulator = shift(accumulator, 1) XOR pack(frame_spikes)

        This compresses an entire temporal sequence (e.g., 300 video frames)
        into a single 10,240-bit binary vector (1.25 KB).

        Args:
            stream: (T, d_input) float32 — T timesteps of input.

        Returns:
            (words_per_vec,) uint32 — packed binary temporal memory vector.
        """
        self.reset()
        T = stream.shape[0]

        accumulator = np.zeros(self.words_per_vec, dtype=np.uint32)

        for t in range(T):
            # SNN step
            spikes = self.step(stream[t])

            # Novelty detection for neurochemistry
            spike_rate = float(np.mean(spikes))
            self.neurochemistry.update(novelty=spike_rate)

            # Project spikes to VSA space and binarize
            projected = self.lsh.project(spikes)
            packed_frame = binarize_and_pack(projected)

            # Temporal binding: cyclic shift + XOR
            accumulator = _cyclic_shift_uint32(accumulator, shift=1)
            accumulator = np.bitwise_xor(accumulator, packed_frame)

        return accumulator

    def encode_stream_with_spikes(
        self, stream: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
        """Encode stream and return intermediate spike data for analysis.

        Returns:
            (temporal_vector, spike_history, arousal_history)
        """
        self.reset()
        T = stream.shape[0]

        accumulator = np.zeros(self.words_per_vec, dtype=np.uint32)
        spike_history = []
        arousal_history = []

        for t in range(T):
            spikes = self.step(stream[t])
            spike_rate = float(np.mean(spikes))
            self.neurochemistry.update(novelty=spike_rate)

            projected = self.lsh.project(spikes)
            packed_frame = binarize_and_pack(projected)

            accumulator = _cyclic_shift_uint32(accumulator, shift=1)
            accumulator = np.bitwise_xor(accumulator, packed_frame)

            spike_history.append(spikes.copy())
            arousal_history.append(self.neurochemistry.arousal)

        return accumulator, spike_history, arousal_history

    def save(self, path: str) -> None:
        """Save SNN state: synaptic weights + neurochemistry.

        The weights encode everything the SNN has learned via STDP.
        Loading them back means the network remembers across sessions.
        """
        np.savez_compressed(
            path,
            W_in=self.W_in,
            cortisol=np.float32(self.neurochemistry.cortisol),
            dopamine=np.float32(self.neurochemistry.dopamine),
            serotonin=np.float32(self.neurochemistry.serotonin),
            oxytocin=np.float32(self.neurochemistry.oxytocin),
            d_input=np.int32(self.d_input),
            n_neurons=np.int32(self.n_neurons),
            d_vsa=np.int32(self.d_vsa),
            neuron_type=np.array([self.neuron_type]),
        )

    def load(self, path: str) -> None:
        """Load saved SNN state: synaptic weights + neurochemistry."""
        data = np.load(path, allow_pickle=True)
        self.W_in = data["W_in"].astype(np.float32)
        self.neurochemistry.cortisol = float(data["cortisol"])
        self.neurochemistry.dopamine = float(data["dopamine"])
        self.neurochemistry.serotonin = float(data["serotonin"])
        self.neurochemistry.oxytocin = float(data["oxytocin"])

    @property
    def spike_dims(self) -> int:
        return self.n_neurons


# ── Utility ───────────────────────────────────────────────────────────────

def _cyclic_shift_uint32(packed: np.ndarray, shift: int = 1) -> np.ndarray:
    """Fast cyclic bit-shift for packed uint32 arrays.

    For shift < 32 (the common case in temporal binding where shift=1):
    uses vectorized 32-bit math without unpacking individual bits.
    ~400x faster than the bit-unpack/roll/repack approach.

    Args:
        packed: (words_per_vec,) uint32.
        shift:  Number of bit positions to shift (positive = left).

    Returns:
        (words_per_vec,) uint32 shifted array.
    """
    total_bits = len(packed) * 32
    shift = shift % total_bits
    if shift == 0:
        return packed.copy()

    if shift < 32:
        # Fast path: pure 32-bit vectorized math (~5 microseconds)
        # 1. Shift all words left by `shift` bits
        left = packed << np.uint32(shift)
        # 2. Extract overflow bits from the top of each word
        overflow = packed >> np.uint32(32 - shift)
        # 3. Roll overflow to the next word (cyclic)
        rolled = np.roll(overflow, 1)
        return (left | rolled).astype(np.uint32)

    # General case: word-level roll + sub-word bit shift
    word_shift = shift // 32
    bit_shift = shift % 32
    rolled_words = np.roll(packed, -word_shift)
    if bit_shift == 0:
        return rolled_words
    left = rolled_words << np.uint32(bit_shift)
    overflow = rolled_words >> np.uint32(32 - bit_shift)
    rolled_overflow = np.roll(overflow, 1)
    return (left | rolled_overflow).astype(np.uint32)
