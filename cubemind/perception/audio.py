"""Audio Perception — microphone → mel spectrogram → SNN temporal encoding.

Captures audio from a microphone, computes mel spectrogram features,
and feeds them through the SNN encoder for temporal VSA binding.

Pipeline:
  microphone → raw PCM → mel spectrogram (numpy, no librosa)
  → SNN temporal encoding → binary VSA temporal memory

Features per frame (hop):
  - N_MELS mel-frequency bands (energy per band)
  - Zero-crossing rate (voiced/unvoiced)
  - RMS energy (loudness)
  - Spectral centroid (brightness)

These map to neurochemistry:
  - Loud sudden sound → cortisol spike (startle)
  - Music/tonal content → serotonin (soothing)
  - Speech detected → oxytocin (social)
  - Silence → gradual neurochemistry decay

Dependencies: sounddevice (installed), numpy
"""

from __future__ import annotations

import math
import time

import numpy as np

from cubemind.perception.snn import SNNEncoder
from cubemind.ops.vsa_bridge import binarize_and_pack, LSHProjector

# sounddevice (optional)
_SD = None
try:
    import sounddevice as _sd_module
    _SD = _sd_module
except ImportError:
    pass


# ── Mel Filterbank (pure numpy, no librosa) ───────────────────────────────

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Create a mel-scale filterbank matrix. No librosa needed."""
    low_mel = _hz_to_mel(0)
    high_mel = _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)
    return filterbank


class AudioFeatureExtractor:
    """Extract mel spectrogram features from raw audio. Pure numpy.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_fft:       FFT window size.
        hop_length:  Hop between frames.
        n_mels:      Number of mel bands.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 40,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._filterbank = _mel_filterbank(n_mels, n_fft, sample_rate)
        self._window = np.hanning(n_fft).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Dimension of the feature vector per frame."""
        return self.n_mels + 3  # mel bands + zcr + rms + centroid

    def extract_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """Extract features from a single audio frame.

        Args:
            audio_frame: (n_fft,) float32 audio samples.

        Returns:
            (feature_dim,) float32 feature vector.
        """
        frame = audio_frame.astype(np.float32)
        if len(frame) < self.n_fft:
            frame = np.pad(frame, (0, self.n_fft - len(frame)))

        # Windowed FFT
        windowed = frame[:self.n_fft] * self._window
        spectrum = np.abs(np.fft.rfft(windowed))

        # Mel spectrogram (log scale)
        mel_energy = self._filterbank @ spectrum
        mel_log = np.log(mel_energy + 1e-8)

        # Zero-crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame[:self.n_fft])))) / (2 * self.n_fft))

        # RMS energy
        rms = float(np.sqrt(np.mean(frame[:self.n_fft] ** 2)))

        # Spectral centroid
        freqs = np.arange(len(spectrum), dtype=np.float32)
        spec_sum = spectrum.sum() + 1e-8
        centroid = float(np.sum(freqs * spectrum) / spec_sum) / len(spectrum)

        return np.concatenate([mel_log, [zcr, rms, centroid]]).astype(np.float32)

    def extract_stream(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from a full audio stream.

        Args:
            audio: (N,) float32 audio samples.

        Returns:
            (n_frames, feature_dim) float32 feature matrix.
        """
        n_frames = max(1, (len(audio) - self.n_fft) // self.hop_length + 1)
        features = np.zeros((n_frames, self.feature_dim), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft]
            features[i] = self.extract_frame(frame)
        return features


# ── Audio SNN Encoder ─────────────────────────────────────────────────────

class AudioEncoder:
    """Audio perception: microphone → mel features → SNN → VSA temporal memory.

    Args:
        sample_rate: Audio sample rate.
        n_mels:      Mel bands.
        snn_neurons: SNN layer size.
        d_vsa:       Binary VSA dimension.
        seed:        Random seed.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        snn_neurons: int = 256,
        d_vsa: int = 2048,
        seed: int = 42,
    ) -> None:
        self.sample_rate = sample_rate
        self.features = AudioFeatureExtractor(
            sample_rate=sample_rate, n_mels=n_mels,
        )
        self.snn = SNNEncoder(
            d_input=self.features.feature_dim,
            n_neurons=snn_neurons,
            d_vsa=d_vsa,
            neuron_type="lif",
            tau=10.0,
            v_threshold=0.12,
            seed=seed,
        )
        self.snn.stdp_lr_potentiate = 0.0003
        self.snn.stdp_lr_depress = 0.00015
        self.snn.stdp_weight_clip = 0.3

    def encode_audio(self, audio: np.ndarray) -> np.ndarray:
        """Encode an audio clip into a single packed binary VSA vector.

        Args:
            audio: (N,) float32 audio samples.

        Returns:
            (words_per_vec,) uint32 packed binary temporal memory.
        """
        feat_matrix = self.features.extract_stream(audio)
        # Normalize features
        feat_std = np.std(feat_matrix, axis=0) + 1e-6
        feat_norm = feat_matrix / feat_std * 0.3
        return self.snn.encode_stream(feat_norm)

    def process_live(self, audio_chunk: np.ndarray) -> dict:
        """Process a live audio chunk (from microphone callback).

        Args:
            audio_chunk: (chunk_size,) float32 samples.

        Returns:
            Dict with features, spike_rate, neurochemistry.
        """
        feat = self.features.extract_frame(audio_chunk)

        # Normalize
        feat_norm = feat / (np.std(feat) + 1e-6) * 0.3

        spikes = self.snn.step(feat_norm)
        spike_rate = float(np.mean(spikes))

        # Audio-specific neurochemistry mapping
        rms = feat[-2]  # RMS energy
        zcr = feat[-3]  # Zero-crossing rate

        # Loud sudden sound → cortisol (startle)
        threat = float(np.clip(rms * 3.0 - 0.5, 0, 0.8))

        # Tonal content (low ZCR) → serotonin-positive valence
        # Noise (high ZCR) → neutral/negative
        tonality = float(np.clip(0.5 - zcr * 2.0, -0.3, 0.5))

        self.snn.neurochemistry.update(
            novelty=spike_rate,
            valence=tonality,
            threat=threat,
            focus=0.3 if rms > 0.1 else 0.0,
        )

        return {
            "features": feat,
            "spike_rate": spike_rate,
            "rms": float(rms),
            "zcr": float(zcr),
            "neurochemistry": self.snn.neurochemistry.to_dict(),
        }

    def reset(self) -> None:
        self.snn.reset()


# ── Live Microphone Capture ───────────────────────────────────────────────

class MicrophoneCapture:
    """Capture audio from a microphone via sounddevice.

    Args:
        device:      Audio device index (None = default).
        sample_rate: Sample rate in Hz.
        chunk_size:  Samples per callback.
    """

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 16000,
        chunk_size: int = 512,
    ) -> None:
        if _SD is None:
            raise ImportError("sounddevice required: pip install sounddevice")
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._stream = None
        self._buffer: list[np.ndarray] = []

    def _callback(self, indata, frames, time_info, status):
        self._buffer.append(indata[:, 0].copy().astype(np.float32))

    def start(self) -> None:
        """Start capturing audio."""
        self._buffer = []
        self._stream = _SD.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop capturing and return all recorded audio.

        Returns:
            (N,) float32 audio samples.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._buffer:
            return np.concatenate(self._buffer)
        return np.zeros(0, dtype=np.float32)

    def read_chunk(self) -> np.ndarray | None:
        """Read the latest chunk from the buffer (non-blocking).

        Returns:
            (chunk_size,) float32 or None if no data.
        """
        if self._buffer:
            return self._buffer.pop(0)
        return None

    def capture_duration(self, duration_s: float) -> np.ndarray:
        """Capture audio for a fixed duration.

        Returns:
            (N,) float32 audio samples.
        """
        self.start()
        time.sleep(duration_s)
        return self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
