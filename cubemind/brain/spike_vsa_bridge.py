"""Spike↔VSA Bridge — Convert between spiking and VSA representations.

Uses grilly's bridge shaders for GPU-accelerated conversion:
  - spikes_to_continuous: spike trains → dense vectors → VSA block-codes
  - continuous_to_spikes: VSA block-codes → dense vectors → Poisson spike trains

This is the glue between the SNN brain (GIFNeuron, Synapsis, SNNFFN)
and the VSA reasoning system (block-codes, bind, bundle, similarity).

Pipeline:
  VSA → continuous_to_spikes → GIFNeuron processing → spikes_to_continuous → VSA

Also provides spike_to_block_code for direct SNN→VSA projection.
"""

from __future__ import annotations

import numpy as np

from cubemind.core.registry import register
from cubemind.ops.block_codes import BlockCodes

# grilly bridge ops
_bridge_available = False
try:
    from grilly.functional.bridge import (
        continuous_to_spikes as _gpu_c2s,
        spikes_to_continuous as _gpu_s2c,
    )
    _bridge_available = True
except ImportError:
    pass


@register("bridge", "spike_vsa")
class SpikeVSABridge:
    """Bidirectional bridge between spike trains and VSA block-codes.

    Args:
        k: VSA blocks.
        l: Block length.
        num_timesteps: Spike timesteps for encoding/decoding.
        coding: Spike coding method ('rate', 'temporal', 'phase').
        seed: Random seed for projection matrix.
    """

    def __init__(
        self,
        k: int = 80,
        l: int = 128,
        num_timesteps: int = 10,
        coding: str = "rate",
        seed: int = 42,
    ) -> None:
        self.k = k
        self.l = l
        self.d_vsa = k * l
        self.num_timesteps = num_timesteps
        self.coding = coding
        self.bc = BlockCodes(k=k, l=l)

        # Random projection for dimensionality matching
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal(
            (self.d_vsa, self.d_vsa)
        ).astype(np.float32) * np.sqrt(2.0 / self.d_vsa)

    def vsa_to_spikes(
        self,
        hv: np.ndarray,
        num_timesteps: int | None = None,
    ) -> np.ndarray:
        """Convert VSA block-code to spike train.

        Args:
            hv: Block-code (k, l) or flat (d_vsa,).
            num_timesteps: Override default timesteps.

        Returns:
            Spike train (num_timesteps, d_vsa) as float32.
        """
        T = num_timesteps or self.num_timesteps
        flat = self.bc.to_flat(hv).astype(np.float32) if hv.ndim == 2 else hv.astype(np.float32)

        if _bridge_available:
            try:
                return _gpu_c2s(
                    flat, num_timesteps=T, coding=self.coding
                )
            except Exception:
                pass

        # Numpy fallback: Poisson rate coding
        rates = np.clip(np.abs(flat), 0, 1)
        rng = np.random.default_rng()
        spikes = (rng.random((T, len(flat))) < rates[np.newaxis, :]).astype(np.float32)
        # Preserve sign information
        spikes *= np.sign(flat)[np.newaxis, :]
        return spikes

    def spikes_to_vsa(
        self,
        spikes: np.ndarray,
    ) -> np.ndarray:
        """Convert spike train to VSA block-code.

        Args:
            spikes: (num_timesteps, d) or (batch, num_timesteps, d) spike train.

        Returns:
            Block-code (k, l) or (batch, k, l).
        """
        batched = spikes.ndim == 3
        if not batched:
            spikes = spikes[np.newaxis, :]

        results = []
        for b in range(spikes.shape[0]):
            s = spikes[b]  # (T, d)

            if _bridge_available:
                try:
                    continuous = _gpu_s2c(s, coding=self.coding)
                    if continuous is not None:
                        # Project to d_vsa if needed
                        if len(continuous) != self.d_vsa:
                            continuous = continuous[:self.d_vsa] if len(continuous) > self.d_vsa \
                                else np.pad(continuous, (0, self.d_vsa - len(continuous)))
                        hv = self.bc.discretize(
                            continuous.reshape(self.k, self.l).astype(np.float32))
                        results.append(hv)
                        continue
                except Exception:
                    pass

            # Numpy fallback: rate decoding (mean over time)
            continuous = s.mean(axis=0).astype(np.float32)
            if len(continuous) != self.d_vsa:
                if len(continuous) > self.d_vsa:
                    continuous = continuous[:self.d_vsa]
                else:
                    padded = np.zeros(self.d_vsa, dtype=np.float32)
                    padded[:len(continuous)] = continuous
                    continuous = padded

            hv = self.bc.discretize(continuous.reshape(self.k, self.l))
            results.append(hv)

        if batched:
            return np.stack(results)
        return results[0]

    def spike_to_block_code(
        self,
        spike_counts: np.ndarray,
    ) -> np.ndarray:
        """Direct conversion: per-neuron spike counts → block-code.

        Faster than full spike train conversion when you only have
        aggregated counts (e.g., from GIFNeuron mean pooling).

        Args:
            spike_counts: (d,) spike count per neuron.

        Returns:
            Block-code (k, l).
        """
        flat = spike_counts.astype(np.float32).ravel()
        if len(flat) != self.d_vsa:
            if len(flat) > self.d_vsa:
                flat = flat[:self.d_vsa]
            else:
                padded = np.zeros(self.d_vsa, dtype=np.float32)
                padded[:len(flat)] = flat
                flat = padded
        return self.bc.discretize(flat.reshape(self.k, self.l))

    def round_trip(self, hv: np.ndarray) -> np.ndarray:
        """VSA → spikes → VSA round trip. Tests fidelity."""
        spikes = self.vsa_to_spikes(hv)
        return self.spikes_to_vsa(spikes)
