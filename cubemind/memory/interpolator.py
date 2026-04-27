"""Temporal memory interpolator -- Hamiltonian flow for sparse-sample recall.

Reconstructs intermediate or extrapolated memory states from a pair of
observed snapshots using one of two genuinely distinct operators:

    linear      : standard convex combination (1 - t) * M0 + t * M1
    hamiltonian : rank-1 unitary flow exp(-i H t) on the analytic signal,
                  giving reversible dynamics + energy conservation +
                  extrapolation past observed endpoints.

The original paper (aura_mono/docs/papers/temporal_memory_interpolation.md)
reported four modes: linear, fourier, hilbert, hamiltonian. We dropped
fourier and hilbert after proving they are bit-exact equivalent to linear
on real-valued inputs: FFT is linear, so
IFFT(lerp(FFT(M0), FFT(M1))) = lerp(M0, M1); and the Hilbert transform is
linear, so Re(lerp(A0, A1)) = lerp(M0, M1). See
``docs/papers/memory_as_manifold.md`` section 3 for the derivation and
``scripts/test_temporal_interpolator.py`` for the numerical verification
(fourier - linear and hilbert - linear both differ by ~1e-16, machine
epsilon).

Hamiltonian mode is the genuinely-distinct operator. Under the paper's
rank-1 Hermitian H = dA * dA^dagger / (||dA||**2 + eps), the unitary flow
exp(-i H t) has a closed form because H has eigenvalues {1, 0, ..., 0}:

    A(t) = A0 + proj_dA(A0) * (exp(-i t) - 1)

where proj_dA(A0) is the projection of A0 onto dA. This is O(n) per
call, avoiding the O(n^3) scipy.linalg.expm route. Reversibility,
unitarity, and shape preservation are verified to machine precision in
the test script.

Public API:

    from cubemind.memory.interpolator import TemporalMemoryInterpolator

    interp = TemporalMemoryInterpolator()
    M_half = interp.interpolate(M0, M1, t=0.5, mode="hamiltonian")

Registration:

    from cubemind.core.registry import get

    interp_cls = get("memory", "temporal_interpolator")
    interp = interp_cls()

Non-goal: midpoint fidelity. Hamiltonian does NOT satisfy A(1) = A1 at
t=1 in general -- it rotates A0 along the dA direction by angle t
rather than moving it to A1. This is by design; the mechanism's value
is reversibility and energy conservation, not endpoint matching. For
endpoint-matching interpolation use ``mode="linear"``.

Future work (not yet implemented, see ``docs/papers/memory_as_manifold.md``
section 6): polar-domain interpolation, Wasserstein-2 optimal transport
for localized features, learned flow matching. These are the genuinely
nonlinear operators that would extend the tool-set beyond linear+hamiltonian.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import hilbert as _hilbert

from cubemind.core.registry import register

Mode = Literal["linear", "hamiltonian"]


@register("memory", "temporal_interpolator")
class TemporalMemoryInterpolator:
    """Sparse-sample memory recall via linear lerp or Hamiltonian flow.

    Args:
        epsilon: Regularizer added to ``||dA||**2`` to avoid division by
            zero when the two observations are identical. Default 1e-12.

    Numerical properties (verified in
    ``scripts/test_temporal_interpolator.py``):

    * Linear mode: bit-exact endpoint reconstruction at t=0 and t=1.
    * Hamiltonian mode at t=0: returns M0 to machine precision.
    * Hamiltonian reversibility: forward-then-backward returns origin
      to max error ~5e-15 for n=128.
    * Hamiltonian unitarity: ||A(t)|| conserved across all t to
      max relative error ~6e-16.

    Shape: input ``M0, M1`` are expected to be 1-D ``np.ndarray`` of the
    same shape ``(n,)``. Output matches input shape.
    """

    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = float(epsilon)

    def interpolate(
        self,
        M0: np.ndarray,
        M1: np.ndarray,
        t: float,
        mode: Mode = "hamiltonian",
    ) -> np.ndarray:
        """Reconstruct a memory state at time ``t`` between (or past) M0, M1.

        Args:
            M0: Observed memory state at reference time t=0.
            M1: Observed memory state at reference time t=1.
            t: Target time. For ``linear`` mode, clipped to [0, 1].
                For ``hamiltonian`` mode, any real value is valid
                (negative = extrapolate backward past M0,
                greater than 1 = extrapolate forward past M1).
            mode: ``"linear"`` for convex combination, ``"hamiltonian"``
                for unitary flow. Default ``"hamiltonian"``.

        Returns:
            Reconstructed memory state, same shape as ``M0``.

        Raises:
            ValueError: If ``mode`` is not one of the supported modes,
                or if ``M0, M1`` shapes differ.
        """
        if M0.shape != M1.shape:
            raise ValueError(
                f"shape mismatch: M0 is {M0.shape}, M1 is {M1.shape}"
            )

        if mode == "linear":
            alpha = float(np.clip(t, 0.0, 1.0))
            return (1.0 - alpha) * M0 + alpha * M1

        if mode == "hamiltonian":
            return self._hamiltonian_flow(M0, M1, float(t))

        raise ValueError(
            f"Unknown interpolation mode: {mode!r}. "
            f"Expected 'linear' or 'hamiltonian'. "
            f"(Paper-1's 'fourier' and 'hilbert' modes were dropped: "
            f"both are bit-exact equivalent to 'linear' on real inputs; "
            f"see docs/papers/memory_as_manifold.md section 3.)"
        )

    def _hamiltonian_flow(
        self,
        M0: np.ndarray,
        M1: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Closed-form rank-1 Hamiltonian evolution.

        Math (see docs/papers/memory_as_manifold.md section 4.2):
            A0, A1 = hilbert(M0), hilbert(M1)
            dA     = A1 - A0
            H      = dA * dA^dagger / (||dA||**2 + eps)   # rank-1 Hermitian
            A(t)   = A0 + proj_dA(A0) * (exp(-i t) - 1)
            M(t)   = Re[A(t)]
        """
        A0 = _hilbert(M0, axis=0)
        A1 = _hilbert(M1, axis=0)
        dA = A1 - A0
        norm_sq = float(np.real(np.vdot(dA, dA)))
        proj_coeff = np.vdot(dA, A0) / (norm_sq + self.epsilon)
        proj_A0 = proj_coeff * dA
        phase = np.exp(-1j * t) - 1.0
        A_t = A0 + proj_A0 * phase
        return np.real(A_t)

    # Note: backward evolution is not a separate method. To evolve from
    # M0 backward past the origin, pass a negative t to ``interpolate``:
    #
    #     M_past = interp.interpolate(M0, M1, t=-0.5, mode="hamiltonian")
    #
    # Reversibility of the Hamiltonian mode holds at the analytic-signal
    # level (forward-then-backward in complex space returns the origin
    # to machine precision). A real-valued round-trip -- going through
    # Re[] and then re-applying hilbert() -- is lossy and is NOT exactly
    # reversible; callers who need perfect reversibility must stay in
    # the analytic-signal domain (access ``_hamiltonian_flow`` internals
    # or drop down to scipy.signal.hilbert directly).
