"""Affective Graph Message Passing via Hormonal Modulation.

Dynamically controls the blending factor alpha in VS-Graph message
passing using the 4-hormone ODE system from the SNN perceptual layer,
creating an emotionally-modulated graph reasoner.

The neurochemical state maps to graph reasoning strategy:
  - High cortisol (threat/stress) -> alpha ~0.7: trust self, consolidate
    existing beliefs, resist neighbor influence.
  - High dopamine (reward/curiosity) -> alpha ~0.3: trust neighbors,
    explore new connections, accept novel information.

This implements the "affect as cognitive control" hypothesis: emotions
don't corrupt reasoning, they tune the exploration-exploitation tradeoff
in graph message passing.

References:
  - Poursiami et al., "VS-Graph", 2025 (associative message passing).
  - Damasio, "Descartes' Error", 1994 (somatic marker hypothesis).
  - Doya, "Modulators of decision making", Nature Neuroscience 2008
    (dopamine = reward prediction, cortisol = urgency/threat).
"""

from __future__ import annotations

import numpy as np

from cubemind.experimental.vs_graph import associative_message_passing
from cubemind.perception.snn import NeurochemicalState


def affective_alpha(state: NeurochemicalState) -> float:
    """Compute dynamic blending alpha from neurochemical state.

    alpha controls self-vs-neighbor blend:
        alpha * self + (1 - alpha) * neighbor

    High alpha -> trust self (consolidate under stress).
    Low alpha -> trust neighbors (explore under curiosity).

    The mapping:
      alpha = 0.3 + 0.4 * (cortisol / (dopamine + cortisol + eps))

    guarantees alpha in [0.3, 0.7]:
      - Pure cortisol: 0.3 + 0.4 * 1.0 = 0.7
      - Pure dopamine: 0.3 + 0.4 * 0.0 = 0.3
      - Balanced: ~0.5

    Args:
        state: Current neurochemical state (dopamine, cortisol, etc.).

    Returns:
        alpha: Blending factor in [0.3, 0.7].
    """
    d = state.dopamine
    c = state.cortisol
    # Cortisol increases alpha (self-trust), dopamine decreases it
    return 0.3 + 0.4 * (c / (d + c + 1e-8))


def affective_message_passing(
    node_hvs: np.ndarray,
    adjacency: np.ndarray,
    state: NeurochemicalState,
    L: int = 2,
) -> np.ndarray:
    """Message passing with hormone-modulated alpha.

    Wraps VS-Graph associative_message_passing with a dynamic alpha
    derived from the current neurochemical state.

    Args:
        node_hvs: (N, d) node hypervectors.
        adjacency: (N, N) adjacency matrix.
        state: Current neurochemical state.
        L: Number of message passing layers.

    Returns:
        (N, d) refined node hypervectors.
    """
    alpha = affective_alpha(state)
    return associative_message_passing(
        node_hvs, adjacency, L=L, alpha=alpha
    )
