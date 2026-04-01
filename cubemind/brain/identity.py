"""Identity — Persistent sense of self for CubeMind.

The brain's identity is a VSA hypervector that encodes:
  - Personality traits (openness, curiosity, caution, empathy, ...)
  - Accumulated experience (Oja-updated from every interaction)
  - Emotional baseline (default neurochemical set-point)
  - Preferences (learned from feedback)

The identity vector modulates:
  - Perception: bind(input, identity) → personalized encoding
  - Memory: identity-weighted retrieval (prefer memories aligned with self)
  - Language: injected into LLM as context (who am I?)
  - Neurochemistry: baseline hormone levels
  - Routing: personality-dependent expert selection

The identity persists across sessions via save/load.
It evolves slowly over time via Oja's rule — not overwritten, adapted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from cubemind.ops.block_codes import BlockCodes

try:
    from cubemind.core import K_BLOCKS, L_BLOCK
except ImportError:
    K_BLOCKS, L_BLOCK = 80, 128


class Identity:
    """CubeMind's persistent sense of self.

    Args:
        name: Identity name.
        k: VSA blocks.
        l: Block length.
        traits: Dict of personality traits (name → float in [-1, 1]).
        baseline_neurochemistry: Default hormone levels.
        oja_eta: Learning rate for identity adaptation.
        seed: Random seed for initial identity vector.
    """

    def __init__(
        self,
        name: str = "CubeMind",
        k: int = K_BLOCKS,
        l: int = L_BLOCK,
        traits: Dict[str, float] | None = None,
        baseline_neurochemistry: Dict[str, float] | None = None,
        oja_eta: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.k = k
        self.l = l
        self.oja_eta = oja_eta
        self.bc = BlockCodes(k=k, l=l)

        # Personality traits
        self.traits = traits or {
            "curiosity": 0.8,
            "empathy": 0.7,
            "caution": 0.3,
            "creativity": 0.9,
            "persistence": 0.8,
            "openness": 0.9,
            "assertiveness": 0.5,
        }

        # Neurochemical baseline (default emotional set-point)
        self.baseline_neurochemistry = baseline_neurochemistry or {
            "dopamine": 0.5,
            "serotonin": 0.6,
            "cortisol": 0.2,
            "noradrenaline": 0.3,
            "oxytocin": 0.4,
        }

        # Identity hypervector: encodes who I am in VSA space
        # Built by binding trait vectors together
        rng = np.random.default_rng(seed)
        self.identity_hv = self._build_identity_hv(rng)

        # Experience accumulator: slowly adapts with every interaction
        self.experience_hv = self.identity_hv.copy()
        self.interaction_count = 0
        self.created_at = time.time()
        self.last_saved = None

    def _build_identity_hv(self, rng: np.random.Generator) -> np.ndarray:
        """Build identity vector from traits via VSA binding."""
        result = np.zeros((self.k, self.l), dtype=np.float32)

        for i, (trait_name, trait_value) in enumerate(self.traits.items()):
            # Each trait gets a unique random vector
            trait_key = self.bc.random_discrete(seed=hash(trait_name) % (2**31))
            # Modulate by trait value: scale the contribution
            result += trait_value * trait_key.astype(np.float32)

        return self.bc.discretize(result)

    def modulate_input(self, input_hv: np.ndarray) -> np.ndarray:
        """Personalize an input by binding with identity.

        The same input will produce different encodings for different identities.

        Args:
            input_hv: Input block-code (k, l).

        Returns:
            Identity-modulated block-code (k, l).
        """
        return self.bc.bind(input_hv, self.identity_hv)

    def modulate_retrieval(
        self,
        query_features: np.ndarray,
        strength: float = 0.1,
    ) -> np.ndarray:
        """Bias memory retrieval toward identity-aligned memories.

        Args:
            query_features: Query vector (d,).
            strength: How much identity influences retrieval.

        Returns:
            Identity-biased query vector.
        """
        id_flat = self.bc.to_flat(self.experience_hv).astype(np.float32)
        # Truncate/pad to match query dim
        d = len(query_features)
        if len(id_flat) > d:
            id_component = id_flat[:d]
        else:
            id_component = np.zeros(d, dtype=np.float32)
            id_component[:len(id_flat)] = id_flat

        # Normalize
        id_norm = np.linalg.norm(id_component)
        if id_norm > 0:
            id_component /= id_norm

        return query_features + strength * id_component

    @property
    def effective_eta(self) -> float:
        """Learning rate decays with experience — personality consolidates.

        Young brain: high eta → fluid personality, easily shaped.
        Mature brain: low eta → stable personality, resistant to change.

        Decay: eta_effective = eta_base / (1 + interactions / consolidation_rate)
        """
        consolidation_rate = 1000  # Half-life in interactions
        return self.oja_eta / (1.0 + self.interaction_count / consolidation_rate)

    @property
    def maturity(self) -> float:
        """How consolidated is the personality? 0=infant, 1=fully mature."""
        return 1.0 - 1.0 / (1.0 + self.interaction_count / 1000.0)

    def adapt(self, experience_hv: np.ndarray) -> None:
        """Adapt identity based on new experience (Oja's rule).

        Personality is fluid at start and consolidates with time.
        Early interactions shape identity strongly; later ones barely nudge it.

        Args:
            experience_hv: Block-code from current experience (k, l).
        """
        self.interaction_count += 1
        eta = self.effective_eta

        # Oja update on the experience accumulator
        exp_flat = self.bc.to_flat(self.experience_hv).astype(np.float32)
        new_flat = self.bc.to_flat(experience_hv).astype(np.float32)

        y = float(np.dot(exp_flat, new_flat))
        exp_flat = exp_flat + eta * y * (new_flat - y * exp_flat)

        self.experience_hv = self.bc.discretize(
            exp_flat.reshape(self.k, self.l))

    def get_system_prompt(self) -> str:
        """Generate LLM system prompt from identity."""
        trait_strs = [f"{t}: {v:.1f}" for t, v in self.traits.items()]
        return (
            f"You are {self.name}, a neuro-vector-symbolic cognitive architecture. "
            f"Your personality traits: {', '.join(trait_strs)}. "
            f"You have experienced {self.interaction_count} interactions. "
            f"You perceive through bio-inspired vision and hearing, "
            f"reason through spiking neural networks, and remember via "
            f"hippocampal episodic memory with place and time cells. "
            f"Respond naturally based on your perceptions and memories."
        )

    def similarity_to(self, other_hv: np.ndarray) -> float:
        """How similar is something to my identity?"""
        return float(self.bc.similarity(self.experience_hv, other_hv))

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save identity to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            identity_hv=self.identity_hv,
            experience_hv=self.experience_hv,
        )

        # Save metadata as JSON alongside
        meta_path = path.with_suffix(".json")
        meta = {
            "name": self.name,
            "k": self.k,
            "l": self.l,
            "traits": self.traits,
            "baseline_neurochemistry": self.baseline_neurochemistry,
            "oja_eta": self.oja_eta,
            "interaction_count": self.interaction_count,
            "created_at": self.created_at,
            "saved_at": time.time(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.last_saved = time.time()

    @classmethod
    def load(cls, path: str | Path) -> "Identity":
        """Load identity from disk."""
        path = Path(path)

        data = np.load(path)
        meta_path = path.with_suffix(".json")

        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        identity = cls(
            name=meta.get("name", "CubeMind"),
            k=meta.get("k", K_BLOCKS),
            l=meta.get("l", L_BLOCK),
            traits=meta.get("traits"),
            baseline_neurochemistry=meta.get("baseline_neurochemistry"),
            oja_eta=meta.get("oja_eta", 0.001),
        )

        identity.identity_hv = data["identity_hv"]
        identity.experience_hv = data["experience_hv"]
        identity.interaction_count = meta.get("interaction_count", 0)
        identity.created_at = meta.get("created_at", time.time())

        return identity

    def __repr__(self) -> str:
        return (f"Identity(name={self.name!r}, "
                f"interactions={self.interaction_count}, "
                f"traits={len(self.traits)})")
