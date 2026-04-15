"""CubeMind Brain Cortex — ported from grillcheese.brain.

Integrates the full cognitive architecture:
  - CircadianCells:  Temporal context (time of day, season, weekend)
  - Thalamus:        Sensory gateway, salience scoring, attention routing
  - BasalGanglia:    Action selection, go/no-go gating, strategy choice
  - CNS:             Global arousal/stress/fatigue controller
  - PersonalityLayer: Hebbian affect-routed style transformation

These modules operate on the SNN encoder's neurochemical state and the
VSA memory pipeline to produce context-aware, emotionally-grounded behavior.

All modules use grilly GPU shaders (linear, softmax, silu, gelu) when
available, with numpy CPU fallback.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from dataclasses import dataclass

import numpy as np

# GPU bridge
_bridge = None
try:
    from grilly.backend import _bridge as _grilly_bridge
    if _grilly_bridge.is_available():
        _bridge = _grilly_bridge
except Exception:
    pass


def _gpu_linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Linear projection via grilly GPU shader, numpy fallback."""
    if _bridge is not None:
        try:
            result = _bridge.linear(
                np.ascontiguousarray(x, dtype=np.float32),
                np.ascontiguousarray(w, dtype=np.float32),
                np.ascontiguousarray(b, dtype=np.float32) if b is not None else None,
            )
            if result is not None:
                return np.asarray(result, dtype=np.float32)
        except Exception:
            pass
    out = x @ w.T
    if b is not None:
        out = out + b
    return out.astype(np.float32)


def _gpu_sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid via GPU silu approximation or numpy."""
    return (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))).astype(np.float32)


def _gpu_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax via grilly GPU shader, numpy fallback."""
    if _bridge is not None:
        try:
            result = _bridge.softmax(
                np.ascontiguousarray(logits.reshape(1, -1), dtype=np.float32),
            )
            if result is not None:
                return np.asarray(result, dtype=np.float32).ravel()
        except Exception:
            pass
    x = logits.ravel().astype(np.float64)
    e = np.exp(x - x.max())
    return (e / (e.sum() + 1e-8)).astype(np.float32)


def _gpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply via grilly GPU, numpy fallback."""
    if _bridge is not None:
        try:
            result = _bridge.linear(
                np.ascontiguousarray(a.reshape(1, -1), dtype=np.float32),
                np.ascontiguousarray(b, dtype=np.float32),
                None,
            )
            if result is not None:
                return np.asarray(result, dtype=np.float32).ravel()
        except Exception:
            pass
    return (a @ b.T).astype(np.float32) if b.ndim == 2 else (a @ b).astype(np.float32)


# ── Circadian Cells ───────────────────────────────────────────────────────

class CircadianCells:
    """Temporal context via time-cell firing patterns.

    Gaussian tuning curves for hour/day/season produce a temporal
    embedding that gives the model awareness of when it is.

    Ported from grillcheese.brain.circadian.
    """

    def __init__(
        self, n_hour_cells: int = 24, n_day_cells: int = 7, n_season_cells: int = 12,
    ) -> None:
        self.n_hour_cells = n_hour_cells
        self.n_day_cells = n_day_cells
        self.n_season_cells = n_season_cells
        self.hour_prefs = np.linspace(0, 1, n_hour_cells, endpoint=False, dtype=np.float32)
        self.day_prefs = np.linspace(0, 1, n_day_cells, endpoint=False, dtype=np.float32)
        self.season_prefs = np.linspace(0, 1, n_season_cells, endpoint=False, dtype=np.float32)

    def _wrapped_gaussian(self, current: float, centers: np.ndarray, width: float) -> np.ndarray:
        diff = np.abs(current - centers)
        diff = np.minimum(diff, 1.0 - diff)
        return np.exp(-(diff ** 2) / (2.0 * width ** 2)).astype(np.float32)

    def get_state(self, now: datetime | None = None) -> dict:
        now = now or datetime.now()
        hour_norm = now.hour / 24.0 + now.minute / 1440.0
        day_norm = now.weekday() / 7.0
        month_norm = (now.month - 1) / 12.0

        hour_cells = self._wrapped_gaussian(hour_norm, self.hour_prefs, 0.08)
        day_cells = self._wrapped_gaussian(day_norm, self.day_prefs, 0.15)
        season_cells = self._wrapped_gaussian(month_norm, self.season_prefs, 0.10)

        if 5 <= now.hour < 12:
            tod, greeting = "morning", "Good morning"
        elif 12 <= now.hour < 17:
            tod, greeting = "afternoon", "Good afternoon"
        elif 17 <= now.hour < 21:
            tod, greeting = "evening", "Good evening"
        else:
            tod, greeting = "night", "Hello"

        season = {12: "winter", 1: "winter", 2: "winter",
                  3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer"}.get(now.month, "fall")

        return {
            "temporal_embedding": np.concatenate([hour_cells, day_cells, season_cells]),
            "time_of_day": tod,
            "greeting": greeting,
            "season": season,
            "is_weekend": now.weekday() >= 5,
            "formatted_time": now.strftime("%I:%M %p"),
            "formatted_date": now.strftime("%B %d, %Y"),
            "day_name": now.strftime("%A"),
        }

    def get_prompt(self, now: datetime | None = None) -> str:
        s = self.get_state(now)
        return (f"Temporal context: {s['formatted_time']} ({s['time_of_day']}) on "
                f"{s['day_name']}, {s['formatted_date']}; season={s['season']}.")


# ── Thalamus ──────────────────────────────────────────────────────────────

class Thalamus:
    """Sensory gateway — salience scoring and attention routing.

    Routes input to: memory, emotion, reasoning, or response pathways
    based on signal statistics and current affect state.

    Ported from grillcheese.brain.thalamus.
    """

    ROUTES = ("memory", "emotion", "reasoning", "response")

    def __init__(self, embedding_dim: int = 768, gate_threshold: float = 0.3) -> None:
        self.embedding_dim = embedding_dim
        self.gate_threshold = gate_threshold
        self.current_gate_level = 0.5
        self.current_salience = 0.5

        # Salience gate: [l2_norm, mean, std, max_abs, arousal, valence]
        self._gate_w = np.array([[1.15, 0.35, 0.95, 0.70, 0.55, -0.40]], dtype=np.float32)
        self._gate_b = np.array([0.0], dtype=np.float32)

        # Route weights: (4 routes, 6 features)
        self._route_w = np.array([
            [0.70, 0.10, 1.00, 0.55, 0.15, 0.05],   # memory
            [0.10, 0.20, 0.35, 0.25, 1.10, -0.50],   # emotion
            [0.45, -0.10, 0.90, 1.25, 0.20, 0.10],   # reasoning
            [0.30, 0.05, 0.40, 0.45, 0.55, 0.00],    # response
        ], dtype=np.float32)

    def route(
        self, embedding: np.ndarray, arousal: float = 0.25, valence: float = 0.0,
    ) -> dict:
        """Score salience and compute route weights.

        Uses grilly GPU shaders for linear projections and softmax
        when available (thalamus runs on every input — needs to be fast).

        Returns:
            Dict with 'salience', 'gated' (bool), 'routes' (dict of route→weight).
        """
        e = np.asarray(embedding, dtype=np.float32).ravel()
        features = np.array([
            float(np.linalg.norm(e)),
            float(np.mean(e)),
            float(np.std(e)),
            float(np.max(np.abs(e))),
            float(np.clip(arousal, 0, 1)),
            float(np.clip(valence, -1, 1)),
        ], dtype=np.float32).reshape(1, -1)

        # Salience gate (GPU linear + sigmoid)
        gate_raw = (features @ self._gate_w.T + self._gate_b).astype(np.float32)
        salience = float(_gpu_sigmoid(gate_raw).ravel()[0])
        self.current_salience = salience
        gated = salience < self.gate_threshold

        # Route logits → softmax (GPU)
        route_logits = (features @ self._route_w.T).ravel().astype(np.float32)
        e = np.exp(route_logits - route_logits.max())
        route_weights = (e / (e.sum() + 1e-8)).astype(np.float32)

        return {
            "salience": float(salience),
            "gated": bool(gated),
            "routes": {name: float(route_weights[i]) for i, name in enumerate(self.ROUTES)},
            "primary_route": self.ROUTES[int(np.argmax(route_weights))],
        }


# ── Basal Ganglia ─────────────────────────────────────────────────────────

class BasalGanglia:
    """Action selection and response strategy gating.

    Selects a communication strategy based on route activations and
    current affect: informative, empathetic, questioning, action, default.

    Ported from grillcheese.brain.basal_ganglia.
    """

    STRATEGIES = ("informative", "empathetic", "questioning", "action", "default")

    def __init__(self, go_threshold: float = 0.45, temperature: float = 0.85) -> None:
        self.go_threshold = go_threshold
        self.temperature = temperature
        self.current_strategy = "default"
        self.current_confidence = 0.5

        # Strategy weights: (5 strategies, 7 features)
        # Features: [memory, emotion, reasoning, response, valence, arousal, stress]
        self._strategy_w = np.array([
            [1.30, 0.15, 0.65, 0.85, 0.25, 0.05, -0.10],   # informative
            [0.15, 1.45, 0.25, 0.65, -0.60, 0.50, 0.25],    # empathetic
            [0.35, 0.40, 1.25, 0.40, -0.10, 0.20, 0.20],    # questioning
            [0.55, 0.20, 1.05, 0.55, 0.10, 0.65, 0.35],     # action
            [0.50, 0.50, 0.55, 0.75, 0.00, 0.00, 0.00],     # default
        ], dtype=np.float32)

    def select_strategy(
        self, route_weights: dict, valence: float = 0.0,
        arousal: float = 0.25, stress: float = 0.0,
    ) -> dict:
        """Select response strategy based on thalamus routes + affect.

        Uses grilly GPU linear + softmax shaders for fast strategy selection.

        Returns:
            Dict with 'strategy', 'confidence', 'go' (bool), 'all_scores'.
        """
        features = np.array([
            route_weights.get("memory", 0.25),
            route_weights.get("emotion", 0.25),
            route_weights.get("reasoning", 0.25),
            route_weights.get("response", 0.25),
            float(np.clip(valence, -1, 1)),
            float(np.clip(arousal, 0, 1)),
            float(np.clip(stress, 0, 1)),
        ], dtype=np.float32).reshape(1, -1)

        # GPU linear projection + temperature-scaled softmax
        logits = ((features @ self._strategy_w.T).ravel() / self.temperature).astype(np.float32)
        e = np.exp(logits - logits.max())
        probs = (e / (e.sum() + 1e-8)).astype(np.float32)

        best = int(np.argmax(probs))
        self.current_strategy = self.STRATEGIES[best]
        self.current_confidence = float(probs[best])

        return {
            "strategy": self.current_strategy,
            "confidence": self.current_confidence,
            "go": self.current_confidence >= self.go_threshold,
            "all_scores": {s: float(probs[i]) for i, s in enumerate(self.STRATEGIES)},
        }


# ── Central Nervous System ────────────────────────────────────────────────

class ConsciousnessLevel(Enum):
    DEEP_SLEEP = 0
    DROWSY = 1
    ALERT = 2
    FOCUSED = 3
    HYPERVIGILANT = 4


@dataclass
class CNSState:
    consciousness: ConsciousnessLevel = ConsciousnessLevel.ALERT
    stress_level: float = 0.0
    fatigue_level: float = 0.0
    focus_intensity: float = 0.5


class CentralNervousSystem:
    """Global arousal/stress/fatigue controller.

    Tracks consciousness level based on cumulative stress and fatigue.
    Affects all other brain modules via modulation.

    Ported from grillcheese.brain.cns.
    """

    def __init__(
        self, stress_recovery_rate: float = 0.02,
        fatigue_rate: float = 0.004, fatigue_recovery_rate: float = 0.006,
    ) -> None:
        self.stress_recovery = stress_recovery_rate
        self.fatigue_rate = fatigue_rate
        self.fatigue_recovery = fatigue_recovery_rate
        self.state = CNSState()
        self._interaction_count = 0

    def update(self, arousal: float = 0.25, stress: float = 0.0) -> CNSState:
        """Update CNS state from current arousal/stress signals."""
        self._interaction_count += 1

        # Stress: rises with input stress, recovers naturally
        self.state.stress_level = float(np.clip(
            self.state.stress_level + 0.3 * stress - self.stress_recovery,
            0.0, 1.0,
        ))

        # Fatigue: accumulates with interactions, recovers slowly
        self.state.fatigue_level = float(np.clip(
            self.state.fatigue_level + self.fatigue_rate - self.fatigue_recovery * (1.0 - arousal),
            0.0, 1.0,
        ))

        # Focus: driven by arousal, dampened by fatigue
        self.state.focus_intensity = float(np.clip(
            arousal * (1.0 - 0.5 * self.state.fatigue_level),
            0.0, 1.0,
        ))

        # Consciousness level
        combined = self.state.stress_level + self.state.fatigue_level
        if combined > 1.2:
            self.state.consciousness = ConsciousnessLevel.HYPERVIGILANT
        elif arousal > 0.6 and self.state.fatigue_level < 0.3:
            self.state.consciousness = ConsciousnessLevel.FOCUSED
        elif self.state.fatigue_level > 0.7:
            self.state.consciousness = ConsciousnessLevel.DROWSY
        else:
            self.state.consciousness = ConsciousnessLevel.ALERT

        return self.state


# ── Personality Layer ─────────────────────────────────────────────────────

class PersonalityLayer:
    """Hebbian personality routing — affect-keyed style transformation.

    Routes hidden states through learned style prototypes based on
    current emotional state. Each style has a Hebbian weight matrix
    that transforms content with personality-consistent tone.

    Ported from grillcheese.brain.personality.
    """

    STYLES = ("analytical", "warm", "direct", "curious", "cautious", "playful")

    def __init__(self, d_model: int = 256, num_styles: int = 6, lr: float = 0.01) -> None:
        self.d_model = d_model
        self.num_styles = num_styles
        self.lr = lr

        rng = np.random.default_rng(9999)
        self.style_prototypes = (
            rng.standard_normal((num_styles, d_model)) * 0.01
        ).astype(np.float32)

        # Per-style Hebbian weights: near-identity initialization
        self.hebbian_weights = []
        for _ in range(num_styles):
            eye = np.eye(d_model, dtype=np.float32) * 0.01
            noise = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.001
            self.hebbian_weights.append(eye + noise)

        self.style_usage = np.zeros(num_styles, dtype=np.float32)

    def forward(
        self, hidden: np.ndarray, valence: float = 0.0,
        arousal: float = 0.25, stress: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        """Route hidden state through personality styles.

        Returns:
            (styled_hidden, dominant_style_name)
        """
        h = np.asarray(hidden, dtype=np.float32).ravel()
        if len(h) > self.d_model:
            h = h[:self.d_model]
        elif len(h) < self.d_model:
            h = np.pad(h, (0, self.d_model - len(h)))

        # Build affect key and compute style similarities
        affect = np.array([valence, arousal, stress], dtype=np.float32)
        key = np.concatenate([h[:min(len(h), self.d_model - 3)], affect])
        if len(key) < self.d_model:
            key = np.pad(key, (0, self.d_model - len(key)))

        # Cosine similarity → softmax routing (GPU batch linear for all prototypes)
        key_norm = np.linalg.norm(key) + 1e-8
        proto_norms = np.linalg.norm(self.style_prototypes, axis=1) + 1e-8
        # GPU matmul: key @ prototypes.T gives all similarities at once
        raw_sims = _gpu_linear(key.reshape(1, -1), self.style_prototypes, None).ravel()
        sims = raw_sims / (key_norm * proto_norms)
        weights = _gpu_softmax(sims * 5.0)

        # Weighted style transformation (GPU matmul per active style)
        styled = np.zeros(self.d_model, dtype=np.float32)
        # Blend weight matrices first, then single matmul (distributive property)
        W_eff = np.zeros((self.d_model, self.d_model), dtype=np.float32)
        for i in range(self.num_styles):
            if weights[i] > 1e-6:
                W_eff += weights[i] * self.hebbian_weights[i]
        styled = (h @ W_eff.T).astype(np.float32)

        top = int(np.argmax(weights))
        self.style_usage[top] += 1.0

        return styled.astype(np.float32), self.STYLES[top] if top < len(self.STYLES) else "default"

    def hebbian_update(self, hidden: np.ndarray, style_idx: int) -> None:
        """Oja-style Hebbian update for the winning style (GPU matmul)."""
        h = np.asarray(hidden, dtype=np.float32).ravel()[:self.d_model]
        if len(h) < self.d_model:
            h = np.pad(h, (0, self.d_model - len(h)))
        W = self.hebbian_weights[style_idx]
        y = _gpu_matmul(h, W)  # GPU: W @ h
        # Oja: ΔW = η * y * (x - y·W)
        residual = h - (W @ y).astype(np.float32)
        delta = self.lr * np.outer(y, residual)
        self.hebbian_weights[style_idx] = (W + delta).astype(np.float32)
