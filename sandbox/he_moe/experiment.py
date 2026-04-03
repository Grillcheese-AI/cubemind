"""HE-MoE: Hilbert-Electrostatic Mixture of Experts.

Hypothesis-driven experiment. Each test validates one hypothesis
from HYPOTHESES.md, building from simplest (H1) to full system (H12).

Run: uv run pytest sandbox/he_moe/experiment.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENTS (isolated, tested individually)
# ═══════════════════════════════════════════════════════════════════════════════


def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    diff = x - y
    return float(np.exp(-np.dot(diff, diff) / (2 * sigma * sigma)))


def rkhs_distance_sq(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    return 2.0 - 2.0 * rbf_kernel(x, y, sigma)


class RandomFourierFeatures:
    def __init__(self, d_input: int, d_rff: int = 128, sigma: float = 1.0,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1.0 / sigma, (d_rff, d_input)).astype(np.float32)
        self.b = rng.uniform(0, 2 * np.pi, d_rff).astype(np.float32)
        self.scale = np.sqrt(2.0 / d_rff)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (self.scale * np.cos(self.W @ x + self.b)).astype(np.float32)


class ChargedExpert:
    def __init__(self, d_input: int, d_output: int, charge: float = 1.0,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.d_input = d_input
        self.d_output = d_output
        self.charge = charge
        self.mu = rng.standard_normal(d_input).astype(np.float32) * 0.1
        std = np.sqrt(2.0 / (d_input + d_output))
        self.w = rng.normal(0, std, (d_output, d_input)).astype(np.float32)
        self.a_trace = 0.0
        self.e_trace = np.zeros(d_output, dtype=np.float32)
        self.gamma_a = 0.95
        self.gamma_e = 0.95
        self.n_uses = 0
        self.cumulative_error_sign = 0.0

    def forward(self, x):
        return (self.w @ x).astype(np.float32)

    def attraction_score(self, x, sigma=1.0):
        return self.charge * rbf_kernel(x, self.mu, sigma)

    def coulomb_force(self, x, sigma=1.0):
        diff = x - self.mu
        dist_sq = np.dot(diff, diff) + 1e-6
        # Capped inverse-square (not cube) to prevent overshoot at small distances
        force = self.charge * diff / (dist_sq + 0.1)
        # Clamp magnitude to prevent instability
        mag = np.linalg.norm(force)
        if mag > 1.0:
            force = force / mag
        return force.astype(np.float32)

    def update_position(self, x, eta=0.01):
        self.mu = (self.mu + eta * self.coulomb_force(x)).astype(np.float32)

    def error_update(self, x, error, eta=0.01):
        """Error-driven delta rule + Oja normalization.

        Delta rule: w += η * error ⊗ x (minimizes prediction error)
        Oja norm: renormalize rows to prevent weight explosion
        """
        # Delta rule: outer product of error and input
        update = np.clip(eta * np.outer(error, x), -0.1, 0.1)
        self.w += update
        # Oja-style row normalization for stability
        norms = np.linalg.norm(self.w, axis=1, keepdims=True)
        self.w /= np.maximum(norms, 1e-8)

    def update_traces(self, activation, error):
        self.a_trace = self.gamma_a * self.a_trace + activation
        self.e_trace = self.gamma_e * self.e_trace + np.clip(error, -10, 10)

    def consolidate(self, eta=0.001):
        """Offline: replay stored error trace as a delta update.

        Uses eligibility: Δw = η * a_trace * outer(e_trace, last_input_direction)
        where last_input_direction ≈ mu (expert's position = centroid of inputs).
        """
        if self.a_trace < 0.01:
            return
        e_norm = np.linalg.norm(self.e_trace)
        mu_norm = np.linalg.norm(self.mu)
        if e_norm > 1e-8 and mu_norm > 1e-8:
            # Delta-style: nudge weights to reduce stored error along input direction
            input_dir = self.mu / mu_norm
            update = np.clip(eta * self.a_trace * np.outer(self.e_trace, input_dir), -0.1, 0.1)
            self.w += update
            norms = np.linalg.norm(self.w, axis=1, keepdims=True)
            self.w /= np.maximum(norms, 1e-8)


class HEMoE:
    def __init__(self, d_input=32, d_output=32, initial_experts=4,
                 max_experts=16, top_k=2, sigma=1.0, eta_force=0.01,
                 eta_oja=0.01, eta_consol=0.001, charge_flip_threshold=0.5,
                 spawn_threshold=0.5, seed=42):
        self.d_input, self.d_output = d_input, d_output
        self.max_experts, self.top_k, self.sigma = max_experts, top_k, sigma
        self.eta_force, self.eta_oja, self.eta_consol = eta_force, eta_oja, eta_consol
        self.charge_flip_threshold, self.spawn_threshold = charge_flip_threshold, spawn_threshold
        self._rng, self._seed, self._step = np.random.default_rng(seed), seed, 0
        self.residual_ema, self._spawn_cooldown = 0.0, 0
        self.experts = [ChargedExpert(d_input, d_output,
                        charge=1.0 if i % 2 == 0 else -1.0, seed=seed + i)
                        for i in range(initial_experts)]
        self.capsules, self.max_capsules = [], 500

    @property
    def n_experts(self):
        return len(self.experts)

    def route(self, x):
        scores = [e.attraction_score(x, self.sigma) for e in self.experts]
        indices = np.argsort(scores)[-self.top_k:][::-1]
        weights = np.array([max(scores[i], 0.01) for i in indices], dtype=np.float32)
        weights /= weights.sum() + 1e-8
        return list(indices), list(weights)

    def forward(self, x):
        indices, weights = self.route(x)
        out = np.zeros(self.d_output, dtype=np.float32)
        for idx, w in zip(indices, weights):
            out += w * self.experts[idx].forward(x)
        return out

    def train_step(self, x, target):
        self._step += 1
        x, target = np.asarray(x, np.float32), np.asarray(target, np.float32)
        indices, weights = self.route(x)
        output = np.zeros(self.d_output, dtype=np.float32)
        for idx, w in zip(indices, weights):
            output += w * self.experts[idx].forward(x)
        error = target - output
        loss = float(np.clip(np.mean(error ** 2), 0, 1e6))

        for idx in indices:
            e = self.experts[idx]
            e.update_position(x, self.eta_force)
            e.error_update(x, error, self.eta_oja)  # delta rule, not PCA
            e.update_traces(float(np.abs(e.attraction_score(x, self.sigma))), error)
            e.n_uses += 1
            e.cumulative_error_sign += float(np.mean(error))

        for idx in range(self.n_experts):
            if idx not in indices:
                self.experts[idx].consolidate(self.eta_consol)

        for e in self.experts:
            if e.n_uses > 10:
                avg = e.cumulative_error_sign / e.n_uses
                if avg < -self.charge_flip_threshold:
                    e.charge = 1.0
                elif avg > self.charge_flip_threshold:
                    e.charge = -1.0

        residual = float(np.linalg.norm(error)) / (float(np.linalg.norm(target)) + 1e-8)
        self.residual_ema = 0.95 * self.residual_ema + 0.05 * residual
        spawned = False
        self._spawn_cooldown = max(0, self._spawn_cooldown - 1)
        if (self.residual_ema > self.spawn_threshold and
                self.n_experts < self.max_experts and self._spawn_cooldown == 0):
            new = ChargedExpert(self.d_input, self.d_output, charge=1.0,
                                seed=self._seed + self.n_experts * 100)
            new.mu = x.copy()
            self.experts.append(new)
            spawned = True
            self._spawn_cooldown = 30

        if float(np.linalg.norm(error)) > 1.0:
            if len(self.capsules) >= self.max_capsules:
                self.capsules.pop(0)
            self.capsules.append({"context": x.copy(), "experts": list(indices),
                                  "error": error.copy(), "step": self._step})

        return {"loss": loss, "n_experts": self.n_experts,
                "charges": [e.charge for e in self.experts],
                "residual_ema": self.residual_ema, "spawned": spawned}

    def sleep_replay(self, n=10):
        if not self.capsules:
            return
        for cap in sorted(self.capsules, key=lambda c: -float(np.linalg.norm(c["error"])))[:n]:
            for eid in cap["experts"]:
                if eid < self.n_experts:
                    self.experts[eid].mu += 0.01 * (cap["context"] - self.experts[eid].mu)


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS TESTS — H1 (simplest) → H12 (full system)
# ═══════════════════════════════════════════════════════════════════════════════


class TestH1_KernelProperties:
    def test_self_similarity_is_one(self):
        x = np.random.randn(16).astype(np.float32)
        assert abs(rbf_kernel(x, x) - 1.0) < 1e-6

    def test_distant_similarity_is_small(self):
        assert rbf_kernel(np.zeros(16), np.ones(16) * 10) < 0.01

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        x, y = rng.standard_normal(16).astype(np.float32), rng.standard_normal(16).astype(np.float32)
        assert abs(rbf_kernel(x, y) - rbf_kernel(y, x)) < 1e-6

    def test_rkhs_distance_monotonic(self):
        x = np.zeros(16, dtype=np.float32)
        assert rkhs_distance_sq(x, x) < rkhs_distance_sq(x, np.ones(16) * 0.1) < rkhs_distance_sq(x, np.ones(16) * 5)


class TestH2_RFF:
    def test_approximation(self):
        rff = RandomFourierFeatures(16, d_rff=512, seed=42)
        rng = np.random.default_rng(42)
        errors = [abs(rbf_kernel(x := rng.standard_normal(16).astype(np.float32),
                                  y := rng.standard_normal(16).astype(np.float32))
                      - float(np.dot(rff.transform(x), rff.transform(y))))
                  for _ in range(100)]
        assert np.mean(errors) < 0.15


class TestH3_ChargedScoring:
    def test_positive_attracts(self):
        e = ChargedExpert(16, 8, charge=1.0, seed=42)
        assert e.attraction_score(e.mu + np.random.randn(16).astype(np.float32) * 0.01) > 0.5

    def test_negative_repels(self):
        e = ChargedExpert(16, 8, charge=-1.0, seed=42)
        assert e.attraction_score(e.mu.copy()) < 0

    def test_asymmetry(self):
        pos = ChargedExpert(16, 8, charge=1.0, seed=42)
        neg = ChargedExpert(16, 8, charge=-1.0, seed=42)
        neg.mu = pos.mu.copy()
        x = pos.mu.copy()
        assert pos.attraction_score(x) > 0 > neg.attraction_score(x)


class TestH4_CoulombForce:
    def test_positive_toward(self):
        e = ChargedExpert(16, 8, charge=1.0, seed=42)
        x = e.mu + np.ones(16, dtype=np.float32) * 0.5
        assert float(np.dot(e.coulomb_force(x), x - e.mu)) > 0

    def test_negative_away(self):
        e = ChargedExpert(16, 8, charge=-1.0, seed=42)
        x = e.mu + np.ones(16, dtype=np.float32) * 0.5
        assert float(np.dot(e.coulomb_force(x), x - e.mu)) < 0


class TestH5_ForceRouting:
    def test_nearest_positive_selected(self):
        m = HEMoE(d_input=8, d_output=8, initial_experts=4, top_k=1, seed=42)
        m.experts[0].charge, m.experts[0].mu = 1.0, np.zeros(8, dtype=np.float32)
        for i in range(1, 4):
            m.experts[i].mu = np.ones(8, dtype=np.float32) * 10 * i
        assert 0 in m.route(np.zeros(8, dtype=np.float32) + 0.01)[0]


class TestH6_PositionDrift:
    def test_moves_toward_cluster(self):
        e = ChargedExpert(8, 8, charge=1.0, seed=42)
        e.mu = np.zeros(8, dtype=np.float32)
        # Start close enough for Coulomb force to be meaningful
        center = np.ones(8, dtype=np.float32) * 0.5
        rng = np.random.default_rng(42)
        dist_before = float(np.linalg.norm(e.mu - center))
        for _ in range(100):
            e.update_position(center + rng.standard_normal(8).astype(np.float32) * 0.05, eta=0.1)
        dist_after = float(np.linalg.norm(e.mu - center))
        assert dist_after < dist_before, f"Didn't move closer: {dist_before:.3f} → {dist_after:.3f}"


class TestH7_OjaNorm:
    def test_norms_bounded(self):
        e = ChargedExpert(16, 8, seed=42)
        rng = np.random.default_rng(42)
        for _ in range(100):
            x = rng.standard_normal(16).astype(np.float32)
            error = rng.standard_normal(8).astype(np.float32)
            e.error_update(x, error, eta=0.01)
        norms = np.linalg.norm(e.w, axis=1)
        assert np.all(norms > 0.5) and np.all(norms < 2.0)


class TestH8_ChargeFlip:
    def test_charges_can_flip(self):
        m = HEMoE(d_input=8, d_output=8, charge_flip_threshold=0.01, eta_force=0.0001, seed=42)
        rng = np.random.default_rng(42)
        for _ in range(100):
            m.train_step(rng.standard_normal(8).astype(np.float32) * 3,
                         rng.standard_normal(8).astype(np.float32) * 3)
        assert any(e.charge != (1.0 if i % 2 == 0 else -1.0) for i, e in enumerate(m.experts))


class TestH9_InactiveConsolidation:
    def test_inactive_weights_change(self):
        m = HEMoE(d_input=8, d_output=8, top_k=1, eta_consol=0.1, seed=42)
        rng = np.random.default_rng(42)
        for e in m.experts:
            e.a_trace, e.e_trace = 1.0, rng.standard_normal(8).astype(np.float32)
        w_before = [e.w.copy() for e in m.experts]
        for _ in range(20):
            m.train_step(rng.standard_normal(8).astype(np.float32),
                         rng.standard_normal(8).astype(np.float32))
        assert any(not np.array_equal(e.w, wb) for e, wb in zip(m.experts, w_before))


class TestH10_HippocampalStorage:
    def test_capsules_stored(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        rng = np.random.default_rng(42)
        for _ in range(50):
            m.train_step(rng.standard_normal(8).astype(np.float32) * 5,
                         rng.standard_normal(8).astype(np.float32) * 5)
        assert len(m.capsules) > 0

    def test_capsule_has_fields(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        m.train_step(np.ones(8, np.float32), np.ones(8, np.float32) * 100)
        if m.capsules:
            cap = m.capsules[-1]
            assert all(k in cap for k in ("context", "error", "experts", "step"))


class TestH11_SleepReplay:
    def test_positions_move(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        rng = np.random.default_rng(42)
        for _ in range(50):
            m.train_step(rng.standard_normal(8).astype(np.float32) * 3,
                         rng.standard_normal(8).astype(np.float32) * 3)
        if not m.capsules:
            pytest.skip("No capsules")
        mu_before = [e.mu.copy() for e in m.experts]
        m.sleep_replay(n=10)
        assert any(not np.array_equal(e.mu, mb) for e, mb in zip(m.experts, mu_before))


class TestH12_FullSystem:
    def test_1000_steps_stable(self):
        m = HEMoE(d_input=16, d_output=16, initial_experts=4, max_experts=12,
                    spawn_threshold=0.3, charge_flip_threshold=0.3,
                    eta_force=0.005, eta_oja=0.005, eta_consol=0.001, seed=42)
        rng = np.random.default_rng(42)
        losses = []
        for _ in range(1000):
            r = m.train_step(rng.standard_normal(16).astype(np.float32),
                             rng.standard_normal(16).astype(np.float32))
            losses.append(r["loss"])
        assert all(np.isfinite(l) for l in losses)
        assert m.n_experts <= 12
        assert np.all(np.isfinite(m.forward(rng.standard_normal(16).astype(np.float32))))

    def test_sleep_after_training(self):
        m = HEMoE(d_input=8, d_output=8, seed=42)
        rng = np.random.default_rng(42)
        for _ in range(200):
            m.train_step(rng.standard_normal(8).astype(np.float32) * 2,
                         rng.standard_normal(8).astype(np.float32) * 2)
        m.sleep_replay(n=20)
        assert np.all(np.isfinite(m.forward(rng.standard_normal(8).astype(np.float32))))
