"""Contrastive Value Learning (CVL).

Numpy-only implementation of Q-value estimation via contrastive learning
of the occupancy measure, based on:
  "Contrastive Value Learning" (Mazoure et al., NeurIPS 2022 Workshop)

Key idea: learn state-action and future-state encoders via InfoNCE such that
their inner product approximates the discounted occupancy ratio, yielding
Q-values without TD learning.

Validates Theorem 4 (CVL convergence).
"""

from __future__ import annotations

import numpy as np


# -- Helpers -------------------------------------------------------------------


def infonce_loss(
    phi_sa: np.ndarray,
    psi_s_future: np.ndarray,
    psi_negatives: np.ndarray,
) -> float:
    """InfoNCE contrastive loss.

    Loss = -log(exp(phi . psi_pos) / (exp(phi . psi_pos) + sum exp(phi . psi_neg)))

    Args:
        phi_sa: State-action encoder output (d,)
        psi_s_future: Future state encoder output -- positive sample (d,)
        psi_negatives: Negative future state samples (n_neg, d)

    Returns:
        Scalar InfoNCE loss.
    """
    pos_logit = phi_sa @ psi_s_future  # scalar
    neg_logits = psi_negatives @ phi_sa  # (n_neg,)

    # Numerically stable log-sum-exp
    all_logits = np.concatenate([[pos_logit], neg_logits])
    max_logit = np.max(all_logits)
    log_denom = max_logit + np.log(np.sum(np.exp(all_logits - max_logit)))

    return float(-(pos_logit - log_denom))


def random_fourier_features(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Approximate the exponential kernel via random Fourier features.

    F(x) = sqrt(2e / d_rff) * cos(W x + b)

    where W ~ N(0, I), b ~ U(0, 2pi).  F(x)^T F(y) approx exp(x^T y).

    Args:
        x: Input vector (d,)
        W: Random projection matrix (d_rff, d), drawn from N(0, I)
        b: Random bias vector (d_rff,), drawn from U(0, 2pi)

    Returns:
        Feature vector (d_rff,)
    """
    d_rff = W.shape[0]
    scale = np.sqrt(2.0 * np.e / d_rff)
    return scale * np.cos(W @ x + b)


class TruncatedGeometric:
    """Truncated geometric distribution sampler."""

    @staticmethod
    def sample(
        p: float,
        min_val: int,
        max_val: int,
        rng: np.random.Generator,
    ) -> int:
        """Sample from a truncated geometric distribution.

        Rejection sampling: draw from Geom(p) and reject values outside
        [min_val, max_val].

        Args:
            p: Success probability (0 < p <= 1)
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            rng: Numpy random generator

        Returns:
            Integer sample in [min_val, max_val]
        """
        while True:
            sample = rng.geometric(p)
            if min_val <= sample <= max_val:
                return int(sample)


# -- Main estimator ------------------------------------------------------------


class ContrastiveValueEstimator:
    """Q-value estimation via contrastive occupancy measure learning.

    Uses separate state-action (phi) and future-state (psi) encoders trained
    with InfoNCE.  Q-values are recovered via random Fourier features of the
    phi embedding dotted with a running reward-weighted average of psi RFF
    features.

    Attributes:
        d_state: State dimensionality
        d_action: Action dimensionality
        d_latent: Encoder output dimensionality
        d_rff: Random Fourier feature dimensionality
        gamma: Discount factor
    """

    def __init__(
        self,
        d_state: int,
        d_action: int,
        d_latent: int = 128,
        d_rff: int = 256,
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.d_state = d_state
        self.d_action = d_action
        self.d_latent = d_latent
        self.d_rff = d_rff
        self.gamma = gamma
        self._rng = rng

        # Phi encoder: Linear(d_state + d_action, d_latent)
        fan_in_phi = d_state + d_action
        scale_phi = np.sqrt(2.0 / fan_in_phi)
        self.W_phi = rng.normal(0, scale_phi, size=(d_latent, fan_in_phi)).astype(
            np.float32
        )
        self.b_phi = np.zeros(d_latent, dtype=np.float32)

        # Psi encoder: Linear(d_state, d_latent)
        scale_psi = np.sqrt(2.0 / d_state)
        self.W_psi = rng.normal(0, scale_psi, size=(d_latent, d_state)).astype(
            np.float32
        )
        self.b_psi = np.zeros(d_latent, dtype=np.float32)

        # RFF parameters
        self.W_rff = rng.standard_normal(size=(d_rff, d_latent)).astype(np.float32)
        self.b_rff = rng.uniform(0, 2 * np.pi, size=d_rff).astype(np.float32)

        # Running average of reward-weighted future features
        self.xi = np.zeros(d_rff, dtype=np.float32)

    # -- Encoders --------------------------------------------------------------

    def encode_state_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Encode a (state, action) pair through the phi encoder.

        Args:
            state: State vector (d_state,)
            action: Action vector (d_action,)

        Returns:
            L2-normalized embedding (d_latent,)
        """
        x = np.concatenate([state, action]).astype(np.float32)
        h = self.W_phi @ x + self.b_phi
        norm = np.linalg.norm(h)
        if norm > 0:
            h = h / norm
        return h

    def encode_future_state(self, state: np.ndarray) -> np.ndarray:
        """Encode a future state through the psi encoder.

        Args:
            state: State vector (d_state,)

        Returns:
            L2-normalized embedding (d_latent,)
        """
        x = state.astype(np.float32)
        h = self.W_psi @ x + self.b_psi
        norm = np.linalg.norm(h)
        if norm > 0:
            h = h / norm
        return h

    # -- Q-value ---------------------------------------------------------------

    def q_value(self, state: np.ndarray, action: np.ndarray) -> float:
        """Estimate Q(s, a) via RFF inner product.

        Q(s,a) = (1 / (1 - gamma)) * F(phi(s,a))^T xi

        Args:
            state: State vector (d_state,)
            action: Action vector (d_action,)

        Returns:
            Scalar Q-value estimate.
        """
        phi = self.encode_state_action(state, action)
        f_phi = random_fourier_features(phi, self.W_rff, self.b_rff)
        return float((1.0 / (1.0 - self.gamma)) * f_phi @ self.xi)

    # -- Updates ---------------------------------------------------------------

    def update_xi(
        self,
        future_states: np.ndarray,
        rewards: np.ndarray,
        beta: float = 0.99,
    ) -> None:
        """Update xi via exponential moving average of reward-weighted RFF features.

        xi <- beta * xi + (1 - beta) * mean_i(F(psi(s_i)) * r_i)

        Args:
            future_states: Array of future states (n, d_state)
            rewards: Corresponding rewards (n,)
            beta: EMA coefficient
        """
        n = len(rewards)
        rff_sum = np.zeros(self.d_rff, dtype=np.float32)
        for i in range(n):
            psi = self.encode_future_state(future_states[i])
            f_psi = random_fourier_features(psi, self.W_rff, self.b_rff)
            rff_sum += f_psi * rewards[i]
        rff_mean = rff_sum / max(n, 1)
        self.xi = beta * self.xi + (1.0 - beta) * rff_mean

    def update_critic(
        self,
        trajectories: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
        lr: float = 1e-3,
    ) -> float:
        """Update phi and psi encoders via InfoNCE on trajectory data.

        For each trajectory element (s, a, s', r):
        - Positive pair: phi(s, a) with psi(s_{t + delta_t})
        - Negatives: psi of random future states from other trajectories

        Updates encoder weights via numerical gradient descent and refreshes xi.

        Args:
            trajectories: List of (state, action, next_state, reward) tuples
            lr: Learning rate

        Returns:
            Mean InfoNCE loss over the batch.
        """
        n = len(trajectories)
        if n < 2:
            return 0.0

        n_neg = min(n - 1, 16)
        total_loss = 0.0

        # Collect all future states and rewards for xi update
        future_states = np.array([t[2] for t in trajectories], dtype=np.float32)
        rewards = np.array([t[3] for t in trajectories], dtype=np.float32)

        for i in range(n):
            state, action, next_state, reward = trajectories[i]

            # Positive: sample a future state with temporal offset
            delta = TruncatedGeometric.sample(
                1.0 - self.gamma, min_val=1, max_val=max(n - i, 1), rng=self._rng
            )
            future_idx = min(i + delta, n - 1)
            s_future = trajectories[future_idx][2]

            phi = self.encode_state_action(state, action)
            psi_pos = self.encode_future_state(s_future)

            # Negatives: random future states from other indices
            other_indices = [j for j in range(n) if j != i]
            neg_indices = self._rng.choice(
                other_indices, size=min(n_neg, len(other_indices)), replace=False
            )
            psi_negs = np.array(
                [self.encode_future_state(trajectories[j][2]) for j in neg_indices],
                dtype=np.float32,
            )

            loss = infonce_loss(phi, psi_pos, psi_negs)
            total_loss += loss

            # Numerical gradient descent on phi encoder weights
            eps = 1e-4

            # Update W_phi
            grad_W_phi = np.zeros_like(self.W_phi)
            for row in range(self.W_phi.shape[0]):
                for col in range(self.W_phi.shape[1]):
                    self.W_phi[row, col] += eps
                    phi_p = self.encode_state_action(state, action)
                    loss_p = infonce_loss(phi_p, psi_pos, psi_negs)
                    self.W_phi[row, col] -= 2 * eps
                    phi_m = self.encode_state_action(state, action)
                    loss_m = infonce_loss(phi_m, psi_pos, psi_negs)
                    self.W_phi[row, col] += eps  # restore
                    grad_W_phi[row, col] = (loss_p - loss_m) / (2 * eps)

            self.W_phi -= lr * grad_W_phi

            # Update b_phi
            grad_b_phi = np.zeros_like(self.b_phi)
            for j in range(self.b_phi.shape[0]):
                self.b_phi[j] += eps
                phi_p = self.encode_state_action(state, action)
                loss_p = infonce_loss(phi_p, psi_pos, psi_negs)
                self.b_phi[j] -= 2 * eps
                phi_m = self.encode_state_action(state, action)
                loss_m = infonce_loss(phi_m, psi_pos, psi_negs)
                self.b_phi[j] += eps
                grad_b_phi[j] = (loss_p - loss_m) / (2 * eps)

            self.b_phi -= lr * grad_b_phi

            # Update W_psi
            grad_W_psi = np.zeros_like(self.W_psi)
            for row in range(self.W_psi.shape[0]):
                for col in range(self.W_psi.shape[1]):
                    self.W_psi[row, col] += eps
                    psi_pos_p = self.encode_future_state(s_future)
                    psi_negs_p = np.array(
                        [
                            self.encode_future_state(trajectories[j][2])
                            for j in neg_indices
                        ],
                        dtype=np.float32,
                    )
                    loss_p = infonce_loss(phi, psi_pos_p, psi_negs_p)
                    self.W_psi[row, col] -= 2 * eps
                    psi_pos_m = self.encode_future_state(s_future)
                    psi_negs_m = np.array(
                        [
                            self.encode_future_state(trajectories[j][2])
                            for j in neg_indices
                        ],
                        dtype=np.float32,
                    )
                    loss_m = infonce_loss(phi, psi_pos_m, psi_negs_m)
                    self.W_psi[row, col] += eps
                    grad_W_psi[row, col] = (loss_p - loss_m) / (2 * eps)

            self.W_psi -= lr * grad_W_psi

            # Update b_psi
            grad_b_psi = np.zeros_like(self.b_psi)
            for j in range(self.b_psi.shape[0]):
                self.b_psi[j] += eps
                psi_pos_p = self.encode_future_state(s_future)
                psi_negs_p = np.array(
                    [
                        self.encode_future_state(trajectories[j][2])
                        for j in neg_indices
                    ],
                    dtype=np.float32,
                )
                loss_p = infonce_loss(phi, psi_pos_p, psi_negs_p)
                self.b_psi[j] -= 2 * eps
                psi_pos_m = self.encode_future_state(s_future)
                psi_negs_m = np.array(
                    [
                        self.encode_future_state(trajectories[j][2])
                        for j in neg_indices
                    ],
                    dtype=np.float32,
                )
                loss_m = infonce_loss(phi, psi_pos_m, psi_negs_m)
                self.b_psi[j] += eps
                grad_b_psi[j] = (loss_p - loss_m) / (2 * eps)

            self.b_psi -= lr * grad_b_psi

        # Update xi with trajectory rewards
        self.update_xi(future_states, rewards)

        return total_loss / n
