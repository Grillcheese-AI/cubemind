"""HMM-VSA learnable reasoning rules.

Hidden Markov Models operating directly on block-code hypervector
representations. Detection uses the forward algorithm; execution uses
posterior-weighted transition prediction. An ensemble of independent
HMM rules naturally specializes during training.

Validates Theorem 2 (HMM = GFSA) and Theorem 8 (convergence).
The forward algorithm uses log-space arithmetic throughout for
numerical stability on long sequences.

Reference: "Learnable Reasoning Rules for NVSA via HMM" (Section 3.2)
"""

from __future__ import annotations

import numpy as np

from cubemind.ops import BlockCodes
from cubemind.core.registry import register


# -- Utilities -----------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along *axis*."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> float | np.ndarray:
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    out = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


# -- HMMRule -------------------------------------------------------------------


@register("detector", "hmm_rule")
class HMMRule:
    """A single HMM reasoning rule grounded in block-code codebook states.

    Parameters
    ----------
    codebook : np.ndarray
        Shape (n_states, k, l) -- block-code vectors for each hidden state.
    temperature : float
        Softmax temperature for emission probabilities.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        codebook: np.ndarray,
        temperature: float = 40.0,
        seed: int = 42,
    ) -> None:
        self.codebook = codebook.astype(np.float64)
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)
        self._bc = BlockCodes(k=codebook.shape[1], l=codebook.shape[2])

        n = codebook.shape[0]
        # Small random perturbation breaks symmetry across ensemble members
        self._log_A = self._rng.normal(0, 0.01, size=(n, n)).astype(np.float64)
        self._log_pi = self._rng.normal(0, 0.01, size=(n,)).astype(np.float64)

    # -- Properties ------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return self.codebook.shape[0]

    @property
    def A(self) -> np.ndarray:
        """Transition matrix (n, n), rows sum to 1."""
        return _softmax(self._log_A, axis=-1)

    @property
    def pi(self) -> np.ndarray:
        """Initial state distribution (n,), sums to 1."""
        return _softmax(self._log_pi, axis=-1)

    # -- Emission --------------------------------------------------------------

    def emission(self, obs: np.ndarray) -> np.ndarray:
        """Compute emission probability vector P(obs | state_i) for all states.

        Parameters
        ----------
        obs : np.ndarray
            Observation block-code vector, shape (k, l).

        Returns
        -------
        np.ndarray
            Probability vector of shape (n_states,), sums to 1.
        """
        sims = self._bc.similarity_batch(obs, self.codebook.astype(np.float32))
        return self._bc.cosine_to_pmf(sims, self.temperature).astype(np.float64)

    # -- Forward algorithm (log-space) -----------------------------------------

    def forward(self, observations: list[np.ndarray]) -> tuple[float, np.ndarray]:
        """Run the forward algorithm in log-space.

        Parameters
        ----------
        observations : list[np.ndarray]
            Sequence of T observation vectors, each (k, l).

        Returns
        -------
        log_likelihood : float
            Total log P(O | model).
        log_alpha : np.ndarray
            Log-alpha matrix of shape (T, n_states).
        """
        T = len(observations)
        n = self.n_states
        log_alpha = np.full((T, n), -np.inf, dtype=np.float64)

        log_pi = np.log(np.clip(self.pi, 1e-300, None))
        log_A = np.log(np.clip(self.A, 1e-300, None))

        # t = 0
        log_B0 = np.log(np.clip(self.emission(observations[0]), 1e-300, None))
        log_alpha[0] = log_pi + log_B0

        # t = 1 .. T-1
        for t in range(1, T):
            log_Bt = np.log(np.clip(self.emission(observations[t]), 1e-300, None))
            for j in range(n):
                log_alpha[t, j] = log_Bt[j] + _logsumexp(log_alpha[t - 1] + log_A[:, j])

        log_likelihood = float(_logsumexp(log_alpha[-1]))
        return log_likelihood, log_alpha

    # -- Viterbi ---------------------------------------------------------------

    def viterbi(self, observations: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Find the most likely state sequence via the Viterbi algorithm.

        Parameters
        ----------
        observations : list[np.ndarray]
            Sequence of T observation vectors, each (k, l).

        Returns
        -------
        state_sequence : np.ndarray
            Most likely state indices, shape (T,).
        log_prob : float
            Log probability of the best path.
        """
        T = len(observations)
        n = self.n_states

        log_pi = np.log(np.clip(self.pi, 1e-300, None))
        log_A = np.log(np.clip(self.A, 1e-300, None))

        # Delta and psi tables
        delta = np.full((T, n), -np.inf, dtype=np.float64)
        psi = np.zeros((T, n), dtype=np.int32)

        # Initialization
        log_B0 = np.log(np.clip(self.emission(observations[0]), 1e-300, None))
        delta[0] = log_pi + log_B0

        # Recursion
        for t in range(1, T):
            log_Bt = np.log(np.clip(self.emission(observations[t]), 1e-300, None))
            for j in range(n):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(candidates))
                delta[t, j] = candidates[psi[t, j]] + log_Bt[j]

        # Termination
        best_last = int(np.argmax(delta[-1]))
        log_prob = float(delta[-1, best_last])

        # Backtrack
        path = np.zeros(T, dtype=np.int32)
        path[-1] = best_last
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path, log_prob

    # -- Detection -------------------------------------------------------------

    def detect(self, observations: list[np.ndarray]) -> float:
        """Total observation-sequence likelihood P(O | model).

        Returns the sum of final alpha values (in probability space).
        """
        log_ll, _ = self.forward(observations)
        return float(np.exp(log_ll))

    # -- Prediction (execution) ------------------------------------------------

    def predict(self, observations: list[np.ndarray]) -> np.ndarray:
        """Predict the next block-code vector via posterior-weighted transition.

        1. Compute posterior P(s_T = i | obs) from forward alphas.
        2. Predict next-state distribution: posterior @ A.
        3. Convert to block-code vector via pmf_to_vector.

        Returns
        -------
        np.ndarray
            Predicted block-code vector, shape (k, l).
        """
        _, log_alpha = self.forward(observations)
        # Posterior of last timestep
        log_alpha_T = log_alpha[-1]
        log_norm = _logsumexp(log_alpha_T)
        posterior = np.exp(log_alpha_T - log_norm)  # (n,)

        # Predicted next-state distribution
        next_dist = posterior @ self.A  # (n,)

        # Convert to block-code vector
        return self._bc.pmf_to_vector(
            self.codebook.astype(np.float32), next_dist.astype(np.float32)
        )

    # -- Backward algorithm (log-space) ----------------------------------------

    def backward(self, observations: list[np.ndarray]) -> np.ndarray:
        """Run the backward algorithm in log-space.

        Returns log-beta matrix of shape (T, n_states).
        """
        T = len(observations)
        n = self.n_states
        log_beta = np.full((T, n), -np.inf, dtype=np.float64)
        log_A = np.log(np.clip(self.A, 1e-300, None))

        # t = T-1: beta(T) = 1 => log_beta = 0
        log_beta[T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            log_Bt1 = np.log(np.clip(self.emission(observations[t + 1]), 1e-300, None))
            for i in range(n):
                log_beta[t, i] = _logsumexp(
                    log_A[i, :] + log_Bt1 + log_beta[t + 1]
                )

        return log_beta

    # -- Training ----------------------------------------------------------------

    def _nll_loss(self, observations: list[np.ndarray], target: np.ndarray) -> float:
        """Negative log-likelihood of the target under predicted next-state distribution.

        Much stronger gradient signal than MSE on reconstructed vectors:
        -log(P(target_state | observations)) has gradient -1/p, which is large
        when the model assigns low probability to the correct answer.
        """
        _, log_alpha = self.forward(observations)
        log_alpha_T = log_alpha[-1]
        log_norm = _logsumexp(log_alpha_T)
        posterior = np.exp(log_alpha_T - log_norm)
        next_dist = posterior @ self.A  # predicted next-state distribution

        # Find which codebook entry best matches the target
        sims = self._bc.similarity_batch(target, self.codebook.astype(np.float32))
        target_idx = int(np.argmax(sims))

        # NLL: -log P(correct entry)
        p_target = float(np.clip(next_dist[target_idx], 1e-12, 1.0))
        return -np.log(p_target)

    def _loss(self, observations: list[np.ndarray], target: np.ndarray) -> float:
        """Loss for gradient computation — uses NLL for strong gradient signal."""
        return self._nll_loss(observations, target)

    def train_step(
        self,
        observations: list[np.ndarray],
        target: np.ndarray,
        lr: float = 0.01,
    ) -> float:
        """One supervised gradient step via central finite differences on NLL.

        Uses negative log-likelihood of the target under the predicted
        next-state distribution. Gradient signal is -1/p (large when wrong).

        For unsupervised pre-training, use train_step_em() separately.

        Returns the loss *before* the update.
        """
        eps = 1e-2  # larger eps for near-uniform transitions
        base_loss = self._loss(observations, target)

        # Gradient for _log_A
        grad_A = np.zeros_like(self._log_A)
        for i in range(self.n_states):
            for j in range(self.n_states):
                self._log_A[i, j] += eps
                loss_plus = self._loss(observations, target)
                self._log_A[i, j] -= 2 * eps
                loss_minus = self._loss(observations, target)
                self._log_A[i, j] += eps  # restore
                grad_A[i, j] = (loss_plus - loss_minus) / (2 * eps)

        # Gradient for _log_pi
        grad_pi = np.zeros_like(self._log_pi)
        for i in range(self.n_states):
            self._log_pi[i] += eps
            loss_plus = self._loss(observations, target)
            self._log_pi[i] -= 2 * eps
            loss_minus = self._loss(observations, target)
            self._log_pi[i] += eps  # restore
            grad_pi[i] = (loss_plus - loss_minus) / (2 * eps)

        # Update
        self._log_A -= lr * grad_A
        self._log_pi -= lr * grad_pi

        return base_loss

    def train_step_em(
        self,
        sequences: list[list[np.ndarray]],
        smoothing: float = 1e-3,
    ) -> float:
        """One Baum-Welch EM update over a batch of observation sequences.

        Computes exact expected transition/initial counts via the
        forward-backward algorithm, then updates parameters in closed form.

        Parameters
        ----------
        sequences : list[list[np.ndarray]]
            Batch of observation sequences.
        smoothing : float
            Laplace smoothing added to counts to prevent zeros.

        Returns
        -------
        float
            Average log-likelihood across the batch (before update).
        """
        n = self.n_states
        A = self.A
        pi = self.pi

        # Accumulators
        A_num = np.full((n, n), smoothing, dtype=np.float64)
        pi_num = np.full((n,), smoothing, dtype=np.float64)
        total_ll = 0.0

        for obs_seq in sequences:
            T = len(obs_seq)
            if T < 2:
                continue

            # Precompute emission log-probs
            log_B = np.zeros((T, n), dtype=np.float64)
            for t in range(T):
                b = self.emission(obs_seq[t])
                log_B[t] = np.log(np.clip(b, 1e-300, None))

            log_A = np.log(np.clip(A, 1e-300, None))
            log_pi = np.log(np.clip(pi, 1e-300, None))

            # Forward
            log_alpha = np.full((T, n), -np.inf, dtype=np.float64)
            log_alpha[0] = log_pi + log_B[0]
            for t in range(1, T):
                for j in range(n):
                    log_alpha[t, j] = log_B[t, j] + _logsumexp(
                        log_alpha[t - 1] + log_A[:, j]
                    )

            # Backward
            log_beta = np.full((T, n), -np.inf, dtype=np.float64)
            log_beta[T - 1] = 0.0
            for t in range(T - 2, -1, -1):
                for i in range(n):
                    log_beta[t, i] = _logsumexp(
                        log_A[i, :] + log_B[t + 1] + log_beta[t + 1]
                    )

            # Sequence log-likelihood
            ll = float(_logsumexp(log_alpha[-1]))
            total_ll += ll

            # Gamma: P(s_t = i | O)
            log_gamma = log_alpha + log_beta
            for t in range(T):
                log_gamma[t] -= _logsumexp(log_gamma[t])

            # Xi: P(s_t = i, s_{t+1} = j | O)
            for t in range(T - 1):
                for i in range(n):
                    for j in range(n):
                        log_xi_ij = (
                            log_alpha[t, i]
                            + log_A[i, j]
                            + log_B[t + 1, j]
                            + log_beta[t + 1, j]
                            - ll
                        )
                        A_num[i, j] += np.exp(log_xi_ij)

            # Initial state counts
            pi_num += np.exp(log_gamma[0])

        # M-step: update parameters
        A_new = A_num / A_num.sum(axis=1, keepdims=True)
        pi_new = pi_num / pi_num.sum()

        # Store as log-parameters (softmax parameterization)
        self._log_A = np.log(np.clip(A_new, 1e-300, None))
        self._log_pi = np.log(np.clip(pi_new, 1e-300, None))

        return total_ll / max(len(sequences), 1)


# -- HMMEnsemble --------------------------------------------------------------


class HMMEnsemble:
    """Ensemble of M independent HMMRule instances.

    Rules are weighted by their detection likelihoods at prediction time.
    A diversity loss discourages mode collapse during training.

    Parameters
    ----------
    codebook : np.ndarray
        Shape (n_states, k, l).
    n_rules : int
        Number of independent HMM rules.
    seed : int
        Base RNG seed; each rule gets ``seed + i``.
    """

    def __init__(
        self,
        codebook: np.ndarray,
        n_rules: int = 8,
        seed: int = 42,
    ) -> None:
        self.rules = [
            HMMRule(codebook, seed=seed + i) for i in range(n_rules)
        ]
        self.n_rules = n_rules

    def predict(
        self, observations: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict next vector as likelihood-weighted average across rules.

        Returns
        -------
        prediction : np.ndarray
            Weighted average block-code vector, shape (k, l).
        weights : np.ndarray
            Per-rule weights (normalized detection likelihoods), shape (n_rules,).
        """
        likelihoods = np.array(
            [rule.detect(observations) for rule in self.rules], dtype=np.float64
        )
        predictions = np.array(
            [rule.predict(observations).astype(np.float64) for rule in self.rules]
        )

        total = likelihoods.sum()
        if total < 1e-300:
            weights = np.ones(self.n_rules, dtype=np.float64) / self.n_rules
        else:
            weights = likelihoods / total

        prediction = np.einsum("r,r...->...", weights, predictions)
        return prediction.astype(np.float32), weights.astype(np.float64)

    def train_step(
        self,
        observations: list[np.ndarray],
        target: np.ndarray,
        lr: float = 0.01,
        diversity_weight: float = 0.01,
    ) -> list[float]:
        """Train each rule independently, then apply diversity pressure.

        The diversity loss is the negative mean pairwise KL divergence
        between transition matrices.

        Returns
        -------
        list[float]
            Per-rule losses (before update).
        """
        losses = [rule.train_step(observations, target, lr) for rule in self.rules]

        # Diversity loss: nudge transition matrices apart
        if diversity_weight > 0 and self.n_rules > 1:
            As = [rule.A for rule in self.rules]
            for m, rule in enumerate(self.rules):
                grad_div = np.zeros_like(rule._log_A)
                for m2 in range(self.n_rules):
                    if m2 == m:
                        continue
                    diff = np.log(np.clip(As[m], 1e-12, None)) - np.log(
                        np.clip(As[m2], 1e-12, None)
                    )
                    grad_div += diff + 1.0
                grad_div /= self.n_rules - 1
                rule._log_A += lr * diversity_weight * grad_div

        return losses


# ══════════════════════════════════════════════════════════════════════════════
# Multi-View HMM — separate HMMs for absolute, delta, row_bundle views
# ══════════════════════════════════════════════════════════════════════════════


class MultiViewHMM:
    """Multi-view HMM with per-view specialization and routing.

    Trains three separate HMMs on different observation views:
      - absolute: raw panel vectors (captures direct patterns)
      - delta: unbind(p_{t+1}, p_t) (captures transformations between panels)
      - row_bundle: bundle per row (captures row-level structure)

    Prediction routes to the view with highest normalized log-likelihood
    (detection confidence), then returns that view's predicted next panel.

    Args:
        codebook: Block-code codebook (n_states, k, l).
        bc: BlockCodes instance for bind/unbind/bundle ops.
        temperature: Emission softmax temperature.
        seed: Random seed.
    """

    VIEW_NAMES = ["absolute", "delta", "row_bundle"]

    def __init__(
        self,
        codebook: np.ndarray,
        bc,
        temperature: float = 40.0,
        seed: int = 42,
    ) -> None:
        self._bc = bc
        self.hmms = {
            "absolute": HMMRule(codebook, temperature=temperature, seed=seed + 100),
            "delta": HMMRule(codebook, temperature=temperature, seed=seed + 200),
            "row_bundle": HMMRule(codebook, temperature=temperature, seed=seed + 300),
        }

    # -- View construction -----------------------------------------------------

    def make_views(self, panel_vecs: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
        """Create three observation views from panel vectors.

        Args:
            panel_vecs: Block-code vectors for panels (8 context or 9 with answer).

        Returns:
            Dict mapping view name to list of observation vectors.
        """
        bc = self._bc
        views = {}

        views["absolute"] = list(panel_vecs)

        deltas = []
        for t in range(len(panel_vecs) - 1):
            deltas.append(bc.unbind(panel_vecs[t + 1], panel_vecs[t]))
        views["delta"] = deltas if deltas else list(panel_vecs)

        row_bundles = []
        if len(panel_vecs) >= 3:
            row_bundles.append(bc.bundle([panel_vecs[0], panel_vecs[1], panel_vecs[2]]))
        if len(panel_vecs) >= 6:
            row_bundles.append(bc.bundle([panel_vecs[3], panel_vecs[4], panel_vecs[5]]))
        remaining = list(panel_vecs[6:])
        if remaining:
            row_bundles.append(
                bc.bundle(remaining) if len(remaining) >= 2 else remaining[0]
            )
        views["row_bundle"] = row_bundles if row_bundles else list(panel_vecs)

        return views

    # -- Training (Baum-Welch EM) ----------------------------------------------

    def train_em(
        self,
        sequences: list[list[np.ndarray]],
        em_epochs: int = 20,
        batch_size: int = 30,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train all view HMMs via Baum-Welch EM.

        Each sequence should include the correct answer appended (9 panels).

        Args:
            sequences: List of full panel sequences (each 9 vectors).
            em_epochs: Number of EM epochs per view.
            batch_size: Batch size for EM updates.
            seed: Random seed for shuffling.
            verbose: Print per-epoch log-likelihood.

        Returns:
            Dict mapping view name to final average log-likelihood.
        """
        # Build per-view observation sequences
        view_seqs: dict[str, list[list[np.ndarray]]] = {v: [] for v in self.VIEW_NAMES}
        for seq in sequences:
            views = self.make_views(seq)
            for vname in view_seqs:
                view_seqs[vname].append(views[vname])

        final_lls = {}
        for vname in self.VIEW_NAMES:
            seqs = view_seqs[vname]
            hmm = self.hmms[vname]
            n_seqs = len(seqs)
            avg_ll = 0.0

            for epoch in range(em_epochs):
                indices = list(range(n_seqs))
                np.random.default_rng(seed + epoch).shuffle(indices)

                epoch_ll = 0.0
                n_batches = 0
                for start in range(0, n_seqs, batch_size):
                    batch_idx = indices[start : start + batch_size]
                    batch = [seqs[i] for i in batch_idx if len(seqs[i]) >= 2]
                    if not batch:
                        continue
                    ll = hmm.train_step_em(batch)
                    epoch_ll += ll
                    n_batches += 1

                avg_ll = epoch_ll / max(n_batches, 1)
                if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                    print(f"    {vname:>12} epoch {epoch+1:2d}/{em_epochs}: "
                          f"avg_ll={avg_ll:.4f}", flush=True)

            final_lls[vname] = avg_ll

        return final_lls

    # -- Prediction (multi-view routing) ---------------------------------------

    def predict(
        self,
        context_vecs: list[np.ndarray],
    ) -> tuple[np.ndarray, str, float]:
        """Predict the next panel via multi-view routing.

        Selects the view with highest normalized log-likelihood (detection
        confidence), then returns that view's predicted next vector.

        Args:
            context_vecs: List of 8 context panel block-codes.

        Returns:
            Tuple of (predicted_vector, best_view_name, best_confidence).
        """
        views = self.make_views(context_vecs)
        best_view = "absolute"
        best_score = -np.inf
        best_prediction = None

        for vname, hmm in self.hmms.items():
            obs_seq = views[vname]
            if len(obs_seq) < 2:
                continue

            try:
                log_ll, _ = hmm.forward(obs_seq)
                norm_ll = log_ll / len(obs_seq)
            except Exception:
                norm_ll = -np.inf

            try:
                predicted = hmm.predict(obs_seq)
            except Exception:
                continue

            if norm_ll > best_score:
                best_score = norm_ll
                best_view = vname
                best_prediction = predicted

        if best_prediction is None:
            best_prediction = self.hmms["absolute"].predict(views["absolute"])

        return best_prediction, best_view, float(best_score)

    def score_candidates(
        self,
        context_vecs: list[np.ndarray],
        candidate_vecs: list[np.ndarray],
    ) -> int:
        """Score candidate answers and return the best index.

        Args:
            context_vecs: List of 8 context panel block-codes.
            candidate_vecs: List of 8 candidate answer block-codes.

        Returns:
            Index of the best-matching candidate (0-7).
        """
        prediction, _, _ = self.predict(context_vecs)
        prediction_disc = self._bc.discretize(prediction)

        scores = np.array([
            self._bc.similarity(prediction_disc, cv) for cv in candidate_vecs
        ])
        return int(np.argmax(scores))
