"""Theory of Mind via VSA hypernetworks.

Models other agents' mental states by observing their behavior and inferring
their hidden goals. Each agent's mind is modeled as an HMM-VSA rule whose
transition matrix captures their behavioral patterns. A HYLA hypernetwork
transforms the HMM posteriors into belief-state hypervectors.

Key equation:
  belief = HYLA_ToM(HMM_forward(observations))
  goal = unbind(belief, current_state)
  Q_social = Q_self + lambda_ToM * sum_i delta(goal_i, bind(state, action))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cubemind.ops import BlockCodes
from cubemind.reasoning.hmm_rule import HMMRule, _logsumexp
from cubemind.execution.hyla import HYLA
from cubemind.telemetry import metrics


# -- Data structures -----------------------------------------------------------


@dataclass
class BeliefState:
    """Represents an inferred belief about another agent's mental state.

    Attributes:
        agent_id: Unique identifier for the modeled agent.
        belief_vector: Block-code belief hypervector of shape (k, l).
        confidence: How confident the model is about this belief.
        inferred_goal: Unbinding result of shape (k, l), or None.
    """

    agent_id: str
    belief_vector: np.ndarray
    confidence: float
    inferred_goal: np.ndarray | None = field(default=None)


# -- Agent Model ---------------------------------------------------------------


class AgentModel:
    """Model of a single observed agent's mind.

    Combines an HMMRule (capturing behavioral transition patterns) with a
    small HYLA hypernetwork (transforming HMM posteriors into belief-state
    hypervectors).

    Args:
        agent_id: Unique identifier for this agent.
        codebook: Shape (n_states, k, l) -- block-code vectors for each state.
        k: Number of blocks in the block-code representation.
        l: Length of each block.
        temperature: Softmax temperature for HMM emission probabilities.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        agent_id: str,
        codebook: np.ndarray,
        k: int,
        l: int,
        temperature: float = 40.0,
        seed: int = 42,
    ) -> None:
        self.agent_id = agent_id
        self.codebook = codebook
        self.k = k
        self.l = l
        self._seed = seed
        self._bc = BlockCodes(k=k, l=l)

        n_states = codebook.shape[0]
        d_vsa = k * l

        # HMM captures this agent's behavioral transition patterns
        self.hmm = HMMRule(codebook, temperature=temperature, seed=seed)

        # HYLA transforms posterior distributions into belief hypervectors
        d_hidden = max(n_states * 2, 16)
        self.hyla = HYLA(
            d_vsa=d_vsa,
            d_hidden=d_hidden,
            d_out=d_vsa,
            k=k,
            l=l,
            seed=seed + 1000,
        )

        self._history: list[np.ndarray] = []

    @property
    def n_states(self) -> int:
        return self.codebook.shape[0]

    def observe(self, observation: np.ndarray) -> None:
        """Add an observation to this agent's behavior history.

        Args:
            observation: Block-code vector (k, l) representing the observed action.
        """
        self._history.append(observation)

    def get_belief(self, current_state: np.ndarray | None = None) -> BeliefState:
        """Compute the current belief about this agent's mental state.

        1. Run HMM forward on the observation history.
        2. Extract the posterior of the last timestep.
        3. Transform via HYLA to get the belief hypervector.
        4. If current_state is provided, unbind to get the inferred goal.

        Args:
            current_state: If provided, shape (k, l). Used for goal inference.

        Returns:
            BeliefState containing belief vector, confidence, and optional goal.
        """
        if not self._history:
            belief_vec = self._bc.random_discrete(seed=0)
            return BeliefState(
                agent_id=self.agent_id,
                belief_vector=belief_vec,
                confidence=0.0,
                inferred_goal=None,
            )

        # Step 1: Run HMM forward algorithm
        log_likelihood, log_alpha = self.hmm.forward(self._history)

        # Step 2: Posterior of last timestep
        log_alpha_T = log_alpha[-1]
        log_norm = _logsumexp(log_alpha_T)
        posterior = np.exp(log_alpha_T - log_norm)

        # Confidence from per-step average log-likelihood
        T = len(self._history)
        avg_log_likelihood = float(log_norm) / T
        confidence = float(np.exp(avg_log_likelihood))
        if not np.isfinite(confidence):
            confidence = 0.0

        # Step 3: Transform posterior to belief via HYLA
        d_vsa = self.k * self.l
        posterior_padded = np.zeros(d_vsa, dtype=np.float32)
        posterior_padded[: len(posterior)] = posterior.astype(np.float32)

        belief_flat = self.hyla.forward(posterior_padded, posterior_padded)
        belief_vec = belief_flat.reshape(self.k, self.l).astype(np.float32)

        # Normalize each block to sum to 1
        block_sums = belief_vec.sum(axis=-1, keepdims=True)
        block_sums = np.where(block_sums == 0, 1.0, block_sums)
        belief_vec = (belief_vec / block_sums).astype(np.float32)

        # Step 4: Optionally unbind to get inferred goal
        inferred_goal = None
        if current_state is not None:
            inferred_goal = self._bc.unbind(belief_vec, current_state)

        return BeliefState(
            agent_id=self.agent_id,
            belief_vector=belief_vec,
            confidence=confidence,
            inferred_goal=inferred_goal,
        )

    def reset(self) -> None:
        """Clear this agent's observation history."""
        self._history.clear()

    def train_step(
        self,
        observations: list[np.ndarray],
        target_behavior: np.ndarray,
        lr: float = 0.01,
    ) -> float:
        """Update HMM transitions from observed behavior.

        Args:
            observations: Sequence of observation vectors, each (k, l).
            target_behavior: The target next-state vector (k, l).
            lr: Learning rate for gradient update.

        Returns:
            Loss before the update.
        """
        return self.hmm.train_step(observations, target_behavior, lr=lr)


# -- Theory of Mind ------------------------------------------------------------


class TheoryOfMind:
    """Multi-agent Theory of Mind via VSA hypernetworks.

    Manages separate AgentModels for each observed agent and computes
    social Q-values that incorporate inferred goals.

    Args:
        n_agents: Expected number of agents (advisory; new agents auto-created).
        codebook: Shape (n_states, k, l) -- shared block-code codebook.
        k: Number of blocks.
        l: Block length.
        lambda_tom: Weight for social Q-value contribution.
        seed: Base RNG seed.
    """

    def __init__(
        self,
        n_agents: int,
        codebook: np.ndarray,
        k: int,
        l: int,
        lambda_tom: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_agents = n_agents
        self.codebook = codebook
        self.k = k
        self.l = l
        self.lambda_tom = lambda_tom
        self._seed = seed
        self._bc = BlockCodes(k=k, l=l)

        self.agents: dict[str, AgentModel] = {}
        self._agent_counter = 0

    def _get_or_create_agent(self, agent_id: str) -> AgentModel:
        """Get an existing agent model or create a new one."""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentModel(
                agent_id=agent_id,
                codebook=self.codebook,
                k=self.k,
                l=self.l,
                seed=self._seed + self._agent_counter * 100,
            )
            self._agent_counter += 1
        return self.agents[agent_id]

    def observe_agent(self, agent_id: str, observation: np.ndarray) -> None:
        """Record an observation of a specific agent's behavior.

        Args:
            agent_id: Unique identifier for the observed agent.
            observation: Block-code vector (k, l) of the observed action.
        """
        agent = self._get_or_create_agent(agent_id)
        agent.observe(observation)

    def update_belief(self, agent_id: str, observation: np.ndarray) -> BeliefState:
        """Update belief about an agent after observing an action.

        Convenience method that observes and then returns the updated belief.

        Args:
            agent_id: The agent observed.
            observation: Block-code vector (k, l) of the observed action.

        Returns:
            Updated BeliefState for the agent.
        """
        self.observe_agent(agent_id, observation)
        agent = self.agents[agent_id]
        belief = agent.get_belief()
        metrics.record("tom.belief_confidence", belief.confidence,
                       tags={"agent": agent_id})
        return belief

    def predict_action(
        self,
        agent_id: str,
        action_codebook: np.ndarray,
    ) -> int:
        """Predict the next action an agent will take.

        Uses the agent's HMM to predict the next state, then finds the
        closest action in the action codebook.

        Args:
            agent_id: The agent whose next action to predict.
            action_codebook: Codebook of possible actions, shape (n_actions, k, l).

        Returns:
            Index of the most likely action in action_codebook.
        """
        agent = self._get_or_create_agent(agent_id)

        if not agent._history:
            return 0

        predicted = agent.hmm.predict(agent._history)
        sims = self._bc.similarity_batch(predicted, action_codebook)
        return int(np.argmax(sims))

    def get_all_beliefs(
        self, current_state: np.ndarray | None = None
    ) -> dict[str, BeliefState]:
        """Get belief states for all observed agents.

        Args:
            current_state: If provided, shape (k, l). Used for goal inference.

        Returns:
            Mapping from agent_id to BeliefState.
        """
        return {
            agent_id: agent.get_belief(current_state)
            for agent_id, agent in self.agents.items()
        }

    def social_q_value(
        self,
        state: np.ndarray,
        action: np.ndarray,
        q_self: float,
    ) -> float:
        """Compute Q_social incorporating inferred goals of other agents.

        Q_social = q_self + lambda_ToM * sum_i confidence_i * delta(goal_i, bind(state, action))

        Args:
            state: Current state, shape (k, l).
            action: Proposed action, shape (k, l).
            q_self: Self Q-value for this state-action pair.

        Returns:
            Social Q-value.
        """
        if not self.agents or self.lambda_tom == 0.0:
            return q_self

        state_action = self._bc.bind(state, action)

        social_contribution = 0.0
        beliefs = self.get_all_beliefs(current_state=state)

        for agent_id, belief in beliefs.items():
            if belief.inferred_goal is None:
                continue
            alignment = self._bc.similarity(belief.inferred_goal, state_action)
            centered_alignment = alignment - 0.5
            social_contribution += belief.confidence * centered_alignment

        return q_self + self.lambda_tom * social_contribution

    def cooperation_score(self, agent_id: str, my_goal: np.ndarray) -> float:
        """Measure how aligned another agent's inferred goal is with my goal.

        Args:
            agent_id: The agent to check alignment with.
            my_goal: My own goal vector, shape (k, l).

        Returns:
            Cosine similarity in [-1, 1] between the two goals.
        """
        agent = self._get_or_create_agent(agent_id)

        if agent._history:
            current_state = agent._history[-1]
        else:
            current_state = self._bc.random_discrete(seed=0)

        belief = agent.get_belief(current_state=current_state)
        if belief.inferred_goal is None:
            return 0.0

        goal_flat = belief.inferred_goal.flatten().astype(np.float64)
        my_flat = my_goal.flatten().astype(np.float64)

        norm_goal = np.linalg.norm(goal_flat)
        norm_my = np.linalg.norm(my_flat)

        if norm_goal < 1e-12 or norm_my < 1e-12:
            return 0.0

        cos_sim = float(np.dot(goal_flat, my_flat) / (norm_goal * norm_my))
        return float(np.clip(cos_sim, -1.0, 1.0))
