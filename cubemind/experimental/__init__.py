"""CubeMind experimental — unstable features for exploration and research."""

from cubemind.experimental.bandits import RuleExplorer, OnlineBanditSolver
from cubemind.experimental.burn_feed import BurnFeed
from cubemind.experimental.theory_of_mind import TheoryOfMind, AgentModel
from cubemind.experimental.convergence import (
    ConvergenceMonitor,
    rhat,
    split_rhat,
    ess,
    check_convergence,
)

__all__ = [
    "RuleExplorer",
    "OnlineBanditSolver",
    "BurnFeed",
    "TheoryOfMind",
    "AgentModel",
    "ConvergenceMonitor",
    "rhat",
    "split_rhat",
    "ess",
    "check_convergence",
]
