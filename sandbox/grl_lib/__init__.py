"""GRL Experiments Library — reusable framework for sandbox experiments.

Independent from cubemind core. Only numpy and standard library.
Experiments import from here. Production code NEVER imports from here.

Modules:
    kernels  — RBF, Matern, RFF kernel functions
    routing  — bandit, UCB, force-based routing strategies
    experts  — base expert class, expert factory
    traces   — eligibility traces, consolidation rules
    memory   — capsule stores, replay mechanisms
    metrics  — ablation helpers, FLOP counters, timers, energy tracking
"""

from .kernels import rbf_kernel, rkhs_distance_sq, RandomFourierFeatures
from .routing import BanditRouter
from .experts import BaseExpert, ExpertFactory
from .traces import EligibilityTrace
from .memory import CapsuleStore
from .metrics import Timer, FlopCounter, AblationTable
