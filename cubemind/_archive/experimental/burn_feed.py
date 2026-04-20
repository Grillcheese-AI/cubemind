"""Live LLM Burn Feed -- continuous ecological state for the CubeMind world model.

Computes the real-time cumulative ecological and financial cost of the LLM
paradigm and encodes it as a block-code vector that feeds into MoWM routing.

Every CubeMind query gets a `burn_context` -- a block-code vector encoding
the current state of planetary destruction from LLMs. The world model uses
this as a penalty signal: routes that align with the burn trajectory get
lower Q-values.

Usage:
    from cubemind.experimental.burn_feed import BurnFeed

    feed = BurnFeed()
    state = feed.now()           # Current burn state as dict
    ctx = feed.context_vector()  # Ready for MoWM routing
    feed.print_status()          # Human-readable summary
"""

from datetime import datetime

import numpy as np

from cubemind.ops import BlockCodes
from cubemind.telemetry import metrics


# -- Constants -----------------------------------------------------------------

CHATGPT_LAUNCH = datetime(2022, 11, 30)
CAPEX_SHIFT = datetime(2026, 2, 1)

HISTORICAL_BURN_USD_PER_SEC = 1500
CAPEX_BURN_USD_PER_SEC = 7399

TRAINING_JOULES = 6.75e17
TRAINING_CO2_TONS = 90_000_000

# Estimated industry inference scale
TOKENS_PER_SEC = 2_083_333_333
JOULES_PER_TOKEN = 6.12
INFERENCE_JOULES_PER_SEC = TOKENS_PER_SEC * JOULES_PER_TOKEN
INFERENCE_CO2_PER_SEC = (INFERENCE_JOULES_PER_SEC / 3_600_000) * (0.48 / 1000)

CO2_PER_DEATH = 4434
EFFICIENCY_RATIO = 200_000_000


class BurnFeed:
    """Live ecological state feed for the CubeMind world model.

    Computes real-time cumulative costs and encodes them as block-code
    vectors for MoWM routing.

    Args:
        k: Number of blocks per vector.
        l: Block length.
        n_levels: Quantization levels for scalar encoding.
    """

    def __init__(self, k: int = 8, l: int = 32, n_levels: int = 64) -> None:
        self.k = k
        self.l = l
        self._bc = BlockCodes(k=k, l=l)
        self.codebook = self._bc.codebook_discrete(n_levels, seed=42)

        # Role vectors for binding different metrics
        self._role_usd = self._bc.random_discrete(seed=100)
        self._role_co2 = self._bc.random_discrete(seed=200)
        self._role_joules = self._bc.random_discrete(seed=300)
        self._role_deaths = self._bc.random_discrete(seed=500)

    def now(self) -> dict:
        """Compute the current burn state.

        Returns:
            Dictionary with timestamp, phase, USD burn, CO2, joules,
            excess deaths, and CubeMind-equivalent costs.
        """
        t = datetime.now()
        total_sec = (t - CHATGPT_LAUNCH).total_seconds()

        if t < CAPEX_SHIFT:
            phase = "historical_ramp"
            usd = total_sec * HISTORICAL_BURN_USD_PER_SEC
        else:
            sec_pre = (CAPEX_SHIFT - CHATGPT_LAUNCH).total_seconds()
            sec_post = (t - CAPEX_SHIFT).total_seconds()
            phase = "capex_hockey_stick"
            usd = (sec_pre * HISTORICAL_BURN_USD_PER_SEC) + (
                sec_post * CAPEX_BURN_USD_PER_SEC
            )

        joules = TRAINING_JOULES + (total_sec * INFERENCE_JOULES_PER_SEC)
        co2 = TRAINING_CO2_TONS + (total_sec * INFERENCE_CO2_PER_SEC)
        deaths = co2 / CO2_PER_DEATH

        return {
            "timestamp": t.isoformat(),
            "days_since_chatgpt": (t - CHATGPT_LAUNCH).days,
            "phase": phase,
            "total_seconds": total_sec,
            "usd_burn": usd,
            "joules": joules,
            "co2_tons": co2,
            "excess_deaths": deaths,
            "cubemind_usd": usd / EFFICIENCY_RATIO,
            "cubemind_co2": co2 / EFFICIENCY_RATIO,
            "cubemind_joules": joules / EFFICIENCY_RATIO,
            "efficiency_ratio": EFFICIENCY_RATIO,
        }

    def _encode_scalar(self, value: float, max_val: float) -> np.ndarray:
        """Encode a scalar to block-code via quantization."""
        n = len(self.codebook)
        idx = int(min(max(value / max_val * (n - 1), 0), n - 1))
        return self.codebook[idx]

    def context_vector(self) -> np.ndarray:
        """Encode the current burn state as a single block-code vector.

        Binds USD, CO2, energy, and deaths into one composite vector.
        This feeds directly into MoWM routing as context.

        Returns:
            Block-code vector (k, l) encoding the full burn state.
        """
        state = self.now()

        # Normalization maxima (projected to end of 2027)
        max_usd = 1e12
        max_co2 = 5e8
        max_joules = 5e18
        max_deaths = 2e5

        usd_vec = self._encode_scalar(state["usd_burn"], max_usd)
        co2_vec = self._encode_scalar(state["co2_tons"], max_co2)
        joules_vec = self._encode_scalar(state["joules"], max_joules)
        deaths_vec = self._encode_scalar(state["excess_deaths"], max_deaths)

        # Bind each metric with its role vector
        bound_usd = self._bc.bind(self._role_usd, usd_vec)
        bound_co2 = self._bc.bind(self._role_co2, co2_vec)
        bound_joules = self._bc.bind(self._role_joules, joules_vec)
        bound_deaths = self._bc.bind(self._role_deaths, deaths_vec)

        # Bundle all into one composite burn state
        composite = self._bc.bundle(
            [bound_usd, bound_co2, bound_joules, bound_deaths],
            normalize=True,
        )

        metrics.record("burn_feed.penalty", self.penalty_score())
        return composite

    def unbind_metric(self, composite: np.ndarray, metric: str) -> np.ndarray:
        """Extract a specific metric from the composite burn vector.

        Args:
            composite: The composite burn state vector (k, l).
            metric: One of 'usd', 'co2', 'joules', 'deaths'.

        Returns:
            The recovered metric vector (k, l).
        """
        roles = {
            "usd": self._role_usd,
            "co2": self._role_co2,
            "joules": self._role_joules,
            "deaths": self._role_deaths,
        }
        role = roles.get(metric)
        if role is None:
            raise ValueError(f"Unknown metric: {metric}. Use: {list(roles.keys())}")
        return self._bc.unbind(composite, role)

    def penalty_score(self) -> float:
        """Compute a routing penalty based on current burn state.

        Returns a value in [0, 1] where 1 = maximum ecological crisis.
        Used by MoWM to penalize routes that would increase the burn.
        """
        state = self.now()
        usd_norm = min(state["usd_burn"] / 1e12, 1.0)
        co2_norm = min(state["co2_tons"] / 5e8, 1.0)
        deaths_norm = min(state["excess_deaths"] / 2e5, 1.0)
        return (usd_norm + co2_norm + deaths_norm) / 3.0

    def print_status(self) -> None:
        """Print human-readable burn status."""
        s = self.now()
        print(f"=== LLM Burn Status ({s['timestamp'][:19]}) ===")
        print(f"  Day {s['days_since_chatgpt']} since ChatGPT | Phase: {s['phase']}")
        print(f"  USD burned:      ${s['usd_burn']:>20,.0f}")
        print(f"  CO2 emitted:     {s['co2_tons']:>20,.0f} tons")
        print(f"  Energy consumed: {s['joules']:>20.2e} J")
        print(f"  Excess deaths:   {s['excess_deaths']:>20,.0f}")
        print("  ---")
        print(f"  CubeMind equiv:  ${s['cubemind_usd']:>20,.2f}")
        print(f"  Efficiency:      {s['efficiency_ratio']:>20,}x")
        print(f"  Penalty score:   {self.penalty_score():.4f}")
