"""Cloud API — FastAPI endpoints for the Decision Oracle pre-funding demo.

Endpoints:
    POST /predict — life decisions (128 futures ranked by plausibility)
    POST /book    — alternate book endings
    POST /train   — train Q-values on scenario data
    GET  /health  — health check

Run with:
    uvicorn cubemind.cloud.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from cubemind.core import K_BLOCKS, L_BLOCK, N_WORLDS
from cubemind.execution.decision_oracle import DecisionOracle
from cubemind.execution.world_encoder import WorldEncoder

app = FastAPI(title="CubeMind Decision Oracle", version="0.2.0")

# ── Configurable dims (env override for tests) ──────────────────────────────

_K = int(os.environ.get("CUBEMIND_K", K_BLOCKS))
_L = int(os.environ.get("CUBEMIND_L", L_BLOCK))
_N = int(os.environ.get("CUBEMIND_N_WORLDS", N_WORLDS))
_D_HIDDEN = int(os.environ.get("CUBEMIND_D_HIDDEN", 32))

# ── World personality archetypes ─────────────────────────────────────────────
# Each world model has a personality that biases its predictions.
# These are assigned round-robin to the 128 worlds.

ARCHETYPES = [
    {"name": "Conservative", "bias": "risk-averse, stability-focused",
     "tpl": "In this cautious scenario, {action} leads to {outcome}."},
    {"name": "Ambitious", "bias": "career-driven, high-reward seeking",
     "tpl": "In this ambitious path, {action} opens up {outcome}."},
    {"name": "Health-First", "bias": "wellness-oriented, stress-reducing",
     "tpl": "Prioritizing health, {action} results in {outcome}."},
    {"name": "Social", "bias": "relationship-focused, community-oriented",
     "tpl": "From a social perspective, {action} means {outcome}."},
    {"name": "Financial", "bias": "money-focused, ROI-driven",
     "tpl": "Financially, {action} produces {outcome}."},
    {"name": "Creative", "bias": "innovation-oriented, unconventional",
     "tpl": "Taking the creative route, {action} sparks {outcome}."},
    {"name": "Pragmatic", "bias": "practical, short-term focused",
     "tpl": "Practically speaking, {action} simply means {outcome}."},
    {"name": "Long-Term", "bias": "strategic, 5-year horizon",
     "tpl": "Looking 5 years out, {action} compounds into {outcome}."},
    {"name": "Worst-Case", "bias": "pessimistic, risk-highlighting",
     "tpl": "In the worst case, {action} could lead to {outcome}."},
    {"name": "Best-Case", "bias": "optimistic, opportunity-finding",
     "tpl": "At best, {action} results in {outcome}."},
    {"name": "Balanced", "bias": "moderate, weighing all factors",
     "tpl": "Balancing all factors, {action} likely means {outcome}."},
    {"name": "Contrarian", "bias": "counter-intuitive, against the grain",
     "tpl": "Surprisingly, {action} actually leads to {outcome}."},
    {"name": "Emotional", "bias": "feelings-focused, intuition-driven",
     "tpl": "Emotionally, {action} feels like {outcome}."},
    {"name": "Analytical", "bias": "data-driven, evidence-based",
     "tpl": "The data suggests {action} correlates with {outcome}."},
    {"name": "Adventurous", "bias": "exploration-oriented, novelty-seeking",
     "tpl": "For the adventurous, {action} leads to {outcome}."},
    {"name": "Cautious-Optimist", "bias": "hopeful but prepared",
     "tpl": "Hoping for the best, {action} likely results in {outcome}."},
]

# Outcome descriptors keyed by (archetype, q_value_range)
OUTCOME_DESCRIPTORS = {
    "high_positive": [
        "significant career advancement and financial stability",
        "a breakthrough opportunity you wouldn't have found otherwise",
        "improved wellbeing and stronger relationships",
        "unexpected doors opening in your professional network",
        "a period of creative renewal and personal growth",
        "compound returns on your time investment",
        "a pivotal turning point toward your long-term goals",
        "freedom to explore options you've been putting off",
    ],
    "moderate_positive": [
        "a manageable transition with moderate upside",
        "incremental progress toward your goals",
        "a useful learning experience with low risk",
        "temporary disruption followed by steady recovery",
        "slightly better work-life balance",
        "new perspectives that shift your priorities",
        "a networking opportunity in disguise",
        "minor financial impact but major clarity",
    ],
    "neutral": [
        "minimal change — things continue roughly as before",
        "a lateral move with neither gain nor loss",
        "a brief pause that barely registers long-term",
        "no measurable impact on your trajectory",
        "a forgettable day in the grand scheme",
        "status quo maintained, deadline still met",
    ],
    "negative": [
        "a setback that requires damage control next week",
        "missed opportunities that are hard to recover",
        "increased stress from cascading consequences",
        "financial strain and damaged professional reputation",
        "isolation from your support network",
        "a pattern of avoidance that compounds over time",
    ],
}

# Book ending archetypes
ENDING_ARCHETYPES = [
    "a redemptive resolution where the protagonist finds peace",
    "a tragic twist — the letter reveals a painful truth",
    "an open ending that leaves the central mystery unsolved",
    "a surprise reunion that recontextualizes everything",
    "a bittersweet conclusion — closure at a cost",
    "a dark revelation that reframes the entire narrative",
    "a hopeful new beginning born from the confrontation",
    "a recursive ending — the story starts over with new meaning",
    "a quiet acceptance — no dramatic resolution, just peace",
    "a shocking betrayal hidden in the final act",
    "an ambiguous ending — the reader decides the truth",
    "a cathartic release after decades of silence",
    "a generational echo — the child repeats the cycle",
    "a liberation — the keeper finally speaks his truth",
    "an unexpected connection between past and present",
    "a mythic resolution that transcends the literal story",
]


def _get_archetype(world_id: int) -> dict:
    return ARCHETYPES[world_id % len(ARCHETYPES)]


def _get_outcome(q_value: float, world_id: int) -> str:
    if q_value > 10:
        pool = OUTCOME_DESCRIPTORS["high_positive"]
    elif q_value > 2:
        pool = OUTCOME_DESCRIPTORS["moderate_positive"]
    elif q_value > -2:
        pool = OUTCOME_DESCRIPTORS["neutral"]
    else:
        pool = OUTCOME_DESCRIPTORS["negative"]
    return pool[world_id % len(pool)]


def _narrate_future(
    world_id: int, q_value: float, plausibility: float, action: str,
) -> str:
    arch = _get_archetype(world_id)
    outcome = _get_outcome(q_value, world_id)
    narrative = arch["tpl"].format(action=action, outcome=outcome)
    confidence = "high" if plausibility > 0.3 else (
        "moderate" if plausibility > 0.1 else "speculative"
    )
    return f"[{arch['name']}] ({confidence} confidence) {narrative}"


def _narrate_ending(world_id: int, plausibility: float) -> str:
    ending = ENDING_ARCHETYPES[world_id % len(ENDING_ARCHETYPES)]
    confidence = "high" if plausibility > 0.3 else (
        "moderate" if plausibility > 0.1 else "speculative"
    )
    return f"({confidence} confidence) The story resolves with {ending}."


# ── Lazy globals ─────────────────────────────────────────────────────────────

_oracle: DecisionOracle | None = None
_encoder: WorldEncoder | None = None


def _get_oracle() -> DecisionOracle:
    global _oracle
    if _oracle is None:
        _oracle = DecisionOracle(
            k=_K, l=_L, n_worlds=_N, d_hidden=_D_HIDDEN,
        )
    return _oracle


def _get_encoder() -> WorldEncoder:
    global _encoder
    if _encoder is None:
        _encoder = WorldEncoder(k=_K, l=_L)
    return _encoder


# ── Schemas ──────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    question: str
    context: dict[str, str] = Field(default_factory=dict)
    top_k: int = Field(default=10, gt=0, le=128)


class FutureItem(BaseModel):
    world_id: int
    archetype: str
    narrative: str
    score: float
    plausibility: float
    q_value: float


class PredictResponse(BaseModel):
    question: str
    n_worlds: int
    elapsed_ms: float
    futures: list[FutureItem]


class BookRequest(BaseModel):
    passage: str
    question: str = "What are the alternate endings?"
    top_k: int = Field(default=10, gt=0, le=128)


class BookResponse(BaseModel):
    passage_preview: str
    n_worlds: int
    elapsed_ms: float
    endings: list[FutureItem]


class TrainRequest(BaseModel):
    scenarios: list[dict] = Field(
        ...,
        description="List of {context: {}, actions: [str]} dicts",
    )
    n_epochs: int = Field(default=5, gt=0, le=50)
    beta: float = Field(default=0.9, gt=0, lt=1)


class TrainResponse(BaseModel):
    epochs: int
    scenarios_trained: int
    q_before: float
    q_after: float
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    n_worlds: int
    trained: bool


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    oracle = _get_oracle()
    trained = float(oracle.cvl.xi.sum()) != 0.0
    return HealthResponse(status="ok", n_worlds=_N, trained=trained)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Evaluate 128 parallel futures for a life decision."""
    oracle = _get_oracle()
    encoder = _get_encoder()

    t0 = time.perf_counter()
    state = encoder.encode_state(req.context)
    action = encoder.encode_action(req.question)
    top = oracle.top_k(state, action, world_prior=state, k=req.top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    futures = [
        FutureItem(
            world_id=f["world_id"],
            archetype=_get_archetype(f["world_id"])["name"],
            narrative=_narrate_future(
                f["world_id"], f["q_value"], f["plausibility"],
                req.question,
            ),
            score=round(float(f["score"]), 4),
            plausibility=round(float(f["plausibility"]), 4),
            q_value=round(float(f["q_value"]), 4),
        )
        for f in top
    ]

    return PredictResponse(
        question=req.question,
        n_worlds=oracle.n_worlds,
        elapsed_ms=round(elapsed_ms, 2),
        futures=futures,
    )


@app.post("/book", response_model=BookResponse)
def book(req: BookRequest) -> BookResponse:
    """Generate 128 alternate book endings ranked by plausibility."""
    oracle = _get_oracle()
    encoder = _get_encoder()

    t0 = time.perf_counter()
    narrative = encoder.encode_narrative(req.passage)
    action = encoder.encode_action(req.question)
    top = oracle.top_k(narrative, action, world_prior=narrative, k=req.top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    endings = [
        FutureItem(
            world_id=f["world_id"],
            archetype=_get_archetype(f["world_id"])["name"],
            narrative=_narrate_ending(
                f["world_id"], f["plausibility"],
            ),
            score=round(float(f["score"]), 4),
            plausibility=round(float(f["plausibility"]), 4),
            q_value=round(float(f["q_value"]), 4),
        )
        for f in top
    ]

    preview = (
        req.passage[:117] + "..." if len(req.passage) > 120 else req.passage
    )
    return BookResponse(
        passage_preview=preview,
        n_worlds=oracle.n_worlds,
        elapsed_ms=round(elapsed_ms, 2),
        endings=endings,
    )


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    """Train CVL Q-values on scenario data.

    Each scenario has a context dict and list of action strings.
    The oracle runs self-play on all (context, action) pairs and
    uses plausibility as the reward signal.
    """
    oracle = _get_oracle()
    encoder = _get_encoder()

    t0 = time.perf_counter()

    training_scenarios = []
    for s in req.scenarios:
        ctx = s.get("context", {})
        actions = s.get("actions", [])
        state = encoder.encode_state(ctx)
        for act in actions:
            action = encoder.encode_action(act)
            training_scenarios.append({
                "state": state,
                "action": action,
                "prior": state,
            })

    stats = oracle.train(
        training_scenarios, n_epochs=req.n_epochs, beta=req.beta,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return TrainResponse(
        epochs=stats["epochs"],
        scenarios_trained=stats["scenarios"],
        q_before=round(stats["mean_q_before"], 4),
        q_after=round(stats["mean_q_after"], 4),
        elapsed_ms=round(elapsed_ms, 2),
    )
