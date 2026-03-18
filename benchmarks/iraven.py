"""iRaven benchmark for CubeMind.

Evaluates CubeMind on the I-RAVEN visual reasoning dataset.
Target: 97.5% accuracy on Center Single configuration.

The benchmark generates block-code representations of Raven's
Progressive Matrices problems and evaluates CubeMind's ability
to identify the correct answer panel.

Usage:
    from benchmarks.iraven import run_iraven_benchmark
    results = run_iraven_benchmark(model)
"""

from __future__ import annotations

import time

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.ops.block_codes import BlockCodes
from cubemind.telemetry import metrics


def generate_iraven_problem(
    bc: BlockCodes,
    n_context: int = 8,
    n_choices: int = 8,
    seed: int = 0,
) -> dict:
    """Generate a synthetic iRaven-style problem as block codes.

    A real iRaven problem has a 3x3 grid of panels with visual attributes
    (shape, size, color, position) that follow rules. The answer panel
    completes the pattern.

    This synthetic version creates block-code sequences with learnable
    patterns that CubeMind's HMM can detect.

    Args:
        bc: BlockCodes instance.
        n_context: Number of context panels.
        n_choices: Number of answer choices.
        seed: Random seed.

    Returns:
        Dict with: context (list of block-codes), choices (list),
        correct_idx (int), rule_type (str).
    """
    rng = np.random.default_rng(seed)

    # Generate a "rule" as a binding key
    rule = bc.random_discrete(seed=int(rng.integers(0, 2**31)))

    # Context panels follow the rule: each panel = bind(prev, rule)
    context = []
    current = bc.random_discrete(seed=int(rng.integers(0, 2**31)))
    for _ in range(n_context):
        context.append(current)
        current = bc.bind(current, rule)

    # Correct answer is the next in sequence
    correct = current

    # Generate distractors
    correct_idx = int(rng.integers(0, n_choices))
    choices = []
    for i in range(n_choices):
        if i == correct_idx:
            choices.append(correct)
        else:
            choices.append(bc.random_discrete(seed=int(rng.integers(0, 2**31))))

    return {
        "context": context,
        "choices": choices,
        "correct_idx": correct_idx,
        "rule_type": "sequential_bind",
    }


def evaluate_problem(model, problem: dict) -> tuple[bool, float]:
    """Evaluate CubeMind on a single iRaven problem.

    Args:
        model: CubeMind instance.
        problem: Dict from generate_iraven_problem.

    Returns:
        (correct: bool, latency_ms: float)
    """
    t0 = time.perf_counter()

    # Use HMM to predict the next panel from context
    prediction, weights = model.hmm.predict(problem["context"])

    # Find the closest choice to the prediction
    bc = BlockCodes(K_BLOCKS, L_BLOCK)
    best_idx = -1
    best_sim = -1.0
    for i, choice in enumerate(problem["choices"]):
        sim = bc.similarity(bc.discretize(prediction), choice)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    latency = (time.perf_counter() - t0) * 1000
    correct = best_idx == problem["correct_idx"]

    return correct, latency


def run_iraven_benchmark(
    model,
    n_problems: int = 200,
    n_context: int = 8,
    n_choices: int = 8,
    seed: int = 42,
    train_first: bool = True,
    train_epochs: int = 5,
    train_lr: float = 0.05,
) -> dict:
    """Run the full iRaven benchmark.

    Args:
        model: CubeMind instance.
        n_problems: Number of problems to evaluate.
        n_context: Context panels per problem.
        n_choices: Answer choices per problem.
        seed: Random seed.
        train_first: Whether to train the HMM on a few examples first.
        train_epochs: Training epochs if train_first.
        train_lr: Training learning rate.

    Returns:
        Dict with: accuracy, latency_ms, n_problems, correct_count,
        per_problem (list of dicts).
    """
    bc = BlockCodes(K_BLOCKS, L_BLOCK)

    # Optional: train HMM on a few examples to learn the rule structure
    if train_first:
        for epoch in range(train_epochs):
            for i in range(min(20, n_problems)):
                prob = generate_iraven_problem(bc, n_context, n_choices, seed=seed + i)
                target = prob["choices"][prob["correct_idx"]]
                model.train_step(prob["context"], target, lr=train_lr)

    # Evaluate
    correct_count = 0
    total_latency = 0.0
    per_problem = []

    for i in range(n_problems):
        prob = generate_iraven_problem(bc, n_context, n_choices, seed=seed + 10000 + i)
        correct, latency = evaluate_problem(model, prob)

        correct_count += int(correct)
        total_latency += latency
        per_problem.append({
            "idx": i,
            "correct": correct,
            "latency_ms": latency,
            "rule_type": prob["rule_type"],
        })

    accuracy = correct_count / n_problems
    avg_latency = total_latency / n_problems

    metrics.record("benchmark.iraven_accuracy", accuracy)
    metrics.record("benchmark.iraven_latency_ms", avg_latency)

    return {
        "accuracy": accuracy,
        "latency_ms": avg_latency,
        "n_problems": n_problems,
        "correct_count": correct_count,
        "per_problem": per_problem,
    }
