"""I-RAVEN-X out-of-distribution abstract reasoning benchmark for CubeMind.

Evaluates CubeMind's integer-domain rule detectors on the I-RAVEN-X dataset,
an extended variant of I-RAVEN with configurable value ranges (maxval) and
grid sizes (n). Problems use pure integer attributes (Type, Size, Color) in
an n x n grid format with 8 candidate answers.

Dataset generation: IBM's I-RAVEN-X
    https://github.com/IBM/iraven-x
    Source: C:\\Users\\grill\\Documents\\GitHub\\iraven-x

Each JSON problem has:
    - rules: list of dicts mapping attributes to rule names
    - rpm: list of (n*n + 7) panel lists, each panel is [entity_dict, ...]
      where entity_dict has Type/Size/Color/Angle as STRING values
    - target: correct answer index (0-7) among the 8 candidates
    - Context panels = rpm[0 : n*n - 1]
    - Candidate panels = rpm[n*n - 1 : n*n + 7]

Rule types: Constant, Progression, Arithmetic, Distribute_Three

Reference results (n=3, 10k problems, seed=1234):
    maxval=10:   98.9%
    maxval=100:  81.2%
    maxval=1000: 79.3%

Usage:
    # Evaluate on pre-generated data
    python -m benchmarks.iravenx --data-dir path/to/json_files

    # Generate and evaluate (requires iraven-x on sys.path)
    python -m benchmarks.iravenx --maxval 100 --n-problems 1000

    # Multiple difficulty levels
    python -m benchmarks.iravenx --maxval 10 100 1000 --n-problems 500

    # Programmatic
    from benchmarks.iravenx import run_iravenx_benchmark
    results = run_iravenx_benchmark(maxvals=[10, 100], n_problems=500)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Attributes used for scoring (Angle is a confounder, excluded)
SCORE_ATTRS = ["Type", "Size", "Color"]

# Rule types present in I-RAVEN-X
RULE_TYPES = ["Constant", "Progression", "Arithmetic", "Distribute_Three"]

# Known reference accuracies (n=3, 10k problems, seed=1234)
REFERENCE_RESULTS = {
    10: 98.9,
    100: 81.2,
    1000: 79.3,
}

# Default path to the I-RAVEN-X generation source
IRAVENX_SRC = Path(r"C:\Users\grill\Documents\GitHub\iraven-x")


# -- Data Loading / Generation ------------------------------------------------


def load_iravenx_json(filepath: str | Path) -> dict:
    """Load an I-RAVEN-X JSON file and return the problems dict.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Dict mapping string indices to problem dicts.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"I-RAVEN-X data file not found: {filepath}")

    logger.info("Loading I-RAVEN-X data from %s", filepath)
    with open(filepath, "r") as f:
        data = json.load(f)

    logger.info("Loaded %d problems from %s", len(data), filepath.name)
    return data


def find_iravenx_files(
    data_dir: str | Path,
    maxval: int | None = None,
    n: int | None = None,
) -> list[Path]:
    """Find I-RAVEN-X JSON files in a directory, optionally filtering.

    Filenames follow the convention:
        center_single{rule}_{strategy}_n_{n}_maxval_{maxval}.json

    Args:
        data_dir: Directory containing JSON files.
        maxval: Filter by maxval (None = all).
        n: Filter by grid size (None = all).

    Returns:
        Sorted list of matching file paths.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    files = sorted(data_dir.glob("center_single*.json"))

    if maxval is not None:
        files = [f for f in files if f"_maxval_{maxval}" in f.name]
    if n is not None:
        files = [f for f in files if f"_n_{n}_" in f.name]

    return files


def generate_iravenx_data(
    n: int = 3,
    maxval: int = 50,
    n_problems: int = 1000,
    seed: int = 1234,
    save_dir: str | Path | None = None,
) -> dict:
    """Generate I-RAVEN-X problems in-process using the IBM generation code.

    Imports the iravenx_task module from the I-RAVEN-X repository and calls
    its generation functions directly.

    Args:
        n: Grid size (3 = standard 3x3).
        maxval: Maximum attribute value (controls difficulty).
        n_problems: Number of problems to generate.
        seed: Random seed for reproducibility.
        save_dir: If provided, save the JSON to this directory.

    Returns:
        Dict mapping string indices to problem dicts.
    """
    # Add iraven-x source to path if needed
    iraven_src = str(IRAVENX_SRC / "src" / "datasets" / "generation")
    if iraven_src not in sys.path:
        sys.path.insert(0, iraven_src)

    try:
        import iravenx_task
    except ImportError:
        raise ImportError(
            f"Cannot import iravenx_task. Ensure I-RAVEN-X source exists at "
            f"{IRAVENX_SRC} or provide pre-generated JSON via --data-dir."
        )

    logger.info(
        "Generating %d I-RAVEN-X problems (n=%d, maxval=%d, seed=%d)",
        n_problems, n, maxval, seed,
    )

    iravenx_task.set_seeds(seed)

    t0 = time.perf_counter()
    samples = {}
    for i in range(n_problems):
        samples[str(i)] = iravenx_task.get_sample(
            n, maxval, rule="", arithmetic_strategy="shuffle"
        )
    elapsed = time.perf_counter() - t0

    logger.info("Generated %d problems in %.1fs", n_problems, elapsed)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"center_single_shuffle_n_{n}_maxval_{maxval}.json"
        save_path = save_dir / filename
        with open(save_path, "w") as f:
            json.dump(samples, f, indent=1, default=str)
        logger.info("Saved to %s", save_path)

    return samples


# -- Problem Parsing -----------------------------------------------------------


def parse_problem(problem: dict, n: int = 3) -> dict:
    """Parse a single I-RAVEN-X problem into context, candidates, target, rules.

    Converts string attribute values to integers and separates the rpm list
    into context panels and candidate panels.

    Args:
        problem: Raw problem dict from I-RAVEN-X JSON.
        n: Grid size (determines context/candidate split).

    Returns:
        Dict with keys:
            context: list of 8 dicts (for n=3; n*n-1 panels)
            candidates: list of 8 dicts
            target: int (correct answer index)
            rules: dict mapping attribute names to rule type strings
    """
    rpm = problem["rpm"]
    n_context = n * n - 1  # 8 for n=3, 24 for n=5, 99 for n=10

    # Context panels: first n*n - 1 panels
    context = []
    for i in range(n_context):
        entity = rpm[i][0]  # Each panel has one entity (center_single)
        context.append({
            "Type": int(entity["Type"]),
            "Size": int(entity["Size"]),
            "Color": int(entity["Color"]),
        })

    # Candidate panels: next 8 panels
    candidates = []
    for i in range(8):
        entity = rpm[n_context + i][0]
        candidates.append({
            "Type": int(entity["Type"]),
            "Size": int(entity["Size"]),
            "Color": int(entity["Color"]),
        })

    # Target
    target = int(problem["target"])

    # Rules
    rules = problem["rules"][0] if problem.get("rules") else {}

    return {
        "context": context,
        "candidates": candidates,
        "target": target,
        "rules": rules,
    }


def get_rule_type(rules: dict) -> str:
    """Extract the dominant rule type from a problem's rule dict.

    I-RAVEN-X assigns a rule per attribute (Type, Size, Color). When all
    three are the same, that's the rule type. Otherwise returns "Mixed".
    """
    attr_rules = [rules.get(a, "Unknown") for a in ["Type", "Size", "Color"]]
    unique = set(attr_rules)
    if len(unique) == 1:
        return unique.pop()
    return "Mixed"


# -- Evaluation ----------------------------------------------------------------


def evaluate_problem(
    context: list[dict],
    candidates: list[dict],
    target: int,
    attrs: list[str] | None = None,
) -> tuple[bool, int, list[float]]:
    """Evaluate a single I-RAVEN-X problem using CubeMind rule detectors.

    Args:
        context: List of context panel attribute dicts.
        candidates: List of 8 candidate attribute dicts.
        target: Correct answer index.
        attrs: Attributes to score on.

    Returns:
        Tuple of (correct, predicted_idx, scores).
    """
    from cubemind.reasoning.rule_detectors import score_candidates

    if attrs is None:
        attrs = SCORE_ATTRS

    scores = score_candidates(context, candidates, attrs=attrs)
    predicted = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = predicted == target

    return correct, predicted, scores


def evaluate_problem_nxn(
    context: list[dict],
    candidates: list[dict],
    target: int,
    n: int,
    attrs: list[str] | None = None,
) -> tuple[bool, int, list[float]]:
    """Evaluate an n x n I-RAVEN-X problem using row-wise rule detectors.

    For n > 3, the standard 3x3 rule detectors cannot be applied directly.
    Instead, we use per-row analysis: check each row for rule consistency,
    then predict the missing value (last cell of the last row) and score
    candidates by attribute match.

    For n = 3, delegates to the standard evaluate_problem.

    Args:
        context: List of (n*n - 1) context panel attribute dicts.
        candidates: List of 8 candidate attribute dicts.
        target: Correct answer index.
        n: Grid size.
        attrs: Attributes to score on.

    Returns:
        Tuple of (correct, predicted_idx, scores).
    """
    if n == 3:
        return evaluate_problem(context, candidates, target, attrs)

    if attrs is None:
        attrs = SCORE_ATTRS

    from cubemind.reasoning.rule_detectors import (
        detect_arithmetic,
        detect_col_constant,
        detect_col_distribute,
        detect_col_progression,
        detect_constant,
        detect_distribute,
        detect_progression,
    )

    # Build n x n grid from context for each attribute, predict the missing
    # value (last cell of last row), then score candidates.
    predictions = {}
    for attr in attrs:
        # Build full grid: context fills row-major, last cell is missing
        grid_vals = [c[attr] for c in context]  # n*n - 1 values
        grid = []
        idx = 0
        for row in range(n):
            row_vals = []
            for col in range(n):
                if row == n - 1 and col == n - 1:
                    row_vals.append(None)
                else:
                    row_vals.append(grid_vals[idx])
                    idx += 1
            grid.append(row_vals)

        # Try detectors on the last 3 rows x last 3 cols (the "corner" sub-grid)
        # This is the most general approach for arbitrary n.
        if n >= 3:
            sub_grid = [
                [grid[n - 3][n - 3], grid[n - 3][n - 2], grid[n - 3][n - 1]],
                [grid[n - 2][n - 3], grid[n - 2][n - 2], grid[n - 2][n - 1]],
                [grid[n - 1][n - 3], grid[n - 1][n - 2], None],
            ]

            for detector_fn in [
                detect_constant,
                detect_progression,
                detect_arithmetic,
                detect_distribute,
                detect_col_constant,
                detect_col_progression,
                detect_col_distribute,
            ]:
                result = detector_fn(sub_grid)
                if result is not None:
                    predictions[attr] = result
                    break

    # Score candidates by matching predictions
    n_cand = len(candidates)
    scores = [0.0] * n_cand
    for attr, predicted_val in predictions.items():
        n_matching = sum(1 for c in candidates if c.get(attr) == predicted_val)
        weight = n_cand / max(n_matching, 1)
        for i, cand in enumerate(candidates):
            if cand.get(attr) == predicted_val:
                scores[i] += weight

    predicted = int(max(range(n_cand), key=lambda i: scores[i]))
    correct = predicted == target

    return correct, predicted, scores


# -- Benchmark Runner ----------------------------------------------------------


def run_iravenx_benchmark(
    maxvals: list[int] | None = None,
    grid_sizes: list[int] | None = None,
    n_problems: int = 1000,
    data_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
    seed: int = 1234,
) -> dict:
    """Run the I-RAVEN-X benchmark across difficulty levels.

    Either loads pre-generated JSON data from data_dir, or generates new
    problems using the I-RAVEN-X generation code.

    Args:
        maxvals: List of maxval settings to test (default: [10, 100, 1000]).
        grid_sizes: List of grid sizes to test (default: [3]).
        n_problems: Number of problems per (maxval, n) combination.
        data_dir: Directory with pre-generated JSON files (skips generation).
        save_dir: Directory to save generated data (only if generating).
        seed: Random seed.

    Returns:
        Dict with overall results and per-configuration breakdowns.
    """
    if maxvals is None:
        maxvals = [10, 100, 1000]
    if grid_sizes is None:
        grid_sizes = [3]

    benchmark_start = time.perf_counter()
    all_results = {}
    total_correct = 0
    total_problems = 0

    for n in grid_sizes:
        for maxval in maxvals:
            config_key = f"n{n}_maxval{maxval}"
            logger.info(
                "Evaluating: grid=%dx%d, maxval=%d", n, n, maxval
            )

            # Load or generate data
            data = None
            data_source = "generated"

            if data_dir is not None:
                files = find_iravenx_files(data_dir, maxval=maxval, n=n)
                if files:
                    data = load_iravenx_json(files[0])
                    data_source = str(files[0].name)
                else:
                    logger.warning(
                        "No pre-generated file found for n=%d, maxval=%d "
                        "in %s. Will generate.",
                        n, maxval, data_dir,
                    )

            if data is None:
                data = generate_iravenx_data(
                    n=n,
                    maxval=maxval,
                    n_problems=n_problems,
                    seed=seed,
                    save_dir=save_dir,
                )
                data_source = "generated"

            # Limit to n_problems
            problem_keys = sorted(data.keys(), key=lambda k: int(k))
            if len(problem_keys) > n_problems:
                problem_keys = problem_keys[:n_problems]

            # Evaluate
            config_start = time.perf_counter()
            correct_count = 0
            per_rule_correct = {r: 0 for r in RULE_TYPES}
            per_rule_correct["Mixed"] = 0
            per_rule_total = {r: 0 for r in RULE_TYPES}
            per_rule_total["Mixed"] = 0
            per_problem = []

            for key in problem_keys:
                raw = data[key]
                parsed = parse_problem(raw, n=n)

                t0 = time.perf_counter()
                if n == 3:
                    ok, pred, scores = evaluate_problem(
                        parsed["context"],
                        parsed["candidates"],
                        parsed["target"],
                    )
                else:
                    ok, pred, scores = evaluate_problem_nxn(
                        parsed["context"],
                        parsed["candidates"],
                        parsed["target"],
                        n=n,
                    )
                latency_ms = (time.perf_counter() - t0) * 1000

                rule_type = get_rule_type(parsed["rules"])
                correct_count += int(ok)
                per_rule_correct[rule_type] = (
                    per_rule_correct.get(rule_type, 0) + int(ok)
                )
                per_rule_total[rule_type] = (
                    per_rule_total.get(rule_type, 0) + 1
                )

                per_problem.append({
                    "idx": int(key),
                    "correct": ok,
                    "predicted": pred,
                    "target": parsed["target"],
                    "rule_type": rule_type,
                    "latency_ms": latency_ms,
                })

            config_elapsed = time.perf_counter() - config_start
            n_eval = len(problem_keys)
            accuracy = correct_count / max(n_eval, 1)

            # Per-rule accuracy
            per_rule_accuracy = {}
            for rule in list(RULE_TYPES) + ["Mixed"]:
                rt = per_rule_total.get(rule, 0)
                rc = per_rule_correct.get(rule, 0)
                if rt > 0:
                    per_rule_accuracy[rule] = {
                        "accuracy": rc / rt,
                        "correct": rc,
                        "total": rt,
                    }

            avg_latency = sum(p["latency_ms"] for p in per_problem) / max(n_eval, 1)

            all_results[config_key] = {
                "n": n,
                "maxval": maxval,
                "accuracy": accuracy,
                "correct_count": correct_count,
                "n_problems": n_eval,
                "avg_latency_ms": avg_latency,
                "wall_clock_s": config_elapsed,
                "data_source": data_source,
                "per_rule": per_rule_accuracy,
                "per_problem": per_problem,
            }

            total_correct += correct_count
            total_problems += n_eval

            # Log reference comparison
            ref = REFERENCE_RESULTS.get(maxval)
            ref_str = f" (ref: {ref:.1f}%)" if ref is not None and n == 3 else ""
            logger.info(
                "  n=%d maxval=%d: %.1f%% (%d/%d) avg=%.2fms%s",
                n, maxval, accuracy * 100, correct_count, n_eval,
                avg_latency, ref_str,
            )

    wall_clock = time.perf_counter() - benchmark_start
    overall_accuracy = total_correct / max(total_problems, 1)

    return {
        "overall_accuracy": overall_accuracy,
        "n_total": total_problems,
        "n_correct": total_correct,
        "wall_clock_s": wall_clock,
        "per_config": all_results,
        "maxvals": maxvals,
        "grid_sizes": grid_sizes,
        "seed": seed,
    }


# -- Pretty Printing -----------------------------------------------------------


def print_results(results: dict) -> None:
    """Print benchmark results in a readable table format."""
    print()
    print("=" * 78)
    print("  I-RAVEN-X Out-of-Distribution Benchmark Results")
    print(f"  Seed: {results['seed']}  |  Wall clock: {results['wall_clock_s']:.1f}s")
    print("=" * 78)

    for config_key, cfg in results["per_config"].items():
        n = cfg["n"]
        maxval = cfg["maxval"]
        acc = cfg["accuracy"] * 100
        n_eval = cfg["n_problems"]
        avg_lat = cfg["avg_latency_ms"]
        ref = REFERENCE_RESULTS.get(maxval)
        ref_str = f"  (ref: {ref:.1f}%)" if ref is not None and n == 3 else ""

        print()
        print(f"  --- Grid {n}x{n}, maxval={maxval} ---")
        print(f"  Overall: {acc:6.1f}%  ({cfg['correct_count']}/{n_eval})"
              f"  avg={avg_lat:.2f}ms{ref_str}")
        print(f"  Source:  {cfg['data_source']}")

        # Per-rule breakdown
        if cfg["per_rule"]:
            print()
            print(f"    {'Rule Type':<22} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
            print("    " + "-" * 54)
            for rule, rdata in sorted(cfg["per_rule"].items()):
                rule_acc = rdata["accuracy"] * 100
                print(
                    f"    {rule:<22} {rule_acc:>9.1f}% "
                    f"{rdata['correct']:>10} {rdata['total']:>8}"
                )

    # Overall summary
    print()
    print("  " + "=" * 74)
    overall_acc = results["overall_accuracy"] * 100
    print(
        f"  Overall: {overall_acc:.1f}% "
        f"({results['n_correct']}/{results['n_total']})"
    )
    print()

    # Context
    print("  Reference baselines (n=3, 10k problems):")
    print("    Random:       12.5% (1/8 choices)")
    for mv, ref_acc in sorted(REFERENCE_RESULTS.items()):
        print(f"    maxval={mv:<5}  {ref_acc:.1f}%")
    print()


# -- CLI Entry Point -----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run the I-RAVEN-X OOD benchmark on CubeMind rule detectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m benchmarks.iravenx --maxval 10 100 1000
  python -m benchmarks.iravenx --data-dir ./data/iravenx --maxval 100
  python -m benchmarks.iravenx --maxval 10 --n-problems 500 --grid-sizes 3 5
  python -m benchmarks.iravenx --maxval 10 --save-dir ./data/iravenx
""",
    )
    parser.add_argument(
        "--maxval",
        nargs="+",
        type=int,
        default=[10, 100, 1000],
        help="Max attribute values to test (default: 10 100 1000)",
    )
    parser.add_argument(
        "--grid-sizes",
        nargs="+",
        type=int,
        default=[3],
        help="Grid sizes to test (default: 3). E.g. 3 5 10",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=1000,
        help="Number of problems per (maxval, grid_size) combo (default: 1000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with pre-generated I-RAVEN-X JSON files",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Save generated data to this directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Save full results to a JSON file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results = run_iravenx_benchmark(
        maxvals=args.maxval,
        grid_sizes=args.grid_sizes,
        n_problems=args.n_problems,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    print_results(results)

    if args.json_output:
        # Strip per-problem details for compact JSON output
        output = {k: v for k, v in results.items() if k != "per_config"}
        output["per_config"] = {}
        for ck, cv in results["per_config"].items():
            output["per_config"][ck] = {
                k: v for k, v in cv.items() if k != "per_problem"
            }
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
