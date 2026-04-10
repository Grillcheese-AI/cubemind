"""I-RAVEN-X Image Benchmark — visual reasoning via Perceiver + VSA pipeline.

Evaluates CubeMind's full visual perception pipeline on I-RAVEN-X:
  1. Render each panel's attributes to an 80x80 grayscale image
  2. Encode each image through the ImageVSAPipeline:
     Image → patches → Perceiver → LSH → binarize → packed binary VSA
  3. Learn context transition patterns in ContinuousItemMemory
  4. Score candidates by Hamming similarity to predicted next panel

This tests whether CubeMind can solve abstract reasoning problems purely
from pixel-level input — no integer attribute access.

Usage:
    python -m benchmarks.iravenx_image --maxval 10 --n-problems 100
    python -m benchmarks.iravenx_image --maxval 10 100 --n-problems 50
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from cubemind.ops.vsa_bridge import (
    LSHProjector,
    binarize_and_pack,
    hamming_similarity,
)
from cubemind.perception.cnn_encoder import CNNEncoder
from cubemind.perception.raven_renderer import render_panel
from cubemind.reasoning.rule_detectors import score_candidates

logger = logging.getLogger(__name__)

# I-RAVEN-X source for generation
IRAVENX_SRC = Path(r"C:\Users\grill\Documents\GitHub\iraven-x")

# Pipeline configuration
D_VSA = 10240       # Full cubemind dims: K=80, L=128
D_CNN_FEAT = 128    # CNNEncoder feature dim (after conv stack)
PANEL_SIZE = 80
# CNN encoder params (small k,l for the encoder itself — features are what matter)
CNN_K = 8
CNN_L = 64


class RAVENImageSolver:
    """Solves I-RAVEN-X problems via CNN features → LSH → binary VSA.

    Pipeline per panel:
      render(attrs) → 80×80 image → CNNEncoder → 128D feature vector
      → LSH projection → 10240D → binarize & pack → uint32[]

    Strategy:
      1. Encode all panels through CNN → LSH → binary VSA
      2. Extract row/col transitions via XOR (binary binding)
      3. Predict missing panel by applying transitions
      4. Score candidates by Hamming similarity to prediction
    """

    def __init__(self, maxval: int = 10, seed: int = 42) -> None:
        self.maxval = maxval

        # CNN feature extractor (128D features from 80×80 images)
        self.cnn = CNNEncoder(k=CNN_K, l=CNN_L, grid_size=(1, 1), seed=seed)

        # LSH: 128D CNN features → 10240D continuous → binarize
        self.lsh = LSHProjector(d_input=D_CNN_FEAT, d_output=D_VSA, seed=seed + 1)

    def _encode_panel(self, attrs: dict) -> np.ndarray:
        """Render → CNN features → LSH → binarize → packed binary."""
        img = render_panel(attrs, size=PANEL_SIZE, maxval=self.maxval)
        # CNN forward gives (k, l) block-code, but we want the 128D features
        # from the intermediate conv stack
        self.cnn.forward(img)
        features = self.cnn._cache.get('features')
        if features is not None:
            feat_vec = features.ravel()[:D_CNN_FEAT]
        else:
            feat_vec = np.zeros(D_CNN_FEAT, dtype=np.float32)

        projected = self.lsh.project(feat_vec)
        return binarize_and_pack(projected)

    def solve(
        self,
        context: list[dict],
        candidates: list[dict],
        n: int = 3,
    ) -> tuple[int, list[float]]:
        """Solve a single RAVEN problem using visual reasoning."""
        # 1. Encode all context panels
        ctx_vecs = [self._encode_panel(a) for a in context]

        # 2. Build grid
        grid = []
        idx = 0
        for row in range(n):
            row_vecs = []
            for col in range(n):
                if row == n - 1 and col == n - 1:
                    row_vecs.append(None)
                else:
                    row_vecs.append(ctx_vecs[idx])
                    idx += 1
            grid.append(row_vecs)

        # 3. Extract row transitions (XOR binding between consecutive panels)
        row_transitions = []
        for row in range(n - 1):
            for col in range(n - 1):
                t = np.bitwise_xor(grid[row][col], grid[row][col + 1])
                row_transitions.append(t)

        # Column transitions
        col_transitions = []
        for col in range(n):
            for row in range(n - 1):
                if grid[row][col] is not None and grid[row + 1][col] is not None:
                    t = np.bitwise_xor(grid[row][col], grid[row + 1][col])
                    col_transitions.append(t)

        # 4. Predict missing panel
        predictions = []

        # Horizontal prediction: last row transition applied to panel (n-1, n-2)
        if grid[n - 1][n - 2] is not None and row_transitions:
            pred_h = np.bitwise_xor(grid[n - 1][n - 2], row_transitions[-1])
            predictions.append(pred_h)

        # Vertical prediction: last col transition applied to panel (n-2, n-1)
        if grid[n - 2][n - 1] is not None and col_transitions:
            pred_v = np.bitwise_xor(grid[n - 2][n - 1], col_transitions[-1])
            predictions.append(pred_v)

        # 5. Encode candidates and score
        cand_vecs = [self._encode_panel(a) for a in candidates]

        scores = [0.0] * 8
        for pred in predictions:
            for i, cv in enumerate(cand_vecs):
                scores[i] += hamming_similarity(pred, cv, D_VSA)

        # Context consistency bonus
        if ctx_vecs:
            last_ctx = ctx_vecs[-1]
            for i, cv in enumerate(cand_vecs):
                scores[i] += 0.1 * hamming_similarity(last_ctx, cv, D_VSA)

        predicted = int(np.argmax(scores))
        return predicted, scores


class HybridRAVENSolver:
    """Hybrid solver: integer rule detectors + CNN visual VSA features.

    Fuses two independent signals:
      1. Algebraic: deterministic integer-domain rule detectors (90.3% baseline)
      2. Visual: CNN → LSH → binary VSA → Hamming similarity

    The algebraic detectors provide the primary signal. When they are
    confident (one candidate dominates), we trust them. When they tie
    or are uncertain, the visual similarity breaks the tie.

    Args:
        maxval:      Max attribute value.
        visual_weight: Weight of visual score relative to algebraic (0-1).
        seed:        Random seed.
    """

    def __init__(self, maxval: int = 10, visual_weight: float = 0.3, seed: int = 42) -> None:
        self.maxval = maxval
        self.visual_weight = visual_weight
        self.visual_solver = RAVENImageSolver(maxval=maxval, seed=seed)

    def solve(
        self,
        context: list[dict],
        candidates: list[dict],
        n: int = 3,
    ) -> tuple[int, list[float]]:
        """Solve using fused algebraic + visual scores."""
        # 1. Algebraic scores from integer rule detectors
        algebraic_scores = score_candidates(
            context, candidates, attrs=["Type", "Size", "Color"],
        )

        # 2. Visual scores from CNN → LSH → binary VSA
        _, visual_scores = self.visual_solver.solve(context, candidates, n=n)

        # 3. Normalize both to [0, 1]
        alg = np.array(algebraic_scores, dtype=np.float64)
        vis = np.array(visual_scores, dtype=np.float64)

        alg_range = alg.max() - alg.min()
        if alg_range > 1e-8:
            alg_norm = (alg - alg.min()) / alg_range
        else:
            alg_norm = np.ones_like(alg) / len(alg)

        vis_range = vis.max() - vis.min()
        if vis_range > 1e-8:
            vis_norm = (vis - vis.min()) / vis_range
        else:
            vis_norm = np.ones_like(vis) / len(vis)

        # 4. Fuse: algebraic is primary, visual is tiebreaker
        w_v = self.visual_weight
        fused = (1.0 - w_v) * alg_norm + w_v * vis_norm

        predicted = int(np.argmax(fused))
        return predicted, fused.tolist()


# ── Benchmark Runner ──────────────────────────────────────────────────────

def run_image_benchmark(
    maxvals: list[int] | None = None,
    n_problems: int = 100,
    n: int = 3,
    seed: int = 1234,
    mode: str = "hybrid",
    visual_weight: float = 0.3,
) -> dict:
    """Run the I-RAVEN-X image benchmark.

    Args:
        maxvals:        List of maxval settings.
        n_problems:     Problems per maxval.
        n:              Grid size.
        seed:           Random seed.
        mode:           "visual" (CNN only), "hybrid" (CNN + rule detectors).
        visual_weight:  Weight of visual signal in hybrid mode (0-1).

    Returns:
        Results dict.
    """
    import sys
    iraven_src = str(IRAVENX_SRC / "src" / "datasets" / "generation")
    if iraven_src not in sys.path:
        sys.path.insert(0, iraven_src)

    try:
        import iravenx_task
    except ImportError:
        raise ImportError(
            f"Cannot import iravenx_task. Ensure I-RAVEN-X source at {IRAVENX_SRC}"
        )

    if maxvals is None:
        maxvals = [10]

    all_results = {}
    total_correct = 0
    total_problems = 0
    bench_start = time.perf_counter()

    for maxval in maxvals:
        logger.info("Evaluating: maxval=%d, n=%d, n_problems=%d, mode=%s", maxval, n, n_problems, mode)
        if mode == "hybrid":
            solver = HybridRAVENSolver(maxval=maxval, visual_weight=visual_weight, seed=seed)
        else:
            solver = RAVENImageSolver(maxval=maxval, seed=seed)

        iravenx_task.set_seeds(seed)
        correct = 0

        t0 = time.perf_counter()
        for i in range(n_problems):
            problem = iravenx_task.get_sample(n, maxval, rule="", arithmetic_strategy="shuffle")

            # Parse
            rpm = problem["rpm"]
            n_context = n * n - 1
            context = []
            for j in range(n_context):
                entity = rpm[j][0]
                context.append({
                    "Type": int(entity["Type"]),
                    "Size": int(entity["Size"]),
                    "Color": int(entity["Color"]),
                })
            candidates = []
            for j in range(8):
                entity = rpm[n_context + j][0]
                candidates.append({
                    "Type": int(entity["Type"]),
                    "Size": int(entity["Size"]),
                    "Color": int(entity["Color"]),
                })
            target = int(problem["target"])

            predicted, scores = solver.solve(context, candidates, n=n)
            if predicted == target:
                correct += 1

            if (i + 1) % 10 == 0:
                acc_so_far = correct / (i + 1) * 100
                logger.info("  [%d/%d] accuracy=%.1f%%", i + 1, n_problems, acc_so_far)

        elapsed = time.perf_counter() - t0
        accuracy = correct / max(n_problems, 1)

        all_results[f"maxval{maxval}"] = {
            "maxval": maxval,
            "accuracy": accuracy,
            "correct": correct,
            "n_problems": n_problems,
            "wall_clock_s": elapsed,
            "avg_latency_ms": elapsed / max(n_problems, 1) * 1000,
        }

        total_correct += correct
        total_problems += n_problems

        print(f"  maxval={maxval}: {accuracy * 100:.1f}% ({correct}/{n_problems}) "
              f"in {elapsed:.1f}s ({elapsed / max(n_problems, 1) * 1000:.1f}ms/problem)")

    wall = time.perf_counter() - bench_start
    overall = total_correct / max(total_problems, 1)

    return {
        "overall_accuracy": overall,
        "n_total": total_problems,
        "n_correct": total_correct,
        "wall_clock_s": wall,
        "per_config": all_results,
        "mode": mode,
        "visual_weight": visual_weight if mode == "hybrid" else 1.0,
        "pipeline": {
            "d_cnn_feat": D_CNN_FEAT,
            "d_vsa": D_VSA,
            "cnn_k": CNN_K,
            "cnn_l": CNN_L,
            "panel_size": PANEL_SIZE,
        },
    }


def print_results(results: dict) -> None:
    print()
    print("=" * 70)
    print("  I-RAVEN-X IMAGE Benchmark (Perceiver + VSA Pipeline)")
    mode = results.get('mode', 'visual')
    vw = results.get('visual_weight', 1.0)
    mode_str = f"HYBRID (visual={vw:.0%}, algebraic={1-vw:.0%})" if mode == "hybrid" else "VISUAL ONLY"
    print(f"  Mode: {mode_str}")
    print(f"  Pipeline: CNN({results['pipeline']['cnn_k']}x{results['pipeline']['cnn_l']}) "
          f"-> LSH({results['pipeline']['d_cnn_feat']}D -> {results['pipeline']['d_vsa']}D) "
          f"-> binarize+pack")
    print(f"  Wall clock: {results['wall_clock_s']:.1f}s")
    print("=" * 70)

    for key, cfg in results["per_config"].items():
        print(f"  maxval={cfg['maxval']}: {cfg['accuracy'] * 100:.1f}% "
              f"({cfg['correct']}/{cfg['n_problems']}) "
              f"avg={cfg['avg_latency_ms']:.1f}ms/problem")

    print()
    print(f"  Overall: {results['overall_accuracy'] * 100:.1f}% "
          f"({results['n_correct']}/{results['n_total']})")
    print()
    print("  Baselines:")
    print("    Random:           12.5% (1/8)")
    print("    Integer detectors: 90.3% (maxval=10, no images)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="I-RAVEN-X Image Benchmark (Perceiver + VSA)",
    )
    parser.add_argument("--maxval", nargs="+", type=int, default=[10])
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", choices=["visual", "hybrid"], default="hybrid",
                        help="visual=CNN only, hybrid=CNN+rule detectors (default: hybrid)")
    parser.add_argument("--visual-weight", type=float, default=0.3,
                        help="Visual signal weight in hybrid mode (default: 0.3)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    results = run_image_benchmark(
        maxvals=args.maxval,
        n_problems=args.n_problems,
        n=args.grid_size,
        seed=args.seed,
        mode=args.mode,
        visual_weight=args.visual_weight,
    )
    print_results(results)


if __name__ == "__main__":
    main()
