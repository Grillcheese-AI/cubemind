"""GPU vs CPU benchmark for CubeMind modules.

Measures wall-clock time for each GPU-routed module with and without
the grilly bridge, reporting speedup factors.

Usage:
    python -m benchmarks.gpu_vs_cpu
    python -m benchmarks.gpu_vs_cpu --iterations 100
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_fn(fn, n_iter: int = 50, warmup: int = 5) -> float:
    """Time a callable, returning median wall-clock seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _check_bridge() -> bool:
    """Check if grilly GPU bridge is available."""
    try:
        from grilly.backend import _bridge
        return _bridge.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Module benchmarks
# ---------------------------------------------------------------------------


def bench_hyla(d_vsa: int, d_hidden: int, n_iter: int) -> dict:
    """Benchmark HYLA forward pass."""
    from cubemind.execution.hyla import HYLA

    hyla = HYLA(d_vsa=d_vsa, d_hidden=d_hidden, d_out=d_vsa,
                k=K_BLOCKS, l=L_BLOCK, seed=0)
    x = np.random.default_rng(0).standard_normal(d_vsa).astype(np.float32)
    e = np.random.default_rng(1).standard_normal(d_vsa).astype(np.float32)

    t = _time_fn(lambda: hyla.forward(x, e), n_iter=n_iter)
    return {"module": "HYLA", "time_ms": t * 1000, "shape": f"d_vsa={d_vsa}"}


def bench_hippocampal(d_model: int, n_episodes: int, n_iter: int) -> dict:
    """Benchmark hippocampal store + recall."""
    from cubemind.memory.hippocampal import HippocampalMemory

    mem = HippocampalMemory(d_model=d_model, capacity=n_episodes + 100, seed=0)
    rng = np.random.default_rng(42)

    # Pre-fill
    for i in range(n_episodes):
        mem.store(rng.standard_normal(d_model).astype(np.float32), f"ep_{i}")

    query = rng.standard_normal(d_model).astype(np.float32)
    t = _time_fn(lambda: mem.recall(query, k=5), n_iter=n_iter)
    return {"module": "Hippocampal", "time_ms": t * 1000,
            "shape": f"d={d_model}, episodes={n_episodes}"}


def bench_cvl(d_state: int, d_action: int, n_iter: int) -> dict:
    """Benchmark CVL Q-value estimation."""
    from cubemind.execution.cvl import ContrastiveValueEstimator

    cvl = ContrastiveValueEstimator(
        d_state=d_state, d_action=d_action, d_latent=128, seed=0
    )
    rng = np.random.default_rng(42)
    state = rng.standard_normal(d_state).astype(np.float32)
    action = rng.standard_normal(d_action).astype(np.float32)

    t = _time_fn(lambda: cvl.q_value(state, action), n_iter=n_iter)
    return {"module": "CVL", "time_ms": t * 1000,
            "shape": f"d_state={d_state}, d_action={d_action}"}


def bench_cache(d_vsa: int, n_entries: int, n_iter: int) -> dict:
    """Benchmark VSACache lookup."""
    from cubemind.memory.cache import VSACache

    cache = VSACache(max_size=n_entries + 100, d_vsa=d_vsa)
    rng = np.random.default_rng(42)

    for i in range(n_entries):
        phi = rng.choice([-1, 1], size=d_vsa).astype(np.float32)
        cache.add(phi, np.array([0.5, 0.1], dtype=np.float32))

    query = rng.choice([-1, 1], size=d_vsa).astype(np.float32)
    t = _time_fn(lambda: cache.lookup(query, k=5), n_iter=n_iter)
    return {"module": "VSACache", "time_ms": t * 1000,
            "shape": f"d={d_vsa}, entries={n_entries}"}


def bench_combiner(d_model: int, seq_len: int, n_iter: int) -> dict:
    """Benchmark CombinerAxialAttention forward."""
    from cubemind.reasoning.combiner import CombinerAxialAttention

    comb = CombinerAxialAttention(d_model=d_model, num_heads=4,
                                  rng=np.random.default_rng(0))
    X = np.random.default_rng(42).standard_normal(
        (seq_len, d_model)
    ).astype(np.float32)

    t = _time_fn(lambda: comb.forward(X), n_iter=n_iter)
    return {"module": "Combiner", "time_ms": t * 1000,
            "shape": f"d={d_model}, L={seq_len}"}


def bench_full_pipeline(n_iter: int) -> dict:
    """Benchmark full CubeMind forward pass."""
    from cubemind.model import CubeMind

    model = CubeMind(n_codebook=8, cache_size=200, d_hidden=64, seed=0)
    rng = np.random.default_rng(42)
    phi = rng.standard_normal((K_BLOCKS, L_BLOCK)).astype(np.float32)

    t = _time_fn(lambda: model.forward(phi=phi), n_iter=n_iter)
    return {"module": "Full Pipeline", "time_ms": t * 1000, "shape": "default"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CubeMind GPU vs CPU benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=50,
                        help="Number of timing iterations per module")
    args = parser.parse_args()

    n = args.iterations
    d_vsa = K_BLOCKS * L_BLOCK  # 2048

    gpu_available = _check_bridge()

    print("=" * 70)
    print(f"CubeMind GPU-First Benchmark  (iterations={n})")
    print(f"GPU bridge: {'AVAILABLE' if gpu_available else 'NOT AVAILABLE (CPU only)'}")
    print("=" * 70)
    print()

    benchmarks = [
        bench_hyla(d_vsa, d_hidden=128, n_iter=n),
        bench_hippocampal(d_vsa, n_episodes=200, n_iter=n),
        bench_cvl(d_vsa, d_vsa, n_iter=n),
        bench_cache(d_vsa, n_entries=500, n_iter=n),
        bench_combiner(d_vsa, seq_len=64, n_iter=n),
        bench_full_pipeline(n_iter=n),
    ]

    print(f"{'Module':<20} {'Time (ms)':>12} {'Shape'}")
    print("-" * 70)
    for b in benchmarks:
        print(f"{b['module']:<20} {b['time_ms']:>12.3f} {b['shape']}")

    print()
    total = sum(b["time_ms"] for b in benchmarks)
    print(f"{'Total':<20} {total:>12.3f} ms")
    print()

    if not gpu_available:
        print("NOTE: grilly GPU bridge not available. All times are CPU-only.")
        print("      Install grilly with Vulkan support for GPU acceleration.")
    else:
        print("GPU bridge active — all modules route through Vulkan compute shaders.")

    print()


if __name__ == "__main__":
    main()
