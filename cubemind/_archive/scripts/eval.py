#!/usr/bin/env python3
"""Evaluate CubeMind on a dataset or benchmark.

Usage:
    python scripts/eval.py --benchmark iraven
    python scripts/eval.py --dataset data/test.json
"""

import argparse
import time


from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.model import CubeMind
from cubemind.ops.block_codes import BlockCodes
from cubemind.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate CubeMind")
    parser.add_argument("--benchmark", type=str, choices=["iraven", "xraven"], default=None)
    parser.add_argument("--n-codebook", type=int, default=16)
    parser.add_argument("--n-hmm-rules", type=int, default=4)
    parser.add_argument("--d-hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bc = BlockCodes(K_BLOCKS, L_BLOCK)

    model = CubeMind(
        n_codebook=args.n_codebook,
        n_hmm_rules=args.n_hmm_rules,
        d_hidden=args.d_hidden,
        seed=args.seed,
    )

    trainer = Trainer(model)

    if args.benchmark == "iraven":
        print("Running iRaven benchmark...")
        try:
            from benchmarks.iraven import run_iraven_benchmark
            results = run_iraven_benchmark(model)
            print(f"  Accuracy: {results['accuracy']:.1%}")
            print(f"  Latency: {results['latency_ms']:.1f} ms/sample")
        except ImportError:
            print("  iRaven benchmark not yet implemented — run synthetic eval instead")
            eval_data = [
                ([bc.random_discrete(seed=i + j * 100) for j in range(5)],
                 bc.random_discrete(seed=i + 999))
                for i in range(50)
            ]
            stats = trainer.evaluate(eval_data)
            print(f"  Synthetic eval: loss={stats['mean_loss']:.4f}, acc={stats['accuracy']:.1%}")
    else:
        # Default: quick synthetic eval
        print("Running synthetic evaluation...")
        eval_data = [
            ([bc.random_discrete(seed=i + j * 100) for j in range(5)],
             bc.random_discrete(seed=i + 999))
            for i in range(100)
        ]
        t0 = time.perf_counter()
        stats = trainer.evaluate(eval_data)
        elapsed = time.perf_counter() - t0

        print(f"  Samples: {stats['n_samples']}")
        print(f"  Loss: {stats['mean_loss']:.4f}")
        print(f"  Accuracy: {stats['accuracy']:.1%}")
        print(f"  Time: {elapsed:.2f}s ({elapsed / stats['n_samples'] * 1000:.1f} ms/sample)")


if __name__ == "__main__":
    main()
