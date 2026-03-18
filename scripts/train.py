#!/usr/bin/env python3
"""Train CubeMind on a dataset.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --epochs 50 --lr 0.01
"""

import argparse
import json
import time

import numpy as np

from cubemind.core import K_BLOCKS, L_BLOCK
from cubemind.model import CubeMind
from cubemind.ops.block_codes import BlockCodes
from cubemind.routing.router import CubeMindRouter
from cubemind.training.trainer import Trainer
from cubemind.telemetry import metrics
from cubemind.telemetry.visualizer import PaperPlotter


def make_synthetic_dataset(bc, n_samples=100, seq_len=5, seed=42):
    """Generate a synthetic dataset for testing."""
    rng = np.random.default_rng(seed)
    dataset = []
    for i in range(n_samples):
        obs = [bc.random_discrete(seed=int(rng.integers(0, 2**31))) for _ in range(seq_len)]
        target = bc.random_discrete(seed=int(rng.integers(0, 2**31)))
        dataset.append((obs, target))
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train CubeMind")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--n-codebook", type=int, default=16)
    parser.add_argument("--n-hmm-rules", type=int, default=4)
    parser.add_argument("--d-hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-figures", type=str, default=None)
    args = parser.parse_args()

    bc = BlockCodes(K_BLOCKS, L_BLOCK)

    print(f"CubeMind Training")
    print(f"  k={K_BLOCKS}, l={L_BLOCK}, d_vsa={K_BLOCKS * L_BLOCK}")
    print(f"  codebook={args.n_codebook}, hmm_rules={args.n_hmm_rules}")
    print(f"  epochs={args.epochs}, lr={args.lr}")
    print()

    # Build model
    model = CubeMind(
        n_codebook=args.n_codebook,
        n_hmm_rules=args.n_hmm_rules,
        d_hidden=args.d_hidden,
        seed=args.seed,
    )

    trainer = Trainer(model)

    # Generate data
    print("Generating synthetic dataset...")
    train_data = make_synthetic_dataset(bc, args.n_train, args.seq_len, args.seed)
    eval_data = make_synthetic_dataset(bc, args.n_eval, args.seq_len, args.seed + 1000)

    # Train
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        stats = trainer.train_epoch(train_data, lr=args.lr)
        eval_stats = trainer.evaluate(eval_data)
        print(f"  Epoch {epoch + 1:3d}: loss={stats['mean_loss']:.4f} "
              f"eval_loss={eval_stats['mean_loss']:.4f} "
              f"acc={eval_stats['accuracy']:.1%}")

    elapsed = time.perf_counter() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final: {model.stats}")

    # Save figures
    if args.save_figures:
        plotter = PaperPlotter(metrics)
        plotter.plot_training_curves(save=f"{args.save_figures}/training.pdf")
        print(f"Figures saved to {args.save_figures}/")


if __name__ == "__main__":
    main()
