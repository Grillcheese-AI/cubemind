"""EGGROLL — Evolution-Guided Rank-1 Learning for MoQE.

Backprop-free training inspired by:
  - EGGROLL (Sarkar et al., 2026): rank-r ES with structured perturbations
  - LayerMatrix (Cloutier, 2024): merit-modulated weight updates

Instead of gradient computation:
  1. Generate N rank-1 perturbations per expert weight matrix
  2. Evaluate loss for each perturbation (parallel forward passes)
  3. Merit-score each perturbation by loss improvement
  4. Update weights with merit-weighted average of best perturbations

Benefits over backprop:
  - Zero optimizer state (no Adam m/v buffers) → ~2MB vs 93GB
  - Integer-compatible (perturbations can be INT8)
  - Embarrassingly parallel (N workers, no gradient sync)
  - No chain rule, no STE, no float16 precision issues

Usage:
    trainer = EggrollTrainer(model, n_workers=64, rank=1)
    for input_ids, labels, teacher in loader:
        stats = trainer.step(input_ids, labels, teacher, temperature=2.0)
        print(f"loss={stats['loss']:.4f} merit={stats['mean_merit']:.3f}")
"""

from __future__ import annotations

import numpy as np

from cubemind.training.moqe_distillation import (
    _dequant_weights,
    _softmax,
)


class EggrollTrainer:
    """EGGROLL + Merit training for MoQE models.

    Args:
        model: MoQEModel instance.
        n_workers: Number of parallel perturbation workers.
        rank: Rank of each perturbation (1 = rank-1, fastest).
        sigma: Perturbation scale (std of random vectors).
        merit_increase: Multiplicative increase for successful weights.
        merit_decay: Multiplicative decay for unsuccessful weights.
        top_k_frac: Fraction of workers to keep (elitist selection).
        seed: Random seed.
    """

    def __init__(
        self,
        model,
        n_workers: int = 64,
        rank: int = 1,
        sigma: float = 0.01,
        merit_increase: float = 1.02,
        merit_decay: float = 0.98,
        top_k_frac: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.n_workers = n_workers
        self.rank = rank
        self.sigma = sigma
        self.merit_increase = merit_increase
        self.merit_decay = merit_decay
        self.top_k = max(1, int(n_workers * top_k_frac))
        self.rng = np.random.default_rng(seed)

        # Initialize per-layer merit scores
        self.merit = {}
        for i, layer in enumerate(model.layers):
            d_out, d_in = layer.w0_int.shape
            self.merit[f"layer{i}_w0"] = np.ones((d_out, d_in), dtype=np.float32)
            self.merit[f"layer{i}_w1"] = np.ones((d_out, d_in), dtype=np.float32)
            self.merit[f"layer{i}_router"] = np.ones(d_in, dtype=np.float32)

        self._step_count = 0

    def _generate_perturbations(
        self, shape: tuple[int, ...],
    ) -> list[np.ndarray]:
        """Generate N rank-r perturbations for a weight matrix.

        Each perturbation is u @ v.T where u is (d_out, rank) and v is (d_in, rank).
        For rank=1: outer product of two random vectors.

        Returns list of N perturbation matrices, each shaped like `shape`.
        """
        d_out, d_in = shape
        perturbations = []
        for _ in range(self.n_workers):
            u = self.rng.standard_normal((d_out, self.rank)).astype(np.float32)
            v = self.rng.standard_normal((d_in, self.rank)).astype(np.float32)
            # Rank-r perturbation: u @ v.T → (d_out, d_in)
            delta = (u @ v.T) * self.sigma
            perturbations.append(delta)
        return perturbations

    def _evaluate_loss(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray,
        teacher_logits,
        temperature: float,
    ) -> float:
        """Evaluate current model loss on a batch."""
        # Forward pass
        logits, router_probs = self.model.forward(input_ids)
        seq_len = len(labels)

        # CE loss
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-8)
        ce = -np.mean(log_probs[np.arange(seq_len), labels])

        # KD loss (if teacher available)
        kd = 0.0
        if teacher_logits is not None and not isinstance(teacher_logits, dict):
            min_vocab = min(logits.shape[-1], teacher_logits.shape[-1])
            s_soft = _softmax(logits[:, :min_vocab] / temperature)
            t_soft = _softmax(teacher_logits[:, :min_vocab].astype(np.float32) / temperature)
            t_soft = np.clip(t_soft, 1e-7, 1.0)
            s_soft = np.clip(s_soft, 1e-7, 1.0)
            kd = float(np.mean(np.sum(
                t_soft * (np.log(t_soft) - np.log(s_soft)), axis=-1
            ))) * (temperature ** 2)
        elif isinstance(teacher_logits, dict) and "top_k_indices" in teacher_logits:
            # Sparse top-k KD
            indices = teacher_logits["top_k_indices"]
            lp = teacher_logits["top_k_logprobs"].astype(np.float32)
            t_probs = _softmax(lp / temperature)
            t_probs = np.clip(t_probs, 1e-7, 1.0)

            s_scaled = logits / temperature
            s_shifted = s_scaled - np.max(s_scaled, axis=-1, keepdims=True)
            s_log = s_shifted - np.log(np.sum(np.exp(s_shifted), axis=-1, keepdims=True) + 1e-8)

            row_idx = np.arange(seq_len)[:, None]
            s_log_topk = s_log[row_idx, indices[:seq_len]]
            kd = float(-np.mean(np.sum(t_probs[:seq_len] * s_log_topk, axis=-1))) * (temperature ** 2)

        return 0.3 * ce + 0.6 * kd + 0.1 * 0.0  # router loss placeholder

    def _apply_perturbation(self, layer_idx: int, expert: int, delta: np.ndarray):
        """Temporarily perturb a layer's dequantized weights."""
        layer = self.model.layers[layer_idx]
        if expert == 0:
            w = _dequant_weights(layer.w0_int, layer.s0, layer.block_size)
        else:
            w = _dequant_weights(layer.w1_int, layer.s1, layer.block_size)
        return w + delta

    def step(
        self,
        input_ids: np.ndarray,
        labels: np.ndarray,
        teacher_logits,
        temperature: float = 2.0,
    ) -> dict:
        """One EGGROLL training step.

        For each layer and expert:
          1. Generate N rank-1 perturbations
          2. Evaluate loss with each perturbation applied
          3. Select top-k by fitness (lowest loss)
          4. Update weights with merit-weighted average of top-k perturbations

        Returns dict with loss, merit stats, etc.
        """
        from cubemind.execution.moqe import (
            _quantize_weights_int4,
            _quantize_weights_int8,
        )

        self._step_count += 1
        base_loss = self._evaluate_loss(input_ids, labels, teacher_logits, temperature)

        total_updates = 0

        for li, layer in enumerate(self.model.layers):
            for expert_id in range(2):
                key = f"layer{li}_w{'0' if expert_id == 0 else '1'}"
                merit = self.merit[key]

                if expert_id == 0:
                    w_orig = _dequant_weights(layer.w0_int, layer.s0, layer.block_size)
                else:
                    w_orig = _dequant_weights(layer.w1_int, layer.s1, layer.block_size)

                # Generate perturbations
                perturbations = self._generate_perturbations(w_orig.shape)

                # Evaluate each perturbation
                fitness = np.zeros(self.n_workers)
                for wi, delta in enumerate(perturbations):
                    # Apply perturbation temporarily
                    w_perturbed = w_orig + delta * merit  # merit-modulated!

                    # Swap into model
                    if expert_id == 0:
                        old_w, old_s = layer.w0_int.copy(), layer.s0.copy()
                        layer.w0_int, s_flat = _quantize_weights_int4(
                            w_perturbed, layer.block_size)
                        num_blocks = (layer.d_model + layer.block_size - 1) // layer.block_size
                        layer.s0 = s_flat[:num_blocks * layer.d_out].reshape(
                            layer.d_out, num_blocks)
                    else:
                        old_w, old_s = layer.w1_int.copy(), layer.s1.copy()
                        layer.w1_int, s_flat = _quantize_weights_int8(
                            w_perturbed, layer.block_size)
                        num_blocks = (layer.d_model + layer.block_size - 1) // layer.block_size
                        layer.s1 = s_flat[:num_blocks * layer.d_out].reshape(
                            layer.d_out, num_blocks)

                    # Evaluate
                    perturbed_loss = self._evaluate_loss(
                        input_ids, labels, teacher_logits, temperature)
                    fitness[wi] = base_loss - perturbed_loss  # positive = improvement

                    # Restore original weights
                    if expert_id == 0:
                        layer.w0_int, layer.s0 = old_w, old_s
                    else:
                        layer.w1_int, layer.s1 = old_w, old_s

                # Select top-k by fitness
                top_indices = np.argsort(fitness)[-self.top_k:]
                top_fitness = fitness[top_indices]

                # Skip if no perturbation improved
                if top_fitness.max() <= 0:
                    # Decay merit for weights that couldn't improve
                    self.merit[key] *= self.merit_decay
                    continue

                # Normalize fitness to weights
                fit_weights = np.maximum(top_fitness, 0)
                fit_sum = fit_weights.sum()
                if fit_sum > 0:
                    fit_weights /= fit_sum
                else:
                    continue

                # Weighted average of top-k perturbations
                avg_delta = np.zeros_like(w_orig)
                for idx, fw in zip(top_indices, fit_weights):
                    avg_delta += fw * perturbations[idx]

                # Apply merit-modulated update
                w_updated = w_orig + avg_delta * merit

                # Re-quantize and store
                if expert_id == 0:
                    layer.w0_int, s_flat = _quantize_weights_int4(
                        w_updated, layer.block_size)
                    num_blocks = (layer.d_model + layer.block_size - 1) // layer.block_size
                    layer.s0 = s_flat[:num_blocks * layer.d_out].reshape(
                        layer.d_out, num_blocks)
                else:
                    layer.w1_int, s_flat = _quantize_weights_int8(
                        w_updated, layer.block_size)
                    num_blocks = (layer.d_model + layer.block_size - 1) // layer.block_size
                    layer.s1 = s_flat[:num_blocks * layer.d_out].reshape(
                        layer.d_out, num_blocks)

                # Update merit: increase for weights where perturbation helped
                improvement_map = np.abs(avg_delta) > 1e-8
                self.merit[key][improvement_map] *= self.merit_increase
                self.merit[key][~improvement_map] *= self.merit_decay

                total_updates += 1

        # Evaluate final loss after all updates
        final_loss = self._evaluate_loss(input_ids, labels, teacher_logits, temperature)

        # Merit statistics
        all_merits = np.concatenate([m.ravel() for m in self.merit.values()])
        _, router_probs = self.model.forward(input_ids)
        eight_bit_frac = float(np.mean(router_probs > 0.5)) if router_probs.ndim > 0 else 0.0

        return {
            "loss": final_loss,
            "base_loss": base_loss,
            "improvement": base_loss - final_loss,
            "mean_merit": float(np.mean(all_merits)),
            "max_merit": float(np.max(all_merits)),
            "min_merit": float(np.min(all_merits)),
            "updates": total_updates,
            "eight_bit_frac": eight_bit_frac,
            "step": self._step_count,
        }

    def memory_bytes(self) -> int:
        """Total memory footprint of EGGROLL trainer (excluding model)."""
        merit_bytes = sum(m.nbytes for m in self.merit.values())
        # Perturbation buffers are transient (allocated/freed per step)
        return merit_bytes

    def memory_mb(self) -> float:
        return self.memory_bytes() / (1024 * 1024)
