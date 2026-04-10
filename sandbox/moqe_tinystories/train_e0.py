"""MoQE E0 — CE-only training on TinyStories via run_offline_distillation.

Uses the production pipeline: OfflineDistillationLoader + moqe_backward + GPU.
No teacher logits needed — CE-only mode with student entropy for router guidance.

Step 1: python -u sandbox/moqe_tinystories/prepare_data.py
Step 2: python -u sandbox/moqe_tinystories/train_e0.py
"""

import numpy as np
from loguru import logger

from cubemind.execution.moqe import ExpertSpec, MoQEModel
from cubemind.training.moqe_distillation import run_offline_distillation


def main():
    # Load vocab size from prepare_data output
    vocab_size = int(np.load("sandbox/moqe_tinystories/data/vocab_size.npy")[0])
    logger.info("Vocab size: {}", vocab_size)

    expert_specs = [
        ExpertSpec(bits=4, specialty="general", target_fraction=0.60),
        ExpertSpec(bits=4, specialty="code", target_fraction=0.20),
        ExpertSpec(bits=8, specialty="factual", target_fraction=0.15),
        ExpertSpec(bits=8, specialty="rare", target_fraction=0.05),
    ]

    model = MoQEModel(
        vocab_size=vocab_size,
        d_model=1024,
        n_layers=12,
        expert_specs=expert_specs,
        top_k=1,
        block_size=32,
        seed=42,
    )

    # Add position embeddings
    rng = np.random.default_rng(42)
    max_seq_len = 768
    model.pos_embed = (rng.standard_normal((max_seq_len, model.d_model)) * 0.02).astype(np.float32)

    # Patch forward to add position embeddings
    _orig_forward = model.forward.__func__ if hasattr(model.forward, '__func__') else None

    def forward_with_pos(self, input_ids):
        seq_len = len(input_ids)
        x = self.embedding[input_ids]
        x = x + self.pos_embed[:seq_len]
        layer_weights = []
        for layer in self.layers:
            outputs, indices, weights = layer.forward_batch(x)
            x = x + outputs
            layer_weights.append(weights)
        logits = (x @ self.out_proj.T).astype(np.float32)
        return logits, layer_weights

    import types
    model.forward = types.MethodType(forward_with_pos, model)

    logger.info("MoQE E0: vocab={}, d={}, layers={}, experts={}",
                vocab_size, model.d_model, model.n_layers, len(expert_specs))

    stats = run_offline_distillation(
        model,
        data_dir="sandbox/moqe_tinystories/data",
        epochs=3,
        max_seq_len=max_seq_len,
        temperature=2.0,
        target_8bit=0.15,
        lr=3e-4,
        save_dir="sandbox/moqe_tinystories/checkpoints",
        save_every=500,
        chunk_gb=2.0,
        optimizer_type="adamw",
    )

    for s in stats:
        logger.info("Epoch {}: loss={:.4f} CE={:.4f} 8bit={:.1f}%",
                     s["epoch"], s["avg_loss"], s["avg_ce"], s["avg_8bit_frac"] * 100)


if __name__ == "__main__":
    main()
