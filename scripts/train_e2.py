"""MoQE E2 multi-teacher training — RAM-optimized.

float16 weights + SGD (no momentum) + 1GB chunks.
Target: <80GB peak on 96GB system.
"""
import numpy as np
from cubemind.execution.moqe import MoQEModel
from cubemind.training.moqe_distillation import run_offline_distillation, load_checkpoint

model = MoQEModel(vocab_size=151936, d_model=2048, n_layers=12)
load_checkpoint(model, "data/checkpoints/checkpoint_e0_b1000.npz")

# Convert frozen weights to float16 to save ~2.4GB RAM
model.embedding = model.embedding.astype(np.float16)
model.out_proj = model.out_proj.astype(np.float16)
print("Starting E2 multi-teacher training (float16 + SGD)...", flush=True)

run_offline_distillation(
    model,
    data_dir="data/logits_512",  # Qwen-vocab, same data as E1 (proven stable)
    epochs=1,
    max_seq_len=512,
    temperature=2.0,
    target_8bit=0.15,
    lr=0.00005,
    save_dir="data/checkpoints",
    save_every=50,
    chunk_gb=1.0,
    optimizer_type="adamw",
)
