#!/bin/bash
# MoQE Student Distillation Training
#
# Usage:
#   ./scripts/train_moqe.sh                    # defaults
#   ./scripts/train_moqe.sh --epochs 10        # override epochs
#   ./scripts/train_moqe.sh --preprocess       # preprocess logits first
#   ./scripts/train_moqe.sh --preprocess --epochs 5 --d-model 256 --n-layers 4

set -euo pipefail
cd "$(dirname "$0")/.."
$env:PATH="$HOME/.local/bin:$PATH"

# ── Defaults ──────────────────────────────────────────────────────────
VOCAB=151936
D_MODEL=256
N_LAYERS=4
EPOCHS=3
MAX_SEQ_LEN=512
TEMPERATURE=2.0
TARGET_8BIT=0.15
LR=3e-4
SRC_DIR="G:/MYDRIVE"
DATA_DIR="data/logits_512"
SAVE_DIR="data/checkpoints"
SAVE_EVERY=100
WORKERS=8
PREPROCESS=false

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --preprocess)     PREPROCESS=true; shift ;;
        --src)            SRC_DIR="$2"; shift 2 ;;
        --data)           DATA_DIR="$2"; shift 2 ;;
        --save-dir)       SAVE_DIR="$2"; shift 2 ;;
        --vocab)          VOCAB="$2"; shift 2 ;;
        --d-model)        D_MODEL="$2"; shift 2 ;;
        --n-layers)       N_LAYERS="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --max-seq-len)    MAX_SEQ_LEN="$2"; shift 2 ;;
        --temperature)    TEMPERATURE="$2"; shift 2 ;;
        --target-8bit)    TARGET_8BIT="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --save-every)     SAVE_EVERY="$2"; shift 2 ;;
        --workers)        WORKERS="$2"; shift 2 ;;
        *)                echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  MoQE Student Distillation"
echo "============================================"
echo "  Model:   vocab=$VOCAB d=$D_MODEL L=$N_LAYERS"
echo "  Data:    $DATA_DIR (seq_len=$MAX_SEQ_LEN)"
echo "  Train:   epochs=$EPOCHS lr=$LR T=$TEMPERATURE"
echo "  Target:  ${TARGET_8BIT} 8-bit fraction"
echo "  Save:    $SAVE_DIR (every $SAVE_EVERY batches)"
echo "============================================"

# ── Step 1: Preprocess logits (optional) ──────────────────────────────
if [ "$PREPROCESS" = true ]; then
    echo ""
    echo "[1/2] Preprocessing teacher logits..."
    uv run python -u scripts/preprocess_logits.py \
        --src "$SRC_DIR" \
        --out "$DATA_DIR" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --workers "$WORKERS"
    echo ""
fi

# ── Step 2: Verify data exists ────────────────────────────────────────
N_FILES=$(find "$DATA_DIR" -name "*.npz" 2>/dev/null | wc -l)
if [ "$N_FILES" -eq 0 ]; then
    echo "ERROR: No .npz files in $DATA_DIR"
    echo "  Run with --preprocess to create them from $SRC_DIR"
    exit 1
fi
echo "Found $N_FILES logit files in $DATA_DIR"

# ── Step 3: Train ─────────────────────────────────────────────────────
echo ""
echo "Starting training..."
uv run python -u -c "
from cubemind.execution.moqe import MoQEModel
from cubemind.training.moqe_distillation import run_offline_distillation
import time, json, os

model = MoQEModel(
    vocab_size=$VOCAB,
    d_model=$D_MODEL,
    n_layers=$N_LAYERS,
    seed=42,
)
print(f'MoQE: v={model.vocab_size} d={model.d_model} L={model.n_layers}')

t = time.perf_counter()
stats = run_offline_distillation(
    model,
    data_dir='$DATA_DIR',
    epochs=$EPOCHS,
    max_seq_len=$MAX_SEQ_LEN,
    temperature=$TEMPERATURE,
    target_8bit=$TARGET_8BIT,
    lr=$LR,
    save_dir='$SAVE_DIR',
    save_every=$SAVE_EVERY,
)
elapsed = time.perf_counter() - t

print()
print('=' * 44)
print('  Training Complete')
print('=' * 44)
total_batches = sum(s['n_batches'] for s in stats)
print(f'  Time:    {elapsed:.1f}s ({total_batches/elapsed:.1f} batch/s)')
for s in stats:
    print(f'  E{s[\"epoch\"]}: loss={s[\"avg_loss\"]:.4f} CE={s[\"avg_ce\"]:.4f} KD={s[\"avg_kd\"]:.4f} 8b={s[\"avg_8bit_frac\"]*100:.1f}%')

# Save training summary
os.makedirs('$SAVE_DIR', exist_ok=True)
summary = {
    'model': {'vocab': $VOCAB, 'd_model': $D_MODEL, 'n_layers': $N_LAYERS},
    'training': {'epochs': $EPOCHS, 'lr': $LR, 'temperature': $TEMPERATURE},
    'elapsed_seconds': elapsed,
    'epoch_stats': stats,
}
with open('$SAVE_DIR/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=float)
print(f'  Summary: $SAVE_DIR/training_summary.json')
"
