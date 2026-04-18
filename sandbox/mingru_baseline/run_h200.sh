#!/usr/bin/env bash
#
# H200 RunPod kickoff — first real big-model run.
#
# Phase 1 of the H200 rollout: LM pretraining on c4_realnewslike with:
#   - Pretrained SentencePiece (grillcheese_spm32k_v2, 32128 vocab)
#   - VSA binding head (instead of tied Linear — structural regularization,
#     lexical decisiveness, faster inference, Colab H100 validated)
#   - Scaled model (~200M params) — 7x the Colab validation size
#   - Full 500M-token c4_realnewslike corpus (NO --subset-tokens trap)
#
# Phase 2 (separate run) will add mixed LM + multitask training on the
# 366K multitask rows from Gemini + SVC once phase 1 validates.
#
# Expected runtime on H200: ~4-6 hours for 25K steps
# Expected cost: ~$15-25 at RunPod H200 pricing
#
# Prereqs on the pod:
#   - git clone cubemind (dev branch)
#   - uv sync + pip install -e .[dev]
#   - /workspace/tokenizer/grillcheese_spm32k_v2.model uploaded
#   - /workspace/data/allenai_c4_realnewslike.500m_tokens.txt uploaded
#
# Run:
#   bash sandbox/mingru_baseline/run_h200.sh

set -euo pipefail

# ── Paths (override via env vars for different pod layouts) ────────────
: "${TOKENIZER_PATH:=/workspace/tokenizer/grillcheese_spm32k_v2.model}"
: "${DATA_PATH:=/workspace/data/allenai_c4_realnewslike.500m_tokens.txt}"
: "${RESULTS_DIR:=/workspace/results_h200}"

# ── Model config (200M params, d=768/L=12/d_ffn=3072) ──────────────────
D_MODEL=768
N_LAYERS=12
D_FFN=3072
SEQ_LEN=1024
VOCAB=32128          # matches grillcheese_spm32k_v2

# ── Training schedule ──────────────────────────────────────────────────
STEPS=25000          # ~6.5B tokens at bs=256/seq=1024
BATCH_SIZE=32        # per-step; grad accum brings effective batch to 256
GRAD_ACCUM=8
LR=6e-4              # slightly below Colab's 8e-4 (bigger model)
MIN_LR=6e-5
WARMUP=1500
WD=0.01
CLIP=1.0
DTYPE=bf16

# ── Eval / logging ─────────────────────────────────────────────────────
LOG_EVERY=50
EVAL_EVERY=500
CKPT_EVERY=1000

# ── Guardrails ─────────────────────────────────────────────────────────
if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "FATAL: tokenizer not found at $TOKENIZER_PATH" >&2
  echo "  upload it from D:/grillcheese_training_data/tokenizer/grillcheese_spm32k_v2.model" >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "FATAL: data file not found at $DATA_PATH" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$(dirname "$0")/../.."   # repo root

echo "========================================================================"
echo "  H200 — phase 1 LM pretraining"
echo "========================================================================"
echo "  tokenizer: $TOKENIZER_PATH"
echo "  data:      $DATA_PATH"
echo "  results:   $RESULTS_DIR"
echo "  model:     d=$D_MODEL L=$N_LAYERS d_ffn=$D_FFN vocab=$VOCAB seq=$SEQ_LEN"
echo "  schedule:  steps=$STEPS warmup=$WARMUP lr=$LR->$MIN_LR dtype=$DTYPE"
echo "  batch:     $BATCH_SIZE x accum=$GRAD_ACCUM = $(( BATCH_SIZE * GRAD_ACCUM )) eff"
echo ""

# Redirect the script's output dir to the results volume so checkpoints
# survive pod restarts.
export RESULTS_DIR_OVERRIDE="$RESULTS_DIR"

python -u sandbox/mingru_baseline/train_torch.py \
  --tokenizer-path "$TOKENIZER_PATH" \
  --vsa-binding-head --vsa-binding-d 10240 \
  --d-model $D_MODEL --n-layers $N_LAYERS --d-ffn $D_FFN \
  --vocab $VOCAB --seq-len $SEQ_LEN \
  --steps $STEPS --batch-size $BATCH_SIZE --grad-accum $GRAD_ACCUM \
  --lr $LR --min-lr $MIN_LR --warmup $WARMUP \
  --wd $WD --clip $CLIP --dtype $DTYPE \
  --log-every $LOG_EVERY --eval-every $EVAL_EVERY --ckpt-every $CKPT_EVERY \
  --prompts sandbox/mingru_baseline/prompts_news.txt \
  --data-path "$DATA_PATH"

echo ""
echo "========================================================================"
echo "  DONE — results in $RESULTS_DIR"
echo "========================================================================"
