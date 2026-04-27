#!/usr/bin/env bash
#
# H200 stage 1-ext — continue LM pretrain on a combined fresh corpus
# (Nemotron CC v2 + Wikibooks, or whatever was flattened into the
# ``pretrain_ext`` corpus). Diagnostic for the Run-1 val PPL plateau.
#
# Run 1 result: val PPL 5.17 after 8000 steps, 589M tokens.
#   -> 2.75 tokens/param (Chinchilla optimal ~20). Badly data-starved.
#
# This run continues the 213M backbone from stage1_lm/best.pt on ~2B
# fresh tokens to disambiguate:
#   * If val PPL drops below 5.17    -> data-limited. Keep this size.
#   * If val PPL plateaus at 5.0-5.2 -> capacity wall. Next run scales up.
#
# Continuation, NOT restart:
#   - Same architecture (must match the checkpoint exactly)
#   - Same hybrid stack (MoE + local attn + hippocampal + hypergrad + binding)
#   - Backbone UNFROZEN (we want the full model to keep learning)
#   - Peak LR 3e-4 (half of Run 1's 6e-4) -- mid-run refresh, not scratch
#   - Short warmup (500 steps) -- model is already in steady state
#
# Prereqs on the pod (build locally with build_pretrain_corpus.py +
# tokenize_local.py, then upload):
#   /workspace/tokenizer/grillcheese_spm32k_v2.model
#   /workspace/data/pretrain_ext_v1.txt
#   /workspace/cubemind/sandbox/mingru_baseline/data/
#       train_pretrain_ext_v1_spm_grillcheese_spm32k_v2.bin
#       val_pretrain_ext_v1_spm_grillcheese_spm32k_v2.bin
#       meta_pretrain_ext_v1_spm_grillcheese_spm32k_v2.json
#       tokenizer_pretrain_ext_v1_spm_grillcheese_spm32k_v2.json
#   /workspace/results_h200/stage1_lm/best.pt

set -euo pipefail

# -- Paths -------------------------------------------------------------
: "${TOKENIZER_PATH:=/workspace/tokenizer/grillcheese_spm32k_v2.model}"
: "${PRETRAIN_DATA_PATH:=/workspace/data/pretrain_ext_v1.txt}"
: "${STAGE1_BEST:=/workspace/results_h200/stage1_lm/best.pt}"
: "${RESULTS_DIR:=/workspace/results_h200}"

# -- Model (MUST match stage1_lm/best.pt architecture exactly) ---------
D_MODEL=768
N_LAYERS=12
D_FFN=3072
SEQ_LEN=768           # matches Run 1 checkpoint (reduced from 1024 mid-run)
VOCAB=32128

# -- Hybrid stack flags (full stack, same as stage 1) ------------------
HYBRID_FLAGS=(
    --vsa-binding-head --vsa-binding-d 10240
    --moe --moe-experts 4 --moe-top-k 2
    --attention --attn-heads 4 --attn-window 128 --attn-every-n 3
    --memory --mem-max 200 --mem-write-threshold 0.4 --mem-every-n 4
    --mem-consolidate-every 1000
    --hypergrad
)

# -- Stage 1-ext schedule (~10h target, ~1.1B tokens processed) --------
S1E_STEPS=15000       # at bs=24/ga=4/seq=768 ~ 1.1B tokens processed
S1E_BATCH=24          # effective 96 -- matches Run 1 (clean diagnostic)
S1E_GRAD_ACCUM=4
S1E_LR=3e-4           # half of Run 1 peak -- continuation LR
S1E_MIN_LR=3e-5
S1E_WARMUP=500        # short, model is already stable

# -- Common ------------------------------------------------------------
DTYPE=bf16
WD=0.01
CLIP=1.0
LOG_EVERY=50
EVAL_EVERY=250        # frequent eval -- catch plateau early
CKPT_EVERY=500

# -- Guardrails --------------------------------------------------------
[[ -f "$TOKENIZER_PATH"     ]] || { echo "FATAL: tokenizer at $TOKENIZER_PATH" >&2; exit 1; }
[[ -f "$PRETRAIN_DATA_PATH" ]] || { echo "FATAL: pretrain data at $PRETRAIN_DATA_PATH" >&2; exit 1; }
[[ -f "$STAGE1_BEST"        ]] || { echo "FATAL: stage 1 best at $STAGE1_BEST -- run stage 1 first" >&2; exit 1; }

cd "$(dirname "$0")/../.."

S1E_DIR="$RESULTS_DIR/stage1_ext_nemotron"
mkdir -p "$S1E_DIR"

echo "========================================================================"
echo "  H200 -- stage 1 extension (fresh pretrain continuation)"
echo "========================================================================"
echo "  tokenizer:    $TOKENIZER_PATH"
echo "  pretrain:     $PRETRAIN_DATA_PATH ($(ls -l "$PRETRAIN_DATA_PATH" | awk '{print $5}') bytes)"
echo "  init from:    $STAGE1_BEST  (Run 1 val PPL ~5.17)"
echo "  results:      $S1E_DIR"
echo "  model:        d=$D_MODEL L=$N_LAYERS d_ffn=$D_FFN vocab=$VOCAB seq=$SEQ_LEN"
echo "  hybrid:       MoE + local attn + hippocampal + hypergrad + binding"
echo ""
echo "  schedule:     $S1E_STEPS steps x bs=$S1E_BATCH x accum=$S1E_GRAD_ACCUM"
echo "  LR:           $S1E_LR -> $S1E_MIN_LR cosine | warmup=$S1E_WARMUP"
echo "  backbone:     UNFROZEN"
echo "========================================================================"
echo ""
echo "  [DIAGNOSTIC]"
echo "    val PPL drops below 5.17     => data-limited, keep this size"
echo "    val PPL plateaus at 5.0-5.2  => capacity wall, scale up next"
echo ""

RESULTS_DIR_OVERRIDE="$S1E_DIR" python -u sandbox/mingru_baseline/train_torch.py \
    --tokenizer-path "$TOKENIZER_PATH" \
    --d-model $D_MODEL --n-layers $N_LAYERS --d-ffn $D_FFN \
    --vocab $VOCAB --seq-len $SEQ_LEN \
    --steps $S1E_STEPS --batch-size $S1E_BATCH --grad-accum $S1E_GRAD_ACCUM \
    --lr $S1E_LR --min-lr $S1E_MIN_LR --warmup $S1E_WARMUP \
    --wd $WD --clip $CLIP --dtype $DTYPE \
    --log-every $LOG_EVERY --eval-every $EVAL_EVERY --ckpt-every $CKPT_EVERY \
    --prompts sandbox/mingru_baseline/prompts_pretrain_ext.txt \
    --data-path "$PRETRAIN_DATA_PATH" \
    --init-from "$STAGE1_BEST" \
    "${HYBRID_FLAGS[@]}"

S1E_BEST="$S1E_DIR/best.pt"
[[ -f "$S1E_BEST" ]] || { echo "FATAL: stage 1-ext produced no best.pt" >&2; exit 1; }

echo ""
echo "========================================================================"
echo "  DONE -- stage 1-ext best at $S1E_BEST"
echo ""
echo "  If PPL dropped below 5.17 (data-limited confirmed):"
echo "    Run stage 1.5 from this checkpoint:"
echo "      STAGE1_BEST=$S1E_BEST \\"
echo "        bash sandbox/mingru_baseline/run_h200_stage15_temporal.sh"
echo ""
echo "  If PPL plateaued at 5.0-5.2 (capacity wall confirmed):"
echo "    Scale up (e.g. d=896 L=14 d_ffn=3584) and re-pretrain from scratch"
echo "    on the SAME combined corpus. Do NOT fold 1.5/1.6 onto this checkpoint"
echo "    at the current size -- capacity-limited fine-tunes waste budget."
echo "========================================================================"
