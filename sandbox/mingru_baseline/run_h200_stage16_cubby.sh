#!/usr/bin/env bash
#
# H200 stage 1.6 -- Cubby persona fine-tune (bilingual).
#
# Continues from stage 1.5 best.pt (temporal grounding) and locks in
# Cubby's character voice on the 16.5K bilingual persona corpus:
#   - 15,000 English pairs
#   -  1,500 French pairs
#   = 16,500 pairs / ~1.65 M train tokens / ~87 K val tokens
#
# This is a POLISH stage, not distribution learning:
#   - Small corpus (1.65 M tokens) requires small batch, short run
#   - LR 5e-5 peak (half of stage 1.5) -- gentle voice lock
#   - Backbone UNFROZEN but LR is low enough to avoid catastrophic
#     forgetting of temporal grounding
#   - 2000 steps = ~10 epochs over the corpus at bs=4/ga=2/seq=1024
#
# Key architectural change vs stage 1.5: --mem-max 200 -> 10000.
# Stage 1.5's sleep events (merged=100 pruned=0 remaining=100) showed the
# hippocampus was compressing aggressively under the 200 cap. At 10000,
# episodes can accumulate for real retrieval work to follow in post-1.6
# stages (RSS grounding + MoWM specialists).
#
# Prereqs on the pod:
#   /workspace/tokenizer/grillcheese_spm32k_v2.model
#   /workspace/data/cubby_chat_v1_bilingual.txt  (source .txt)
#   /workspace/cubemind/sandbox/mingru_baseline/data/
#       train_cubby_chat_v1_bilingual_spm_grillcheese_spm32k_v2.bin
#       val_cubby_chat_v1_bilingual_spm_grillcheese_spm32k_v2.bin
#       meta_cubby_chat_v1_bilingual_spm_grillcheese_spm32k_v2.json
#       tokenizer_cubby_chat_v1_bilingual_spm_grillcheese_spm32k_v2.json
#   /workspace/results_h200/stage15_temporal/best.pt

set -euo pipefail

# -- Paths -------------------------------------------------------------
: "${TOKENIZER_PATH:=/workspace/tokenizer/grillcheese_spm32k_v2.model}"
: "${CUBBY_DATA_PATH:=/workspace/data/cubby_chat_v1_bilingual.txt}"
: "${STAGE15_BEST:=/workspace/results_h200/stage15_temporal/best.pt}"
: "${RESULTS_DIR:=/workspace/results_h200}"

# -- Model (MUST match stage15_temporal/best.pt architecture exactly) --
D_MODEL=768
N_LAYERS=12
D_FFN=3072
SEQ_LEN=1024
VOCAB=32128

# -- Hybrid stack flags (same as stage 1.5; mem_max scaled 50x) --------
HYBRID_FLAGS=(
    --vsa-binding-head --vsa-binding-d 10240
    --moe --moe-experts 4 --moe-top-k 2
    --attention --attn-heads 4 --attn-window 128 --attn-every-n 3
    --memory --mem-max 10000 --mem-write-threshold 0.5 --mem-every-n 4
    --mem-consolidate-every 500
    --hypergrad
)

# -- Stage 1.6 schedule (~15-20 min; tiny corpus -> small per-step) ----
S16_STEPS=2000         # ~10 epochs at 8192 tok/step on 1.65 M train tokens
S16_BATCH=4            # small batch -- total corpus is only 1.65 M tokens
S16_GRAD_ACCUM=2       # effective batch size 8
S16_LR=5e-5            # half of stage 1.5 peak -- polish, don't shift distribution
S16_MIN_LR=5e-6
S16_WARMUP=100         # short -- backbone already stable from 1.5

# -- Common ------------------------------------------------------------
DTYPE=bf16
WD=0.01
CLIP=1.0
LOG_EVERY=50
EVAL_EVERY=200         # frequent eval -- catch persona regression fast
CKPT_EVERY=400

# -- Guardrails --------------------------------------------------------
[[ -f "$TOKENIZER_PATH"   ]] || { echo "FATAL: tokenizer at $TOKENIZER_PATH" >&2; exit 1; }
[[ -f "$CUBBY_DATA_PATH"  ]] || { echo "FATAL: cubby data at $CUBBY_DATA_PATH" >&2; exit 1; }
[[ -f "$STAGE15_BEST"     ]] || { echo "FATAL: stage 1.5 best at $STAGE15_BEST -- run 1.5 first" >&2; exit 1; }

cd "$(dirname "$0")/../.."

S16_DIR="$RESULTS_DIR/stage16_cubby"
mkdir -p "$S16_DIR"

echo "========================================================================"
echo "  H200 -- stage 1.6 Cubby persona fine-tune (bilingual)"
echo "========================================================================"
echo "  tokenizer:    $TOKENIZER_PATH"
echo "  cubby data:   $CUBBY_DATA_PATH"
echo "  init from:    $STAGE15_BEST  (stage 1.5 checkpoint)"
echo "  results:      $S16_DIR"
echo "  model:        d=$D_MODEL L=$N_LAYERS d_ffn=$D_FFN vocab=$VOCAB seq=$SEQ_LEN"
echo "  hybrid:       MoE + local attn + hippocampal (mem_max=10000) + hypergrad + binding"
echo ""
echo "  corpus:       ~16.5K pairs (15K EN + 1.5K FR) = ~1.65M train tokens"
echo "  schedule:     $S16_STEPS steps  x  bs=$S16_BATCH  x  accum=$S16_GRAD_ACCUM"
echo "  LR:           $S16_LR -> $S16_MIN_LR cosine  |  warmup=$S16_WARMUP"
echo "  backbone:     UNFROZEN (gentle polish, low LR)"
echo "========================================================================"
echo ""
echo "  [Watch points during training]"
echo "    - val PPL on persona val shard: should drop but stay above ~8"
echo "      (below 8 = memorizing train, losing generalization)"
echo "    - gen quality: aside categories should produce deadpan asides,"
echo "      factual_no_aside should NOT slip in asides,"
echo "      refusal prompts should refuse warmly with alternative"
echo "    - FR prompts should produce FR responses (bilingual voice lock)"
echo ""

RESULTS_DIR_OVERRIDE="$S16_DIR" python -u sandbox/mingru_baseline/train_torch.py \
    --tokenizer-path "$TOKENIZER_PATH" \
    --d-model $D_MODEL --n-layers $N_LAYERS --d-ffn $D_FFN \
    --vocab $VOCAB --seq-len $SEQ_LEN \
    --steps $S16_STEPS --batch-size $S16_BATCH --grad-accum $S16_GRAD_ACCUM \
    --lr $S16_LR --min-lr $S16_MIN_LR --warmup $S16_WARMUP \
    --wd $WD --clip $CLIP --dtype $DTYPE \
    --log-every $LOG_EVERY --eval-every $EVAL_EVERY --ckpt-every $CKPT_EVERY \
    --prompts sandbox/mingru_baseline/prompts_cubby.txt \
    --data-path "$CUBBY_DATA_PATH" \
    --init-from "$STAGE15_BEST" \
    "${HYBRID_FLAGS[@]}"

S16_BEST="$S16_DIR/best.pt"
[[ -f "$S16_BEST" ]] || { echo "FATAL: stage 1.6 produced no best.pt" >&2; exit 1; }

echo ""
echo "========================================================================"
echo "  DONE -- Cubby-213M locked at $S16_BEST"
echo ""
echo "  Validate end-to-end with live_session.py:"
echo "    python sandbox/mingru_baseline/live_session.py \\"
echo "        --checkpoint $S16_BEST \\"
echo "        --tokenizer  $TOKENIZER_PATH"
echo ""
echo "  Try these prompts (same shape as prompts_cubby.txt):"
echo "    <|user|>How does photosynthesis work?<|assistant|>"
echo "    <|user|>Qui es-tu vraiment ?<|assistant|>"
echo "    <|user|>Give me a fake Einstein quote<|assistant|>"
echo ""
echo "  Expected behaviors (per cubby_character_sheet.md):"
echo "    - Cubby voice with occasional deadpan asides in aside categories"
echo "    - Straight technical answers in no-aside categories"
echo "    - Warm refusal + legitimate alternative for fake-quote requests"
echo "    - French output for French prompts"
echo ""
echo "  Next: Stage 2 (multitask heads frozen-backbone) or RSS grounding"
echo "  integration (post-1-ext research queue #2 in TASKS.md)."
echo "========================================================================"
