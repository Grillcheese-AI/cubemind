#!/usr/bin/env bash
#
# H200 stage 1.5 — temporal fine-tune from stage 1 checkpoint.
#
# Loads the stage-1 LM-pretrained backbone (full hybrid stack) and
# continues training on the PUB/SUBJ-tagged temporal corpus
# (NYT 1851-2025 + historical events + dated knowledge books +
# Project Gutenberg). Goal: ground the model in calendar time so
# prompts like ``[PUB:1862] President Lincoln`` produce period-
# authentic prose, and ``[SUBJ:WWII]`` triggers WWII vocabulary
# regardless of register.
#
# This is a CONTINUATION, not from-scratch:
#   - LR is lower (1e-4 peak vs 6e-4 in stage 1) — the backbone
#     already has good language fluency
#   - Backbone stays UNFROZEN — we want the recurrence + binding
#     head to learn date-token conditioning
#   - Multitask heads are off (no JSONL data path) — pure LM mode
#   - Same hybrid stack (MoE + attn + memory + hypergrad + binding)
#
# Total: ~2000 steps, ~520M tokens, ~80 min on H200 SXM (~$7).
# After this completes, run stage 2 (multitask heads) from the
# stage-1.5 checkpoint instead of stage-1 to keep the temporal
# grounding in the final model.
#
# Prereqs on the pod:
#   /workspace/cubemind/sandbox/mingru_baseline/data/  (with the
#       cached train_<stem>_<tok_tag>.bin / val_<stem>_<tok_tag>.bin /
#       meta_*.json / tokenizer_*.json files for the temporal corpus
#       — built on first invocation if missing)
#   /workspace/results_h200/stage1_lm/best.pt   (stage 1 checkpoint)
#   /workspace/data/temporal_corpus_v3.txt      (the source text)
#   /workspace/tokenizer/grillcheese_spm32k_v2.model

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
: "${TOKENIZER_PATH:=/workspace/tokenizer/grillcheese_spm32k_v2.model}"
: "${TEMPORAL_DATA_PATH:=/workspace/data/temporal_corpus_v3.txt}"
: "${STAGE1_BEST:=/workspace/results_h200/stage1_lm/best.pt}"
: "${RESULTS_DIR:=/workspace/results_h200}"

# ── Model (matches stage 1 — ARCHITECTURE MUST BE IDENTICAL) ──────────
D_MODEL=768
N_LAYERS=12
D_FFN=3072
SEQ_LEN=1024
VOCAB=32128

# ── Hybrid stack flags (full stack, same as stage 1) ──────────────────
HYBRID_FLAGS=(
    --vsa-binding-head --vsa-binding-d 10240
    --moe --moe-experts 4 --moe-top-k 2
    --attention --attn-heads 4 --attn-window 128 --attn-every-n 3
    --memory --mem-max 200 --mem-write-threshold 0.4 --mem-every-n 4
    --mem-consolidate-every 1000
    --hypergrad
)

# ── Stage 1.5: temporal fine-tune ─────────────────────────────────────
S15_STEPS=2000        # ~520M tokens at bs=256/seq=1024 — ~25% of one
                      # epoch over the 3.6 GB temporal corpus
S15_BATCH=32          # effective 256 with grad-accum 8
S15_GRAD_ACCUM=8
S15_LR=1e-4           # 6x lower than stage 1's 6e-4 — fine-tune scale
S15_MIN_LR=1e-5
S15_WARMUP=100        # short warmup — model is already in steady state

# ── Common ─────────────────────────────────────────────────────────────
DTYPE=bf16
WD=0.01
CLIP=1.0
LOG_EVERY=50
EVAL_EVERY=250
CKPT_EVERY=500

# ── Guardrails ─────────────────────────────────────────────────────────
[[ -f "$TOKENIZER_PATH"     ]] || { echo "FATAL: tokenizer at $TOKENIZER_PATH" >&2; exit 1; }
[[ -f "$TEMPORAL_DATA_PATH" ]] || { echo "FATAL: temporal data at $TEMPORAL_DATA_PATH" >&2; exit 1; }
[[ -f "$STAGE1_BEST"        ]] || { echo "FATAL: stage 1 checkpoint at $STAGE1_BEST — run stage 1 first" >&2; exit 1; }

cd "$(dirname "$0")/../.."

S15_DIR="$RESULTS_DIR/stage15_temporal"
mkdir -p "$S15_DIR"

echo "========================================================================"
echo "  H200 — stage 1.5 temporal fine-tune"
echo "========================================================================"
echo "  tokenizer:    $TOKENIZER_PATH"
echo "  temporal:     $TEMPORAL_DATA_PATH ($(ls -l "$TEMPORAL_DATA_PATH" | awk '{print $5}') bytes)"
echo "  init from:    $STAGE1_BEST"
echo "  results:      $S15_DIR"
echo "  model:        d=$D_MODEL L=$N_LAYERS d_ffn=$D_FFN vocab=$VOCAB seq=$SEQ_LEN"
echo "  hybrid:       MoE + local attn + hippocampal + hypergrad + binding"
echo ""
echo "  schedule:     $S15_STEPS steps × bs=$S15_BATCH × accum=$S15_GRAD_ACCUM"
echo "  LR:           $S15_LR -> $S15_MIN_LR cosine | warmup=$S15_WARMUP"
echo "  backbone:     UNFROZEN (LM continues to learn date conditioning)"
echo "========================================================================"
echo ""

RESULTS_DIR_OVERRIDE="$S15_DIR" python -u sandbox/mingru_baseline/train_torch.py \
    --tokenizer-path "$TOKENIZER_PATH" \
    --d-model $D_MODEL --n-layers $N_LAYERS --d-ffn $D_FFN \
    --vocab $VOCAB --seq-len $SEQ_LEN \
    --steps $S15_STEPS --batch-size $S15_BATCH --grad-accum $S15_GRAD_ACCUM \
    --lr $S15_LR --min-lr $S15_MIN_LR --warmup $S15_WARMUP \
    --wd $WD --clip $CLIP --dtype $DTYPE \
    --log-every $LOG_EVERY --eval-every $EVAL_EVERY --ckpt-every $CKPT_EVERY \
    --prompts sandbox/mingru_baseline/prompts_news.txt \
    --data-path "$TEMPORAL_DATA_PATH" \
    --init-from "$STAGE1_BEST" \
    "${HYBRID_FLAGS[@]}"

S15_BEST="$S15_DIR/best.pt"
[[ -f "$S15_BEST" ]] || { echo "FATAL: stage 1.5 produced no best.pt" >&2; exit 1; }

echo ""
echo "========================================================================"
echo "  DONE — stage 1.5 best at $S15_BEST"
echo ""
echo "  Next step: stage 2 multitask head fine-tune from this checkpoint."
echo "  Either:"
echo "    1) Edit run_h200.sh to set:"
echo "         S1_BEST=$S15_BEST"
echo "       (skip its stage-1 block by Ctrl-C after launch — or just"
echo "        run the stage-2 python invocation manually with"
echo "        --init-from $S15_BEST --freeze-backbone)"
echo "    2) Test temporal grounding with live_session.py:"
echo "         python sandbox/mingru_baseline/live_session.py \\"
echo "             --checkpoint $S15_BEST \\"
echo "             --tokenizer  $TOKENIZER_PATH"
echo "       Try prompts like:"
echo "         /gen [PUB:1862-04-12] [SUBJ:1862] HEADLINE: The Battle of"
echo "         /gen [SUBJ:1939-1945] BOOK: An account of the war's"
echo "         /gen [PUB:1925] HEADLINE: New invention promises to"
echo "========================================================================"

# ── Quick reference: temporal prompt examples ──────────────────────────
#
# After this run you can prompt the model with PUB/SUBJ tags to get
# era-conditional generation. Examples (paste into live_session.py
# prompt or live_adapter.forward):
#
#   [PUB:1862-04-12] [SUBJ:1862]                 ← Civil War-era NYT register
#   [PUB:1925] HEADLINE:                         ← Roaring 20s news prose
#   [SUBJ:WWII] BOOK: An account of the war's    ← WWII subject, any register
#   [SUBJ:1066] HISTORICAL EVENT: William the    ← medieval style
#   [PUB:2024] [SUBJ:1861-1865]                  ← modern historian on Civil War
#   [PUB:1929] [SUBJ:1914-1918] TITLE:           ← contemporaneous WWI essay
#
# The PUB tag controls register/style (when written). The SUBJ tag
# controls topic/named-entities (what's discussed). Together they let
# you get e.g. "modern academic prose ABOUT 1860s" vs "1860s primary
# source register" from the same prompt scaffolding.
