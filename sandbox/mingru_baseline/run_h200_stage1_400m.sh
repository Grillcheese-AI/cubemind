#!/usr/bin/env bash
#
# H200 stage 1 — 400M LM pretrain (final architecture, bilingual EN/FR).
#
# Replaces the 213M Cubby line. Architecture is L=18, FFN=4096,
# d_model=864, MoE-4/top-2, attention every-3, hippocampal every-4
# → ~402M params (~411M with --vsa-binding-head). The 213M run hit a
# capacity wall at val PPL ~5.0 with abundant data; 2× the params at
# the same data tier should buy a meaningful PPL drop and leave usable
# capacity for Stage 2 multitask + Stage 1.6 persona transfer.
#
# Architecture math (vocab=32128, no VSA-head):
#   embed:    32128 × 864                    = 27.76M
#   18 × MoE: 18 × 12 × 864²                 = 161.24M  (4 experts × MinGRU)
#   18 × GLU: 18 × 3 × 864 × 4096            = 191.10M  (SwiGLU FFN)
#   6  × attn:6 × 4 × 864²                   = 17.92M   (every 3rd layer)
#   5  × mem: 5 × 864²                       = 3.73M    (every 4th layer)
#   norms+gates+biases                       = 0.34M
#   ─────────────────────────────────────────────────
#   total                                    ≈ 402.1M
#
# Bilingual mix (~85% EN / ~15% FR):
#   English: Nemotron CC v2 + Wikibooks + Gutenberg + factual books
#            → ~1.7B EN tokens (~85%)
#   French:  Wikipedia (wikimedia/wikipedia config 20231101.fr)
#            → ~300M FR tokens (~15%, ~400k articles capped)
#   Total:   ~2.0B tokens at seq=1024, bs=64×accum=4 = 256k tok/step
#            × 8000 steps = 2.05B tokens seen ≈ 1 epoch.
#
# MoE health: --moe-aux-weight 0.01 enables Switch-style load balancing
# so all 4 experts learn (without it routing collapses to 1-2 experts).
# train_torch.py prints per-layer expert utilization every
# log_every × 5 = 250 steps (override with --moe-log-every) and flags
# any expert below 50% of ideal share.
#
# ─── PREREQS — build the bilingual corpus ONCE locally, then upload ───
#
# 1. Pull the French Wikipedia slice (capped at 15% of total token budget):
#
#    python sandbox/mingru_baseline/download_frwiki.py \
#        --output D:\grillcheese_training_data\jsonl\frwiki_400k.jsonl \
#        --max-records 400000 \
#        --min-chars 500
#
# 2. Flatten EN + FR into one shuffled-by-source pretrain.txt:
#
#    python sandbox/mingru_baseline/build_pretrain_corpus.py \
#        --input   D:\grillcheese_training_data\unified\nemotron_cc_v2_high_quality.1b_tokens.jsonl \
#        --input-cap 0 \
#        --input   D:\grillcheese_training_data\jsonl\wikibooks_corpus.jsonl \
#        --input-cap 0 \
#        --input   D:\grillcheese_training_data\jsonl\frwiki_400k.jsonl \
#        --input-cap 400000 \
#        --txt-dir E:\RAW_TEXTS\realms\gutemberg_books_unclassified \
#        --txt-dir E:\RAW_TEXTS\realms\factual_books \
#        --output  D:\grillcheese_training_data\pretrain_ext_v2_bilingual.txt
#
# 3. Tokenize once locally (the .bin shards travel to the pod):
#
#    python sandbox/mingru_baseline/tokenize_local.py \
#        --data-path D:\grillcheese_training_data\pretrain_ext_v2_bilingual.txt \
#        --tokenizer-path D:\grillcheese_training_data\tokenizer\grillcheese_spm32k_v2.model
#
# 4. Upload to pod:
#    /workspace/tokenizer/grillcheese_spm32k_v2.model
#    /workspace/data/pretrain_ext_v2_bilingual.txt
#    /workspace/cubemind/sandbox/mingru_baseline/data/
#        train_pretrain_ext_v2_bilingual_spm_grillcheese_spm32k_v2.bin
#        val_pretrain_ext_v2_bilingual_spm_grillcheese_spm32k_v2.bin
#        meta_*.json  tokenizer_*.json

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
: "${TOKENIZER_PATH:=/workspace/tokenizer/grillcheese_spm32k_v2.model}"
: "${PRETRAIN_DATA_PATH:=/workspace/data/pretrain_ext_v2_bilingual.txt}"
: "${RESULTS_DIR:=/workspace/results_h200}"

# ── Model — 400M FINAL architecture ────────────────────────────────────
# d=864 chosen to land ~400M with L=18, FFN=4096:
#   - divisible by attn_n_heads (4 → d_head=216, 6 → 144, 12 → 72)
#   - divisible by 32 (bf16 tensor-core friendly on H200)
D_MODEL=864
N_LAYERS=18
D_FFN=4096
SEQ_LEN=1024
VOCAB=32128

# ── Hybrid stack (full stack + MoE health-check enabled) ───────────────
HYBRID_FLAGS=(
    --vsa-binding-head --vsa-binding-d 10240
    --moe --moe-experts 4 --moe-top-k 2
    --moe-aux-weight 0.01
    --moe-warn-threshold 0.5
    --attention --attn-heads 4 --attn-window 128 --attn-every-n 3
    --memory --mem-max 200 --mem-write-threshold 0.4 --mem-every-n 4
    --mem-consolidate-every 1000
    --hypergrad
)

# ── Stage 1: from-scratch bilingual LM pretrain ────────────────────────
# 8000 steps × 256k tok/step = 2.05B tokens = ~1 epoch over the
# bilingual corpus. Chinchilla optimal for 400M is ~8B; we're paying
# real money — 2B is the "good enough to know if the architecture
# works" tier. Extend to 16-32k steps once we know routing is healthy.
S1_STEPS=8000
S1_BATCH=64           # per-step micro-batch (effective 256 with accum=4)
S1_GRAD_ACCUM=4
S1_LR=4e-4            # peak — MoE tolerates higher LR than dense
S1_MIN_LR=2e-5
S1_WARMUP=500         # smooth ramp — long enough for routing to stabilize

# ── Common ─────────────────────────────────────────────────────────────
DTYPE=bf16
WD=0.01
CLIP=1.0
LOG_EVERY=50
EVAL_EVERY=250
CKPT_EVERY=500
# MoE log fires every LOG_EVERY × 5 = 250 steps by default — same cadence
# as eval so each eval row pairs with a routing-health row. Override
# with --moe-log-every if you want it tighter.

# ── Guardrails ─────────────────────────────────────────────────────────
[[ -f "$TOKENIZER_PATH"      ]] || { echo "FATAL: tokenizer at $TOKENIZER_PATH" >&2; exit 1; }
[[ -f "$PRETRAIN_DATA_PATH"  ]] || { echo "FATAL: pretrain data at $PRETRAIN_DATA_PATH" >&2; exit 1; }

cd "$(dirname "$0")/../.."

S1_DIR="$RESULTS_DIR/stage1_400m"
mkdir -p "$S1_DIR"

echo "========================================================================"
echo "  H200 — stage 1 LM pretrain (400M, bilingual EN/FR ~85/15)"
echo "========================================================================"
echo "  tokenizer:    $TOKENIZER_PATH"
echo "  pretrain:     $PRETRAIN_DATA_PATH"
echo "  results:      $S1_DIR"
echo "  model:        d=$D_MODEL L=$N_LAYERS d_ffn=$D_FFN vocab=$VOCAB seq=$SEQ_LEN"
echo "  hybrid:       MoE-4/top-2 + attn(every-3) + hippo(every-4) + hypergrad + binding"
echo "  MoE health:   aux_weight=0.01 (Switch LB)  warn_threshold=0.5×ideal"
echo "  language mix: EN ~85% (Nemotron + books) / FR ~15% (Wikipedia)"
echo ""
echo "  schedule:     $S1_STEPS steps × bs=$S1_BATCH × accum=$S1_GRAD_ACCUM"
echo "                ≈ 2.05B tokens at seq_len=$SEQ_LEN"
echo "  LR:           $S1_LR -> $S1_MIN_LR cosine | warmup=$S1_WARMUP"
echo "========================================================================"
echo ""

RESULTS_DIR_OVERRIDE="$S1_DIR" python -u sandbox/mingru_baseline/train_torch.py \
    --tokenizer-path "$TOKENIZER_PATH" \
    --d-model $D_MODEL --n-layers $N_LAYERS --d-ffn $D_FFN \
    --vocab $VOCAB --seq-len $SEQ_LEN \
    --steps $S1_STEPS --batch-size $S1_BATCH --grad-accum $S1_GRAD_ACCUM \
    --lr $S1_LR --min-lr $S1_MIN_LR --warmup $S1_WARMUP \
    --wd $WD --clip $CLIP --dtype $DTYPE \
    --log-every $LOG_EVERY --eval-every $EVAL_EVERY --ckpt-every $CKPT_EVERY \
    --prompts sandbox/mingru_baseline/prompts_pretrain_ext.txt \
    --data-path "$PRETRAIN_DATA_PATH" \
    "${HYBRID_FLAGS[@]}"

S1_BEST="$S1_DIR/best.pt"
[[ -f "$S1_BEST" ]] || { echo "FATAL: stage 1 produced no best.pt" >&2; exit 1; }

echo ""
echo "========================================================================"
echo "  DONE — stage 1 (400M bilingual) best at $S1_BEST"
echo ""
echo "  What to look for in the [moe] log lines:"
echo "    - All experts in the 15-35% range (ideal 25% per expert for"
echo "      M=4, since token_frac is normalised across top_k slots)."
echo "    - 'COLLAPSED' tag on any layer means routing went to <12.5%"
echo "      on one expert. If this persists past step ~1000, raise"
echo "      --moe-aux-weight to 0.02 or 0.05 and restart from checkpoint."
echo ""
echo "  Sanity-check French generation:"
echo "    python sandbox/mingru_baseline/cubby_infer.py \\"
echo "        --checkpoint $S1_BEST \\"
echo "        --tokenizer  $TOKENIZER_PATH"
echo "  Then prompt: 'La Révolution française de 1789 a' / 'Paris est la'"
echo ""
echo "  Next: stage 1.5 temporal fine-tune from \$S1_BEST (edit"
echo "  run_h200_stage15_temporal.sh: D_MODEL=864 N_LAYERS=18 D_FFN=4096"
echo "  STAGE1_BEST=$S1_BEST)."
echo "========================================================================"
