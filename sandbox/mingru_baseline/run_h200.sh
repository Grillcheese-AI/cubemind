#!/usr/bin/env bash
#
# H200 RunPod kickoff — full two-stage cubemind training, single session.
#
# STAGE 1 — LM pretrain on c4_realnewslike with the full hybrid stack:
#   - MinGRU backbone, GLU FFN
#   - MoE-MinGRU mixer (specialization)
#   - Local sliding-window attention every 3 layers
#   - Hippocampal episodic memory every 4 layers + dopamine-gated writes
#   - SurpriseTracker hypergrad scaling on each MinGRU layer
#   - VSA binding head (fixed bipolar codebook, bf16-ready)
#   - Pretrained grillcheese_spm32k_v2 tokenizer (32128 vocab, opcodes
#     + multitask tags + VSA roles all single-token)
#
# STAGE 2 — Multitask head fine-tune on combined Gemini + SVC rows:
#   - Same architecture, backbone + LM head FROZEN
#   - Only the 5 MindForge LoRA aux heads (opcode/intent/schema/rule/
#     validity) and the binding head's query projection update
#   - Trains in ~30 min on the 366K multitask rows
#
# Total: ~5-7 hours on H200, single pod, ~$15-25.
#
# Adding new heads later (future-VSA, MoQE, hippocampal forge, etc.):
# each becomes its own cheap stage-2 run reusing the same stage-1
# checkpoint — Colab/local fine, no H200 required.
#
# Prereqs on the pod (after scripts/runpod_h200_bootstrap.sh):
#   /workspace/tokenizer/grillcheese_spm32k_v2.model
#   /workspace/data/allenai_c4_realnewslike.500m_tokens.txt
#   /workspace/data/multitask_combined.jsonl
#     (concat of multitask_gemini_v1.jsonl + multitask_svc_v1.jsonl —
#      see "data prep" section at the bottom for the cat command)

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
: "${TOKENIZER_PATH:=/workspace/tokenizer/grillcheese_spm32k_v2.model}"
# LM corpus for stage 1. Default = c4 alone, but you can concat
# OpenThoughts-114k chat-formatted reasoning traces for richer signal:
#
#   python sandbox/mingru_baseline/extract_openthoughts.py \
#       --input D:/grillcheese_training_data/jsonl/OpenThoughts-114k.jsonl \
#       --output D:/grillcheese_training_data/openthoughts_chat.txt
#   cat allenai_c4_realnewslike.500m_tokens.txt openthoughts_chat.txt \
#       > stage1_lm_combined.txt
#   # then point LM_DATA_PATH at stage1_lm_combined.txt
: "${LM_DATA_PATH:=/workspace/data/allenai_c4_realnewslike.500m_tokens.txt}"
: "${MT_DATA_PATH:=/workspace/data/multitask_combined.jsonl}"
: "${RESULTS_DIR:=/workspace/results_h200}"

# ── Model ──────────────────────────────────────────────────────────────
D_MODEL=768
N_LAYERS=12
D_FFN=3072
SEQ_LEN=1024
VOCAB=32128

# ── Hybrid stack toggles (all ON for full cubemind architecture) ───────
HYBRID_FLAGS=(
    --vsa-binding-head --vsa-binding-d 10240
    --moe --moe-experts 4 --moe-top-k 2
    --attention --attn-heads 4 --attn-window 128 --attn-every-n 3
    --memory --mem-max 200 --mem-write-threshold 0.4 --mem-every-n 4
    --mem-consolidate-every 1000
    --hypergrad
)

# ── Stage 1: LM pretrain ───────────────────────────────────────────────
S1_STEPS=20000        # ~5.2B tokens at bs=256/seq=1024
S1_BATCH=32           # effective 256 with grad-accum 8
S1_GRAD_ACCUM=8
S1_LR=6e-4
S1_MIN_LR=6e-5
S1_WARMUP=1500

# ── Stage 2: multitask head fine-tune ──────────────────────────────────
S2_STEPS=3000         # 366K rows / bs=64 ≈ 5700 rows/epoch → ~33 epochs
S2_BATCH=64           # smaller batch, multitask rows are short
S2_GRAD_ACCUM=2
S2_LR=2e-4            # smaller LR — only heads + binding query train
S2_MIN_LR=2e-5
S2_WARMUP=200

# ── Common ─────────────────────────────────────────────────────────────
DTYPE=bf16
WD=0.01
CLIP=1.0
LOG_EVERY=50
EVAL_EVERY=500
CKPT_EVERY=1000

# ── Guardrails ─────────────────────────────────────────────────────────
[[ -f "$TOKENIZER_PATH" ]] || { echo "FATAL: tokenizer at $TOKENIZER_PATH" >&2; exit 1; }
[[ -f "$LM_DATA_PATH"   ]] || { echo "FATAL: LM data at $LM_DATA_PATH" >&2; exit 1; }
[[ -f "$MT_DATA_PATH"   ]] || { echo "FATAL: multitask data at $MT_DATA_PATH (run cat to combine)" >&2; exit 1; }

# Validate multitask label ranges fit the head sizes — the Colab pre-flight
# discovered the SVC+Gemini combined file has out-of-range schema/rule/opcode
# IDs that crash CE with "t >= 0 && t < n_classes" on first batch. Run the
# scrubber once if the ranges aren't safe.
echo ""; echo ">>> Pre-flight: check multitask label ranges"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<PY
import json, io, sys
limits = {"opcode_id": 55, "intent_id": 6, "schema_id": 16,
          "rule_id": 32, "validity": 2}
maxes = {}
n = 0
with io.open("$MT_DATA_PATH", "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        n += 1
        try: row = json.loads(line)
        except Exception: continue
        for k, lim in limits.items():
            if k in row: maxes[k] = max(maxes.get(k, 0), int(row[k]))
ok = True
for k, lim in limits.items():
    seen = maxes.get(k)
    if seen is None:
        print(f"  WARN {k}: missing in all rows (head will get default 0)")
    elif seen >= lim:
        print(f"  FAIL {k}: max={seen} >= head size {lim}")
        ok = False
    else:
        print(f"  OK   {k}: max={seen} < head size {lim}")
if not ok:
    print("\\n  → run the scrubber first:")
    print(f"     python sandbox/mingru_baseline/scrub_multitask.py "
          f"--input  $MT_DATA_PATH "
          f"--output \${{MT_DATA_PATH%.jsonl}}_clean.jsonl")
    print("     then point MT_DATA_PATH at the _clean.jsonl")
    sys.exit(1)
PY

cd "$(dirname "$0")/../.."

S1_DIR="$RESULTS_DIR/stage1_lm"
S2_DIR="$RESULTS_DIR/stage2_multitask"
mkdir -p "$S1_DIR" "$S2_DIR"

echo "========================================================================"
echo "  H200 — full two-stage cubemind training"
echo "========================================================================"
echo "  tokenizer:    $TOKENIZER_PATH"
echo "  LM data:      $LM_DATA_PATH"
echo "  MT data:      $MT_DATA_PATH"
echo "  results:      $RESULTS_DIR"
echo "  model:        d=$D_MODEL L=$N_LAYERS d_ffn=$D_FFN vocab=$VOCAB seq=$SEQ_LEN"
echo "  hybrid:       MoE + local attn + hippocampal + hypergrad + binding"
echo ""
echo "  stage 1:      $S1_STEPS steps × bs=$S1_BATCH × accum=$S1_GRAD_ACCUM"
echo "  stage 2:      $S2_STEPS steps × bs=$S2_BATCH × accum=$S2_GRAD_ACCUM (frozen base)"
echo "========================================================================"
echo ""

# ── STAGE 1 ────────────────────────────────────────────────────────────
echo ""; echo ">>> STAGE 1: LM pretrain (~5-6 hours)"; echo ""
RESULTS_DIR_OVERRIDE="$S1_DIR" python -u sandbox/mingru_baseline/train_torch.py \
    --tokenizer-path "$TOKENIZER_PATH" \
    --d-model $D_MODEL --n-layers $N_LAYERS --d-ffn $D_FFN \
    --vocab $VOCAB --seq-len $SEQ_LEN \
    --steps $S1_STEPS --batch-size $S1_BATCH --grad-accum $S1_GRAD_ACCUM \
    --lr $S1_LR --min-lr $S1_MIN_LR --warmup $S1_WARMUP \
    --wd $WD --clip $CLIP --dtype $DTYPE \
    --log-every $LOG_EVERY --eval-every $EVAL_EVERY --ckpt-every $CKPT_EVERY \
    --prompts sandbox/mingru_baseline/prompts_news.txt \
    --data-path "$LM_DATA_PATH" \
    "${HYBRID_FLAGS[@]}"

# Move the stage-1 best checkpoint somewhere stage 2 won't clobber.
S1_BEST="$S1_DIR/best.pt"
[[ -f "$S1_BEST" ]] || { echo "FATAL: stage 1 produced no best.pt" >&2; exit 1; }
echo ""; echo ">>> STAGE 1 done. Best checkpoint: $S1_BEST"

# ── STAGE 2 ────────────────────────────────────────────────────────────
echo ""; echo ">>> STAGE 2: multitask head fine-tune on frozen base (~30-60 min)"; echo ""
RESULTS_DIR_OVERRIDE="$S2_DIR" python -u sandbox/mingru_baseline/train_torch.py \
    --tokenizer-path "$TOKENIZER_PATH" \
    --d-model $D_MODEL --n-layers $N_LAYERS --d-ffn $D_FFN \
    --vocab $VOCAB --seq-len $SEQ_LEN \
    --steps $S2_STEPS --batch-size $S2_BATCH --grad-accum $S2_GRAD_ACCUM \
    --lr $S2_LR --min-lr $S2_MIN_LR --warmup $S2_WARMUP \
    --wd $WD --clip $CLIP --dtype $DTYPE \
    --log-every $LOG_EVERY --eval-every $EVAL_EVERY --ckpt-every $CKPT_EVERY \
    --use-jsonl-dataset --jsonl-path "$MT_DATA_PATH" \
    --aux-opcode-loss-weight   0.4 \
    --aux-intent-loss-weight   0.2 \
    --aux-schema-loss-weight   0.2 \
    --aux-rule-loss-weight     0.2 \
    --aux-validity-loss-weight 0.1 \
    --init-from "$S1_BEST" --freeze-backbone \
    "${HYBRID_FLAGS[@]}"

echo ""
echo "========================================================================"
echo "  DONE — stage 1 in $S1_DIR, stage 2 in $S2_DIR"
echo "  Pull both back with:"
echo "    scp -r root@<pod-ip>:$RESULTS_DIR ./"
echo "========================================================================"

# ── Data-prep notes (run on the pod ONCE before this script) ───────────
#
# Combine the Gemini and SVC multitask outputs into one JSONL the
# trainer can read. The two sources have aligned schemas — Gemini's
# 36K + SVC's 330K = 366K rows total.
#
#   cat /workspace/data/multitask_gemini_v1.jsonl \
#       /workspace/data/multitask_svc_v1.jsonl \
#       > /workspace/data/multitask_combined.jsonl
#
# Optional: post-process to bucket low-frequency schemas into "other":
#   python sandbox/mingru_baseline/postprocess_multitask.py \
#       --input  /workspace/data/multitask_combined.jsonl \
#       --output /workspace/data/multitask_combined_clean.jsonl \
#       --top-schemas 16 --top-rules 32
#   # then point MT_DATA_PATH at the _clean.jsonl
