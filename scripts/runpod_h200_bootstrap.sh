#!/usr/bin/env bash
#
# RunPod H200 bootstrap — PyTorch-only, no Vulkan/grilly dependencies.
#
# Unlike scripts/runpod_bootstrap.sh (which targets the Vulkan/grilly
# backend), this script is for the H200 pretraining path where we use
# PyTorch + CUDA directly. No Vulkan SDK, no grilly native extensions.
#
# Prepares a RunPod H200 pod for running
# sandbox/mingru_baseline/run_h200.sh:
#   1. Installs uv + Python 3.12 (if not present)
#   2. Clones cubemind (dev branch) to /workspace/cubemind
#   3. Installs Python deps
#   4. Creates /workspace/tokenizer and /workspace/data placeholders
#   5. Prints the next-step upload commands
#
# Data upload (run from your laptop AFTER bootstrap finishes):
#
#   # Tokenizer (~1 MB)
#   scp D:/grillcheese_training_data/tokenizer/grillcheese_spm32k_v2.model \
#       root@<pod-ip>:/workspace/tokenizer/
#
#   # c4_realnewslike corpus (~2 GB)
#   scp D:/grillcheese_training_data/unified/allenai_c4_realnewslike.500m_tokens.txt \
#       root@<pod-ip>:/workspace/data/
#
# Then on the pod:
#   cd /workspace/cubemind
#   bash sandbox/mingru_baseline/run_h200.sh

set -euo pipefail

: "${WORKSPACE:=/workspace}"
: "${CUBEMIND_REPO:=https://github.com/Grillcheese-AI/cubemind.git}"
: "${CUBEMIND_BRANCH:=dev}"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m   %s\n' "$*"; }
die()  { printf '  \033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Sanity check ────────────────────────────────────────────────────
bold "[1/5] Sanity check"
command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi — wrong pod type"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
ok "GPU: $GPU (${VRAM} MiB)"
if [[ ! "$GPU" =~ H200|H100|A100|Blackwell ]]; then
    warn "GPU is $GPU — expected H200 (config is tuned for H200's 141 GB HBM3)"
fi

# ── 2. Python + uv ─────────────────────────────────────────────────────
bold "[2/5] Python + uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed"
else
    ok "uv already present: $(uv --version)"
fi

# ── 3. Clone cubemind ──────────────────────────────────────────────────
bold "[3/5] cubemind repo"
CUBEMIND_DIR="$WORKSPACE/cubemind"
if [[ -d "$CUBEMIND_DIR/.git" ]]; then
    cd "$CUBEMIND_DIR"
    git fetch origin
    git checkout "$CUBEMIND_BRANCH"
    git pull origin "$CUBEMIND_BRANCH"
    ok "cubemind pulled: $(git rev-parse --short HEAD) on $CUBEMIND_BRANCH"
else
    mkdir -p "$WORKSPACE"
    git clone --branch "$CUBEMIND_BRANCH" "$CUBEMIND_REPO" "$CUBEMIND_DIR"
    ok "cubemind cloned to $CUBEMIND_DIR"
fi

# ── 4. Python env ──────────────────────────────────────────────────────
bold "[4/5] Python environment"
cd "$CUBEMIND_DIR"
# Explicit packages for the H200 path — we don't need grilly's Vulkan
# machinery, but we do need torch with CUDA, sentencepiece, and
# the data-pipeline deps. Pinning minimum versions to what the code
# was validated against.
uv venv --python 3.12 2>&1 | tail -3
uv pip install \
    'torch>=2.4' \
    'numpy>=1.24' \
    'tokenizers>=0.20' \
    'sentencepiece>=0.2' \
    'python-dotenv>=1.0' \
    'pymupdf>=1.24' \
    'scikit-learn>=1.5' 2>&1 | tail -5
# Verify CUDA is live
CUDA_OK=$(uv run python -c "import torch; print('ok' if torch.cuda.is_available() else 'no')" 2>&1 | tail -1)
[[ "$CUDA_OK" == "ok" ]] && ok "torch.cuda.is_available() = True" \
                        || die "torch.cuda NOT available — check the pod image"

# ── 5. Data directories ────────────────────────────────────────────────
bold "[5/5] Data placeholders"
mkdir -p "$WORKSPACE/tokenizer" "$WORKSPACE/data" "$WORKSPACE/results_h200"
ok "$WORKSPACE/tokenizer  (upload grillcheese_spm32k_v2.model here)"
ok "$WORKSPACE/data       (upload allenai_c4_realnewslike.500m_tokens.txt here)"
ok "$WORKSPACE/results_h200  (persistent — survives pod restarts)"

# ── Done ───────────────────────────────────────────────────────────────
cat <<EOF

========================================================================
  Bootstrap complete.
========================================================================

  Next steps:

  1. Upload files FROM YOUR LAPTOP (scp / rsync / SFTP via the RunPod UI):

       D:/grillcheese_training_data/tokenizer/grillcheese_spm32k_v2.model
         -> $WORKSPACE/tokenizer/

       D:/grillcheese_training_data/unified/allenai_c4_realnewslike.500m_tokens.txt
         -> $WORKSPACE/data/

  2. Launch training:

       cd $CUBEMIND_DIR
       bash sandbox/mingru_baseline/run_h200.sh

  3. Monitor:

       tail -f $WORKSPACE/results_h200/training.log
       watch -n2 nvidia-smi

  4. When done, pull checkpoints back:

       scp -r root@<pod-ip>:$WORKSPACE/results_h200 ./

EOF
