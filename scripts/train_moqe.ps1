# MoQE Student Distillation Training
#
# Usage:
#   .\scripts\train_moqe.ps1                    # defaults
#   .\scripts\train_moqe.ps1 -Epochs 10         # override epochs
#   .\scripts\train_moqe.ps1 -Preprocess        # preprocess logits first
#   .\scripts\train_moqe.ps1 -Preprocess -DModel 256 -NLayers 4

param(
    [switch]$Preprocess,
    [string]$Src      = "G:\MYDRIVE",
    [string]$Data     = "data\logits_512",
    [string]$SaveDir  = "data\checkpoints",
    [int]$Vocab       = 151936,
    [int]$DModel      = 2048,
    [int]$NLayers     = 12, # 24 moqe experts (4bit, 8bit = 2 experts per layer) 32 tokens per expert = 768 most stable for local training / distillation
    [int]$Epochs      = 3,
    [int]$MaxSeqLen   = 768,
    [float]$Temperature = 2.0,
    [float]$Target8bit  = 0.15,
    [string]$LR       = "0.000303",
    [int]$SaveEvery   = 50,
    [int]$Workers     = 8
)

$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "============================================"
Write-Host "  MoQE Student Distillation"
Write-Host "============================================"
Write-Host "  Model:   vocab=$Vocab d=$DModel L=$NLayers"
Write-Host "  Data:    $Data (seq_len=$MaxSeqLen)"
Write-Host "  Train:   epochs=$Epochs lr=$LR T=$Temperature"
Write-Host "  Target:  $Target8bit 8-bit fraction"
Write-Host "  Save:    $SaveDir (every $SaveEvery batches)"
Write-Host "============================================"

# Step 1: Preprocess logits (optional)
if ($Preprocess) {
    Write-Host ""
    Write-Host "[1/2] Preprocessing teacher logits..."
    uv run python -u scripts/preprocess_logits.py `
        --src $Src --out $Data `
        --max-seq-len $MaxSeqLen --workers $Workers
    Write-Host ""
}

# Step 2: Verify data exists
$nFiles = (Get-ChildItem -Path $Data -Filter "*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
if ($nFiles -eq 0) {
    Write-Host "ERROR: No .npz files in $Data" -ForegroundColor Red
    Write-Host "  Run with -Preprocess to create them from $Src"
    Pop-Location
    exit 1
}
Write-Host "Found $nFiles logit files in $Data"

# Step 3: Train
Write-Host ""
Write-Host "Starting training..."

$pyCode = @"
from cubemind.execution.moqe import MoQEModel
from cubemind.training.moqe_distillation import run_offline_distillation
import time, json, os

model = MoQEModel(
    vocab_size=$Vocab,
    d_model=$DModel,
    n_layers=$NLayers,
    seed=42,
)
print(f'MoQE: v={model.vocab_size} d={model.d_model} L={model.n_layers}')

t = time.perf_counter()
stats = run_offline_distillation(
    model,
    data_dir=r'$Data',
    epochs=$Epochs,
    max_seq_len=$MaxSeqLen,
    temperature=$Temperature,
    target_8bit=$Target8bit,
    lr=$LR,
    save_dir=r'$SaveDir',
    save_every=$SaveEvery,
)
elapsed = time.perf_counter() - t

print()
print('=' * 44)
print('  Training Complete')
print('=' * 44)
total_batches = sum(s['n_batches'] for s in stats)
print(f'  Time:    {elapsed:.1f}s ({total_batches/elapsed:.1f} batch/s)')
for s in stats:
    print(f'  E{s["epoch"]}: loss={s["avg_loss"]:.4f} CE={s["avg_ce"]:.4f} KD={s["avg_kd"]:.4f} 8b={s["avg_8bit_frac"]*100:.1f}%')

os.makedirs(r'$SaveDir', exist_ok=True)
summary = {
    'model': {'vocab': $Vocab, 'd_model': $DModel, 'n_layers': $NLayers},
    'training': {'epochs': $Epochs, 'lr': $LR, 'temperature': $Temperature},
    'elapsed_seconds': elapsed,
    'epoch_stats': stats,
}
with open(os.path.join(r'$SaveDir', 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=float)
print(f'  Summary: $SaveDir/training_summary.json')
"@

uv run python -u -c $pyCode

Pop-Location
