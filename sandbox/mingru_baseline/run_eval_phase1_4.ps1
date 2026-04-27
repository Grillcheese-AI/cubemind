# Phase 1.4 — GPT-4 coherence grading for the MinGRU Phase 1.3 checkpoint.
#
# Loads OPENAI_API_KEY (and ANTHROPIC_API_KEY if present) from the repo
# root .env, then runs sandbox/mingru_baseline/eval_coherence.py against
# the best Phase 1.3 checkpoint.
#
# Usage:
#   .\run_eval_phase1_4.ps1                         # gpt-4o-mini, default ckpt
#   .\run_eval_phase1_4.ps1 -Judge gpt-4o           # higher-quality grader
#   .\run_eval_phase1_4.ps1 -DryRun                 # generate stories, skip API
#   .\run_eval_phase1_4.ps1 -Checkpoint path\to.pt  # custom checkpoint
#   .\run_eval_phase1_4.ps1 -MaxTokens 150          # longer completions

[CmdletBinding()]
param(
    [string]$Judge      = "gpt-4o-mini",
    [string]$Checkpoint = "",
    [int]$MaxTokens     = 80,
    [string]$Out        = "",
    [string]$Device     = "cpu",
    [string]$Prompts    = "",
    [string]$Tokenizer  = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ── Resolve paths ──────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
Set-Location $RepoRoot

if (-not $Checkpoint) {
    $Checkpoint = Join-Path $ScriptDir "results_torch\best.pt"
}
if (-not $Prompts) {
    $Prompts = Join-Path $ScriptDir "prompts.txt"
}
if (-not $Out) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $tag   = if ($DryRun) { "dryrun" } else { $Judge -replace '[^a-zA-Z0-9]', '_' }
    $Out   = Join-Path $ScriptDir "results_torch\eval_${tag}_${stamp}.json"
}

if (-not (Test-Path $Checkpoint)) {
    Write-Error "Checkpoint not found: $Checkpoint"
    exit 1
}
if (-not (Test-Path $Prompts)) {
    Write-Error "Prompts file not found: $Prompts"
    exit 1
}

# ── Load .env ──────────────────────────────────────────────────────────
$EnvFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Error "No .env at $EnvFile (need OPENAI_API_KEY for grading)"
    exit 1
}

$loadedKeys = @()
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) { return }
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$') {
        $key   = $Matches[1]
        $value = $Matches[2].Trim('"').Trim("'")
        # Whitelist — don't pollute env with unrelated repo-local vars.
        if ($key -in @("OPENAI_API_KEY", "ANTHROPIC_API_KEY")) {
            Set-Item -Path "Env:$key" -Value $value
            $loadedKeys += $key
        }
    }
}

if ($loadedKeys.Count -eq 0 -and -not $DryRun) {
    Write-Error "No API key found in .env (need OPENAI_API_KEY or ANTHROPIC_API_KEY). Use -DryRun to skip the LLM call."
    exit 1
}

# ── Banner ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "===== Phase 1.4 Coherence Grading =====" -ForegroundColor Cyan
Write-Host "  repo       : $RepoRoot"
Write-Host "  checkpoint : $Checkpoint"
Write-Host "  prompts    : $Prompts"
Write-Host "  judge      : $(if ($DryRun) {'(dry-run, no API call)'} else {$Judge})"
Write-Host "  device     : $Device"
Write-Host "  max tokens : $MaxTokens"
Write-Host "  output     : $Out"
if ($loadedKeys.Count -gt 0) {
    Write-Host "  env loaded : $($loadedKeys -join ', ')" -ForegroundColor DarkGray
}
Write-Host ""

# ── Run ────────────────────────────────────────────────────────────────
$pyArgs = @(
    "run", "python", "-u",
    "sandbox/mingru_baseline/eval_coherence.py",
    "--checkpoint",     $Checkpoint,
    "--prompts",        $Prompts,
    "--judge",          $Judge,
    "--max-new-tokens", "$MaxTokens",
    "--out",            $Out,
    "--device",         $Device
)
if ($DryRun) {
    $pyArgs += "--skip-grade"
}
if ($Tokenizer) {
    $pyArgs += @("--tokenizer", $Tokenizer)
}

# Force unbuffered Python so streaming output appears live.
$Env:PYTHONUNBUFFERED = "1"

Write-Host "uv $($pyArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ""
& uv @pyArgs
$exit = $LASTEXITCODE

if ($exit -ne 0) {
    Write-Host ""
    Write-Error "eval_coherence.py exited with code $exit"
    exit $exit
}

Write-Host ""
Write-Host "===== Done — results: $Out =====" -ForegroundColor Green
