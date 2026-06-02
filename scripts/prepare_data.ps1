<#
.SYNOPSIS
    tinybit data preparation (Windows / PowerShell).

.DESCRIPTION
    Downloads and tokenizes datasets for training tinybit. This is the Windows
    counterpart of scripts/prepare_data.sh — both are thin wrappers that invoke
    the shared scripts/prepare_data.py with the same environment variables.

    Configuration is via environment variables (set them before calling, e.g.
    `$env:TOTAL_TOKENS = "1500000000"`):
        TOTAL_TOKENS   total desired tokens (default 500_000_000)
        MIN_TOKENS     fail if fewer than this are collected (default 75% of TOTAL_TOKENS)
        SEQ_LEN        for val-set sizing (default 1024)
        DATA_PROFILE   general | coding   (default: general; alias: PROFILE)
        HF_TOKEN       optional — enables gated datasets (the-stack-smol, etc.)
        ENABLE_GATED   1 to attempt gated datasets when HF_TOKEN is set (default 1)
        CUSTOM_CHAT_DIR     dir of your own {"messages":[...]} JSONL/TXT files (default: datasets/)
        CUSTOM_CHAT_EPOCHS  times to repeat each custom file (default 10; 0 disables)

.PARAMETER OutputDir
    Output directory for train.bin / val.bin (default: data).

.EXAMPLE
    $env:DATA_PROFILE = "general"; $env:TOTAL_TOKENS = "1500000000"
    .\scripts\prepare_data.ps1 data
#>
[CmdletBinding()]
param(
    [string]$OutputDir = "data"
)

$ErrorActionPreference = "Stop"

$OutputDir = $OutputDir.TrimEnd('/', '\')
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Preparing data in $OutputDir ..."

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Prefer `python`, fall back to `python3` (matches the .sh behaviour).
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $python) {
    Write-Error "Python not found on PATH. Install Python 3 and required packages: pip install datasets tokenizers tqdm numpy"
    exit 1
}

& $python.Source (Join-Path $scriptDir "prepare_data.py") $OutputDir
exit $LASTEXITCODE
