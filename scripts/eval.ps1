<#
.SYNOPSIS
    Run the tinybit test/evaluation suite (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/eval.sh. Runs the cargo test suite.

.PARAMETER ModelPath
    Optional model checkpoint path (informational; default models/tinybit-micro.safetensors).
.PARAMETER ConfigPath
    Optional model config path (informational; default configs/micro.toml).
#>
[CmdletBinding()]
param(
    [string]$ModelPath  = "models/tinybit-micro.safetensors",
    [string]$ConfigPath = "configs/micro.toml"
)

$ErrorActionPreference = "Stop"

Write-Host "Running cargo tests..."
cargo test --workspace
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done."
