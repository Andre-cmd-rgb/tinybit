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

# Native tools (gcloud/gsutil/git/cargo/python) write normal output to stderr;
# under 'Stop', Windows PowerShell 5.1 turns that into a spurious terminating
# NativeCommandError even on success. Use 'Continue' and judge success by
# $LASTEXITCODE (which every native call below already checks).
$ErrorActionPreference = "Continue"

Write-Host "Running cargo tests..."
cargo test --workspace
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done."
