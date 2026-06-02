<#
.SYNOPSIS
    Shared helpers for the tinybit GCP PowerShell scripts.

.DESCRIPTION
    Dot-sourced by gcp_*.ps1. Resolves the repository root and, if present,
    dot-sources a local `.tinybit.env.ps1` so you can keep project/bucket
    settings out of your shell profile, e.g.:

        # .tinybit.env.ps1  (git-ignored)
        $env:GCP_PROJECT = "my-project"
        $env:GCP_BUCKET  = "gs://my-bucket"

    This mirrors the `.tinybit.env` that the bash scripts source. Also provides
    Require-Cmd / Require-Env helpers.
#>

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$envFile = Join-Path $script:RepoRoot ".tinybit.env.ps1"
if (Test-Path $envFile) {
    . $envFile
}

function Get-RepoRoot { return $script:RepoRoot }

function Require-Cmd {
    param([Parameter(Mandatory)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "$Name not installed or not on PATH"
        exit 1
    }
}

function Require-Env {
    param([Parameter(Mandatory)][string]$Name)
    $val = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($val)) {
        Write-Error "$Name not set (set `$env:$Name or create .tinybit.env.ps1)"
        exit 1
    }
    return $val
}

# Strip a trailing slash from a gs:// bucket URL.
function Trim-Bucket {
    param([Parameter(Mandatory)][string]$Bucket)
    return $Bucket.TrimEnd('/')
}
