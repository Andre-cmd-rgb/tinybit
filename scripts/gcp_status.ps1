<#
.SYNOPSIS
    Show status of a tinybit GCP training run (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/gcp_status.sh. Reads
    gs://$GCP_BUCKET/runs/<RUN_ID>/status.json and lists checkpoints and the tail
    of training.log. Also shows any matching live VMs.

.PARAMETER RunId
    Run id. If omitted, uses gs://$GCP_BUCKET/latest_run.txt.

.EXAMPLE
    .\scripts\gcp_status.ps1
    .\scripts\gcp_status.ps1 20260527-141500-micro
#>
[CmdletBinding()]
param(
    [string]$RunId
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_tinybit_env.ps1")

$GcpBucket  = Trim-Bucket (Require-Env "GCP_BUCKET")
$GcpProject = Require-Env "GCP_PROJECT"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = (& gsutil -q cat "$GcpBucket/latest_run.txt" 2>$null)
    if ($RunId) { $RunId = $RunId.Trim() }
    if ([string]::IsNullOrWhiteSpace($RunId)) {
        Write-Error "No RUN_ID given and gs://.../latest_run.txt is empty."
        exit 1
    }
}

$prefix = "$GcpBucket/runs/$RunId"
Write-Host "RUN_ID  : $RunId"
Write-Host "bucket  : $prefix"
Write-Host ""

Write-Host "== status.json =="
$status = & gsutil -q cat "$prefix/status.json" 2>$null
if ($status) { $status } else { Write-Host "(no status.json yet)" }
Write-Host ""

Write-Host "== launch.json =="
$launch = & gsutil -q cat "$prefix/launch.json" 2>$null
if ($launch) { $launch } else { Write-Host "(no launch.json)" }
Write-Host ""

foreach ($marker in @("DONE", "FAILED")) {
    & gsutil -q stat "$prefix/$marker.json" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "== $marker.json =="
        & gsutil -q cat "$prefix/$marker.json"
        Write-Host ""
    }
}

Write-Host "== checkpoints =="
$ckpts = & gsutil -q ls "$prefix/checkpoints/" 2>$null
if ($ckpts) { $ckpts | Select-Object -Last 20 } else { Write-Host "(none)" }
Write-Host ""

Write-Host "== training.log (tail 30) =="
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "_tinybit_tail.log"
& gsutil -q cp "$prefix/logs/training.log" $tmp 2>$null
if ($LASTEXITCODE -eq 0 -and (Test-Path $tmp)) {
    Get-Content $tmp -Tail 30
} else {
    Write-Host "(no training.log yet)"
}
Write-Host ""

Write-Host "== live VMs matching tinybit =="
& gcloud compute instances list --project="$GcpProject" `
    --filter="name~^tinybit-" `
    --format='table(name,zone,status,machineType.basename(),scheduling.provisioningModel)' 2>$null
