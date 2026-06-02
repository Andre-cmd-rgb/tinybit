<#
.SYNOPSIS
    Force an immediate checkpoint/log sync from a live VM (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/gcp_sync_now.sh. SSHes into the VM and pushes
    the current status, logs, and checkpoints to the run's bucket prefix.

.PARAMETER InstanceName
    The VM instance name.
.PARAMETER Zone
    The VM zone.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$InstanceName,
    [Parameter(Mandatory)][string]$Zone
)

# Native tools (gcloud/gsutil/git/cargo/python) write normal output to stderr;
# under 'Stop', Windows PowerShell 5.1 turns that into a spurious terminating
# NativeCommandError even on success. Use 'Continue' and judge success by
# $LASTEXITCODE (which every native call below already checks).
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "_tinybit_env.ps1")

$GcpProject = Require-Env "GCP_PROJECT"
$GcpBucket  = Trim-Bucket (Require-Env "GCP_BUCKET")

# Remote command runs in bash on the (Linux) VM — keep it as a single bash -c.
$remote = @"
set -uo pipefail
RUN_ID="`$(awk -F\" '/"run_id":/ {print `$4; exit}' /var/log/tinybit-status.json 2>/dev/null)"
if [ -z "`$RUN_ID" ]; then echo "no status.json on VM"; exit 1; fi
PFX="$GcpBucket/runs/`$RUN_ID"
cd /workspace/tinybit 2>/dev/null || cd /root
gsutil -q cp /var/log/tinybit-status.json    "`$PFX/status.json"        || true
gsutil -q cp /var/log/tinybit-bootstrap.log  "`$PFX/logs/bootstrap.log" || true
gsutil -q cp /var/log/tinybit-training.log   "`$PFX/logs/training.log"  || true
if [ -d checkpoints ]; then
  gsutil -m -q rsync -r checkpoints/ "`$PFX/checkpoints/" || true
fi
echo "synced to `$PFX"
"@

Write-Host "Triggering sync on $InstanceName ($Zone)..."
& gcloud compute ssh $InstanceName --zone=$Zone --project=$GcpProject --command=$remote
exit $LASTEXITCODE
