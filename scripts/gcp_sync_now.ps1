<#
.SYNOPSIS
    Force an immediate checkpoint/log sync from a live VM (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/gcp_sync_now.sh. Pushes the VM's current
    status, logs, and checkpoints to the run's bucket prefix.

    NOTE: `gcloud compute ssh` HANGS on Windows (it buffers all output and, on a
    first run, blocks on an interactive key-passphrase prompt), so this script
    does NOT use it. Instead it drives the native ssh.exe directly:
      1. ensures a passphrase-less key at ~/.ssh/google_compute_engine,
      2. installs its public half into the VM's ssh-keys metadata (OS Login off),
      3. connects to the VM's external IP and runs the sync as root.
    The remote script is base64-encoded to avoid cross-shell quoting issues.

.PARAMETER InstanceName
    The VM instance name.
.PARAMETER Zone
    The VM zone.
.PARAMETER User
    Linux username to create/connect as (default: the current Windows user).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$InstanceName,
    [Parameter(Mandatory)][string]$Zone,
    [string]$User = $env:USERNAME
)

# Native tools (gcloud/gsutil/git/ssh) write normal output to stderr; under
# 'Stop', Windows PowerShell 5.1 turns that into a spurious terminating
# NativeCommandError even on success. Use 'Continue' and judge success by
# $LASTEXITCODE.
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "_tinybit_env.ps1")

$GcpProject = Require-Env "GCP_PROJECT"
$GcpBucket  = Trim-Bucket (Require-Env "GCP_BUCKET")
Require-Cmd "gcloud"
Require-Cmd "ssh"

# ---- 1. ensure a passphrase-less SSH key -------------------------------------
$keyPath = Join-Path $env:USERPROFILE ".ssh\google_compute_engine"
$sshDir  = Split-Path $keyPath
if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Force -Path $sshDir | Out-Null }
if (-not (Test-Path $keyPath)) {
    Write-Host "Creating SSH key $keyPath ..."
    # PowerShell 5.1 swallows an empty -N "", so run ssh-keygen via cmd.
    & cmd /c "ssh-keygen -t rsa -b 2048 -f `"$keyPath`" -N `"`" -C $User -q"
    if (-not (Test-Path $keyPath)) { Write-Error "[FATAL] ssh-keygen failed"; exit 1 }
}
$pub = ((Get-Content "$keyPath.pub" -Raw) -replace "\r?\n", "").Trim()

# ---- 2. install the pubkey into the VM's ssh-keys metadata --------------------
# add-metadata replaces the whole ssh-keys value, so merge: keep other users'
# keys, drop any stale entry for our user, then append ours.
$existing = (& gcloud compute instances describe $InstanceName --zone=$Zone --project=$GcpProject `
    --format="value(metadata.items.filter(key:ssh-keys).extract(value).flatten())" 2>$null) -join "`n"
if (-not ($existing -and $existing.Contains($pub))) {
    Write-Host "Installing SSH key into $InstanceName metadata ..."
    $userPrefix = "${User}:"
    $lines = @()
    if ($existing) {
        $lines += ($existing -split "`n" | Where-Object { $_.Trim() -ne "" -and -not $_.StartsWith($userPrefix) })
    }
    $lines += "${User}:$pub"
    $mf = New-TemporaryFile
    [System.IO.File]::WriteAllText($mf.FullName, (($lines -join "`n") + "`n"), (New-Object System.Text.UTF8Encoding($false)))
    & gcloud compute instances add-metadata $InstanceName --zone=$Zone --project=$GcpProject `
        --metadata-from-file="ssh-keys=$($mf.FullName)"
    $rc = $LASTEXITCODE
    Remove-Item $mf.FullName -Force
    if ($rc -ne 0) { Write-Error "[FATAL] failed to set ssh-keys metadata"; exit 1 }
    Start-Sleep -Seconds 5   # let the guest agent apply the new key
}

# ---- 3. external IP ----------------------------------------------------------
$ip = (& gcloud compute instances describe $InstanceName --zone=$Zone --project=$GcpProject `
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>$null)
if ($ip) { $ip = ([string]$ip).Trim() }
if ([string]::IsNullOrWhiteSpace($ip)) { Write-Error "[FATAL] no external IP for $InstanceName"; exit 1 }

# ---- 4. remote sync (run as root; base64 to dodge cross-shell quoting) --------
$remote = @"
set -uo pipefail
export PATH=/snap/bin:/usr/local/bin:/usr/bin:/bin:`$PATH
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
$b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($remote))

Write-Host "Triggering sync on $InstanceName ($ip) ..."
$sshArgs = @(
    "-i", $keyPath,
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=NUL",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "$User@$ip",
    "echo $b64 | base64 -d | sudo bash"
)
& ssh.exe @sshArgs
exit $LASTEXITCODE
