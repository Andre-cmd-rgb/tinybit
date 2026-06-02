<#
.SYNOPSIS
    Continuously fetch and tail the bootstrap/training log for a run (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/gcp_tail_logs.sh. Polls the run's log object in
    the bucket every 10 seconds and prints newly-appended bytes. Ctrl-C to stop.

.PARAMETER RunId
    Run id. If omitted, uses gs://$GCP_BUCKET/latest_run.txt.
.PARAMETER Which
    'training' (default) or 'bootstrap'.
#>
[CmdletBinding()]
param(
    [string]$RunId,
    [ValidateSet("training", "bootstrap")]
    [string]$Which = "training"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_tinybit_env.ps1")

$GcpBucket = Trim-Bucket (Require-Env "GCP_BUCKET")

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = (& gsutil -q cat "$GcpBucket/latest_run.txt" 2>$null)
    if ($RunId) { $RunId = $RunId.Trim() }
    if ([string]::IsNullOrWhiteSpace($RunId)) {
        Write-Error "No RUN_ID and latest_run.txt empty"
        exit 1
    }
}

$file = if ($Which -eq "bootstrap") { "logs/bootstrap.log" } else { "logs/training.log" }
$prefix = "$GcpBucket/runs/$RunId"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "_tinybit-tail-$RunId-$Which.log"

Write-Host "Tailing $prefix/$file (Ctrl-C to stop)"
$lastBytes = 0
while ($true) {
    $tmpNew = "$tmp.new"
    & gsutil -q cp "$prefix/$file" $tmpNew 2>$null
    if ($LASTEXITCODE -eq 0 -and (Test-Path $tmpNew)) {
        $newBytes = (Get-Item $tmpNew).Length
        if ($newBytes -gt $lastBytes) {
            $fs = [System.IO.File]::OpenRead($tmpNew)
            try {
                $fs.Seek($lastBytes, [System.IO.SeekOrigin]::Begin) | Out-Null
                $buf = New-Object byte[] ($newBytes - $lastBytes)
                $read = $fs.Read($buf, 0, $buf.Length)
                [Console]::Out.Write([System.Text.Encoding]::UTF8.GetString($buf, 0, $read))
            } finally {
                $fs.Close()
            }
            $lastBytes = $newBytes
        }
        Move-Item -Force $tmpNew $tmp
    }
    Start-Sleep -Seconds 10
}
