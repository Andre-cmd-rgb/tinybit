<#
.SYNOPSIS
    Stop (but don't delete) a tinybit training VM (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/gcp_stop_vm.sh. Useful if you want to resume
    later from the same disk. Asks for confirmation unless -Force (or $env:FORCE=1).

.PARAMETER InstanceName
    The VM instance name.
.PARAMETER Zone
    The VM zone.
.PARAMETER Force
    Skip the confirmation prompt.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$InstanceName,
    [Parameter(Mandatory)][string]$Zone,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_tinybit_env.ps1")

$GcpProject = Require-Env "GCP_PROJECT"
if ($env:FORCE -eq "1") { $Force = $true }

Write-Host "About to stop instance: $InstanceName in $Zone  (project: $GcpProject)"
if (-not $Force) {
    $ans = Read-Host "Type 'stop' to confirm"
    if ($ans -ne "stop") { Write-Host "aborted"; exit 1 }
}

& gcloud compute instances stop $InstanceName --zone=$Zone --project=$GcpProject
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Stopped $InstanceName."
