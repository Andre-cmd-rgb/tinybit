<#
.SYNOPSIS
    Delete a tinybit training VM and its boot disk (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/gcp_delete_vm.sh. Asks for confirmation
    (type the instance name) unless -Force (or $env:FORCE=1). Always free the GPU
    when you're done debugging.

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

Write-Host "About to DELETE instance: $InstanceName in $Zone  (project: $GcpProject)"
Write-Host "This will release the GPU and destroy the boot disk."
if (-not $Force) {
    $ans = Read-Host "Type the instance name to confirm"
    if ($ans -ne $InstanceName) { Write-Host "aborted (you typed: $ans)"; exit 1 }
}

& gcloud compute instances delete $InstanceName --zone=$Zone --project=$GcpProject --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Deleted $InstanceName."
