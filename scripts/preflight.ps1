<#
.SYNOPSIS
    Local preflight checks before paying for a GCP GPU VM (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/preflight.sh. Verifies gcloud/gsutil/cargo/
    rustc/python, GCP_PROJECT/GCP_BUCKET reachability, repo state, configs, the
    cloud startup template, scripts, tokenizer, and a cargo check.

.PARAMETER ModelSize
    micro | bit | qbit (default: micro).
#>
[CmdletBinding()]
param(
    [string]$ModelSize = "micro"
)

. (Join-Path $PSScriptRoot "_tinybit_env.ps1")
$RepoRoot = Get-RepoRoot

$script:Pass = 0
$script:Warn = 0
$script:Fail = 0

function Ok($m)   { Write-Host ("  [ok]   {0}" -f $m); $script:Pass++ }
function Warn2($m){ Write-Host ("  [warn] {0}" -f $m); $script:Warn++ }
function Err($m)  { Write-Host ("  [FAIL] {0}" -f $m); $script:Fail++ }
function Step($m) { Write-Host ""; Write-Host ("== {0} ==" -f $m) }

function Has($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

Step "tools"
if (Has "gcloud")  { Ok ("gcloud installed: " + ((& gcloud --version 2>$null) | Select-Object -First 1)) } else { Err "gcloud missing" }
if (Has "gsutil")  { Ok "gsutil installed" } else { Err "gsutil missing" }
if (Has "cargo")   { Ok ("cargo  installed: " + (& cargo --version)) } else { Err "cargo missing" }
if (Has "rustc")   { Ok ("rustc  installed: " + (& rustc --version)) } else { Err "rustc missing" }
$py = if (Has "python") { "python" } elseif (Has "python3") { "python3" } else { $null }
if ($py) { Ok ("python installed: " + (& $py --version 2>&1)) } else { Err "python missing" }

Step "env"
if ($env:GCP_PROJECT) { Ok "GCP_PROJECT=$($env:GCP_PROJECT)" } else { Err "GCP_PROJECT not set" }
if ($env:GCP_BUCKET)  { Ok "GCP_BUCKET=$($env:GCP_BUCKET)" }  else { Err "GCP_BUCKET not set" }

Step "gcloud auth"
if (Has "gcloud") {
    $act = (& gcloud config get-value account 2>$null)
    if ($act -and $act -ne "(unset)") { Ok "active account: $act" } else { Err "no active gcloud account (run: gcloud auth login)" }
}

Step "bucket access"
if ($env:GCP_BUCKET -and (Has "gsutil")) {
    $b = $env:GCP_BUCKET.TrimEnd('/')
    & gsutil ls "$b/" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Ok "$($env:GCP_BUCKET) reachable"
        $probe = "$b/_preflight_probe.txt"
        "preflight" | & gsutil -q cp - $probe 2>$null
        if ($LASTEXITCODE -eq 0) {
            & gsutil -q rm $probe 2>$null
            Ok "bucket writable"
        } else {
            Err "bucket not writable"
        }
    } else {
        Err "cannot list $($env:GCP_BUCKET)"
    }
}

Step "repo state"
if (Test-Path (Join-Path $RepoRoot "VERSION")) {
    Ok ("VERSION: " + ((Get-Content (Join-Path $RepoRoot "VERSION")) -join ""))
} else {
    Warn2 "VERSION file missing"
}
if (Has "git") {
    & git -C $RepoRoot rev-parse HEAD 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Ok ("git commit: " + (& git -C $RepoRoot rev-parse --short HEAD))
        & git -C $RepoRoot diff --quiet HEAD 2>$null
        if ($LASTEXITCODE -ne 0) { Warn2 "git working tree is dirty (set `$env:FORCE=1 to launch anyway)" } else { Ok "git working tree clean" }
    } else {
        Warn2 "not a git repo (or no commits)"
    }
}

Step "configs"
if (Test-Path (Join-Path $RepoRoot "configs/$ModelSize.toml")) {
    Ok "configs/$ModelSize.toml present"
} else {
    Err "configs/$ModelSize.toml not found"
}

Step "cloud startup template"
$template = Join-Path $RepoRoot "scripts/cloud/startup.sh"
if (Test-Path $template) {
    Ok "template present"
    $placeholders = [regex]::Matches((Get-Content $template -Raw), '__[A-Z_]+__') | ForEach-Object { $_.Value } | Sort-Object -Unique
    if ($placeholders) {
        Ok ("found placeholders: " + ($placeholders -join ' '))
    } else {
        Err "template has no placeholders — looks corrupted"
    }
} else {
    Err "missing $template"
}

Step "scripts"
foreach ($f in @("gcp_launch.sh","prepare_data.sh","gcp_status.sh","gcp_tail_logs.sh","gcp_sync_now.sh","gcp_stop_vm.sh","gcp_delete_vm.sh","prepare_data.py")) {
    if (Test-Path (Join-Path $RepoRoot "scripts/$f")) { Ok "scripts/$f present" } else { Err "scripts/$f missing" }
}

Step "tokenizer"
if (Test-Path (Join-Path $RepoRoot "tokenizer.json")) {
    Ok "tokenizer.json present locally"
} else {
    Warn2 "tokenizer.json missing locally — VM will download from HuggingFace"
}

Step "python datasets package (data-prep precheck)"
if ($py) {
    & $py -c "import datasets, tokenizers, tqdm" 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "datasets/tokenizers/tqdm importable locally" } else { Warn2 "datasets/tokenizers/tqdm not installed locally (VM installs its own copy)" }
}

Step "build sanity (cargo check)"
Push-Location $RepoRoot
try {
    & cargo check --workspace --offline 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Ok "cargo check --offline passes"
    } else {
        & cargo check --workspace 2>&1 | Select-Object -Last 5 | Out-File -FilePath (Join-Path ([System.IO.Path]::GetTempPath()) "preflight-check.log")
        if ($LASTEXITCODE -eq 0) { Ok "cargo check passes" } else { Err "cargo check failed — see preflight-check.log" }
    }
} finally {
    Pop-Location
}

Step "result"
Write-Host ("  passes: {0}   warns: {1}   fails: {2}" -f $script:Pass, $script:Warn, $script:Fail)
if ($script:Fail -gt 0) {
    Write-Host "FAILED — fix the [FAIL] items before launching a GPU VM."
    exit 1
}
Write-Host "OK — ready to launch."
