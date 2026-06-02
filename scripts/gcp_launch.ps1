<#
.SYNOPSIS
    tinybit GCP launcher — L4 only (Windows / PowerShell).

.DESCRIPTION
    Windows counterpart of scripts/gcp_launch.sh. Tries every zone in order until
    ONE L4 VM is created, then stops. Generates a per-run RUN_ID, uploads the
    local repo to gs://$GCP_BUCKET/$GCS_REPO_PREFIX, renders cloud/startup.sh with
    substitutions, and attaches it as VM metadata.

    Required env:
        GCP_PROJECT     Google Cloud project id
        GCP_BUCKET      gs://your-bucket   (no trailing slash)

    Optional env (all have defaults) — same names/semantics as gcp_launch.sh:
        PROVISIONING_MODEL  STANDARD | SPOT | comma list (default: STANDARD)
        DATA_TOKENS         desired tokens      (default: 20000000)
        MIN_TOKENS          min acceptable      (default: 75% of DATA_TOKENS)
        TRAIN_STEPS         training steps      (default: 2000)
        CUDA_VERSION        apt suffix          (default: 12-8)
        CUDA_DIR            toolkit prefix      (default: /usr/local/cuda-12.8)
        GCP_ZONES           override zone list  (space-separated)
        SYNC_INTERVAL       bucket sync seconds (default: 120)
        KEEP_VM_ON_FAILURE  0|1                 (default: 0)
        HF_TOKEN            HuggingFace token   (optional)
        RUN_ID              override run id     (default: auto)
        GCS_REPO_PREFIX     bucket prefix       (default: tinybit)
        SKIP_UPLOAD         1 to skip rsync     (default: 0)
        FORCE               1 to bypass guards  (default: 0)
        RESET_RUN           0|1                 (default: 0)
        CUSTOM_CHAT_EPOCHS  identity/tools repeats (default: 10)
        TRAIN_CONFIG        path inside the repo to a training config TOML

.PARAMETER ModelSize
    micro | bit | qbit | micro-coding | bit-coding | qbit-coding (default: micro).

.EXAMPLE
    $env:GCP_PROJECT = "my-proj"; $env:GCP_BUCKET = "gs://my-bucket"
    $env:DATA_TOKENS = "1500000000"; $env:TRAIN_CONFIG = "configs/train-micro-l4.toml"
    .\scripts\gcp_launch.ps1 micro
#>
[CmdletBinding()]
param(
    [string]$ModelSize = "micro"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_tinybit_env.ps1")

$RepoRoot = Get-RepoRoot

function Env-Or([string]$name, [string]$default) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrEmpty($v)) { return $default } else { return $v }
}

# ---------- script version & git info -----------------------------------------
$ScriptVersion = (Get-Content (Join-Path $RepoRoot "VERSION") -ErrorAction SilentlyContinue) -join ""
if (-not $ScriptVersion) { $ScriptVersion = "unknown" }
$GitCommit = (& git -C $RepoRoot rev-parse --short HEAD 2>$null)
if (-not $GitCommit) { $GitCommit = "unknown" }
& git -C $RepoRoot diff --quiet HEAD 2>$null
$GitDirty = if ($LASTEXITCODE -ne 0) { " (dirty)" } else { "" }

$GcpProject = Require-Env "GCP_PROJECT"
$GcpBucket  = Trim-Bucket (Require-Env "GCP_BUCKET")

$Provisioning      = Env-Or "PROVISIONING_MODEL" "STANDARD"
$DataTokens        = Env-Or "DATA_TOKENS" "20000000"
$TrainSteps        = Env-Or "TRAIN_STEPS" "2000"
$CudaVersion       = Env-Or "CUDA_VERSION" "12-8"
$CudaDir           = Env-Or "CUDA_DIR" "/usr/local/cuda-12.8"
$SyncInterval      = Env-Or "SYNC_INTERVAL" "120"
$KeepVmOnFailure   = Env-Or "KEEP_VM_ON_FAILURE" "0"
$HfTokenVal        = Env-Or "HF_TOKEN" ""
$GcsRepoPrefix     = Env-Or "GCS_REPO_PREFIX" "tinybit"
$SkipUpload        = Env-Or "SKIP_UPLOAD" "0"
$Force             = Env-Or "FORCE" "0"
$TrainConfig       = Env-Or "TRAIN_CONFIG" ""
$ResetRun          = Env-Or "RESET_RUN" "0"
$CustomChatEpochs  = Env-Or "CUSTOM_CHAT_EPOCHS" "10"

if ($TrainConfig) {
    if (-not (Test-Path (Join-Path $RepoRoot $TrainConfig))) {
        Write-Error "[FATAL] TRAIN_CONFIG=$TrainConfig not found in repo"
        exit 1
    }
}

$MinTokens = Env-Or "MIN_TOKENS" ""
if (-not $MinTokens) {
    $MinTokens = [string]([int64]$DataTokens * 3 / 4)
}

$RunId = Env-Or "RUN_ID" ""
if (-not $RunId) {
    $RunId = "{0}-{1}" -f ([DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")), $ModelSize
}

# Default zone list — EU first, then US, then Asia.
$DefaultZones = @(
    "europe-west4-a","europe-west4-b","europe-west4-c",
    "europe-west1-b","europe-west1-c",
    "europe-west3-a","europe-west3-b",
    "europe-west2-a","europe-west2-b","europe-west2-c",
    "europe-north1-a","europe-north1-b","europe-north1-c",
    "us-central1-a","us-central1-b","us-central1-c","us-central1-f",
    "us-west1-a","us-west1-b","us-west1-c","us-west4-a","us-west4-c",
    "us-east1-b","us-east1-c","us-east1-d",
    "us-east4-a","us-east4-b","us-east4-c",
    "us-east5-a","us-east5-b","us-east5-c",
    "us-south1-a","us-south1-b",
    "asia-east1-a","asia-east1-b",
    "asia-southeast1-a","asia-southeast1-b","asia-southeast1-c"
)
$zonesEnv = Env-Or "GCP_ZONES" ""
if ($zonesEnv) {
    $Zones = $zonesEnv -split '\s+' | Where-Object { $_ }
} else {
    $Zones = $DefaultZones
}

# Hardware: L4 only.
$ProfileId       = "l4"
$MachineType     = "g2-standard-4"
$AcceleratorType = "nvidia-l4"
$DiskType        = "pd-ssd"
$CostHint        = "~0.71"
$ImageFamily     = "common-cu129-ubuntu-2204-nvidia-580"
$ImageProject    = "deeplearning-platform-release"
$BootDiskSize    = "200GB"

# ---------- banner ------------------------------------------------------------
@(
    "============================================================",
    " tinybit GCP launcher  v$ScriptVersion  commit=$GitCommit$GitDirty",
    " run_id        : $RunId",
    " model         : $ModelSize",
    " hardware      : L4 ($MachineType + $AcceleratorType, ~$CostHint/hr on-demand)",
    " provisioning  : $Provisioning",
    " data tokens   : $DataTokens  (min $MinTokens)",
    " train steps   : $TrainSteps",
    " cuda          : $CudaDir  (apt cuda-toolkit-$CudaVersion)",
    " project       : $GcpProject",
    " bucket        : $GcpBucket",
    " repo prefix   : $GcsRepoPrefix",
    " sync interval : ${SyncInterval}s",
    " HF token      : $(if ($HfTokenVal) { 'set' } else { 'unset' })",
    " train config  : $(if ($TrainConfig) { $TrainConfig } else { '<inline default>' })",
    " zones (count) : $($Zones.Count)",
    "============================================================"
) | ForEach-Object { Write-Host $_ }

if ($GitDirty -and $Force -ne "1") {
    Write-Host "[warn] git working tree is dirty. Uncommitted code will not match what's uploaded."
    Write-Host "       set `$env:FORCE=1 to launch anyway, or commit/stash first."
    exit 1
}

# Validate PROVISIONING_MODEL up-front.
$provList = $Provisioning -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
foreach ($p in $provList) {
    if ($p -ne "STANDARD" -and $p -ne "SPOT") {
        Write-Error "[FATAL] PROVISIONING_MODEL contains unknown entry '$p' (allowed: STANDARD, SPOT)"
        exit 1
    }
}

# ---------- preflight ---------------------------------------------------------
Require-Cmd "gcloud"
Require-Cmd "gsutil"

$activeAcct = (& gcloud config get-value account 2>$null)
if (-not $activeAcct -or $activeAcct -eq "(unset)") {
    Write-Error "[FATAL] gcloud has no active account. Run: gcloud auth login"
    exit 1
}

& gsutil ls "$GcpBucket/" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "[FATAL] cannot list $GcpBucket — does it exist and do you have access?"
    exit 1
}
$probe = "$GcpBucket/_launcher_probe_$RunId.txt"
"probe $RunId" | & gsutil -q cp - $probe
if ($LASTEXITCODE -ne 0) { Write-Error "[FATAL] cannot write to bucket"; exit 1 }
& gsutil -q rm $probe 2>$null
Write-Host "[ok] bucket reachable and writable"

# ---------- upload repo -------------------------------------------------------
if ($SkipUpload -eq "1") {
    Write-Host "[info] SKIP_UPLOAD=1 — assuming $GcpBucket/$GcsRepoPrefix is current"
} else {
    Write-Host "Uploading repo to $GcpBucket/$GcsRepoPrefix (excluding target/, data/, checkpoints*/) ..."
    $excl = '(^|.*/)target/.*$|(^|.*/)data/.*\.bin$|(^|.*/)checkpoints.*/.*$|^\.git/.*$|.*\.safetensors$'
    & gsutil -m -q rsync -r -d -x $excl "$RepoRoot/" "$GcpBucket/$GcsRepoPrefix/"
    if ($LASTEXITCODE -ne 0) { Write-Error "[FATAL] repo upload failed"; exit 1 }
    Write-Host "[ok] repo uploaded"
}

# ---------- render startup script ---------------------------------------------
$template = Join-Path $PSScriptRoot "cloud/startup.sh"
if (-not (Test-Path $template)) { Write-Error "[FATAL] missing cloud startup template: $template"; exit 1 }
$templateText = Get-Content $template -Raw

function Render-Startup([string]$zone) {
    $subs = @{
        "__RUN_ID__"             = $RunId
        "__MODEL_SIZE__"         = $ModelSize
        "__GCS_BUCKET__"         = $GcpBucket
        "__GCS_REPO_PREFIX__"    = $GcsRepoPrefix
        "__DATA_TOKENS__"        = $DataTokens
        "__MIN_TOKENS__"         = $MinTokens
        "__TRAIN_STEPS__"        = $TrainSteps
        "__CUDA_VERSION__"       = $CudaVersion
        "__CUDA_DIR__"           = $CudaDir
        "__KEEP_VM_ON_FAILURE__" = $KeepVmOnFailure
        "__SYNC_INTERVAL__"      = $SyncInterval
        "__HF_TOKEN__"           = $HfTokenVal
        "__SCRIPT_VERSION__"     = $ScriptVersion
        "__TRAIN_CONFIG__"       = $TrainConfig
        "__RESET_RUN__"          = $ResetRun
        "__CUSTOM_CHAT_EPOCHS__" = $CustomChatEpochs
        "__ZONE__"               = $zone
        "__MACHINE__"            = $MachineType
        "__ACCELERATOR__"        = $AcceleratorType
    }
    $text = $templateText
    foreach ($k in $subs.Keys) {
        $text = $text.Replace($k, $subs[$k])
    }
    # The startup script runs on Linux — it MUST have LF line endings.
    $text = $text -replace "`r`n", "`n"
    $tmp = New-TemporaryFile
    # Write without a BOM and with LF endings.
    [System.IO.File]::WriteAllText($tmp.FullName, $text, (New-Object System.Text.UTF8Encoding($false)))
    return $tmp.FullName
}

# ---------- try (provisioning × zone) combinations ----------------------------
$success = $false
$instanceName = ""
$selectedZone = ""
$selectedProvisioning = ""
$startupRendered = ""

foreach ($prov in $provList) {
    foreach ($zone in $Zones) {
        $instanceName = "tinybit-$ProfileId-" + ([DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss"))
        Write-Host ""
        Write-Host "------------------------------------------------------------"
        Write-Host " attempt: prov=$prov zone=$zone"
        Write-Host "          machine=$MachineType accel=$AcceleratorType  on-demand $CostHint/hr"
        Write-Host " instance: $instanceName"
        Write-Host "------------------------------------------------------------"

        if ($startupRendered -and (Test-Path $startupRendered)) { Remove-Item -Force $startupRendered -ErrorAction SilentlyContinue }
        $startupRendered = Render-Startup $zone

        $flags = @(
            "--maintenance-policy=TERMINATE",
            "--accelerator=type=$AcceleratorType,count=1"
        )
        if ($prov -eq "SPOT") {
            $flags += "--provisioning-model=SPOT"
            $flags += "--instance-termination-action=STOP"
        }

        $metadata = "run-id=$RunId,tinybit-script-version=$ScriptVersion,tinybit-model=$ModelSize,tinybit-profile=$ProfileId,tinybit-provisioning=$prov"

        & gcloud compute instances create $instanceName `
            --project=$GcpProject `
            --zone=$zone `
            --machine-type=$MachineType `
            --image-family=$ImageFamily `
            --image-project=$ImageProject `
            --boot-disk-size=$BootDiskSize `
            --boot-disk-type=$DiskType `
            --scopes=storage-full `
            --metadata=$metadata `
            --metadata-from-file=startup-script=$startupRendered `
            @flags 2>&1 | ForEach-Object { Write-Host $_ }

        if ($LASTEXITCODE -eq 0) {
            $success = $true
            $selectedZone = $zone
            $selectedProvisioning = $prov
            break
        } else {
            Write-Host "[miss] $zone/$prov unavailable — trying next"
            Start-Sleep -Seconds 3
        }
    }
    if ($success) { break }
}

if ($startupRendered -and (Test-Path $startupRendered)) { Remove-Item -Force $startupRendered -ErrorAction SilentlyContinue }

if (-not $success) {
    Write-Host ""
    Write-Host "[FATAL] no (provisioning x zone) combination accepted the VM create request."
    Write-Host "        Tried provisioning  : $($provList -join ' ')"
    Write-Host "        Tried $($Zones.Count) zones. Request a reservation or retry later."
    exit 1
}

# ---------- record run metadata -----------------------------------------------
$launchedAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
$runMeta = [ordered]@{
    run_id             = $RunId
    model_size         = $ModelSize
    instance_name      = $instanceName
    zone               = $selectedZone
    profile            = $ProfileId
    machine_type       = $MachineType
    accelerator        = $AcceleratorType
    provisioning_model = $selectedProvisioning
    data_tokens        = [int64]$DataTokens
    min_tokens         = [int64]$MinTokens
    train_steps        = [int64]$TrainSteps
    cuda_version       = $CudaVersion
    cuda_dir           = $CudaDir
    sync_interval      = [int64]$SyncInterval
    script_version     = $ScriptVersion
    git_commit         = $GitCommit
    launched_at        = $launchedAt
}
$runMetaFile = New-TemporaryFile
($runMeta | ConvertTo-Json) | Set-Content -Path $runMetaFile.FullName -Encoding utf8
& gsutil -q cp $runMetaFile.FullName "$GcpBucket/runs/$RunId/launch.json" 2>$null
$RunId | & gsutil -q cp - "$GcpBucket/latest_run.txt" 2>$null
Remove-Item -Force $runMetaFile.FullName -ErrorAction SilentlyContinue

@"

============================================================
 LAUNCHED
   instance:     $instanceName
   zone:         $selectedZone
   hardware:     L4 ($MachineType + $AcceleratorType)
   provisioning: $selectedProvisioning
   run_id:       $RunId
   bucket:       $GcpBucket/runs/$RunId/
   image:        $ImageFamily ($ImageProject)
============================================================

 Tail bootstrap log (serial console):
   gcloud compute instances get-serial-port-output $instanceName --zone=$selectedZone --project=$GcpProject | Select-Object -Last 200

 SSH in:
   gcloud compute ssh $instanceName --zone=$selectedZone --project=$GcpProject

 Watch status:
   .\scripts\gcp_status.ps1 $RunId

 Tail GCS-side training log:
   .\scripts\gcp_tail_logs.ps1 $RunId

 Stop / delete the VM when needed:
   .\scripts\gcp_stop_vm.ps1   $instanceName $selectedZone
   .\scripts\gcp_delete_vm.ps1 $instanceName $selectedZone
"@ | Write-Host
