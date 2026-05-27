#!/bin/bash
# tinybit unified GCP launcher.
#
# Tries multiple zones × hardware profiles until ONE VM is created, then stops.
# Generates a per-run RUN_ID, uploads the local repo to gs://$GCP_BUCKET/$GCS_REPO_PREFIX,
# renders cloud/startup.sh with substitutions, and attaches it as VM metadata.
#
# Required env:
#   GCP_PROJECT     Google Cloud project id
#   GCP_BUCKET      gs://your-bucket   (no trailing slash)
#
# Optional env (all have defaults):
#   PROFILE              "l4"|"t4"|"l4,t4"   (comma-separated priority list)
#   PROVISIONING_MODEL   STANDARD|SPOT       (default: STANDARD)
#   DATA_TOKENS          desired tokens      (default: 20000000)
#   MIN_TOKENS           min acceptable      (default: 75% of DATA_TOKENS)
#   TRAIN_STEPS          training steps      (default: 2000)
#   CUDA_VERSION         apt suffix          (default: 12-8)
#   CUDA_DIR             toolkit prefix      (default: /usr/local/cuda-12.8)
#   GCP_ZONES            override zone list  (space-separated)
#   SYNC_INTERVAL        bucket sync seconds (default: 120)
#   KEEP_VM_ON_FAILURE   0|1                 (default: 0)
#   HF_TOKEN             HuggingFace token   (optional, for gated datasets)
#   RUN_ID               override run id     (default: auto)
#   GCS_REPO_PREFIX      bucket prefix       (default: tinybit)
#   SKIP_UPLOAD          1 to skip rsync     (default: 0)
#   FORCE                1 to bypass guards  (default: 0)
#
# Usage:
#   ./scripts/gcp_launch.sh [nano|micro|small|base]

set -Eeuo pipefail

# ---------- script version & git info -----------------------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SCRIPT_VERSION="$(cat "$REPO_ROOT/VERSION" 2>/dev/null || echo unknown)"
GIT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=""
if (cd "$REPO_ROOT" && ! git diff --quiet HEAD 2>/dev/null); then
  GIT_DIRTY=" (dirty)"
fi

MODEL_SIZE="${1:-nano}"
: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GCP_BUCKET:?Set GCP_BUCKET (gs://...)}"

# strip trailing slash if any
GCP_BUCKET="${GCP_BUCKET%/}"

PROFILE="${PROFILE:-l4,t4}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-STANDARD}"
DATA_TOKENS="${DATA_TOKENS:-20000000}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
CUDA_VERSION="${CUDA_VERSION:-12-8}"
CUDA_DIR="${CUDA_DIR:-/usr/local/cuda-12.8}"
SYNC_INTERVAL="${SYNC_INTERVAL:-120}"
KEEP_VM_ON_FAILURE="${KEEP_VM_ON_FAILURE:-0}"
HF_TOKEN_VAL="${HF_TOKEN:-}"
GCS_REPO_PREFIX="${GCS_REPO_PREFIX:-tinybit}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
FORCE="${FORCE:-0}"

# Default MIN_TOKENS = 75% of DATA_TOKENS
if [ -z "${MIN_TOKENS:-}" ]; then
  MIN_TOKENS=$(( DATA_TOKENS * 3 / 4 ))
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-${MODEL_SIZE}}"

# Default zone list: prefer US (broadest GPU stock), then EU.
DEFAULT_ZONES=(
  us-central1-a us-central1-b us-central1-c
  us-west1-a us-west1-b us-west4-a us-west4-c
  us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c
  europe-west1-b europe-west1-c
  europe-west4-a europe-west4-b europe-west4-c
  europe-west3-a europe-west3-b
  europe-west2-a europe-west2-b
)
if [ -n "${GCP_ZONES:-}" ]; then
  read -r -a ZONES <<< "$GCP_ZONES"
else
  ZONES=("${DEFAULT_ZONES[@]}")
fi

# Hardware profile table: profile_id|machine|accelerator
declare -A PROFILE_MACHINE=(
  [l4]=g2-standard-4
  [t4]=n1-standard-4
)
declare -A PROFILE_ACCEL=(
  [l4]=nvidia-l4
  [t4]=nvidia-tesla-t4
)

# Image: deep-learning common image with CUDA 12.9 runtime + nvidia driver 580.
# (We pin CUDA 12.8 toolkit inside startup.sh to match cudarc.)
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="200GB"

# ---------- banner ------------------------------------------------------------
printf '%s\n' \
  "============================================================" \
  " tinybit GCP launcher  v$SCRIPT_VERSION  commit=$GIT_COMMIT$GIT_DIRTY" \
  " run_id        : $RUN_ID" \
  " model         : $MODEL_SIZE" \
  " profiles      : $PROFILE" \
  " provisioning  : $PROVISIONING_MODEL" \
  " data tokens   : $DATA_TOKENS  (min $MIN_TOKENS)" \
  " train steps   : $TRAIN_STEPS" \
  " cuda          : $CUDA_DIR  (apt cuda-toolkit-$CUDA_VERSION)" \
  " project       : $GCP_PROJECT" \
  " bucket        : $GCP_BUCKET" \
  " repo prefix   : $GCS_REPO_PREFIX" \
  " sync interval : ${SYNC_INTERVAL}s" \
  " HF token      : $([ -n "$HF_TOKEN_VAL" ] && echo set || echo unset)" \
  " zones (count) : ${#ZONES[@]}" \
  "============================================================"

if [ -n "$GIT_DIRTY" ] && [ "$FORCE" != "1" ]; then
  echo "[warn] git working tree is dirty. Uncommitted code will not match what's uploaded."
  echo "       set FORCE=1 to launch anyway, or commit/stash first."
  exit 1
fi

# ---------- preflight ---------------------------------------------------------
command -v gcloud >/dev/null || { echo "gcloud not installed"; exit 1; }
command -v gsutil >/dev/null || { echo "gsutil not installed"; exit 1; }

ACTIVE_ACCT="$(gcloud config get-value account 2>/dev/null || true)"
if [ -z "$ACTIVE_ACCT" ] || [ "$ACTIVE_ACCT" = "(unset)" ]; then
  echo "[FATAL] gcloud has no active account. Run: gcloud auth login"
  exit 1
fi

# Verify bucket exists & writable.
if ! gsutil ls "$GCP_BUCKET/" >/dev/null 2>&1; then
  echo "[FATAL] cannot list $GCP_BUCKET — does it exist and do you have access?"
  exit 1
fi
PROBE="$GCP_BUCKET/_launcher_probe_${RUN_ID}.txt"
printf 'probe %s\n' "$RUN_ID" | gsutil -q cp - "$PROBE" || { echo "[FATAL] cannot write to bucket"; exit 1; }
gsutil -q rm "$PROBE" || true
echo "[ok] bucket reachable and writable"

# ---------- upload repo -------------------------------------------------------
if [ "$SKIP_UPLOAD" = "1" ]; then
  echo "[info] SKIP_UPLOAD=1 — assuming $GCP_BUCKET/$GCS_REPO_PREFIX is current"
else
  echo "Uploading repo to $GCP_BUCKET/$GCS_REPO_PREFIX (excluding target/, data/, checkpoints*/) …"
  gsutil -m -q rsync -r -d \
    -x '(^|.*/)target/.*$|(^|.*/)data/.*\.bin$|(^|.*/)checkpoints.*/.*$|^\.git/.*$|.*\.safetensors$' \
    "$REPO_ROOT/" "$GCP_BUCKET/$GCS_REPO_PREFIX/"
  echo "[ok] repo uploaded"
fi

# ---------- render startup script ---------------------------------------------
TEMPLATE="$HERE/cloud/startup.sh"
[ -r "$TEMPLATE" ] || { echo "[FATAL] missing cloud startup template: $TEMPLATE"; exit 1; }
STARTUP_RENDERED="$(mktemp /tmp/tinybit-startup.XXXXXX.sh)"
trap 'rm -f "$STARTUP_RENDERED"' EXIT

render_startup() {
  local zone="$1" machine="$2" accel="$3"
  # Use python to do safe placeholder substitution — sed risks corrupting HF token / special chars.
  python3 - "$TEMPLATE" "$STARTUP_RENDERED" <<PY
import os, sys
src, dst = sys.argv[1], sys.argv[2]
subs = {
    "__RUN_ID__":             os.environ["RUN_ID"],
    "__MODEL_SIZE__":         os.environ["MODEL_SIZE"],
    "__GCS_BUCKET__":         os.environ["GCP_BUCKET"],
    "__GCS_REPO_PREFIX__":    os.environ["GCS_REPO_PREFIX"],
    "__DATA_TOKENS__":        os.environ["DATA_TOKENS"],
    "__MIN_TOKENS__":         os.environ["MIN_TOKENS"],
    "__TRAIN_STEPS__":        os.environ["TRAIN_STEPS"],
    "__CUDA_VERSION__":       os.environ["CUDA_VERSION"],
    "__CUDA_DIR__":           os.environ["CUDA_DIR"],
    "__KEEP_VM_ON_FAILURE__": os.environ["KEEP_VM_ON_FAILURE"],
    "__SYNC_INTERVAL__":      os.environ["SYNC_INTERVAL"],
    "__HF_TOKEN__":           os.environ.get("HF_TOKEN_VAL", ""),
    "__SCRIPT_VERSION__":     os.environ["SCRIPT_VERSION"],
    "__ZONE__":               os.environ["ZONE_INFO"],
    "__MACHINE__":            os.environ["MACHINE_INFO"],
    "__ACCELERATOR__":        os.environ["ACCEL_INFO"],
}
with open(src, "r") as f: text = f.read()
for k, v in subs.items():
    text = text.replace(k, v)
with open(dst, "w") as f: f.write(text)
PY
}

# ---------- try (zone × profile) combinations --------------------------------
SUCCESS=0
INSTANCE_NAME=""
SELECTED_ZONE=""
SELECTED_PROFILE=""

# Iterate zone-major so we try the same hardware in many zones first.
IFS=',' read -r -a PROFILE_LIST <<< "$PROFILE"

for profile in "${PROFILE_LIST[@]}"; do
  machine="${PROFILE_MACHINE[$profile]:-}"
  accel="${PROFILE_ACCEL[$profile]:-}"
  if [ -z "$machine" ]; then
    echo "[warn] unknown profile '$profile' — skipping"; continue
  fi

  for zone in "${ZONES[@]}"; do
    INSTANCE_NAME="tinybit-${profile}-$(date -u +%Y%m%d-%H%M%S)"
    echo
    echo "------------------------------------------------------------"
    echo " attempt: profile=$profile zone=$zone machine=$machine accel=$accel"
    echo " instance: $INSTANCE_NAME"
    echo "------------------------------------------------------------"

    export RUN_ID MODEL_SIZE GCP_BUCKET GCS_REPO_PREFIX DATA_TOKENS MIN_TOKENS \
           TRAIN_STEPS CUDA_VERSION CUDA_DIR KEEP_VM_ON_FAILURE SYNC_INTERVAL \
           HF_TOKEN_VAL SCRIPT_VERSION
    ZONE_INFO="$zone" MACHINE_INFO="$machine" ACCEL_INFO="$accel" \
    render_startup "$zone" "$machine" "$accel"

    INSTANCE_FLAGS=(
      --maintenance-policy=TERMINATE
      --accelerator="type=$accel,count=1"
    )
    if [ "$PROVISIONING_MODEL" = "SPOT" ]; then
      INSTANCE_FLAGS+=(--provisioning-model=SPOT --instance-termination-action=STOP)
    fi

    if gcloud compute instances create "$INSTANCE_NAME" \
         --project="$GCP_PROJECT" \
         --zone="$zone" \
         --machine-type="$machine" \
         --image-family="$IMAGE_FAMILY" \
         --image-project="$IMAGE_PROJECT" \
         --boot-disk-size="$BOOT_DISK_SIZE" \
         --boot-disk-type=pd-ssd \
         --scopes=storage-full \
         --metadata=run-id="$RUN_ID",tinybit-script-version="$SCRIPT_VERSION",tinybit-model="$MODEL_SIZE" \
         --metadata-from-file=startup-script="$STARTUP_RENDERED" \
         "${INSTANCE_FLAGS[@]}" 2>&1
    then
      SUCCESS=1
      SELECTED_ZONE="$zone"
      SELECTED_PROFILE="$profile"
      break 2
    else
      echo "[miss] $zone/$profile unavailable — trying next"
      sleep 3
    fi
  done
done

if [ "$SUCCESS" != "1" ]; then
  echo
  echo "[FATAL] no (zone × profile) combination accepted the VM create request."
  echo "        Tried profiles: ${PROFILE_LIST[*]}"
  echo "        Tried ${#ZONES[@]} zones. Request a reservation or retry later."
  exit 1
fi

# ---------- record run metadata ----------------------------------------------
RUN_META="$(mktemp)"
trap 'rm -f "$STARTUP_RENDERED" "$RUN_META"' EXIT
cat > "$RUN_META" <<JSON
{
  "run_id": "$RUN_ID",
  "model_size": "$MODEL_SIZE",
  "instance_name": "$INSTANCE_NAME",
  "zone": "$SELECTED_ZONE",
  "profile": "$SELECTED_PROFILE",
  "machine_type": "${PROFILE_MACHINE[$SELECTED_PROFILE]}",
  "accelerator": "${PROFILE_ACCEL[$SELECTED_PROFILE]}",
  "provisioning_model": "$PROVISIONING_MODEL",
  "data_tokens": $DATA_TOKENS,
  "min_tokens": $MIN_TOKENS,
  "train_steps": $TRAIN_STEPS,
  "cuda_version": "$CUDA_VERSION",
  "cuda_dir": "$CUDA_DIR",
  "sync_interval": $SYNC_INTERVAL,
  "script_version": "$SCRIPT_VERSION",
  "git_commit": "$GIT_COMMIT",
  "launched_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
gsutil -q cp "$RUN_META" "$GCP_BUCKET/runs/$RUN_ID/launch.json" || true
printf '%s\n' "$RUN_ID" | gsutil -q cp - "$GCP_BUCKET/latest_run.txt" || true

cat <<EOF

============================================================
 LAUNCHED
   instance: $INSTANCE_NAME
   zone:     $SELECTED_ZONE
   profile:  $SELECTED_PROFILE
   run_id:   $RUN_ID
   bucket:   $GCP_BUCKET/runs/$RUN_ID/
   image:    $IMAGE_FAMILY ($IMAGE_PROJECT)
============================================================

 Tail bootstrap log (serial console):
   gcloud compute instances get-serial-port-output $INSTANCE_NAME \\
     --zone=$SELECTED_ZONE --project=$GCP_PROJECT | tail -200

 SSH in:
   gcloud compute ssh $INSTANCE_NAME --zone=$SELECTED_ZONE --project=$GCP_PROJECT

 Watch status:
   ./scripts/gcp_status.sh $RUN_ID

 Tail GCS-side training log:
   ./scripts/gcp_tail_logs.sh $RUN_ID

 Stop / delete the VM when needed:
   ./scripts/gcp_stop_vm.sh   $INSTANCE_NAME $SELECTED_ZONE
   ./scripts/gcp_delete_vm.sh $INSTANCE_NAME $SELECTED_ZONE
EOF
