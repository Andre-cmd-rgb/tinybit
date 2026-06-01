#!/bin/bash
# tinybit GCP launcher — L4 only.
#
# Tries every zone in order until ONE L4 VM is created, then stops.
# Generates a per-run RUN_ID, uploads the local repo to gs://$GCP_BUCKET/$GCS_REPO_PREFIX,
# renders cloud/startup.sh with substitutions, and attaches it as VM metadata.
#
# Required env:
#   GCP_PROJECT     Google Cloud project id
#   GCP_BUCKET      gs://your-bucket   (no trailing slash)
#
# Optional env (all have defaults):
#   PROVISIONING_MODEL   STANDARD | SPOT | comma list (e.g. "STANDARD,SPOT").
#                        With a list, the launcher tries each mode in order so
#                        you can prefer on-demand and fall back to spot when
#                        on-demand capacity is gone. Default: "STANDARD".
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
#   TRAIN_CONFIG         path inside the repo to a training config TOML
#                        (e.g. configs/train-micro-l4.toml). Empty falls back
#                        to an inline generated config parameterized by
#                        TRAIN_STEPS.
#
# Usage:
#   ./scripts/gcp_launch.sh [micro|bit|qbit|micro-coding|bit-coding|qbit-coding]
#   (micro≈50M batch 11 is the validated target; bit≈100M / qbit≈150M use the
#    smaller-batch configs/train-{bit,qbit}-l4.toml to fit the L4's 24 GB.)
#
# A "-coding" suffix selects configs/<size>-coding.toml and the code-heavy data
# profile (startup.sh derives DATA_PROFILE from the name). Pair it with the same
# train config as the general sibling, e.g.
#   TRAIN_CONFIG=configs/train-micro-l4.toml ./scripts/gcp_launch.sh micro-coding

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO_ROOT/.tinybit.env" ] && source "$REPO_ROOT/.tinybit.env"

# ---------- script version & git info -----------------------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SCRIPT_VERSION="$(cat "$REPO_ROOT/VERSION" 2>/dev/null || echo unknown)"
GIT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=""
if (cd "$REPO_ROOT" && ! git diff --quiet HEAD 2>/dev/null); then
  GIT_DIRTY=" (dirty)"
fi

MODEL_SIZE="${1:-micro}"
: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GCP_BUCKET:?Set GCP_BUCKET (gs://...)}"

# strip trailing slash if any
GCP_BUCKET="${GCP_BUCKET%/}"

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
TRAIN_CONFIG="${TRAIN_CONFIG:-}"
# RESET_RUN=1 wipes stale data/checkpoints on a reused/warm disk so a redeploy
# that changed the data mix re-tokenizes and trains from step 0 (vs --resume).
# Fresh launches default 0 (empty disk → nothing to clear).
RESET_RUN="${RESET_RUN:-0}"
# Times to repeat each datasets/*.jsonl (identity/tools) file when tokenizing.
# The documented "definitive micro run" uses 50 (~0.5% of tokens) so the model
# reliably learns it is tinybit + the tool-call protocol. Launcher default 10.
CUSTOM_CHAT_EPOCHS="${CUSTOM_CHAT_EPOCHS:-10}"

# If a train config path was given, verify it exists locally so we fail fast
# before paying for a VM.
if [ -n "$TRAIN_CONFIG" ]; then
  if [ ! -r "$REPO_ROOT/$TRAIN_CONFIG" ]; then
    echo "[FATAL] TRAIN_CONFIG=$TRAIN_CONFIG not found in repo" >&2
    exit 1
  fi
fi

# Default MIN_TOKENS = 75% of DATA_TOKENS
if [ -z "${MIN_TOKENS:-}" ]; then
  MIN_TOKENS=$(( DATA_TOKENS * 3 / 4 ))
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-${MODEL_SIZE}}"

# Default zone list — EU first, then US, then Asia. L4 is broadly available;
# listing many zones helps work around regional capacity exhaustion.
DEFAULT_ZONES=(
  europe-west4-a europe-west4-b europe-west4-c
  europe-west1-b europe-west1-c
  europe-west3-a europe-west3-b
  europe-west2-a europe-west2-b europe-west2-c
  europe-north1-a europe-north1-b europe-north1-c
  us-central1-a us-central1-b us-central1-c us-central1-f
  us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c
  us-east1-b us-east1-c us-east1-d
  us-east4-a us-east4-b us-east4-c
  us-east5-a us-east5-b us-east5-c
  us-south1-a us-south1-b
  asia-east1-a asia-east1-b
  asia-southeast1-a asia-southeast1-b asia-southeast1-c
)
if [ -n "${GCP_ZONES:-}" ]; then
  read -r -a ZONES <<< "$GCP_ZONES"
else
  ZONES=("${DEFAULT_ZONES[@]}")
fi

# Hardware: L4 only.
PROFILE_ID="l4"
MACHINE_TYPE="g2-standard-4"
ACCELERATOR_TYPE="nvidia-l4"
DISK_TYPE="pd-ssd"
COST_HINT="~0.71"

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
  " hardware      : L4 ($MACHINE_TYPE + $ACCELERATOR_TYPE, ~$COST_HINT/hr on-demand)" \
  " provisioning  : $PROVISIONING_MODEL" \
  " data tokens   : $DATA_TOKENS  (min $MIN_TOKENS)" \
  " train steps   : $TRAIN_STEPS" \
  " cuda          : $CUDA_DIR  (apt cuda-toolkit-$CUDA_VERSION)" \
  " project       : $GCP_PROJECT" \
  " bucket        : $GCP_BUCKET" \
  " repo prefix   : $GCS_REPO_PREFIX" \
  " sync interval : ${SYNC_INTERVAL}s" \
  " HF token      : $([ -n "$HF_TOKEN_VAL" ] && echo set || echo unset)" \
  " train config  : ${TRAIN_CONFIG:-<inline default>}" \
  " zones (count) : ${#ZONES[@]}" \
  "============================================================"

if [ -n "$GIT_DIRTY" ] && [ "$FORCE" != "1" ]; then
  echo "[warn] git working tree is dirty. Uncommitted code will not match what's uploaded."
  echo "       set FORCE=1 to launch anyway, or commit/stash first."
  exit 1
fi

# Validate PROVISIONING_MODEL up-front so typos fail before any remote calls.
IFS=',' read -r -a _PROVISIONING_CHECK <<< "$PROVISIONING_MODEL"
for p in "${_PROVISIONING_CHECK[@]}"; do
  case "$p" in
    STANDARD|SPOT) ;;
    *)
      echo "[FATAL] PROVISIONING_MODEL contains unknown entry '$p' (allowed: STANDARD, SPOT)" >&2
      exit 1
      ;;
  esac
done

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
  local zone="$1"
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
    "__TRAIN_CONFIG__":       os.environ.get("TRAIN_CONFIG", ""),
    "__RESET_RUN__":          os.environ.get("RESET_RUN", "0"),
    "__CUSTOM_CHAT_EPOCHS__": os.environ.get("CUSTOM_CHAT_EPOCHS", "10"),
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

# ---------- try (provisioning × zone) combinations ----------------------------
SUCCESS=0
INSTANCE_NAME=""
SELECTED_ZONE=""
SELECTED_PROVISIONING=""

IFS=',' read -r -a PROVISIONING_LIST <<< "$PROVISIONING_MODEL"

# Outer loop: provisioning. Inner: zone. With PROVISIONING_MODEL="STANDARD,SPOT"
# we first try every zone on-demand, then retry the whole grid on spot.
for prov in "${PROVISIONING_LIST[@]}"; do
  for zone in "${ZONES[@]}"; do
    INSTANCE_NAME="tinybit-${PROFILE_ID}-$(date -u +%Y%m%d-%H%M%S)"
    echo
    echo "------------------------------------------------------------"
    echo " attempt: prov=$prov zone=$zone"
    echo "          machine=$MACHINE_TYPE accel=$ACCELERATOR_TYPE  on-demand $COST_HINT/hr"
    echo " instance: $INSTANCE_NAME"
    echo "------------------------------------------------------------"

    export RUN_ID MODEL_SIZE GCP_BUCKET GCS_REPO_PREFIX DATA_TOKENS MIN_TOKENS \
           TRAIN_STEPS CUDA_VERSION CUDA_DIR KEEP_VM_ON_FAILURE SYNC_INTERVAL \
           HF_TOKEN_VAL SCRIPT_VERSION TRAIN_CONFIG RESET_RUN CUSTOM_CHAT_EPOCHS
    ZONE_INFO="$zone" MACHINE_INFO="$MACHINE_TYPE" ACCEL_INFO="$ACCELERATOR_TYPE" \
    render_startup "$zone"

    INSTANCE_FLAGS=(
      --maintenance-policy=TERMINATE
      --accelerator="type=$ACCELERATOR_TYPE,count=1"
    )
    if [ "$prov" = "SPOT" ]; then
      INSTANCE_FLAGS+=(--provisioning-model=SPOT --instance-termination-action=STOP)
    fi

    if gcloud compute instances create "$INSTANCE_NAME" \
         --project="$GCP_PROJECT" \
         --zone="$zone" \
         --machine-type="$MACHINE_TYPE" \
         --image-family="$IMAGE_FAMILY" \
         --image-project="$IMAGE_PROJECT" \
         --boot-disk-size="$BOOT_DISK_SIZE" \
         --boot-disk-type="$DISK_TYPE" \
         --scopes=storage-full \
         --metadata=run-id="$RUN_ID",tinybit-script-version="$SCRIPT_VERSION",tinybit-model="$MODEL_SIZE",tinybit-profile="$PROFILE_ID",tinybit-provisioning="$prov" \
         --metadata-from-file=startup-script="$STARTUP_RENDERED" \
         "${INSTANCE_FLAGS[@]}" 2>&1
    then
      SUCCESS=1
      SELECTED_ZONE="$zone"
      SELECTED_PROVISIONING="$prov"
      break 2
    else
      echo "[miss] $zone/$prov unavailable — trying next"
      sleep 3
    fi
  done
done

if [ "$SUCCESS" != "1" ]; then
  echo
  echo "[FATAL] no (provisioning × zone) combination accepted the VM create request."
  echo "        Tried provisioning  : ${PROVISIONING_LIST[*]}"
  echo "        Tried ${#ZONES[@]} zones. Request a reservation or retry later."
  exit 1
fi

# ---------- record run metadata -----------------------------------------------
RUN_META="$(mktemp)"
trap 'rm -f "$STARTUP_RENDERED" "$RUN_META"' EXIT
cat > "$RUN_META" <<JSON
{
  "run_id": "$RUN_ID",
  "model_size": "$MODEL_SIZE",
  "instance_name": "$INSTANCE_NAME",
  "zone": "$SELECTED_ZONE",
  "profile": "$PROFILE_ID",
  "machine_type": "$MACHINE_TYPE",
  "accelerator": "$ACCELERATOR_TYPE",
  "provisioning_model": "$SELECTED_PROVISIONING",
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
   instance:     $INSTANCE_NAME
   zone:         $SELECTED_ZONE
   hardware:     L4 ($MACHINE_TYPE + $ACCELERATOR_TYPE)
   provisioning: $SELECTED_PROVISIONING
   run_id:       $RUN_ID
   bucket:       $GCP_BUCKET/runs/$RUN_ID/
   image:        $IMAGE_FAMILY ($IMAGE_PROJECT)
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
