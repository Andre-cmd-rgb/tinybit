#!/bin/bash
# tinybit cloud startup script — runs on a fresh GCP VM as root via metadata.
#
# This script is meant to be rendered from gcp_launch.sh by substituting the
# placeholders below (sed-style "__VAR__" tokens) before being passed as
# --metadata-from-file=startup-script=...
#
# Placeholders (must all be substituted before upload):
#   __RUN_ID__              — unique run id, e.g. 20260527-141500-nano
#   __MODEL_SIZE__          — nano|micro|small|base
#   __GCS_BUCKET__          — gs://bucket-name  (no trailing slash)
#   __GCS_REPO_PREFIX__     — repo prefix inside bucket, e.g. tinybit
#   __DATA_TOKENS__         — desired total training tokens
#   __MIN_TOKENS__          — minimum acceptable token count to proceed
#   __TRAIN_STEPS__         — total training steps
#   __CUDA_VERSION__        — e.g. 12-8 (apt suffix) or empty to skip CUDA pin
#   __CUDA_DIR__            — e.g. /usr/local/cuda-12.8
#   __KEEP_VM_ON_FAILURE__  — 0|1
#   __SYNC_INTERVAL__       — seconds between checkpoint syncs (default 120)
#   __HF_TOKEN__            — optional HF token, or empty
#   __SCRIPT_VERSION__      — launcher version string
#   __TRAIN_CONFIG__        — path to a checked-in train config (relative to
#                              repo root, e.g. "configs/train-quality.toml")
#                              or empty to fall back to the inline default
#                              parameterized by __TRAIN_STEPS__.
#   __ZONE__ / __MACHINE__ / __ACCELERATOR__ — informational, recorded in status.json

set -Eeuo pipefail

# ---------- environment hardening ----------------------------------------------
# GCP startup scripts run as root with no shell init: HOME is unset, PATH is
# minimal. Set everything explicitly.
#
# /snap/bin is critical — on the GCP deep-learning image, gsutil and gcloud are
# installed as snap apps, so without /snap/bin every `gsutil cp` in this script
# silently no-ops because of the `command -v gsutil` guard, and *nothing* makes
# it back to the bucket.
export HOME=/root
export USER=root
export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.cargo/bin:/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd /root

# ---------- run parameters (filled by launcher) -------------------------------
RUN_ID="__RUN_ID__"
MODEL_SIZE="__MODEL_SIZE__"
GCS_BUCKET="__GCS_BUCKET__"
GCS_REPO_PREFIX="__GCS_REPO_PREFIX__"
DATA_TOKENS="__DATA_TOKENS__"
MIN_TOKENS="__MIN_TOKENS__"
TRAIN_STEPS="__TRAIN_STEPS__"
CUDA_VERSION="__CUDA_VERSION__"
CUDA_DIR="__CUDA_DIR__"
KEEP_VM_ON_FAILURE="__KEEP_VM_ON_FAILURE__"
SYNC_INTERVAL="__SYNC_INTERVAL__"
HF_TOKEN_VAL="__HF_TOKEN__"
SCRIPT_VERSION="__SCRIPT_VERSION__"
TRAIN_CONFIG_OVERRIDE="__TRAIN_CONFIG__"
ZONE_INFO="__ZONE__"
MACHINE_INFO="__MACHINE__"
ACCELERATOR_INFO="__ACCELERATOR__"

WORKDIR=/workspace/tinybit
GCS_RUN_PREFIX="$GCS_BUCKET/runs/$RUN_ID"
BOOTSTRAP_LOG=/var/log/tinybit-bootstrap.log
TRAINING_LOG=/var/log/tinybit-training.log
STATUS_PATH=/var/log/tinybit-status.json
STAGE_CURRENT="boot"

mkdir -p "$WORKDIR" /var/log
: > "$BOOTSTRAP_LOG"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1

# ---------- helpers -----------------------------------------------------------
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

# Resolve gsutil even when PATH is wrong. On the GCP deep-learning image it's
# at /snap/bin/gsutil; on other images it may be /usr/bin/gsutil or installed
# via the Google Cloud SDK to /opt/google-cloud-sdk/bin/gsutil.
GSUTIL_BIN=""
for _cand in gsutil /snap/bin/gsutil /usr/bin/gsutil /opt/google-cloud-sdk/bin/gsutil /usr/lib/google-cloud-sdk/bin/gsutil; do
  if [ "$_cand" = "gsutil" ]; then
    if command -v gsutil >/dev/null 2>&1; then GSUTIL_BIN="$(command -v gsutil)"; break; fi
  elif [ -x "$_cand" ]; then
    GSUTIL_BIN="$_cand"; break
  fi
done
if [ -z "$GSUTIL_BIN" ]; then
  echo "[boot-warn] gsutil not found on PATH or common install locations — bucket sync will be disabled"
fi
gs() {
  if [ -n "$GSUTIL_BIN" ]; then "$GSUTIL_BIN" "$@"; else return 1; fi
}

write_status() {
  local stage="$1"; local extra="${2:-}"
  local step="${LAST_STEP:-0}"; local ckpt="${LAST_CKPT:-}"
  cat > "$STATUS_PATH" <<JSON
{
  "run_id": "$RUN_ID",
  "stage": "$stage",
  "model_size": "$MODEL_SIZE",
  "script_version": "$SCRIPT_VERSION",
  "zone": "$ZONE_INFO",
  "machine_type": "$MACHINE_INFO",
  "accelerator": "$ACCELERATOR_INFO",
  "step": $step,
  "last_checkpoint": "$ckpt",
  "updated_at": "$(ts)"$extra
}
JSON
  gs -q cp "$STATUS_PATH" "$GCS_RUN_PREFIX/status.json" || true
}

log_stage() {
  STAGE_CURRENT="$1"
  log "[stage] $1"
  write_status "$1"
  sync_logs
}

sync_logs() {
  gs -q cp "$BOOTSTRAP_LOG" "$GCS_RUN_PREFIX/logs/bootstrap.log" || true
  if [ -f "$TRAINING_LOG" ]; then
    gs -q cp "$TRAINING_LOG" "$GCS_RUN_PREFIX/logs/training.log" || true
  fi
}

sync_checkpoints() {
  if [ -d "$WORKDIR/checkpoints" ]; then
    gs -m -q rsync -r "$WORKDIR/checkpoints/" "$GCS_RUN_PREFIX/checkpoints/" || true
  fi
}

write_marker() {
  local name="$1"; local reason="${2:-}"
  local marker=/tmp/tinybit-$name.json
  cat > "$marker" <<JSON
{
  "run_id": "$RUN_ID",
  "result": "$name",
  "stage_at_event": "$STAGE_CURRENT",
  "reason": "$reason",
  "script_version": "$SCRIPT_VERSION",
  "at": "$(ts)"
}
JSON
  gs -q cp "$marker" "$GCS_RUN_PREFIX/$name.json" || true
}

maybe_shutdown() {
  if [ "$KEEP_VM_ON_FAILURE" = "1" ]; then
    log "KEEP_VM_ON_FAILURE=1 → leaving VM running for debugging"
    return
  fi
  log "Shutting VM down in 60s (set KEEP_VM_ON_FAILURE=1 to skip)…"
  sleep 60
  shutdown -h now || poweroff || true
}

on_err() {
  local code=$?
  local line=${BASH_LINENO[0]:-?}
  log "[FAILED] stage=$STAGE_CURRENT line=$line code=$code cmd=${BASH_COMMAND}"
  write_status "failed" ", \"failed_stage\": \"$STAGE_CURRENT\", \"failed_line\": $line, \"failed_command\": $(printf '%s' "$BASH_COMMAND" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null || echo '"unknown"')"
  sync_logs
  sync_checkpoints
  write_marker FAILED "$STAGE_CURRENT line $line"
  maybe_shutdown
  exit "$code"
}
trap on_err ERR

LAST_STEP=0
LAST_CKPT=""

log "=============================================================="
log " tinybit cloud startup  v$SCRIPT_VERSION  run_id=$RUN_ID"
log " model=$MODEL_SIZE zone=$ZONE_INFO machine=$MACHINE_INFO accel=$ACCELERATOR_INFO"
log " bucket=$GCS_BUCKET  data_tokens=$DATA_TOKENS  steps=$TRAIN_STEPS"
log " cuda=$CUDA_DIR (apt=$CUDA_VERSION)  sync_interval=${SYNC_INTERVAL}s"
log "=============================================================="

# ---------- swap_setup --------------------------------------------------------
# Add swap early so that data-prep and cargo build cannot OOM the VM.
# L4 VMs (g2-standard-4) ship with 16 GB RAM and zero swap.
log_stage swap_setup
SWAPFILE=/swapfile
if ! swapon --show | grep -q "$SWAPFILE" 2>/dev/null; then
  if [ ! -f "$SWAPFILE" ]; then
    log "Allocating 32 GB swap at $SWAPFILE…"
    fallocate -l 32G "$SWAPFILE" 2>/dev/null \
      || dd if=/dev/zero of="$SWAPFILE" bs=1G count=32 status=progress
    chmod 600 "$SWAPFILE"
    mkswap "$SWAPFILE"
  fi
  swapon "$SWAPFILE"
  log "Swap active: $(swapon --show --noheadings)"
else
  log "Swap already active — skipping"
fi

# ---------- install_deps ------------------------------------------------------
log_stage install_deps
apt-get update -q
apt-get install -y --no-install-recommends \
  curl ca-certificates gnupg2 wget jq screen tmux \
  build-essential pkg-config libssl-dev \
  git python3 python3-pip python3-numpy
command -v pkg-config >/dev/null
pkg-config --libs --cflags openssl >/dev/null
dpkg -s libssl-dev >/dev/null

# ---------- install_cuda ------------------------------------------------------
log_stage install_cuda
if [ -n "$CUDA_VERSION" ] && [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
  log "Installing cuda-toolkit-$CUDA_VERSION (cudarc requires a supported toolkit)"
  cd /tmp
  KEYRING=/tmp/cuda-keyring.deb
  if [ ! -f "$KEYRING" ]; then
    wget -q -O "$KEYRING" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  fi
  dpkg -i "$KEYRING" || apt-get install -f -y
  apt-get update -q
  apt-get install -y "cuda-toolkit-$CUDA_VERSION"
fi

if [ -x "$CUDA_DIR/bin/nvcc" ]; then
  export CUDA_ROOT="$CUDA_DIR"
  export CUDA_PATH="$CUDA_DIR"
  export CUDA_TOOLKIT_ROOT_DIR="$CUDA_DIR"
  export PATH="$CUDA_DIR/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
fi

# ---------- cuda_check --------------------------------------------------------
log_stage cuda_check
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || log "[warn] nvidia-smi failed (driver not ready?)"
else
  log "[warn] nvidia-smi missing — CUDA build will still be attempted"
fi
if [ -x "$CUDA_DIR/bin/nvcc" ]; then
  "$CUDA_DIR/bin/nvcc" --version
else
  log "[FATAL] nvcc not found at $CUDA_DIR/bin/nvcc"
  false
fi

# ---------- install_rust ------------------------------------------------------
log_stage install_rust
if [ ! -x /root/.cargo/bin/cargo ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
source /root/.cargo/env
cargo --version
rustc --version

# ---------- copy_repo ---------------------------------------------------------
log_stage copy_repo
mkdir -p /workspace
if [ ! -d "$WORKDIR/.git" ] && [ ! -f "$WORKDIR/Cargo.toml" ]; then
  gs -m -q rsync -r "$GCS_BUCKET/$GCS_REPO_PREFIX" "$WORKDIR"
else
  log "[info] repo already present at $WORKDIR — rsyncing updates"
  gs -m -q rsync -r "$GCS_BUCKET/$GCS_REPO_PREFIX" "$WORKDIR" || true
fi
chmod +x "$WORKDIR"/scripts/*.sh 2>/dev/null || true
chmod +x "$WORKDIR"/scripts/cloud/*.sh 2>/dev/null || true
cd "$WORKDIR"

# ---------- cargo_build -------------------------------------------------------
log_stage cargo_build
BIN="target/release/tinybit"
if [ "${FORCE_REBUILD:-0}" = "1" ]; then
  log "FORCE_REBUILD=1 — wiping target/release/build/cudarc-* and candle-*"
  rm -rf target/release/build/cudarc-* target/release/deps/libcudarc* \
         target/release/build/candle-* target/release/deps/libcandle* || true
fi
if [ ! -x "$BIN" ]; then
  RUSTFLAGS='-C target-cpu=native' \
    cargo build --release -p tinybit-cli --features cuda
else
  log "[info] $BIN already built — skipping cargo build (set FORCE_REBUILD=1 to force)"
fi
test -x "$BIN"

# ---------- prepare_data ------------------------------------------------------
log_stage prepare_data
mkdir -p data
if [ -s data/train.bin ] && [ -s data/val.bin ] && [ "${FORCE_DATA:-0}" != "1" ]; then
  log "[info] data/train.bin and data/val.bin already present — skipping (FORCE_DATA=1 to redo)"
else
  python3 -m pip install --break-system-packages --quiet datasets tokenizers tqdm \
    || python3 -m pip install --user --quiet datasets tokenizers tqdm
  if [ -n "$HF_TOKEN_VAL" ]; then
    export HF_TOKEN="$HF_TOKEN_VAL"
  fi
  TOTAL_TOKENS="$DATA_TOKENS" MIN_TOKENS="$MIN_TOKENS" \
    bash ./scripts/prepare_data.sh data/
fi
test -s data/train.bin
test -s data/val.bin

# ---------- restore_checkpoints -----------------------------------------------
# If this is a fresh VM (e.g., relaunching after preemption on a new instance),
# download any existing checkpoints from GCS so the trainer can resume from the
# last saved step rather than starting from scratch.
log_stage restore_checkpoints
mkdir -p "$WORKDIR/checkpoints"
if [ -z "$(ls -A "$WORKDIR/checkpoints/" 2>/dev/null)" ]; then
  if gs ls "$GCS_RUN_PREFIX/checkpoints/" >/dev/null 2>&1; then
    log "Downloading existing checkpoints from GCS..."
    gs -m -q rsync -r "$GCS_RUN_PREFIX/checkpoints/" "$WORKDIR/checkpoints/" \
      && log "Checkpoints restored from GCS" \
      || log "[warn] checkpoint restore failed — training will start from scratch"
  else
    log "[info] No existing checkpoints in GCS for this run — starting fresh"
  fi
else
  log "[info] Local checkpoints already present — skipping GCS restore"
fi

# ---------- prepare_training_config ------------------------------------------
log_stage prepare_training_config
if [ -n "$TRAIN_CONFIG_OVERRIDE" ] && [ -r "$TRAIN_CONFIG_OVERRIDE" ]; then
  TRAIN_CONFIG_PATH="$TRAIN_CONFIG_OVERRIDE"
  log "[info] using checked-in train config: $TRAIN_CONFIG_PATH"
else
  TRAIN_CONFIG_PATH="configs/train-cloud.toml"
  log "[info] generating inline train config at $TRAIN_CONFIG_PATH (total_steps=$TRAIN_STEPS)"
  cat > "$TRAIN_CONFIG_PATH" <<TOML
train_data     = "data/train.bin"
val_data       = "data/val.bin"
checkpoint_dir = "checkpoints/"

batch_size     = 4
grad_accum     = 4
total_steps    = $TRAIN_STEPS
peak_lr        = 3e-4
weight_decay   = 0.01
grad_clip      = 1.0

save_every     = 100
eval_every     = 50
eval_batches   = 5

smoke_test_steps = 0
TOML
fi

# ---------- start_training ----------------------------------------------------
log_stage start_training
if pgrep -fa "target/release/tinybit train" >/dev/null 2>&1; then
  log "[guard] training process already running — refusing to start a second one"
else
  : > "$TRAINING_LOG"
  setsid bash -c '
    RUST_LOG=tinybit_train=info,info ./target/release/tinybit train \
      --model-config "configs/'"$MODEL_SIZE"'.toml" \
      --train-config "'"$TRAIN_CONFIG_PATH"'" \
      --resume
  ' </dev/null >>"$TRAINING_LOG" 2>&1 &
  TRAIN_PID=$!
  echo "$TRAIN_PID" > /var/run/tinybit-train.pid
  log "training pid=$TRAIN_PID log=$TRAINING_LOG"
fi

# ---------- sync_loop ---------------------------------------------------------
log_stage sync_loop
TRAIN_PID="$(cat /var/run/tinybit-train.pid 2>/dev/null || echo 0)"
while kill -0 "$TRAIN_PID" >/dev/null 2>&1; do
  # discover latest checkpoint for status.json
  LATEST="$(ls -1 checkpoints/step_*.safetensors 2>/dev/null | sort | tail -n1 || true)"
  if [ -n "$LATEST" ]; then
    LAST_CKPT="$(basename "$LATEST")"
    LAST_STEP="$(echo "$LAST_CKPT" | sed -E 's/step_0*([0-9]+)\.safetensors/\1/')"
  fi
  write_status "training"
  sync_logs
  sync_checkpoints
  sleep "$SYNC_INTERVAL"
done

# ---------- final_sync --------------------------------------------------------
log_stage final_sync
LATEST="$(ls -1 checkpoints/step_*.safetensors 2>/dev/null | sort | tail -n1 || true)"
if [ -n "$LATEST" ]; then
  LAST_CKPT="$(basename "$LATEST")"
  LAST_STEP="$(echo "$LAST_CKPT" | sed -E 's/step_0*([0-9]+)\.safetensors/\1/')"
fi
sync_logs
sync_checkpoints

# Decide success vs failure by looking at training log tail and exit residue.
if grep -qE "^Error|panicked at|fatal" "$TRAINING_LOG" 2>/dev/null; then
  write_status "failed" ", \"failed_stage\": \"training\""
  write_marker FAILED "training reported errors — see logs/training.log"
  maybe_shutdown
  exit 1
fi

write_status "done"
write_marker DONE "training reached completion"

# ---------- shutdown ----------------------------------------------------------
log_stage shutdown
log "Training complete — shutting down in 60s."
sleep 60
shutdown -h now || poweroff || true
