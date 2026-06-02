#!/bin/bash
# tinybit cloud startup script — runs on a fresh GCP VM as root via metadata.
#
# This script is meant to be rendered from gcp_launch.sh by substituting the
# placeholders below (sed-style "__VAR__" tokens) before being passed as
# --metadata-from-file=startup-script=...
#
# Placeholders (must all be substituted before upload):
#   __RUN_ID__              — unique run id, e.g. 20260527-141500-micro
#   __MODEL_SIZE__          — micro|bit|qbit (optionally with a -coding
#                              suffix, e.g. micro-coding → configs/micro-coding.toml
#                              + the coding data profile)
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
#                              repo root, e.g. "configs/train-micro-l4.toml")
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
RESET_RUN="__RESET_RUN__"
CUSTOM_CHAT_EPOCHS="__CUSTOM_CHAT_EPOCHS__"   # times to repeat datasets/*.jsonl (identity/tools)
ZONE_INFO="__ZONE__"
MACHINE_INFO="__MACHINE__"
ACCELERATOR_INFO="__ACCELERATOR__"

# Model family → data profile. A "*-coding" model size (e.g. micro-coding, which
# uses configs/micro-coding.toml) trains on the code-heavy data mix; everything
# else uses the general mix. See scripts/prepare_data.sh.
case "$MODEL_SIZE" in
  *-coding) DATA_PROFILE="coding" ;;
  *)        DATA_PROFILE="general" ;;
esac

WORKDIR=/workspace/tinybit
GCS_RUN_PREFIX="$GCS_BUCKET/runs/$RUN_ID"
BOOTSTRAP_LOG=/var/log/tinybit-bootstrap.log
TRAINING_LOG=/var/log/tinybit-training.log
DATAPREP_LOG=/var/log/tinybit-dataprep.log
STATUS_PATH=/var/log/tinybit-status.json
STAGE_CURRENT="boot"

# Data-prep watchdog thresholds. prepare_data.py is self-healing (it kills and
# restarts a wedged HF stream), so these are the OUTER backstop that guarantees
# the run can never sit silently: if the prep stops updating its heartbeat, or
# stops writing new tokens for too long, the watchdog kills it and FAILS the run
# loudly (FAILED.json + shutdown) instead of burning GPU-hours doing nothing.
PREP_WATCH_INTERVAL="${PREP_WATCH_INTERVAL:-30}"     # how often to sync + check
PREP_HEARTBEAT_LIMIT="${PREP_HEARTBEAT_LIMIT:-900}"  # 15 min: heartbeat frozen => wedged parent
PREP_PROGRESS_LIMIT="${PREP_PROGRESS_LIMIT:-2700}"   # 45 min: no new tokens => stuck

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

# Push data-prep visibility (log + structured progress + failure marker) to the
# bucket so a long prep is observable in real time and a failure is recorded.
sync_dataprep() {
  [ -f "$DATAPREP_LOG" ] && gs -q cp "$DATAPREP_LOG" "$GCS_RUN_PREFIX/logs/dataprep.log" || true
  [ -f "$WORKDIR/data/prepare_progress.json" ] && \
    gs -q cp "$WORKDIR/data/prepare_progress.json" "$GCS_RUN_PREFIX/data/prepare_progress.json" || true
  [ -f "$WORKDIR/data/prepare_FAILED.json" ] && \
    gs -q cp "$WORKDIR/data/prepare_FAILED.json" "$GCS_RUN_PREFIX/data/prepare_FAILED.json" || true
}

# Run data preparation under a watchdog. prepare_data.py self-heals stalls, but
# this guarantees the run can NEVER hang silently (the failure mode that froze
# the 2026-06-02 run): it streams logs+progress to the bucket continuously and,
# if the prep wedges (heartbeat frozen) or stops producing tokens for too long,
# kills the whole prep process group and returns non-zero so the run fails loudly.
# Returns: 0 = prep succeeded, non-zero = prep failed or was declared wedged.
run_data_prep_watched() {
  : > "$DATAPREP_LOG"
  # New session/process group so the watchdog can kill python + its mp workers.
  setsid bash -c '
    set -Eeuo pipefail
    cd "'"$WORKDIR"'"
    [ -n "'"$HF_TOKEN_VAL"'" ] && export HF_TOKEN="'"$HF_TOKEN_VAL"'"
    export TOTAL_TOKENS="'"$DATA_TOKENS"'" MIN_TOKENS="'"$MIN_TOKENS"'" DATA_PROFILE="'"$DATA_PROFILE"'"
    export CUSTOM_CHAT_EPOCHS="'"${CUSTOM_CHAT_EPOCHS:-10}"'"
    exec bash ./scripts/prepare_data.sh data/
  ' </dev/null >>"$DATAPREP_LOG" 2>&1 &
  local prep_pid=$!
  local pgid="$prep_pid"
  log "data prep pid=$prep_pid (heartbeat limit ${PREP_HEARTBEAT_LIMIT}s, progress limit ${PREP_PROGRESS_LIMIT}s, log=$DATAPREP_LOG)"

  local progress="$WORKDIR/data/prepare_progress.json"
  local last_tw=-1
  local last_tw_change; last_tw_change="$(date +%s)"

  while kill -0 "$prep_pid" 2>/dev/null; do
    sleep "$PREP_WATCH_INTERVAL"
    local now; now="$(date +%s)"
    sync_dataprep
    sync_logs

    if [ -f "$progress" ]; then
      local hb tw
      hb="$(jq -r '.heartbeat_unix // 0' "$progress" 2>/dev/null || echo 0)"
      tw="$(jq -r '.total_written // 0' "$progress" 2>/dev/null || echo 0)"
      LAST_STEP=0
      write_status "prepare_data" ", \"data_tokens_written\": ${tw:-0}, \"data_heartbeat_age_s\": $(( now - ${hb:-0} ))"

      if [ "${hb:-0}" -gt 0 ] && [ "$(( now - hb ))" -gt "$PREP_HEARTBEAT_LIMIT" ]; then
        log "[watchdog] FATAL: data-prep heartbeat stale $(( now - hb ))s > ${PREP_HEARTBEAT_LIMIT}s — prep is WEDGED, killing"
        kill -TERM -- "-$pgid" 2>/dev/null || true; sleep 5; kill -KILL -- "-$pgid" 2>/dev/null || true
        return 124
      fi
      if [ "${tw:-0}" != "$last_tw" ]; then
        last_tw="${tw:-0}"; last_tw_change="$now"
      elif [ "$(( now - last_tw_change ))" -gt "$PREP_PROGRESS_LIMIT" ]; then
        log "[watchdog] FATAL: data-prep wrote no new tokens for $(( now - last_tw_change ))s (stuck at ${tw} tokens) — killing"
        kill -TERM -- "-$pgid" 2>/dev/null || true; sleep 5; kill -KILL -- "-$pgid" 2>/dev/null || true
        return 124
      fi
    fi
  done

  local rc=0
  wait "$prep_pid" || rc=$?   # capture prep exit code without tripping set -e
  sync_dataprep
  log "data prep finished rc=$rc"
  return "$rc"
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
log " model=$MODEL_SIZE (data_profile=$DATA_PROFILE) zone=$ZONE_INFO machine=$MACHINE_INFO accel=$ACCELERATOR_INFO"
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

# ---------- reset_run (fresh redeploy onto a warm/reused disk) -----------------
# When RESET_RUN=1 — a redeploy that CHANGES the data mix or otherwise wants a
# clean run on a reused VM's warm disk — wipe the stale tokenized data and old
# checkpoints. Otherwise prepare_data would skip (seeing the old data/*.bin) and
# start_training would --resume the previous run. With them gone, prepare_data
# re-tokenizes the new mix and training starts from step 0. Default 0 → no-op
# (fresh-VM launches have an empty disk anyway).
# RESET_RUN is honored ONCE per disk: the first boot wipes stale data/checkpoints
# and re-tokenizes, then drops a sentinel. Subsequent boots (e.g. a GCP host-
# maintenance TERMINATE + auto-restart during a multi-day run) see the sentinel
# and DO NOT wipe — so training resumes from the last checkpoint instead of
# restarting from step 0 and re-tokenizing. Without this guard a single restart
# would throw away days of progress on a long run.
RESET_SENTINEL="$WORKDIR/.reset_done"
if [ "$RESET_RUN" = "1" ] && [ ! -f "$RESET_SENTINEL" ]; then
  log_stage reset_run
  log "RESET_RUN=1 (first boot) — clearing stale data/*.bin and checkpoints for a fresh run"
  rm -f "$WORKDIR"/data/*.bin 2>/dev/null || true
  rm -rf "$WORKDIR"/checkpoints/* 2>/dev/null || true
  touch "$RESET_SENTINEL"
  FORCE_DATA=1
elif [ "$RESET_RUN" = "1" ]; then
  log "RESET_RUN=1 but reset already performed on this disk ($RESET_SENTINEL exists) — NOT wiping; resuming from checkpoints."
fi

# ---------- cargo_build -------------------------------------------------------
log_stage cargo_build
BIN="target/release/tinybit"
if [ "${FORCE_REBUILD:-0}" = "1" ]; then
  log "FORCE_REBUILD=1 — wiping target/release/build/cudarc-* and candle-*"
  rm -rf target/release/build/cudarc-* target/release/deps/libcudarc* \
         target/release/build/candle-* target/release/deps/libcandle* || true
fi
# Always build: cargo is incremental, so a fresh VM does a full build (~20 min)
# but a REUSED/rebooted VM with a warm target/ only recompiles changed crates
# (~minutes) — what a code redeploy needs. (The old "skip if binary exists" guard
# silently ran STALE code after redeploying onto an existing disk.)
RUSTFLAGS='-C target-cpu=native' \
  cargo build --release -p tinybit-cli --features cuda
test -x "$BIN"

# ---------- prepare_data ------------------------------------------------------
# Data is cached per-run in the bucket so a relaunch (SPOT preemption, or a code
# redeploy that resumes from a checkpoint) does NOT re-tokenize the full token
# budget (~hours for 1.5B). Resolution order: local disk → bucket cache →
# fresh prep (which then populates the cache). FORCE_DATA=1 forces a fresh prep.
log_stage prepare_data
mkdir -p data
if [ -s data/train.bin ] && [ -s data/val.bin ] && [ "${FORCE_DATA:-0}" != "1" ]; then
  log "[info] data/train.bin and data/val.bin already present — skipping (FORCE_DATA=1 to redo)"
else
  if [ "${FORCE_DATA:-0}" != "1" ] && gs ls "$GCS_RUN_PREFIX/data/train.bin" >/dev/null 2>&1; then
    log "Restoring cached data from $GCS_RUN_PREFIX/data/ (skips re-tokenization)…"
    gs -q cp "$GCS_RUN_PREFIX/data/train.bin" data/train.bin || true
    gs -q cp "$GCS_RUN_PREFIX/data/val.bin"   data/val.bin   || true
  fi
  if [ ! -s data/train.bin ] || [ ! -s data/val.bin ]; then
    log "Preparing data from scratch (no usable cache)…"
    python3 -m pip install --break-system-packages --quiet datasets tokenizers tqdm \
      || python3 -m pip install --user --quiet datasets tokenizers tqdm
    # Watched, resumable prep. Returns non-zero on failure OR if the watchdog
    # declares the prep wedged — either way the ERR trap fires, FAILED.json is
    # written, and the VM shuts down instead of hanging forever (the 06-02 bug).
    run_data_prep_watched
    # Cache for future relaunches of this run (best-effort; never fatal).
    log "Caching prepared data to $GCS_RUN_PREFIX/data/ …"
    gs -q cp data/train.bin "$GCS_RUN_PREFIX/data/train.bin" || true
    gs -q cp data/val.bin   "$GCS_RUN_PREFIX/data/val.bin"   || true
  fi
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

batch_size     = 2
grad_accum     = 16
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

# ---------- free_gpu_memory ---------------------------------------------------
# GCP deep-learning images run CUDA MPS (Multi-Process Service) + Jupyter which
# hold an exclusive CUDA context. Any process using CUDA after that gets OOM.
# We must stop MPS, kill all GPU-using processes, and verify the GPU is free
# before starting training.
log_stage free_gpu_memory

# 1. Stop CUDA MPS server (runs on DL images to share GPU between frameworks).
#    If MPS holds the context and we don't stop it, cudarc gets CUDA_ERROR_OUT_OF_MEMORY.
log "Stopping CUDA MPS if running..."
echo quit | nvidia-cuda-mps-control 2>/dev/null && sleep 3 || true
pkill -9 -f 'nvidia-cuda-mps' 2>/dev/null || true

# 2. Kill all remaining GPU-using processes
log "Killing remaining GPU/ML processes..."
pkill -9 -f 'jupyter|tensorboard|python3 -m|/.local/lib' 2>/dev/null || true
sleep 5

# 3. Kill any process still holding GPU memory (shows in compute-apps)
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r pid _mem; do
        pid="${pid// /}"
        [ -z "$pid" ] && continue
        comm="$(cat /proc/$pid/comm 2>/dev/null || echo unknown)"
        log "[gpu-free] killing pid=$pid ($comm) holding GPU memory"
        kill -9 "$pid" 2>/dev/null || true
      done
  sleep 5
fi

# 4. Explicitly set GPU visibility (overrides any CUDA_VISIBLE_DEVICES=-1 set by DL image)
export CUDA_VISIBLE_DEVICES=0

# 5. CUDA pre-flight: verify we can actually allocate on the GPU at the driver level.
#    If this fails, we get a clear diagnostic instead of a mysterious OOM mid-init.
log "Running CUDA pre-flight allocation test..."
python3 - <<'CUTEST' || { log "[FATAL] CUDA pre-flight failed — see diagnostic above"; false; }
import ctypes, sys

def try_int(fn, *args):
    rc = fn(*args)
    return rc

lib = None
for name in ('libcuda.so.1', 'libcuda.so'):
    try:
        lib = ctypes.CDLL(name)
        break
    except OSError:
        pass
if lib is None:
    print("[cuda-preflight] ERROR: cannot load libcuda.so.1")
    sys.exit(1)

rc = lib.cuInit(0)
print(f"[cuda-preflight] cuInit(0) = {rc} ({'OK' if rc == 0 else 'FAIL'})")
if rc != 0:
    sys.exit(1)

count = ctypes.c_int()
lib.cuDeviceGetCount(ctypes.byref(count))
print(f"[cuda-preflight] device count = {count.value}")
if count.value == 0:
    print("[cuda-preflight] ERROR: CUDA_VISIBLE_DEVICES hid all GPUs — check env")
    sys.exit(1)

dev = ctypes.c_int()
rc = lib.cuDeviceGet(ctypes.byref(dev), 0)
print(f"[cuda-preflight] cuDeviceGet(0) rc={rc} dev={dev.value}")

ctx = ctypes.c_void_p()
rc = lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
print(f"[cuda-preflight] cuDevicePrimaryCtxRetain rc={rc} ctx={ctx.value}")
if rc != 0:
    sys.exit(1)

rc = lib.cuCtxSetCurrent(ctx)
print(f"[cuda-preflight] cuCtxSetCurrent rc={rc}")

# Try a 256 MB allocation to confirm full VRAM access
ptr = ctypes.c_void_p()
size = 256 * 1024 * 1024
rc = lib.cuMemAlloc_v2(ctypes.byref(ptr), size)
print(f"[cuda-preflight] cuMemAlloc(256MB) rc={rc} ({'OK' if rc == 0 else 'FAIL - OOM'})")
if rc == 0:
    lib.cuMemFree_v2(ptr)
    print("[cuda-preflight] PASS: GPU memory allocation works")
else:
    lib.cuDevicePrimaryCtxRelease_v2(dev)
    sys.exit(1)

lib.cuDevicePrimaryCtxRelease_v2(dev)
CUTEST

nvidia-smi --query-gpu=index,name,memory.free,memory.total,compute_mode --format=csv || true
log "GPU free — proceeding to training"

# ---------- start_training ----------------------------------------------------
log_stage start_training
if pgrep -fa "target/release/tinybit train" >/dev/null 2>&1; then
  log "[guard] training process already running — refusing to start a second one"
else
  : > "$TRAINING_LOG"
  setsid bash -c '
    export CUDA_VISIBLE_DEVICES=0
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
