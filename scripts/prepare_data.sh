#!/bin/bash
# Usage: ./scripts/prepare_data.sh [output_dir]
# Downloads and tokenizes datasets for training tinybit.
#
# Env vars:
#   TOTAL_TOKENS   total desired tokens
#   MIN_TOKENS     fail if fewer than this are collected
#   SEQ_LEN        for val-set sizing
#   DATA_PROFILE   general | coding   (alias: PROFILE)
#   HF_TOKEN       optional — enables gated datasets
#   ENABLE_GATED   1 to attempt gated datasets when HF_TOKEN is set
#   CUSTOM_CHAT_DIR
#   CUSTOM_CHAT_EPOCHS
#
# Robust streaming controls:
#   STREAM_TIMEOUT             seconds without one example before declaring a stall
#   MAX_STALLS                 consecutive stalls before killing/restarting stream
#   MAX_RESTARTS_PER_DATASET   0 = unlimited restarts
#   FLUSH_EVERY                flush token buffer every N tokens
#   RESUME_DATA_PREP           1 = resume from prepare_progress.json + _tokens_tmp.bin
#
# This wrapper is intentionally strict:
# - Python runs unbuffered, so cloud logs update immediately.
# - HuggingFace streams are allowed to stall, but not freeze forever.
# - Data prep can resume from partial progress after reset/preemption.
# - The VM should not silently skip the broken dataset unless the Python script
#   explicitly decides to fail or move on.

set -Eeuo pipefail

OUTPUT_DIR="${1:-data}"
OUTPUT_DIR="${OUTPUT_DIR%/}"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/prepare_data.py"

if [ ! -f "$PY_SCRIPT" ]; then
  echo "ERROR: missing $PY_SCRIPT"
  exit 1
fi

echo "Preparing data in $OUTPUT_DIR ..."
echo "Using script: $PY_SCRIPT"

# Make Python logs visible immediately in /var/log/tinybit-bootstrap.log.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# HuggingFace/fsspec timeouts. These do not solve every C-level hang by
# themselves, but they help normal network failures fail faster.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_DATASETS_IN_MEMORY_MAX_SIZE="${HF_DATASETS_IN_MEMORY_MAX_SIZE:-0}"

# Robust restart/resume behavior for prepare_data.py.
export STREAM_TIMEOUT="${STREAM_TIMEOUT:-45}"
export MAX_STALLS="${MAX_STALLS:-2}"
export MAX_RESTARTS_PER_DATASET="${MAX_RESTARTS_PER_DATASET:-0}"
export FLUSH_EVERY="${FLUSH_EVERY:-500000}"
export RESUME_DATA_PREP="${RESUME_DATA_PREP:-1}"

echo "DATA_PROFILE=${DATA_PROFILE:-${PROFILE:-general}}"
echo "TOTAL_TOKENS=${TOTAL_TOKENS:-unset}"
echo "MIN_TOKENS=${MIN_TOKENS:-unset}"
echo "STREAM_TIMEOUT=$STREAM_TIMEOUT"
echo "MAX_STALLS=$MAX_STALLS"
echo "MAX_RESTARTS_PER_DATASET=$MAX_RESTARTS_PER_DATASET"
echo "FLUSH_EVERY=$FLUSH_EVERY"
echo "RESUME_DATA_PREP=$RESUME_DATA_PREP"
echo "HF_TOKEN=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
echo

exec python3 -u "$PY_SCRIPT" "$OUTPUT_DIR"
