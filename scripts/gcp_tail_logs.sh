#!/bin/bash
# Continuously fetch and tail the bootstrap + training log for a run.
#
# Usage:
#   ./scripts/gcp_tail_logs.sh [RUN_ID] [bootstrap|training]
#
# Default is "training" if data/train.bin built, otherwise "bootstrap".
# Polls every 10 seconds.

set -uo pipefail

: "${GCP_BUCKET:?Set GCP_BUCKET (gs://...)}"
GCP_BUCKET="${GCP_BUCKET%/}"

RUN_ID="${1:-}"
if [ -z "$RUN_ID" ]; then
  RUN_ID="$(gsutil -q cat "$GCP_BUCKET/latest_run.txt" 2>/dev/null | tr -d '[:space:]' || true)"
  if [ -z "$RUN_ID" ]; then echo "No RUN_ID and latest_run.txt empty"; exit 1; fi
fi

WHICH="${2:-training}"
case "$WHICH" in
  bootstrap) FILE="logs/bootstrap.log" ;;
  training)  FILE="logs/training.log" ;;
  *) echo "second arg must be 'bootstrap' or 'training'"; exit 1 ;;
esac

PREFIX="$GCP_BUCKET/runs/$RUN_ID"
TMP="/tmp/_tinybit-tail-${RUN_ID}-${WHICH}.log"
: > "$TMP"
echo "Tailing $PREFIX/$FILE (Ctrl-C to stop)"
LAST_BYTES=0
while true; do
  if gsutil -q cp "$PREFIX/$FILE" "$TMP.new" 2>/dev/null; then
    NEW_BYTES=$(stat -c %s "$TMP.new" 2>/dev/null || stat -f %z "$TMP.new")
    if [ "$NEW_BYTES" -gt "$LAST_BYTES" ]; then
      dd if="$TMP.new" bs=1 skip="$LAST_BYTES" count=$((NEW_BYTES - LAST_BYTES)) 2>/dev/null
      LAST_BYTES="$NEW_BYTES"
    fi
    mv "$TMP.new" "$TMP"
  fi
  sleep 10
done
