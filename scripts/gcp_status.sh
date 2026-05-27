#!/bin/bash
# Show status of a tinybit GCP training run.
#
# Usage:
#   ./scripts/gcp_status.sh                       # uses latest_run.txt
#   ./scripts/gcp_status.sh <RUN_ID>
#
# Reads gs://$GCP_BUCKET/runs/<RUN_ID>/status.json and lists checkpoints and
# tail of training.log. Also shows any matching live VMs.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../.tinybit.env
[ -f "$REPO_ROOT/.tinybit.env" ] && source "$REPO_ROOT/.tinybit.env"

: "${GCP_BUCKET:?Set GCP_BUCKET (gs://...) or create .tinybit.env}"
: "${GCP_PROJECT:?Set GCP_PROJECT or create .tinybit.env}"
GCP_BUCKET="${GCP_BUCKET%/}"

RUN_ID="${1:-}"
if [ -z "$RUN_ID" ]; then
  RUN_ID="$(gsutil -q cat "$GCP_BUCKET/latest_run.txt" 2>/dev/null | tr -d '[:space:]' || true)"
  if [ -z "$RUN_ID" ]; then
    echo "No RUN_ID given and gs://.../latest_run.txt is empty." >&2
    exit 1
  fi
fi

PREFIX="$GCP_BUCKET/runs/$RUN_ID"
echo "RUN_ID  : $RUN_ID"
echo "bucket  : $PREFIX"
echo

echo "== status.json =="
gsutil -q cat "$PREFIX/status.json" 2>/dev/null || echo "(no status.json yet)"
echo

echo "== launch.json =="
gsutil -q cat "$PREFIX/launch.json" 2>/dev/null || echo "(no launch.json)"
echo

for marker in DONE FAILED; do
  if gsutil -q stat "$PREFIX/$marker.json" 2>/dev/null; then
    echo "== $marker.json =="
    gsutil -q cat "$PREFIX/$marker.json"
    echo
  fi
done

echo "== checkpoints =="
gsutil -q ls "$PREFIX/checkpoints/" 2>/dev/null | tail -20 || echo "(none)"
echo

echo "== training.log (tail 30) =="
gsutil -q cp "$PREFIX/logs/training.log" /tmp/_tinybit_tail.log 2>/dev/null \
  && tail -30 /tmp/_tinybit_tail.log \
  || echo "(no training.log yet)"
echo

echo "== live VMs matching tinybit =="
gcloud compute instances list --project="$GCP_PROJECT" \
  --filter="name~^tinybit-" \
  --format='table(name,zone,status,machineType.basename(),scheduling.provisioningModel)' \
  2>/dev/null || true
