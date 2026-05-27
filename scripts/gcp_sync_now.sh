#!/bin/bash
# Force an immediate checkpoint/log sync from a live VM.
#
# Usage:
#   ./scripts/gcp_sync_now.sh <INSTANCE_NAME> <ZONE>

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO_ROOT/.tinybit.env" ] && source "$REPO_ROOT/.tinybit.env"

VM="${1:?Usage: $0 <INSTANCE_NAME> <ZONE>}"
ZONE="${2:?Usage: $0 <INSTANCE_NAME> <ZONE>}"
: "${GCP_PROJECT:?Set GCP_PROJECT or create .tinybit.env}"
: "${GCP_BUCKET:?Set GCP_BUCKET or create .tinybit.env}"
GCP_BUCKET="${GCP_BUCKET%/}"

echo "Triggering sync on $VM ($ZONE)…"
gcloud compute ssh "$VM" --zone="$ZONE" --project="$GCP_PROJECT" --command='
  set -uo pipefail
  RUN_ID="$(awk -F\" "/\"run_id\":/ {print \$4; exit}" /var/log/tinybit-status.json 2>/dev/null)"
  if [ -z "$RUN_ID" ]; then echo "no status.json on VM"; exit 1; fi
  PFX="'"$GCP_BUCKET"'/runs/$RUN_ID"
  cd /workspace/tinybit 2>/dev/null || cd /root
  gsutil -q cp /var/log/tinybit-status.json    "$PFX/status.json"           || true
  gsutil -q cp /var/log/tinybit-bootstrap.log  "$PFX/logs/bootstrap.log"    || true
  gsutil -q cp /var/log/tinybit-training.log   "$PFX/logs/training.log"     || true
  if [ -d checkpoints ]; then
    gsutil -m -q rsync -r checkpoints/ "$PFX/checkpoints/" || true
  fi
  echo "synced to $PFX"
'
