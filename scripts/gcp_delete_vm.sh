#!/bin/bash
# Delete a tinybit training VM (and its boot disk). Asks for confirmation
# unless FORCE=1. Always free the GPU when you're done debugging.
#
# Usage:
#   ./scripts/gcp_delete_vm.sh <INSTANCE_NAME> <ZONE>

set -Eeuo pipefail

VM="${1:?Usage: $0 <INSTANCE_NAME> <ZONE>}"
ZONE="${2:?Usage: $0 <INSTANCE_NAME> <ZONE>}"
: "${GCP_PROJECT:?Set GCP_PROJECT}"

echo "About to DELETE instance: $VM in $ZONE  (project: $GCP_PROJECT)"
echo "This will release the GPU and destroy the boot disk."
if [ "${FORCE:-0}" != "1" ]; then
  printf "Type the instance name to confirm: "
  read -r ans
  [ "$ans" = "$VM" ] || { echo "aborted (you typed: $ans)"; exit 1; }
fi

gcloud compute instances delete "$VM" \
  --zone="$ZONE" --project="$GCP_PROJECT" --quiet
echo "Deleted $VM."
