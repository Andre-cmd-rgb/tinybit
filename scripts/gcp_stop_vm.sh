#!/bin/bash
# Stop (but don't delete) a tinybit training VM. Useful if you want to resume
# later from the same disk. Asks for confirmation unless FORCE=1.
#
# Usage:
#   ./scripts/gcp_stop_vm.sh <INSTANCE_NAME> <ZONE>

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO_ROOT/.tinybit.env" ] && source "$REPO_ROOT/.tinybit.env"

VM="${1:?Usage: $0 <INSTANCE_NAME> <ZONE>}"
ZONE="${2:?Usage: $0 <INSTANCE_NAME> <ZONE>}"
: "${GCP_PROJECT:?Set GCP_PROJECT or create .tinybit.env}"

echo "About to stop instance: $VM in $ZONE  (project: $GCP_PROJECT)"
if [ "${FORCE:-0}" != "1" ]; then
  printf "Type 'stop' to confirm: "
  read -r ans
  [ "$ans" = "stop" ] || { echo "aborted"; exit 1; }
fi

gcloud compute instances stop "$VM" \
  --zone="$ZONE" --project="$GCP_PROJECT"
echo "Stopped $VM."
