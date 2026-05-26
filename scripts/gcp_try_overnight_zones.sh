#!/bin/bash
# Try multiple GPU zones until an overnight training VM starts.
#
# Required env:
#   GCP_PROJECT=tinybit-run-0
#   GCP_BUCKET=gs://tinybit-run-0-tiny-bit
#
# Optional env:
#   GCP_MACHINE_TYPE=n1-standard-4
#   GCP_ACCELERATOR_TYPE=nvidia-tesla-t4
#   DATA_TOKENS=20000000
#   TRAIN_STEPS=2000
#   PROVISIONING_MODEL=STANDARD
#   GCP_ZONES="europe-west1-b europe-west1-c ..."
#
# Usage: ./scripts/gcp_try_overnight_zones.sh [nano|micro|small]

set -euo pipefail

MODEL_SIZE="${1:-nano}"
: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GCP_BUCKET:?Set GCP_BUCKET}"

DEFAULT_ZONES=(
  europe-west1-b
  europe-west1-c
  europe-west4-a
  europe-west4-b
  europe-west4-c
  europe-west3-a
  europe-west3-b
  europe-west2-a
  europe-west2-b
  europe-west6-b
  europe-west6-c
  us-central1-a
  us-central1-b
  us-central1-c
  us-west1-a
  us-west1-b
  us-west1-c
  us-east1-b
  us-east1-c
  us-east1-d
  us-east4-a
  us-east4-c
  us-west4-a
  us-west4-c
)

if [ -n "${GCP_ZONES:-}" ]; then
  read -r -a ZONES <<< "$GCP_ZONES"
else
  ZONES=("${DEFAULT_ZONES[@]}")
fi

for zone in "${ZONES[@]}"; do
  echo "Trying zone: $zone"
  if GCP_ZONE="$zone" ./scripts/gcp_overnight_train.sh "$MODEL_SIZE"; then
    echo "Started training in $zone"
    exit 0
  fi
  echo "Zone failed: $zone"
  sleep 5
done

echo "No listed zone had capacity. Try again later or request a reservation/quota increase." >&2
exit 1
