#!/bin/bash
# CPU training on n2-highcpu-32 spot instance.
# Slower but usually cheaper than GPU. Best for nano/micro models.
# n2-highcpu-32: 32 vCPUs, 32GB RAM.
# nano model (10M): ~2-3 days. micro (50M): ~1-2 weeks.
#
# Usage: ./scripts/gcp_spot_train.sh [nano|micro]

set -euo pipefail

MODEL_SIZE="${1:-nano}"
PROJECT="${GCP_PROJECT:-your-gcp-project-id}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-${REGION}-a}"
INSTANCE_NAME="tiny-bit-cpu-$(date +%s)"
BUCKET="${GCP_BUCKET:-gs://your-bucket-tiny-bit}"

echo "Launching CPU spot instance for model: $MODEL_SIZE"

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="n2-highcpu-32" \
  --image-family="debian-12" \
  --image-project="debian-cloud" \
  --boot-disk-size="100GB" \
  --provisioning-model="SPOT" \
  --instance-termination-action="STOP" \
  --scopes="storage-full" \
  --metadata=startup-script="
#!/bin/bash
set -euo pipefail
apt-get update -q && apt-get install -y curl build-essential screen
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source \$HOME/.cargo/env

gsutil -m cp -r ${BUCKET}/tiny-bit/ /workspace/
cd /workspace/tiny-bit

RUSTFLAGS='-C target-cpu=native' cargo build --release -p tiny-bit-cli
./scripts/prepare_data.sh data/

screen -dm -S train bash -c '
  ./target/release/tiny-bit train \
    --model-config configs/${MODEL_SIZE}.toml \
    --train-config configs/train.toml \
    --resume 2>&1 | tee training.log
'

while true; do sleep 600; gsutil -m rsync -r checkpoints/ ${BUCKET}/checkpoints/ || true; done
"

echo "CPU spot instance $INSTANCE_NAME created."
