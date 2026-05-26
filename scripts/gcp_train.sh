#!/bin/bash
# GCP training script — spot L4 GPU.
#
# Requirements: gcloud CLI installed and authenticated
# Usage: ./scripts/gcp_train.sh [nano|micro|small|base]

set -euo pipefail

MODEL_SIZE="${1:-small}"
PROJECT="${GCP_PROJECT:-your-gcp-project-id}"       # set GCP_PROJECT env var
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-${REGION}-a}"
INSTANCE_NAME="tiny-bit-train-$(date +%s)"
BUCKET="${GCP_BUCKET:-gs://your-bucket-tiny-bit}"    # set GCP_BUCKET env var

echo "Launching training instance for model: $MODEL_SIZE"
echo "Project: $PROJECT | Zone: $ZONE | Bucket: $BUCKET"

# Create preemptible L4 GPU instance
gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="g2-standard-8" \
  --image-family="common-cu129-ubuntu-2204-nvidia-580" \
  --image-project="deeplearning-platform-release" \
  --boot-disk-size="200GB" \
  --boot-disk-type="pd-ssd" \
  --provisioning-model="SPOT" \
  --instance-termination-action="STOP" \
  --maintenance-policy="TERMINATE" \
  --scopes="storage-full" \
  --metadata=startup-script="
#!/bin/bash
set -euo pipefail
apt-get update -q && apt-get install -y curl build-essential screen python3-pip
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source \$HOME/.cargo/env
export PATH=\"/usr/local/cuda/bin:\$PATH\"
export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}\"
nvidia-smi
nvcc --version
python3 -m pip install --break-system-packages datasets tokenizers tqdm || python3 -m pip install --user datasets tokenizers tqdm

# Fetch code from GCS
gsutil -m cp -r ${BUCKET}/tiny-bit/ /workspace/
cd /workspace/tiny-bit

# Build with CUDA support
RUSTFLAGS='-C target-cpu=native' cargo build --release -p tiny-bit-cli --features cuda

# Prepare data
TOTAL_TOKENS=${TOTAL_TOKENS:-500000000} ./scripts/prepare_data.sh data/

# Run smoke test first
echo 'Running smoke test...'
./target/release/tiny-bit train \
  --model-config configs/${MODEL_SIZE}.toml \
  --smoke-test
echo 'Smoke test passed — starting full training...'

# Start training in screen (survives SSH disconnect)
screen -dm -S train bash -c '
  ./target/release/tiny-bit train \
    --model-config configs/${MODEL_SIZE}.toml \
    --train-config configs/train.toml \
    --resume 2>&1 | tee training.log
'

# Sync checkpoints to GCS every 10 minutes
while true; do
  sleep 600
  gsutil -m rsync -r checkpoints/ ${BUCKET}/checkpoints/ || true
done
"

echo "Instance $INSTANCE_NAME created."
echo "To monitor: gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE"
echo "To SSH: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
