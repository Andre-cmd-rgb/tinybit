#!/bin/bash
# Fast GCP CUDA smoke train on a spot L4 GPU.
#
# Requirements:
#   export GCP_PROJECT="your-project-id"
#   export GCP_BUCKET="gs://your-bucket"
#   gsutil -m rsync -r . "$GCP_BUCKET/tiny-bit"
#
# Usage: ./scripts/gcp_smoke_train.sh [nano|micro]

set -euo pipefail

MODEL_SIZE="${1:-nano}"
: "${GCP_PROJECT:?Set GCP_PROJECT to your Google Cloud project id}"
: "${GCP_BUCKET:?Set GCP_BUCKET to a gs:// bucket containing tiny-bit/}"

REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-${REGION}-a}"
INSTANCE_NAME="tiny-bit-smoke-$(date +%s)"

echo "Launching fast CUDA smoke train for model: $MODEL_SIZE"
echo "Project: $GCP_PROJECT | Zone: $ZONE | Bucket: $GCP_BUCKET"

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$ZONE" \
  --machine-type="g2-standard-8" \
  --image-family="common-cu129-ubuntu-2204-nvidia-580" \
  --image-project="deeplearning-platform-release" \
  --boot-disk-size="100GB" \
  --boot-disk-type="pd-ssd" \
  --provisioning-model="SPOT" \
  --instance-termination-action="STOP" \
  --maintenance-policy="TERMINATE" \
  --scopes="storage-full" \
  --metadata=startup-script="
#!/bin/bash
set -euo pipefail
apt-get update -q && apt-get install -y curl build-essential screen
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source \$HOME/.cargo/env
export PATH=\"/usr/local/cuda/bin:\$PATH\"
export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}\"
nvidia-smi
nvcc --version

gsutil -m cp -r ${GCP_BUCKET}/tiny-bit/ /workspace/
cd /workspace/tiny-bit

RUSTFLAGS='-C target-cpu=native' cargo build --release -p tiny-bit-cli --features cuda

python3 - <<'PYTHON'
from pathlib import Path

def write_tokens(path, chunks, seq_len):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        for i in range(chunks * seq_len):
            token = 1 + (i % 997)
            f.write(token.to_bytes(4, 'little'))

write_tokens('data/train.bin', chunks=128, seq_len=512)
write_tokens('data/val.bin', chunks=16, seq_len=512)
PYTHON

cat > configs/train-smoke.toml <<'TOML'
train_data     = \"data/train.bin\"
val_data       = \"data/val.bin\"
checkpoint_dir = \"checkpoints-smoke/\"

batch_size     = 2
grad_accum     = 1
total_steps    = 20
peak_lr        = 3e-4
weight_decay   = 0.01
grad_clip      = 1.0

save_every     = 10
eval_every     = 5
eval_batches   = 2

smoke_test_steps = 20
TOML

./target/release/tiny-bit train \
  --model-config configs/${MODEL_SIZE}.toml \
  --train-config configs/train-smoke.toml \
  --smoke-test 2>&1 | tee training-smoke.log

gsutil -m rsync -r checkpoints-smoke/ ${GCP_BUCKET}/checkpoints-smoke/ || true
gsutil cp training-smoke.log ${GCP_BUCKET}/training-smoke.log || true
echo 'Smoke train complete; shutting down instance.'
shutdown -h now
"

echo "Instance $INSTANCE_NAME created."
echo "Serial log: gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE --project=$GCP_PROJECT"
echo "SSH:        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$GCP_PROJECT"
