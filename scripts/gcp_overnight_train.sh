#!/bin/bash
# Bounded overnight GCP training run on a spot L4 GPU.
#
# Required env:
#   GCP_PROJECT=tinybit-run-0
#   GCP_BUCKET=gs://tinybit-run-0-tiny-bit
#
# Optional env:
#   GCP_ZONE=us-central1-a
#   GCP_MACHINE_TYPE=g2-standard-4
#   DATA_TOKENS=20000000
#   TRAIN_STEPS=2000
#   PROVISIONING_MODEL=STANDARD  # or SPOT
#
# Usage: ./scripts/gcp_overnight_train.sh [nano|micro|small]

set -euo pipefail

MODEL_SIZE="${1:-nano}"
: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GCP_BUCKET:?Set GCP_BUCKET}"

REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-${REGION}-a}"
MACHINE_TYPE="${GCP_MACHINE_TYPE:-g2-standard-4}"
DATA_TOKENS="${DATA_TOKENS:-20000000}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-STANDARD}"
INSTANCE_NAME="tiny-bit-overnight-$(date +%s)"

echo "Launching overnight training for model: $MODEL_SIZE"
echo "Project: $GCP_PROJECT | Zone: $ZONE | Bucket: $GCP_BUCKET"
echo "Machine type: $MACHINE_TYPE"
echo "Data tokens: $DATA_TOKENS | Train steps: $TRAIN_STEPS | Provisioning: $PROVISIONING_MODEL"

INSTANCE_FLAGS=(--maintenance-policy="TERMINATE")
if [ "$PROVISIONING_MODEL" = "SPOT" ]; then
  INSTANCE_FLAGS+=(--provisioning-model="SPOT" --instance-termination-action="STOP")
fi

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="common-cu129-ubuntu-2204-nvidia-580" \
  --image-project="deeplearning-platform-release" \
  --boot-disk-size="200GB" \
  --boot-disk-type="pd-ssd" \
  "${INSTANCE_FLAGS[@]}" \
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

gsutil -m cp -r ${GCP_BUCKET}/tiny-bit/ /workspace/
cd /workspace/tiny-bit

RUSTFLAGS='-C target-cpu=native' cargo build --release -p tiny-bit-cli --features cuda
TOTAL_TOKENS=${DATA_TOKENS} ./scripts/prepare_data.sh data/

cat > configs/train-overnight.toml <<'TOML'
train_data     = \"data/train.bin\"
val_data       = \"data/val.bin\"
checkpoint_dir = \"checkpoints-overnight/\"

batch_size     = 4
grad_accum     = 4
total_steps    = ${TRAIN_STEPS}
peak_lr        = 3e-4
weight_decay   = 0.01
grad_clip      = 1.0

save_every     = 100
eval_every     = 50
eval_batches   = 5

smoke_test_steps = 0
TOML

screen -dm -S train bash -c '
  set -euo pipefail
  RUST_LOG=tiny_bit_train=info ./target/release/tiny-bit train \
    --model-config configs/${MODEL_SIZE}.toml \
    --train-config configs/train-overnight.toml \
    --resume 2>&1 | tee training-overnight.log
  touch /tmp/tiny-bit-train.done
'

while [ ! -f /tmp/tiny-bit-train.done ]; do
  sleep 600
  gsutil -m rsync -r checkpoints-overnight/ ${GCP_BUCKET}/checkpoints-overnight/ || true
  gsutil cp training-overnight.log ${GCP_BUCKET}/training-overnight.log || true
done

gsutil -m rsync -r checkpoints-overnight/ ${GCP_BUCKET}/checkpoints-overnight/ || true
gsutil cp training-overnight.log ${GCP_BUCKET}/training-overnight.log || true
echo 'Overnight training complete; shutting down instance.'
shutdown -h now
"

echo "Instance $INSTANCE_NAME created."
echo "Serial log: gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE --project=$GCP_PROJECT"
echo "SSH:        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$GCP_PROJECT"
