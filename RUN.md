# Running a training job that produces non-garbage output

`tinybit` trains on **NVIDIA L4 only** (`g2-standard-4` on GCP). L4 has 24 GB
VRAM, broad zone availability, and ~$0.22/hr SPOT — the cheapest GPU that
fits the 50M `micro` model with sequence length 512 and the RWKV-7 sequential
WKV scan in candle.

## TL;DR

Pick the training config that matches the model preset. Each is sized so the
L4 is well-utilized but won't OOM.

| Model preset | Train config                  | Time / Cost (SPOT) |
|--------------|-------------------------------|--------------------|
| `nano`       | `configs/train-nano-l4.toml`  | ~5-7 h, ~$1-2      |
| `micro`      | `configs/train-micro-l4.toml` | ~15-22 h, ~$4-6    |

`small` (258M) and `base` (501M) are kept as architectural presets for
inference of pre-trained checkpoints but do not fit on a single L4 for
training — they were the targets of the deleted A100/H100 train configs.

```bash
export GCP_PROJECT=tinybit-run-0
export GCP_BUCKET=gs://tinybit-run-0       # your bucket
export HF_TOKEN=hf_xxx                     # optional — see below

# 50M micro (main target):
DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-micro-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT \
  ./scripts/gcp_launch.sh micro

# 25M nano (smoke / fast iteration):
DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-nano-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT \
  ./scripts/gcp_launch.sh nano
```

## Quota

L4 quota is granted out of the box on most GCP projects — no quota request
needed. If `gcloud compute instances create` fails with `Quota 'NVIDIA_L4_GPUS'
exceeded`, request a limit of 1 in any US/EU region from the console:

```
https://console.cloud.google.com/iam-admin/quotas?project=$GCP_PROJECT
```

For SPOT specifically, the metric is `Preemptible NVIDIA L4 GPUs`.

Inspect what you currently have from the CLI:

```bash
for r in us-central1 us-east1 us-east4 us-east5 us-west1 us-west4 \
         europe-west1 europe-west3 europe-west4; do
  echo "=== $r ==="
  gcloud compute regions describe "$r" --project="$GCP_PROJECT" \
    --format='value(quotas)' 2>/dev/null | tr ';' '\n' \
    | grep -E "NVIDIA_L4_GPUS" \
    | grep -v "limit=0.0$"
done
```

### Hardware

| Profile | Machine        | GPU       | Mem  | On-demand* | Spot*   |
|---------|----------------|-----------|------|------------|---------|
| `l4`    | g2-standard-4  | nvidia-l4 | 24GB | ~$0.71/hr  | ~$0.22  |

\* approximate, US zones.

### Provisioning fallback

`PROVISIONING_MODEL` is a comma-separated list. Common pattern:

```
PROVISIONING_MODEL=STANDARD,SPOT
```

The launcher walks every zone on on-demand first; if everything is sold out
it retries the same zones on spot. Spot can be evicted at any time — your
checkpoints continue uploading to GCS, so re-launching after eviction resumes
from where you left off.

You can also flip the order to prefer cheap-spot first:
`PROVISIONING_MODEL=SPOT,STANDARD`.

Watch it:

```bash
./scripts/gcp_status.sh                          # uses latest_run.txt
./scripts/gcp_tail_logs.sh <RUN_ID> training     # once training starts
```

## What `configs/train-micro-l4.toml` does

```
batch_size     = 2       # × max_seq_len 512 (from configs/micro.toml)
grad_accum     = 32      # → effective batch 64  →  64 * 512 = 32_768 tokens / step
total_steps    = 25000   # → 25000 * 32_768 ≈ 819M token-passes
peak_lr        = 3e-4    # WSD: 2% warmup, 78% stable, 20% cosine decay to 3e-5
weight_decay   = 0.01
grad_clip      = 1.0     # global L2 norm clipping

save_every     = 500     # ~50 checkpoints over the run; pruned to best-3 + recent-3
eval_every     = 500
eval_batches   = 20
```

Paired with `configs/micro.toml` (50M params), this is roughly Chinchilla-
optimal (16 toks/param) — enough for genuinely coherent English, simple
question following, and recognizable knowledge from the FineWeb-Edu /
Wikipedia / OpenHermes mix.

Why the small `batch_size`: candle's sequential WKV scan in
`crates/tinybit-core/src/model/time_mix.rs` retains every intermediate
state in the autograd graph for backward, so peak training VRAM scales
linearly with `batch_size × max_seq_len × num_layers`. On a 24 GB L4 with
the 16-layer 50M micro, `batch_size = 2` at `max_seq_len = 512` is the
largest microbatch that reliably fits.

If you're tighter on time:

| Budget   | Model | DATA_TOKENS | total_steps | Expected output                    |
|----------|-------|-------------|-------------|------------------------------------|
| ~3-4h L4 | nano  | 200M        | 6000        | Coherent words, no real reasoning  |
| ~5-7h L4 | nano  | 1B          | 15000       | Sentences, weak QA, some facts     |
| ~10-15h  | micro | 1B          | 25000       | Paragraphs, QA, basic instructions |
| ~20-30h  | micro | 1.5B        | 50000       | Better instruction following       |

Override per-run with env vars:

```bash
DATA_TOKENS=500000000 \
TRAIN_STEPS=15000 \
  ./scripts/gcp_launch.sh micro
```

(When `TRAIN_CONFIG` is set, `TRAIN_STEPS` is ignored — the value in the
TOML wins. Leave `TRAIN_CONFIG` unset to use the inline generated config
parameterized by `TRAIN_STEPS`.)

## HF_TOKEN

Without `HF_TOKEN`, the data prep skips `bigcode/the-stack-smol` (gated) and
redistributes its 5% weight to the other datasets. That's fine for natural
language quality, but model won't see code. Set `HF_TOKEN` to enable code.
Other datasets in the default mix are not gated.

## Verifying the run produced non-garbage output

```bash
# Pull the final checkpoint from the bucket
gsutil cp "$GCP_BUCKET/runs/<RUN_ID>/checkpoints/step_*.safetensors" \
          /tmp/final.safetensors
gsutil cp "$GCP_BUCKET/runs/<RUN_ID>/checkpoints/step_*.json" \
          /tmp/final.json
mkdir -p checkpoints/final && mv /tmp/final.safetensors /tmp/final.json checkpoints/final/

# Get the tokenizer
curl -sSfL -o tokenizer.json \
  https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.json

# Chat
tinybit chat \
  --config configs/micro.toml \
  --model checkpoints/final/<filename>.safetensors \
  --tokenizer tokenizer.json
```

Healthy signs:

- `val_loss` log lines decrease steadily from ~10.4 (the random init for
  vocab=32008) to under 5.0 by step 5k, under 4.0 by step 15k, under 3.5
  by step 25k.
- `gnorm` log values mostly between 0.05 and 2.0; rarely above 5.0.
- Final chat output is grammatical English on most prompts, even if
  factually unreliable.

Unhealthy signs to investigate:

- `gnorm` consistently above 10 → reduce `peak_lr` to 1e-4.
- `loss` flat or rising → check the data — usually means `train.bin`
  contains zeros or a tokenization mismatch.
- Many `skipping update — non-finite` warnings → reduce `peak_lr` and/or
  `grad_clip`.
- `DriverError(CUDA_ERROR_OUT_OF_MEMORY)` at training start → another
  process is holding the GPU. The startup script's `free_gpu_memory`
  stage stops MPS and kills GPU processes; if it fails, SSH in and run
  `nvidia-smi` to confirm.
