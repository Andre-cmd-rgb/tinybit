# Running a training job that produces non-garbage output

The earlier overnight run produced garbage for **three reasons**, none of
them related to the model code:

1. **Too few parameters for the data budget.** It used `nano` (10M params),
   which struggles to produce coherent English even at large data scales.
2. **Too few tokens.** Only ~17M tokens reached the trainer — Chinchilla
   would want ~200M for a 10M-param model, and our gated/buggy dataset spec
   capped collection well below the 20M target.
3. **Too few steps.** 2k steps × ~16k tokens/step = 32M token-passes, ie one
   third of an epoch over the data we did get.

This file is the recipe that fixes all three.

## TL;DR

Pick the training config that matches the GPU you're using. Each one is
sized so the GPU is well-utilized but won't OOM.

| Model preset | GPU profile | Train config                     | Time / Cost (SPOT)  |
|--------------|-------------|----------------------------------|---------------------|
| `nano`       | any         | `configs/train.toml`             | for smoke tests     |
| `micro`      | l4 / a100   | `configs/train-quality.toml`     | ~14h L4 / ~4h A100  |
| `small`      | l4          | `configs/train-small-l4.toml`    | ~30-40h, ~$10-15    |
| `small`      | a100 (40GB) | `configs/train-small-a100.toml`  | ~15-20h, ~$17-22    |
| `small`      | h100 / a100-80 | `configs/train-small-h100.toml` | ~6-10h H100, ~14-18h A100-80 |

```bash
export GCP_PROJECT=tinybit-run-0
export GCP_BUCKET=gs://tinybit-run-0-tinybit       # your bucket
export HF_TOKEN=hf_xxx                              # optional — see below

# 150M model, L4 SPOT only (works without quota requests):
DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-small-l4.toml \
PROFILE=l4 \
PROVISIONING_MODEL=SPOT \
  ./scripts/gcp_launch.sh small

# 150M model, A100 SPOT (requires A100 quota — see "Quota requests" below):
DATA_TOKENS=2000000000 \
TRAIN_CONFIG=configs/train-small-a100.toml \
PROFILE=a100,a100-80 \
PROVISIONING_MODEL=SPOT \
  ./scripts/gcp_launch.sh small

# 150M model, H100/A100-80 SPOT:
DATA_TOKENS=2000000000 \
TRAIN_CONFIG=configs/train-small-h100.toml \
PROFILE=h100,a100-80 \
PROVISIONING_MODEL=SPOT \
  ./scripts/gcp_launch.sh small
```

## Quota requests

New GCP projects start with **zero quota** for A100 / A100-80 / H100 (this
is GCP's default, not specific to this project). L4 and T4 work out of the
box; everything bigger needs a quota request first.

Direct quota console (filter the page for the GPU you want):

```
https://console.cloud.google.com/iam-admin/quotas?project=$GCP_PROJECT
```

Search for the exact metric name from the table below, tick its checkbox,
"Edit Quotas", request limit `1` in a US region. First requests are
usually auto-approved within minutes for spot quota, hours for on-demand.

| GPU                | On-demand metric                  | SPOT metric                              |
|--------------------|-----------------------------------|-------------------------------------------|
| A100 40GB          | `NVIDIA A100 GPUs`                | `Preemptible NVIDIA A100 GPUs`            |
| A100 80GB          | `NVIDIA A100 80GB GPUs`           | `Preemptible NVIDIA A100 80GB GPUs`       |
| H100 80GB          | `NVIDIA H100 GPUs`                | `Preemptible NVIDIA H100 GPUs`            |

Inspect what you currently have from the CLI:

```bash
for r in us-central1 us-east1 us-east4 us-east5 us-west1 us-west4; do
  echo "=== $r ==="
  gcloud compute regions describe "$r" --project="$GCP_PROJECT" \
    --format='value(quotas)' 2>/dev/null | tr ';' '\n' \
    | grep -E "A100|H100|NVIDIA_L4_GPUS|NVIDIA_T4_GPUS" \
    | grep -v "limit=0.0$"
done
```

### Hardware profiles

| Profile   | Machine          | GPU                | Mem  | On-demand* | Spot* | Relative speed (vs L4) |
|-----------|------------------|--------------------|------|------------|-------|------------------------|
| `t4`      | n1-standard-4    | nvidia-tesla-t4    | 16GB | ~$0.35/hr  | ~$0.11| 0.5×                    |
| `l4`      | g2-standard-4    | nvidia-l4          | 24GB | ~$0.71/hr  | ~$0.22| 1× (baseline)           |
| `a100`    | a2-highgpu-1g    | nvidia-tesla-a100  | 40GB | ~$3.67/hr  | ~$1.10| 3-4×                    |
| `a100-80` | a2-ultragpu-1g   | nvidia-a100-80gb   | 80GB | ~$5.07/hr  | ~$1.50| 3-4× (more headroom)    |
| `h100`    | a3-highgpu-1g    | nvidia-h100-80gb   | 80GB | ~$11.00/hr | ~$3.30| 6-8×                    |

\* approximate, US zones, varies by region; the launcher logs the per-attempt cost hint.

`PROFILE` is a comma-separated priority list. The launcher tries each
profile across every candidate zone, then moves to the next profile.
A100/H100 capacity is much tighter than L4, so listing fallbacks is
recommended.

### Provisioning fallback

`PROVISIONING_MODEL` is also a comma-separated list. Common pattern:

```
PROVISIONING_MODEL=STANDARD,SPOT
```

The launcher runs the whole (profile × zone) grid on on-demand first;
if everything is sold out it retries the same grid on spot. Spot can
be evicted at any time — your checkpoints continue uploading to GCS,
so re-launching after eviction resumes from where you left off.

You can also flip the order to prefer cheap-spot first:
`PROVISIONING_MODEL=SPOT,STANDARD`.

Watch it:

```bash
./scripts/gcp_status.sh                          # uses latest_run.txt
./scripts/gcp_tail_logs.sh <RUN_ID> training     # once training starts
```

## What `configs/train-quality.toml` does

```
batch_size     = 4       # × max_seq_len 1024 (from configs/micro.toml)
grad_accum     = 8       # → effective batch 32  →  32 * 1024 = 32_768 tokens / step
total_steps    = 30000   # → 30000 * 32_768 ≈ 983M token-passes
peak_lr        = 3e-4    # WSD: 2% warmup, 78% stable, 20% cosine decay to 3e-5
weight_decay   = 0.01
grad_clip      = 1.0     # global L2 norm clipping (actually applied now)

save_every     = 500     # ~60 checkpoints over the run; pruned to best-3 + recent-3
eval_every     = 200
eval_batches   = 20
```

Paired with `configs/micro.toml` (50M params), this is roughly Chinchilla-
optimal (20 toks/param) — enough for genuinely coherent English, simple
question following, and recognizable knowledge from the FineWeb-Edu /
Wikipedia / OpenHermes mix.

If you're tighter on time:

| Budget       | Hardware | Model  | DATA_TOKENS | total_steps | Expected output                      |
|--------------|----------|--------|-------------|-------------|--------------------------------------|
| ~3-4h L4     | l4       | nano   | 200M        | 6000        | Coherent words, no real reasoning    |
| ~5-7h L4     | l4       | micro  | 500M        | 15000       | Sentences, weak QA, some facts       |
| ~10-14h L4   | l4       | micro  | 1B          | 30000       | Paragraphs, QA, basic instructions   |
| ~3-4h A100   | a100     | micro  | 1B          | 30000       | Same as 14h L4 above, ~4× faster     |
| ~6-8h A100   | a100-80  | small  | 2B          | 45000       | Genuinely useful output for a 150M   |
| ~24h L4      | l4       | small  | 3B          | 60000       | Better-than-nano "useful" output     |
| ~6h H100     | h100     | small  | 3B          | 60000       | Same as 24h L4, ~4× faster           |

Override per-run with env vars:

```bash
DATA_TOKENS=500000000 \
TRAIN_STEPS=15000 \
TRAIN_CONFIG=configs/train-quality.toml \
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

## What was wrong before, and what's been fixed

- ✅ `grad_clip` is now actually applied (was previously ignored). Global L2
  clipping prevents loss spikes from poisoning Adam's second-moment estimates.
- ✅ NaN/Inf losses now skip the optimizer step instead of corrupting the
  weights. A `tracing::warn` is emitted with the step number.
- ✅ Checkpoint pruning is wired into the trainer (keeps best-3 by val loss
  and most-recent-3 by step). Disk usage stays bounded.
- ✅ Failed datasets in `prepare_data.sh` redistribute their token budget so
  total collection actually hits the target.
- ✅ `MIN_TOKENS` floor fails fast before paying for the GPU if data prep
  collected too little.
- ✅ Chat/serve commands no longer crash on the vocab mismatch — the
  tokenizer is vocab-aware and the chat template uses plain-text role
  markers.

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
  by step 30k.
- `gnorm` log values mostly between 0.05 and 2.0; rarely above 5.0.
- Final chat output is grammatical English on most prompts, even if
  factually unreliable.

Unhealthy signs to investigate:

- `gnorm` consistently above 10 → reduce `peak_lr` to 1e-4.
- `loss` flat or rising → check the data — usually means `train.bin`
  contains zeros or a tokenization mismatch.
- Many `skipping update — non-finite` warnings → reduce `peak_lr` and/or
  `grad_clip`.
