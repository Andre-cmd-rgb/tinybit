# Training Guide

tinybit trains RWKV-7 language models on GCP using a single GPU VM. This guide covers
everything from data preparation to launching a run and monitoring it.

---

## Quick start — 150M model on L4

```bash
export GCP_PROJECT=your-project-id
export GCP_BUCKET=gs://your-bucket

DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-small-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT \
  ./scripts/gcp_launch.sh small
```

That's it. The launcher finds the first available zone with an L4, uploads the repo,
installs CUDA 12.8, prepares ~1 B tokens of training data, compiles the binary, and starts
training — all unattended. Checkpoints sync to `$GCP_BUCKET/runs/<run_id>/checkpoints/`
every 120 s.

---

## Hardware and cost

| GPU   | Machine         | VRAM  | RAM  | On-demand | SPOT       | Typical run (150 M, 30 k steps) |
|-------|-----------------|-------|------|-----------|------------|----------------------------------|
| T4    | n1-standard-4   | 16 GB | 15 G | ~$0.35/hr | ~$0.11/hr  | 55–65 h · **$20–24** (SPOT: $6–8) |
| **L4**| g2-standard-4   | 24 GB | 16 G | ~$0.71/hr | ~$0.22/hr  | 30–40 h · **$22–29** (SPOT: $7–9) |
| A100  | a2-highgpu-1g   | 40 GB | 85 G | ~$3.67/hr | ~$1.10/hr  | 10–15 h · **$37–55** (SPOT: $11–17) |
| H100  | a3-highgpu-1g   | 80 GB | 234G | ~$11/hr   | ~$3.30/hr  | 4–6 h   · **$44–66** (SPOT: $13–20) |

Costs are estimates for US zones. SPOT prices are roughly 30 % of on-demand but the VM can
be preempted — training resumes from the latest checkpoint on relaunch, so multi-day L4 SPOT
runs typically cost $10–15 all-in after 1–2 preemptions.

**Recommended for most training:** L4 SPOT. Best cost-efficiency, 24 GB VRAM is enough for
the 150 M model with `batch_size=8`, and the 200 GB SSD boot disk keeps data prep fast.

---

## Model sizes

| Name  | Params | Layers | d_model | d_ffn | Recommended GPU |
|-------|--------|--------|---------|-------|-----------------|
| nano  | ~10 M  | 6      | 256     | 512   | T4 or CPU       |
| micro | ~50 M  | 12     | 512     | 1024  | T4 or L4        |
| small | ~150 M | 18     | 768     | 2048  | **L4** (this guide) |
| base  | ~400 M | 32     | 1024    | 2048  | A100 or H100    |

---

## Configuration files

### Model config (`configs/small.toml`)
Defines the model architecture. Do not change `vocab_size` or `max_seq_len` after
training starts — these are baked into checkpoints.

### Training config (`configs/train-small-l4.toml`)
Controls the training run. Key fields:

```toml
batch_size  = 8     # sequences per microbatch (per GPU forward pass)
grad_accum  = 4     # microbatches before one optimizer step
total_steps = 30000 # optimizer steps to run
peak_lr     = 3e-4  # warmup target and stable-phase LR
grad_clip   = 1.0   # global L2 norm clip
save_every  = 500   # save checkpoint every N optimizer steps
eval_every  = 500   # compute val loss every N optimizer steps
```

**Token budget:**
```
effective_batch = batch_size × seq_len × grad_accum
               = 8 × 1024 × 4 = 32,768 tokens/step

total_tokens = total_steps × effective_batch
             = 30,000 × 32,768 ≈ 983 M ≈ 1 B tokens
```

Set `DATA_TOKENS=1500000000` (50 % headroom over the training budget) so the data
preparation collects enough tokens even if some datasets fail or are skipped.

---

## Launch options

```bash
./scripts/gcp_launch.sh [nano|micro|small|base]
```

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT` | (required) | GCP project ID |
| `GCP_BUCKET` | (required) | `gs://bucket-name` |
| `PROFILE` | `l4,t4` | GPU profiles to try, in order |
| `PROVISIONING_MODEL` | `STANDARD` | `STANDARD`, `SPOT`, or `STANDARD,SPOT` |
| `DATA_TOKENS` | `20000000` | Target training tokens |
| `MIN_TOKENS` | 75% of DATA_TOKENS | Abort if fewer are collected |
| `TRAIN_CONFIG` | (inline default) | Path to a checked-in train config TOML |
| `TRAIN_STEPS` | `2000` | Steps (ignored when `TRAIN_CONFIG` is set) |
| `HF_TOKEN` | (unset) | HuggingFace token for gated datasets |
| `KEEP_VM_ON_FAILURE` | `0` | Set to `1` to keep VM alive on error |
| `SYNC_INTERVAL` | `120` | Seconds between checkpoint syncs to GCS |

**Example — 150M model, prefer SPOT, fall back to on-demand:**
```bash
GCP_PROJECT=my-project \
GCP_BUCKET=gs://my-bucket \
PROFILE=l4,t4 \
PROVISIONING_MODEL=SPOT,STANDARD \
DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-small-l4.toml \
  ./scripts/gcp_launch.sh small
```

**Example — fastest available GPU (cost-no-object):**
```bash
PROFILE=h100,a100-80,a100,l4 \
PROVISIONING_MODEL=STANDARD \
DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-small-l4.toml \
  ./scripts/gcp_launch.sh small
```

---

## Monitoring

```bash
# Watch GCS-synced status
./scripts/gcp_status.sh <run_id>

# Tail training log in real time from GCS
./scripts/gcp_tail_logs.sh <run_id>

# SSH into the VM
gcloud compute ssh tinybit-l4-YYYYMMDD-HHMMSS --zone=us-central1-a --project=my-project
```

The training log prints one line per optimizer step:
```
step=1000 lr=3.00e-04 loss=3.2847 gnorm=0.842
step=1000 val_loss=3.3112
```

Expected loss trajectory for a 150M model trained on 1 B tokens:
- Step 500:   loss ≈ 5.5–6.0 (early warmup)
- Step 2000:  loss ≈ 4.0–4.5
- Step 10000: loss ≈ 3.0–3.5
- Step 30000: loss ≈ 2.4–2.8 (language modeling perplexity ~11–16)

---

## Resuming after preemption

SPOT VMs use `--instance-termination-action=STOP`, so the disk is preserved on
preemption. To resume, simply restart the same instance:

```bash
gcloud compute instances start tinybit-l4-YYYYMMDD-HHMMSS \
  --zone=us-central1-a --project=my-project
```

The startup script re-runs on every boot. It skips data prep and cargo build if
the outputs are already on disk, downloads any GCS checkpoints missing locally,
and restarts training with `--resume` from the last checkpoint.

**If the VM was deleted** (e.g., after hitting the maximum preemption count or
a manual restart via `gcp_launch.sh` with the same `RUN_ID`):

```bash
RUN_ID=20260527-091332-small \
DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-small-l4.toml \
  ./scripts/gcp_launch.sh small
```

The `restore_checkpoints` stage downloads existing checkpoints from
`gs://bucket/runs/<run_id>/checkpoints/` onto the new VM before training
starts. Training resumes automatically from the latest saved step.

To force a clean restart from step 0, set `FORCE_DATA=1` or `FORCE_REBUILD=1`,
or use a different `RUN_ID`.

---

## Data preparation

Data is prepared by `scripts/prepare_data.sh` and produces two binary files:

- `data/train.bin` — training tokens (uint32, little-endian)
- `data/val.bin`   — validation tokens (uint32, little-endian)

**Memory design:** tokens are written directly to disk using numpy (4 bytes/token).
Peak RAM during data prep is bounded to ~16 MB (one flush buffer) regardless of total
token count. A 32 GB swap file is created early in startup as an additional safety net.

Datasets used (all streamed, no full download):
- **FineWeb-Edu** (40%): high-quality web text filtered for educational content
- **Wikipedia EN** (30%): encyclopedia articles
- **OpenHermes-2.5** (15%): instruction-following conversations
- **dolphin-r1 nonreasoning** (10%): chat data
- **the-stack-smol Python** (5%): code (requires `HF_TOKEN`)

---

## Local smoke test

Verify the training pipeline works before launching on GCP:

```bash
# Prepare a tiny dataset locally
TOTAL_TOKENS=200000 MIN_TOKENS=100000 ./scripts/prepare_data.sh data/

# Run 200 steps to check the pipeline
cargo build --release -p tinybit-cli
./target/release/tinybit train \
  --model-config configs/nano.toml \
  --train-config configs/train.toml \
  --smoke-test
```

The smoke test should complete in under 5 minutes on CPU, ending at loss < 8.

---

## Checkpoint format

Checkpoints are pairs of files in `checkpoints/`:
- `step_NNNNNNN.safetensors` — model weights
- `step_NNNNNNN.json` — metadata (step, loss, tokens_seen, config, timestamp)

The trainer keeps the **3 best** (lowest val_loss) and the **3 most recent** checkpoints,
deleting the rest to prevent disk fill-up on long runs.
