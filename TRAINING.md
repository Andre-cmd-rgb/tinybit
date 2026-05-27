# Training Guide

tinybit trains RWKV-7 language models on a single NVIDIA L4 GPU on GCP. This
guide covers everything from data preparation to launching a run and
monitoring it. **L4 is the only supported hardware** — the launcher,
configs, and startup script are all tuned for `g2-standard-4 + nvidia-l4`.

---

## Quick start

Set your project and bucket once:

```bash
export GCP_PROJECT=your-project-id
export GCP_BUCKET=gs://your-bucket
```

Then pick your model size:

**25M nano (~5–7 h, ~$1–2 SPOT):**
```bash
DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-nano-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT ./scripts/gcp_launch.sh nano
```

**50M micro (~15–22 h, ~$4–6 SPOT) — recommended:**
```bash
DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-micro-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT ./scripts/gcp_launch.sh micro
```

The launcher finds the first available zone with an L4, uploads the repo,
installs CUDA 12.8, prepares ~1 B tokens of training data, compiles the
binary, and starts training — all unattended. Checkpoints sync to
`$GCP_BUCKET/runs/<run_id>/checkpoints/` every 120 s.

---

## Hardware and cost

| GPU   | Machine        | VRAM  | RAM  | On-demand | SPOT       | Typical run (50 M micro, 25 k steps) |
|-------|----------------|-------|------|-----------|------------|--------------------------------------|
| L4    | g2-standard-4  | 24 GB | 16 G | ~$0.71/hr | ~$0.22/hr  | 15–22 h · ~$11 on-demand (SPOT: $4–6) |

Costs are estimates for US zones. SPOT prices are roughly 30 % of on-demand
but the VM can be preempted — training resumes from the latest checkpoint
on relaunch, so SPOT runs typically cost $4–6 all-in after 1–2 preemptions.

---

## Model sizes

All models use a **3.5× FFN expansion ratio** (d_ffn = 3.5 × d_model),
matching the original RWKV-7 paper. This gives significantly more knowledge
capacity per parameter compared to the 2× ratio commonly used in smaller
implementations.

| Name  | Params  | Layers | d_model | d_ffn | FFN ratio | Trainable on L4? |
|-------|---------|--------|---------|-------|-----------|------------------|
| nano  | ~25 M   | 9      | 320     | 1120  | 3.5×      | Yes              |
| micro | ~50 M   | 16     | 384     | 1344  | 3.5×      | Yes              |
| small | ~258 M  | 13     | 1024    | 3584  | 3.5×      | Inference only   |
| base  | ~501 M  | 17     | 1280    | 4480  | 3.5×      | Inference only   |

`small` and `base` remain in the codebase as architectural presets so the
inference path can load pre-trained checkpoints, but training them requires
A100/H100-class hardware that is not supported by the launcher in this
repo.

---

## Configuration files

### Model config (`configs/micro.toml`)
Defines the model architecture. Do not change `vocab_size` or `max_seq_len`
after training starts — these are baked into checkpoints.

```toml
vocab_size  = 32008
num_layers  = 16
d_model     = 384
d_ffn       = 1344
num_heads   = 6
head_dim    = 64
max_seq_len = 512   # L4-tuned: longer sequences OOM in the candle WKV scan
dropout     = 0.05
spec_heads  = 0
ternary_ffn = false
int8_time   = false
```

### Training config (`configs/train-micro-l4.toml`)
Controls the training run. Key fields:

```toml
batch_size  = 6     # sequences per microbatch (per GPU forward pass)
grad_accum  = 11    # microbatches before one optimizer step
total_steps = 25000 # optimizer steps to run
peak_lr     = 3e-4  # warmup target and stable-phase LR
grad_clip   = 1.0   # global L2 norm clip
save_every  = 500   # save checkpoint every N optimizer steps
eval_every  = 500   # compute val loss every N optimizer steps
```

**Token budget:**
```
effective_batch = batch_size × seq_len × grad_accum
               = 6 × 512 × 11 = 33_792 tokens/step

total_tokens = total_steps × effective_batch
             = 25_000 × 33_792 ≈ 845 M tokens
```

Set `DATA_TOKENS=1500000000` (~75 % headroom over the training budget) so
data preparation collects enough tokens even if some datasets fail or are
skipped.

Why `batch_size = 6`: the RWKV-7 WKV scan in
`crates/tinybit-core/src/model/time_mix.rs` is a candle-side sequential
loop that retains every intermediate state in the autograd graph for
backward. Peak training VRAM scales linearly with
`batch_size × max_seq_len × num_layers`. On a 24 GB L4 with the 16-layer
50M micro at `max_seq_len = 512`, `batch_size = 6` peaks at ~19.5 GB with
~3 GB headroom — a balance between throughput and OOM safety. B=4 is
safer but ~30 % slower; B=8 lands right at the OOM cliff under candle's
pool allocator overhead.

---

## Launch options

```bash
./scripts/gcp_launch.sh [nano|micro]
```

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT` | (required) | GCP project ID |
| `GCP_BUCKET` | (required) | `gs://bucket-name` |
| `PROVISIONING_MODEL` | `STANDARD` | `STANDARD`, `SPOT`, or `STANDARD,SPOT` |
| `DATA_TOKENS` | `20000000` | Target training tokens |
| `MIN_TOKENS` | 75% of DATA_TOKENS | Abort if fewer are collected |
| `TRAIN_CONFIG` | (inline default) | Path to a checked-in train config TOML |
| `TRAIN_STEPS` | `2000` | Steps (ignored when `TRAIN_CONFIG` is set) |
| `HF_TOKEN` | (unset) | HuggingFace token for gated datasets |
| `KEEP_VM_ON_FAILURE` | `0` | Set to `1` to keep VM alive on error |
| `SYNC_INTERVAL` | `120` | Seconds between checkpoint syncs to GCS |
| `GCP_ZONES` | (broad list) | Space-separated override of zones to try |

**Example — prefer SPOT, fall back to on-demand:**
```bash
GCP_PROJECT=my-project \
GCP_BUCKET=gs://my-bucket \
PROVISIONING_MODEL=SPOT,STANDARD \
DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-micro-l4.toml \
  ./scripts/gcp_launch.sh micro
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

Expected loss trajectory for the 50M micro model on ~1 B tokens:
- Step 500:   loss ≈ 5.5–6.0 (early warmup)
- Step 2000:  loss ≈ 4.0–4.5
- Step 10000: loss ≈ 3.0–3.5
- Step 25000: loss ≈ 2.6–3.0 (language modeling perplexity ~13–20)

---

## Resuming after preemption

SPOT VMs use `--instance-termination-action=STOP`, so the disk is preserved
on preemption. To resume, simply restart the same instance:

```bash
gcloud compute instances start tinybit-l4-YYYYMMDD-HHMMSS \
  --zone=us-central1-a --project=my-project
```

The startup script re-runs on every boot. It skips data prep and cargo
build if the outputs are already on disk, downloads any GCS checkpoints
missing locally, and restarts training with `--resume` from the last
checkpoint.

**If the VM was deleted** (e.g., after hitting the maximum preemption count
or a manual restart via `gcp_launch.sh` with the same `RUN_ID`):

```bash
RUN_ID=20260527-091332-micro \
DATA_TOKENS=1500000000 \
TRAIN_CONFIG=configs/train-micro-l4.toml \
  ./scripts/gcp_launch.sh micro
```

The `restore_checkpoints` stage downloads existing checkpoints from
`gs://bucket/runs/<run_id>/checkpoints/` onto the new VM before training
starts. Training resumes automatically from the latest saved step.

To force a clean restart from step 0, set `FORCE_DATA=1` or
`FORCE_REBUILD=1`, or use a different `RUN_ID`.

---

## Data preparation

Data is prepared by `scripts/prepare_data.sh` and produces two binary
files:

- `data/train.bin` — training tokens (uint32, little-endian)
- `data/val.bin`   — validation tokens (uint32, little-endian)

**Memory design:** tokens are written directly to disk using numpy
(4 bytes/token). Peak RAM during data prep is bounded to ~16 MB (one flush
buffer) regardless of total token count. A 32 GB swap file is created
early in startup as an additional safety net.

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
  --train-config configs/train-nano-l4.toml \
  --smoke-test
```

The smoke test should complete in under 15 minutes on CPU, ending at loss < 8.

---

## Checkpoint format

Checkpoints are pairs of files in `checkpoints/`:
- `step_NNNNNNN.safetensors` — model weights
- `step_NNNNNNN.json` — metadata (step, loss, tokens_seen, config, timestamp)

The trainer keeps the **3 best** (lowest val_loss) and the **3 most recent**
checkpoints, deleting the rest to prevent disk fill-up on long runs.
