# Running a training job that produces non-garbage output

`tinybit` trains on **NVIDIA L4 only** (`g2-standard-4` on GCP). L4 has 24 GB
VRAM, broad zone availability, and ~$0.22/hr SPOT — the cheapest GPU that
fits the 50M `micro` model with sequence length 512 and the RWKV-7 WKV scan.
On CUDA the scan uses the fused kernel by default (see CLAUDE.md design
decision 16); the unfused candle loop is the CPU / `TINYBIT_FUSED_WKV=off` path.

## TL;DR

Pick the training config that matches the model preset. Each is sized so the
L4 is well-utilized but won't OOM.

| Model preset | Train config                  | Time / Cost                          |
|--------------|-------------------------------|--------------------------------------|
| `nano`       | `configs/train-nano-l4.toml`  | unverified post-fix (faster than micro) |
| `micro`      | `configs/train-micro-l4.toml` | ~6.5-7 days, ~$115-125 on-demand (measured) |

> **Throughput note (measured 2026-05-28, post LayerNorm fix):** the `micro`
> run holds **~23 s/step** on L4 (~1,440 tok/s) with the fused WKV kernel, so
> 25k steps takes ~6.7 days. Earlier docs quoted ~15-22 h — that was the
> *frozen-gradient* era, when a LayerNorm backward bug pruned the graph and
> backward did almost no work. Correct training (gradients through all 16
> layers) is ~7x slower per step. SPOT halves the $/hr but a multi-day run
> will be preempted repeatedly; on-demand is the realistic baseline.

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
batch_size     = 6       # × max_seq_len 512 (from configs/micro.toml)
grad_accum     = 11      # → effective batch 66  →  66 * 512 = 33_792 tokens / step
total_steps    = 25000   # → 25000 * 33_792 ≈ 845M token-passes
peak_lr        = 3e-4    # WSD: 2% warmup, 78% stable, 20% cosine decay to 3e-5
weight_decay   = 0.01
grad_clip      = 1.0     # global L2 norm clipping

save_every     = 500     # ~50 checkpoints over the run; pruned to best-3 + recent-3
eval_every     = 500
eval_batches   = 20
```

Paired with `configs/micro.toml` (50M params), this is roughly Chinchilla-
optimal (17 toks/param) — enough for genuinely coherent English, simple
question following, and recognizable knowledge from the FineWeb-Edu /
Wikipedia / OpenHermes mix.

Why this `batch_size`: with the fused WKV CUDA kernel (the default), peak
training VRAM at `batch_size = 6`, `max_seq_len = 512` measures **~12.5 GB**
on the L4 — well under the 24 GB cap. (The old "~19.5 GB / B=8 is the OOM
cliff" rationale described candle's *unfused* sequential scan, which retained
the full `O(T·dh²)` per-layer state graph; that path is now CPU /
`TINYBIT_FUSED_WKV=off` only.) There is real headroom to raise `batch_size`
on CUDA, but the configs are **not yet re-tuned** — raise it only after a
live run confirms loss is unaffected (see CLAUDE.md design decision 16).

If you're tighter on time, fewer steps trade quality for wall-clock. At the
measured ~23 s/step for `micro`, total time ≈ `total_steps × 23 s`:

| total_steps | Model | DATA_TOKENS | ~Wall time | Expected output                    |
|-------------|-------|-------------|------------|------------------------------------|
| 6000        | micro | 200M        | ~38 h      | Coherent words, weak structure     |
| 15000       | micro | 1B          | ~4 days    | Sentences, weak QA, some facts     |
| 25000       | micro | 1.5B        | ~6.7 days  | Paragraphs, QA, basic instructions |
| 50000       | micro | 1.5B        | ~13 days   | Better instruction following       |

(`nano` is smaller/faster per step but its post-fix throughput is unmeasured.)

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
- `gnorm` is the *pre-clip* global norm; `grad_clip = 1.0` caps the applied
  update. Early-warmup spikes into the tens (observed 4–70 over the first
  ~40 steps) are normal. What matters: it stays finite and trends down as
  the LR schedule decays.
- Final chat output is grammatical English on most prompts, even if
  factually unreliable.

Unhealthy signs to investigate:

- `gnorm` near-zero and unchanging (e.g. ~0.007) with `loss` stuck → the
  backward graph is pruned (this was the pre-2026-05-28 LayerNorm bug).
- `gnorm` growing without bound or going NaN/inf → reduce `peak_lr` to 1e-4.
- `loss` flat or rising → check the data — usually means `train.bin`
  contains zeros or a tokenization mismatch.
- Many `skipping update — non-finite` warnings → reduce `peak_lr` and/or
  `grad_clip`.
- `DriverError(CUDA_ERROR_OUT_OF_MEMORY)` at training start → another
  process is holding the GPU. The startup script's `free_gpu_memory`
  stage stops MPS and kills GPU processes; if it fails, SSH in and run
  `nvidia-smi` to confirm.
