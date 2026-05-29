# Running a training job that produces non-garbage output

`tinybit` trains on **NVIDIA L4 only** (`g2-standard-4` on GCP). L4 has 24 GB
VRAM, broad zone availability, and ~$0.22/hr SPOT — the cheapest GPU that
fits the 50M `micro` model with sequence length 512 and the RWKV-7 WKV scan.
On CUDA the scan uses the fused kernel by default (see CLAUDE.md design
decision 16); the unfused candle loop is the CPU / `TINYBIT_FUSED_WKV=off` path.

## TL;DR

Pick the training config that matches the model preset. Each is sized so the
L4 is well-utilized but won't OOM.

| Model preset | Train config                  | Time / Cost (25k steps)                          |
|--------------|-------------------------------|--------------------------------------------------|
| `nano`       | `configs/train-nano-l4.toml`  | faster than micro (unmeasured post-fix)          |
| `micro`      | `configs/train-micro-l4.toml` | **~1.5–2.2 days projected, ~$25–40 on-demand** (was ~4.4 days) |

> **Throughput note.** The fused WKV scan's BACKWARD dominated the training step.
> A **2026-05-29 fix** (replacing a per-timestep shared-memory `atomicAdd` storm
> in the `dv` reduction with a conflict-free column reduction — see CLAUDE.md
> design decision 16) made the backward **~9× faster** with the model numerically
> unchanged (all `cuda_*` parity tests pass at T=512). Measured end-to-end on a
> local RTX A2000, a full micro step is **~2–3× faster** across shapes; the WKV
> contribution at the real batch (b=11, t=512) fell from ~4.4 s to ~0.5 s per step
> (16 layers).
>
> | Era | s/step (L4, b11 bf16) | 25k steps | Note |
> |-----|----|----|----|
> | frozen-gradient (pre-2026-05-28) | — | ~15–22 h | LayerNorm backward bug pruned the graph; "fast" but not learning |
> | post-LayerNorm-fix, pre-bf16 | ~23 | ~6.7 d | batch 6 |
> | bf16 + batch 11 (2026-05-28, measured) | **15.2** | ~4.4 d | 2.23k tok/s |
> | + WKV backward fix (2026-05-29, projected) | **~5–7.5** | **~1.5–2.2 d** | confirm from live tok/s |
>
> The last row is **projected from local A2000 measurements**, not measured on L4 —
> update it once a live run reports its tok/s. Because the model is unchanged, an
> in-progress run can be **resumed** from its latest checkpoint with the new code
> to pick up the speedup. SPOT halves the $/hr but a multi-day run is preempted
> repeatedly; on-demand is the realistic baseline.

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
batch_size     = 11      # × max_seq_len 512 (from configs/micro.toml)
grad_accum     = 6       # → effective batch 66  →  66 * 512 = 33_792 tokens / step
total_steps    = 25000   # → 25000 * 33_792 ≈ 845M token-passes
peak_lr        = 3e-4    # WSD: 2% warmup, 78% stable, 20% cosine decay to 3e-5
weight_decay   = 0.01
grad_clip      = 1.0     # global L2 norm clipping
bf16           = true    # mixed precision (block matmuls bf16; norms/WKV/loss f32)

save_every     = 500     # ~50 checkpoints over the run; pruned to best-3 + recent-3
eval_every     = 500
eval_batches   = 20
```

Paired with `configs/micro.toml` (50M params), this is roughly Chinchilla-
optimal (17 toks/param) — enough for genuinely coherent English, simple
question following, and recognizable knowledge from the FineWeb-Edu /
Wikipedia / OpenHermes mix.

Why this `batch_size`: with the fused WKV CUDA kernel (the default) plus bf16,
`batch_size = 11` keeps the L4's SMs busier — the kernel launches one block per
`(batch, head)`, so batch 11 → 66 blocks (vs 36 at batch 6), and a larger
microbatch also cuts the grad-accum iteration count. Live L4 (2026-05-28):
batch 11 / accum 6 runs **15.18 s/step (2.23k tok/s)**; **batch 12 OOMs
immediately**, so do not raise this without re-measuring. (The old "~19.5 GB /
B=8 is the OOM cliff" rationale described candle's *unfused* sequential scan,
which retained the full `O(T·dh²)` per-layer state graph; that path is now CPU /
`TINYBIT_FUSED_WKV=off` only.) See CLAUDE.md design decision 16.

If you're tighter on time, fewer steps trade quality for wall-clock. Wall time ≈
`total_steps × s/step`. The post-fix column uses ~6 s/step (mid of the projected
~5–7.5 s/step — confirm from your live run); the pre-fix column is the measured
15.2 s/step for reference:

| total_steps | Model | DATA_TOKENS | Pre-fix ~time | **Post-fix ~time (projected)** | Expected output                    |
|-------------|-------|-------------|---------------|--------------------------------|------------------------------------|
| 6000        | micro | 200M        | ~25 h         | **~10 h**                      | Coherent words, weak structure     |
| 15000       | micro | 1B          | ~63 h         | **~25 h**                      | Sentences, weak QA, some facts     |
| 25000       | micro | 1.5B        | ~4.4 days     | **~1.7 days**                  | Paragraphs, QA, basic instructions |
| 50000       | micro | 1.5B        | ~8.8 days     | **~3.5 days**                  | Better instruction following       |

Because the 3× step speedup is "free" (same model), a good use of it is to spend
the saved time on **more tokens**: e.g. 50k steps (~3.5 days) is now cheaper than
the old 25k run (~4.4 days) and trains a meaningfully stronger model (~34 tok/param
vs ~17). (`nano` is smaller/faster per step but its post-fix throughput is unmeasured.)

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
