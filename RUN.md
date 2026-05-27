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

```bash
export GCP_PROJECT=tinybit-run-0
export GCP_BUCKET=gs://tinybit-run-0-tinybit       # your bucket
export HF_TOKEN=hf_xxx                              # optional — see below

# Single command:
DATA_TOKENS=1000000000 \
TRAIN_CONFIG=configs/train-quality.toml \
  ./scripts/gcp_launch.sh micro
```

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

| Budget    | Model  | DATA_TOKENS | total_steps | Expected output                      |
|-----------|--------|-------------|-------------|--------------------------------------|
| ~3-4h L4  | nano   | 200M        | 6000        | Coherent words, no real reasoning    |
| ~5-7h L4  | micro  | 500M        | 15000       | Sentences, weak QA, some facts       |
| ~10-14h L4| micro  | 1B          | 30000       | Paragraphs, QA, basic instructions   |
| ~24h L4   | small  | 3B          | 60000       | Better-than-nano "useful" output     |

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
