# Training guide

tinybit trains RWKV-7 language models on a single **NVIDIA L4** GPU on GCP.
**L4 is the only supported hardware** — the launcher, configs, and startup
script are tuned for `g2-standard-4 + nvidia-l4` (24 GB VRAM, broad zone
availability, ~$0.22/hr SPOT). This guide covers data prep, launching a run,
monitoring it, and verifying the result.

For the model-variant matrix (general vs coding, sizes), see [MODELS.md](MODELS.md).

---

## Quick start

```bash
export GCP_PROJECT=your-project-id
export GCP_BUCKET=gs://your-bucket

# 50M micro (general) — the main target
DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-micro-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT ./scripts/gcp_launch.sh micro

# 25M nano (fast iteration)
DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-nano-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT ./scripts/gcp_launch.sh nano

# Coding variant — same train config, code-heavy data (gated → set HF_TOKEN)
HF_TOKEN=hf_xxx DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-micro-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT ./scripts/gcp_launch.sh micro-coding
```

The launcher finds the first available zone with an L4, uploads the repo,
installs CUDA 12.8, prepares the data, compiles with the `cuda` feature, and
starts training — unattended. Checkpoints sync to
`$GCP_BUCKET/runs/<run_id>/checkpoints/` every 120 s.

---

## Hardware and cost

| GPU | Machine | VRAM | RAM | On-demand | SPOT | 50M micro · 25k steps |
|-----|---------|------|-----|-----------|------|------------------------|
| L4  | g2-standard-4 | 24 GB | 16 GB | ~$0.71/hr | ~$0.22/hr | **~1.9 days · ~$32 on-demand** |

The 25k-step `micro` run measures **6.5 s/step** (≈5.2k tok/s, fused WKV kernel +
bf16) on a live L4 — ~1.9 days. This is **2.33× faster** than the pre-optimization
15.2 s/step, after the WKV-backward fix (see "Throughput" below). SPOT is ~⅓ the
on-demand price but a multi-day run is preempted repeatedly; checkpoints resume
from GCS, so SPOT works, but on-demand is the realistic single-run baseline.

`small` (258M) and `base` (501M) are inference-only on an L4 — they need
A100/H100-class hardware to train, which the launcher does not support.

---

## Configuration

### Model config (`configs/micro.toml`)
Architecture only. **Do not change `vocab_size` or `max_seq_len` after training
starts** — they are baked into checkpoints.

```toml
vocab_size  = 32008   # 32000 LLaMA vocab + 8 reserved tool-marker slots
num_layers  = 16
d_model     = 384
d_ffn       = 1344
num_heads   = 6
head_dim    = 64
max_seq_len = 512     # L4-tuned; see "VRAM" note below
```

### Training config (`configs/train-micro-l4.toml`)

```toml
batch_size  = 11    # sequences per microbatch
grad_accum  = 6     # microbatches per optimizer step  → effective batch 66
total_steps = 25000
peak_lr     = 3e-4  # WSD: 2% warmup, ~78% stable, 20% cosine decay to 3e-5
grad_clip   = 1.0
bf16        = true  # block matmuls bf16; norms / WKV scan / loss stay f32
save_every  = 500
eval_every  = 500
```

**Token budget:** `11 × 512 × 6 = 33,792 tokens/step`; `25,000 × 33,792 ≈ 0.85 B
tokens` (~17 tok/param, near Chinchilla-optimal). Set `DATA_TOKENS=1500000000`
(headroom over the training budget so prep still succeeds if a dataset is
skipped).

**Optimizer:** AdamW is the **validated default** — the loss targets below assume
it. Muon (`optimizer = "muon"`) is opt-in and experimental.

**VRAM / batch size:** `batch_size = 11` keeps the L4's SMs busy (the fused WKV
kernel launches one block per `(batch, head)` → 66 blocks). `batch 12` OOMs, so
do not raise it without re-measuring on a live L4.

---

## Data preparation

`scripts/prepare_data.sh` streams open datasets, tokenizes them, and writes two
memory-mapped binary files (`u32`, little-endian):

- `data/train.bin` — training tokens
- `data/val.bin`   — validation tokens (head of the stream)

Select the family mix with `DATA_PROFILE`:

| Profile | Mix |
|---------|-----|
| `general` (default) | FineWeb-Edu, Cosmopedia v2, TinyStories, OpenHermes/dolphin chat, a little code |
| `coding` | The Stack (Python/Rust/JS/C/Go), technical chat, some prose |

```bash
DATA_PROFILE=general TOTAL_TOKENS=1500000000 ./scripts/prepare_data.sh data/
```

On GCP the profile is derived automatically from the model name (a `*-coding`
size → `coding`).

**Prompt-format consistency.** Conversation datasets are formatted with the
*exact* chat template `tinybit chat` uses at inference:

```
system:
{system}
user:
{user}
assistant:
{assistant}
```

So the turn structure the model trains on matches what it sees at chat time.
Plain corpora (web/wiki/code files) are tokenized as raw text. The template
lives in `crates/tinybit-core/src/tokenizer.rs`; the data script mirrors it.

**Code datasets are gated** on HuggingFace — set `HF_TOKEN` to include them
(essential for the `coding` profile). Without it they are skipped and the weight
is redistributed.

**Memory:** tokens stream straight to disk (~16 MB peak RAM regardless of total
size); the L4 startup adds 32 GB swap as a safety net.

---

## Monitoring

```bash
./scripts/gcp_status.sh   <run_id>   # stage, step, last checkpoint
./scripts/gcp_tail_logs.sh <run_id>  # tail training log from GCS
```

The training log prints one line per optimizer step:

```
step=1000 lr=3.00e-04 loss=3.2847 gnorm=0.842
step=1000 val_loss=3.3112
```

Set `TINYBIT_PROFILE=1` to log per-step forward / backward / optimizer time
(device synced at phase boundaries; zero overhead when unset).

---

## Verifying the run produced non-garbage output

Pull the final checkpoint and chat with it:

```bash
gsutil cp "$GCP_BUCKET/runs/<run_id>/checkpoints/step_*.safetensors" /tmp/final.safetensors
tinybit chat --config configs/micro.toml --model /tmp/final.safetensors
# or measure it:
tinybit eval --config configs/micro.toml --model /tmp/final.safetensors --data data/val.bin
```

**Healthy signs:**
- `val_loss` falls steadily from ~10.4 (the vocab=32008 random-init floor) toward
  the trajectory below.
- `gnorm` is the *pre-clip* global norm. Early-warmup spikes into the tens
  (observed 4–70 over the first ~40 steps) are normal; what matters is that it
  stays finite and trends down as the LR decays.
- Final chat output is grammatical English on most prompts (even if factually
  unreliable).

**Unhealthy signs:**
- `gnorm` near-zero and unchanging (e.g. ~0.007) with `loss` stuck near ln(vocab)
  → the backward graph is pruned (this was the pre-fix candle-LayerNorm bug;
  `tests/grad_flow.rs` now guards it).
- `gnorm` → ∞ or NaN → lower `peak_lr` to 1e-4.
- `loss` flat or rising → check the data (zeros / tokenization mismatch).
- Many `skipping update — non-finite` warnings → lower `peak_lr` and/or `grad_clip`.

**Projected loss trajectory** (50M micro, ~0.85 B tokens, AdamW; estimates):

| Step | loss | note |
|------|------|------|
| 500   | ~6–7    | LR still warming |
| 2000  | ~4.5–5.0 | words forming |
| 10000 | ~3.5–4.0 | sentences, weak QA |
| 25000 | ~3.0–3.8 | paragraphs, basic instructions (perplexity ~20–45) |

A 50M model at ~17 tok/param lands well short of GPT-2-small quality. Expect
coherent short English and simple instruction following — not reliable
factuality or reasoning.

---

## Resuming after preemption

SPOT VMs use `--instance-termination-action=STOP`, so the disk survives. Restart
the same instance and the startup script re-runs: it skips data prep and the
build if their outputs exist, downloads any GCS checkpoints missing locally, and
restarts training with `--resume` from the last step.

```bash
gcloud compute instances start tinybit-l4-YYYYMMDD-HHMMSS --zone=<zone> --project=$GCP_PROJECT
```

If the VM was deleted, relaunch with the same `RUN_ID`; the `restore_checkpoints`
stage pulls existing checkpoints from `gs://bucket/runs/<run_id>/checkpoints/`
before training resumes. To force a clean restart, use a new `RUN_ID` or set
`FORCE_DATA=1` / `FORCE_REBUILD=1`.

---

## Launch options

```bash
./scripts/gcp_launch.sh [nano|micro|nano-coding|micro-coding]
```

| Variable | Default | Description |
|---|---|---|
| `GCP_PROJECT` | (required) | GCP project ID |
| `GCP_BUCKET` | (required) | `gs://bucket-name` |
| `PROVISIONING_MODEL` | `STANDARD` | `STANDARD`, `SPOT`, or `STANDARD,SPOT` |
| `DATA_TOKENS` | `20000000` | Target training tokens (use 1.5e9 for a real run) |
| `TRAIN_CONFIG` | (inline default) | Path to a checked-in train config TOML |
| `HF_TOKEN` | (unset) | HuggingFace token for gated code datasets |
| `KEEP_VM_ON_FAILURE` | `0` | Keep the VM alive on error for debugging |
| `SYNC_INTERVAL` | `120` | Seconds between checkpoint syncs to GCS |
| `GCP_ZONES` | (broad list) | Space-separated zone override |

---

## Throughput: where the time goes

A `micro` fwd+bwd step is only ~6–7 TFLOP of matmul plus a tiny WKV scan — the
historical slowness was **stalls, not FLOPs**. Two fixes removed them:

1. **Per-step sync stalls.** `global_grad_norm` once did a host read per
   parameter (~150–200 forced CUDA syncs/step); the loss synced per microbatch.
   Both now accumulate on-device and sync **once per step**.
2. **WKV backward `dv` atomic storm.** The scan's backward reduced `dv` with a
   per-timestep storm of shared-memory `atomicAdd`s (all threads contending on
   the same addresses). Replacing it with a conflict-free column reduction cut
   the backward **~9×** and the whole scan ~8× — numerically unchanged (the
   `cuda_*` parity tests pass at T=512), so checkpoints stay compatible. This is
   the 15.2 → 6.5 s/step (2.33×) win.
3. **Single-GEMM linear projections.** Every projection (the 6 time-mix + 3
   channel-mix matmuls per layer, plus the LM head) used candle's `broadcast_left`
   + batched matmul: `B` small GEMMs with the weight replicated across the batch.
   `linear_flat` now flattens `(B,T,D)→(B*T,D)` and runs one large GEMM — one big
   cuBLAS call instead of `B` small ones, and no weight broadcast. Numerically
   exact (deterministic unit test pins it; checkpoints unchanged). **Measured on
   CPU: 1.5–1.7× per projection, 1.30× on a full micro fwd+bwd step.** The CUDA
   magnitude is unmeasured here — the single large GEMM (especially the
   `d_model × 32008` head) should help, but confirm on an L4 with `TINYBIT_PROFILE=1`.

A chunked-parallel scan could raise occupancy further but is unmotivated now that
the scan is no longer the bottleneck. See `CLAUDE.md` design decisions 16 and 20
for the kernel/matmul details and regression guards.

---

## Local smoke test

```bash
TOTAL_TOKENS=200000 MIN_TOKENS=100000 ./scripts/prepare_data.sh data/
cargo build --release -p tinybit-cli
./target/release/tinybit train \
  --model-config configs/nano.toml \
  --train-config configs/train-nano-l4.toml --smoke-test
```

Completes in under ~15 min on CPU, ending at loss < 8.

---

## Checkpoint format

Pairs of files in `checkpoints/`:
- `step_NNNNNNN.safetensors` — model weights
- `step_NNNNNNN.json` — metadata (step, loss, tokens_seen, config, timestamp)

The trainer keeps the **best 3** (lowest val_loss) and the **most recent 3**,
deleting the rest to bound disk usage on long runs.
