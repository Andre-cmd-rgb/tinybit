# tinybit

A local AI assistant built on **RWKV-7** with **ternary BitLinear** quantization. Configurable from 25M to 500M parameters. Train on Google Cloud L4 free credits; run on Linux, macOS (M-series), or Windows.

Apache 2.0 — all Rust, no C++ compiler required.

---

## Features

- **RWKV-7 architecture** — recurrent transformer with O(1) memory at inference (no KV cache)
- **BitLinear quantization** — ternary weights `{-1, 0, +1}` via BitNet b1.58 STE
- **Built-in tools** — calculator, time, todos, notes, calendar (SQLite-backed, user-extensible)
- **Four size presets** — nano (25M), micro (50M), small (258M), base (501M)
- **Muon + AdamW optimizer** — Newton-Schulz gradient orthogonalization for weight matrices
- **OpenAI-compatible HTTP server** — drop-in for local inference
- **Speculative decoding heads** — optional extra heads on small/base for faster sampling

---

## Quick start

```bash
# Prerequisites: Rust stable ≥ 1.82 (see rust-toolchain.toml)
git clone <this-repo> && cd tinybit

# Build everything
cargo build --release

# Interactive chat (loads a model checkpoint)
tinybit chat --model checkpoints/nano/latest.safetensors --config configs/nano.toml

# HTTP server (OpenAI-compatible)
tinybit serve --model checkpoints/micro/latest.safetensors --config configs/micro.toml --port 8080

# Download tokenizer
tinybit download --out data/tokenizer.json
```

---

## Model presets

All models use 3.5× FFN expansion (d_ffn = 3.5 × d_model) matching the RWKV-7 paper.

| Preset | Params  | Layers | d_model | d_ffn | Notes |
|--------|---------|--------|---------|-------|-------|
| nano   | ~25M    | 9      | 320     | 1120  | Fast iteration |
| micro  | ~50M    | 16     | 384     | 1344  | Main L4 target, ~1.5–2.2 days (projected, post WKV-backward fix) |
| small  | ~258M   | 13     | 1024    | 3584  | Architecture only — too big to train on L4 |
| base   | ~501M   | 17     | 1280    | 4480  | Architecture only — too big to train on L4 |

Config files live in `configs/`. Override any field:

```toml
# configs/nano.toml
vocab_size  = 32008
num_layers  = 9
d_model     = 320
d_ffn       = 1120
num_heads   = 5
head_dim    = 64
max_seq_len = 512
```

---

## Training

### 1. Prepare data (on GCP or locally)

```bash
# Download + tokenize FineWeb-Edu, Wikipedia, The Stack Smol
bash scripts/prepare_data.sh data/

# This creates data/train.bin and data/val.bin (packed u32 token IDs)
```

### 2. Train

```bash
# Local smoke test (CPU, ~15 min)
cargo build --release -p tinybit-cli
./target/release/tinybit train \
  --model-config configs/nano.toml \
  --train-config configs/train-nano-l4.toml \
  --smoke-test

# GCP — L4-only launcher
export GCP_PROJECT="your-project-id"
export GCP_BUCKET="gs://your-bucket"

# 25M nano (fast iteration)
DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-nano-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT ./scripts/gcp_launch.sh nano

# 50M micro — 25k steps ≈ 1.5–2.2 days on L4, ~$25–40 on-demand (projected post WKV fix; see RUN.md)
DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-micro-l4.toml \
PROVISIONING_MODEL=STANDARD,SPOT ./scripts/gcp_launch.sh micro

# Watch the run
./scripts/gcp_status.sh <RUN_ID>           # stage, step, last checkpoint
./scripts/gcp_tail_logs.sh <RUN_ID>        # tail training log live
```

Output layout in the bucket:

```
gs://$GCP_BUCKET/runs/<RUN_ID>/
  launch.json                # what was provisioned
  status.json                # latest stage, step, checkpoint
  DONE.json | FAILED.json    # terminal marker
  logs/bootstrap.log
  logs/training.log
  checkpoints/step_000NNNN.safetensors  (+ .json meta)
gs://$GCP_BUCKET/latest_run.txt
```

Failure handling: any error before training starts uploads `FAILED.json` and
shuts the VM down (set `KEEP_VM_ON_FAILURE=1` to keep it for debugging).

Key training config fields (`configs/train-micro-l4.toml`):

```toml
batch_size     = 6       # sequences per microbatch (L4-tuned)
grad_accum     = 11      # microbatches per optimizer step
total_steps    = 25000   # optimizer steps
peak_lr        = 3e-4
grad_clip      = 1.0
save_every     = 500
eval_every     = 500
```

### 3. WSD learning rate schedule

- **Warmup**: linear ramp over first 2% of steps
- **Stable**: constant LR
- **Decay**: cosine decay over final 20% of steps

### 4. Optimizer

- **Muon** for all 2D weight matrices (w_q, w_k, w_v, w_o, w_g1, w_g2, BitLinear weights)
- **AdamW** for embeddings, LayerNorm params, biases, 1D tensors

---

## Tool system

The model can call tools via structured tokens:

```
<|tool_call|>{"tool":"calculator","args":{"expr":"2^10"}}<|end_tool_call|>
```

Results are injected as:

```
<|tool_result|>1024<|end_tool_result|>
```

### Built-in tools

| Tool | Description |
|------|-------------|
| `time` | Current date, time, timezone |
| `calculator` | Math expressions via `evalexpr` (e.g. `2+2`, `12^2`, `sin(3.14)`) |
| `todos` | Add / list / complete / delete tasks (SQLite) |
| `notes` | Save and full-text search notes (SQLite FTS5) |
| `calendar` | Add / list / delete calendar events (SQLite) |

### Adding custom tools

Implement the `Tool` trait in `tinybit-tools`:

```rust
use tinybit_tools::tool::{Tool, ToolOutput};

struct MyTool;

impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Does something useful" }
    fn execute(&self, args_json: &str) -> anyhow::Result<ToolOutput> {
        // parse args_json, do work, return ToolOutput
        Ok(ToolOutput { content: "result".to_string(), is_error: false })
    }
}

// Register
registry.register(Box::new(MyTool));
```

---

## HTTP server (OpenAI-compatible)

```bash
tinybit serve --port 8080
```

Endpoints:

- `POST /v1/chat/completions` — chat completion (streaming and non-streaming)
- `GET /v1/models` — list loaded model

Example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinybit",
    "messages": [{"role": "user", "content": "What is 12^2?"}]
  }'
```

---

## Model export

```bash
# Export to safetensors
tinybit convert --model checkpoints/micro/latest.safetensors --out model.safetensors --format safetensors

# Export to GGUF (for llama.cpp compatibility)
tinybit convert --model checkpoints/micro/latest.safetensors --out model.gguf --format gguf
```

---

## Running tests

```bash
# All tests
cargo test

# Specific suites
cargo test -p tinybit-tests --test model_correctness
cargo test -p tinybit-tests --test tool_system
cargo test -p tinybit-tests --test training_smoke
cargo test -p tinybit-tests --test quantize
```

Gate tests that must pass before any real training run:
- `test_forward_shapes_nano` — logits shape and finite check
- `test_inference_step_matches_train` — step-by-step matches full-sequence forward (tolerance 1e-3)
- `smoke_train_nano_100_steps` — initial loss ≤ 2× ln(vocab_size)

---

## Architecture overview

```
Token IDs (B, T)
    │
    ▼ EmbeddingHead.embed()
Hidden (B, T, D)
    │
    ▼ × num_layers
┌─────────────────────────────┐
│  LayerNorm                  │
│  TimeMix (RWKV-7 WKV scan)  │
│  + residual                 │
│  LayerNorm                  │
│  ChannelMix (gated FFN)     │
│  + residual                 │
└─────────────────────────────┘
    │
    ▼ EmbeddingHead.lm_head()  (tied weights, scaled by 1/√d_model)
Logits (B, T, vocab_size)
```

**TimeMix** (RWKV-7 WKV):
- Token-shifted lerp inputs → r, k, v, gate
- Recurrent state: `S_t = S_{t-1} * decay + k_t ⊗ v_t`
- Output: `y_t = r_t @ S_t`, group-normalized, gated

**ChannelMix** (RWKV-7 FFN):
- `k = SiLU(W_k · x_k)`
- `v = W_v · k²`
- `r = sigmoid(W_r · x_r)`
- `out = r * v`

**BitLinear**:
- RMSNorm on input, then linear projection
- At quantization time: ternary weights `{-1, 0, +1}`, int8 activations
- Training: full float32 weights (STE for gradients)

---

## Workspace layout

```
crates/
  tinybit-core/    — model, config, tokenizer, state, quantize
  tinybit-tools/   — tool trait, registry, built-in tools
  tinybit-infer/   — inference engine, sampler, session, tool processor
  tinybit-train/   — trainer, optimizer, scheduler, loss, checkpoint, data
  tinybit-cli/     — CLI (chat, serve, train, convert, download)
tests/              — integration tests (workspace member)
configs/            — TOML config files for each model size
scripts/            — GCP provisioning and data preparation
data/               — downloaded datasets (not committed)
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
