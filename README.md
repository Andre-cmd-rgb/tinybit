# tinybit

A small, fast, **local-first** AI assistant built on **RWKV-7** with **ternary
BitLinear** quantization. Shipped in three sizes, ~50M to ~150M parameters. Train
cheaply on a Google Cloud L4; run locally on Linux, macOS (Apple Silicon), or
Windows.

All Rust. No C++ compiler. No cloud calls at inference time. Apache 2.0.

---

## What tinybit is (and isn't)

tinybit is a small assistant that is useful because it is **optimized,
measurable, tool-aware, and honest about its limits**. It is meant to be hacked
on and inspected.

It is **not** a ChatGPT clone. A 50M-parameter model is not going to match a
large cloud model on knowledge or reasoning, and tinybit doesn't pretend
otherwise. It wins by being small, local, fast, controllable, easy to train,
and easy to read.

V1.0 is **local CLI inference first**: `chat`, `eval`, `train`, `convert`,
`download`. There is intentionally no HTTP server.

---

## Highlights

- **RWKV-7 architecture** — recurrent, O(1) memory at inference (no KV cache);
  state is a fixed-size matrix per layer regardless of context length.
- **Two model families** — *general* (`micro`/`bit`/`qbit`) and
  *coding* (`*-coding`). Same architecture, different training data + persona.
  See [MODELS.md](MODELS.md).
- **BitLinear quantization** — ternary weights `{-1, 0, +1}` (BitNet b1.58 STE),
  packed on export.
- **Built-in local tools** — calculator, time, todos, notes, calendar
  (SQLite-backed, user-extensible) via a stable token protocol.
- **Stable prompt format** — the exact `system:/user:/assistant:` template used
  in `tinybit chat` is the one the training data is formatted with.
- **Cheap training** — a 50M `micro` model is ~1.9 days / ~$32 on a single L4
  (measured post-optimization; see [TRAINING.md](TRAINING.md)).
- **Fused WKV CUDA kernel** — the RWKV-7 scan runs as a single fused
  forward/backward op on CUDA (≈25× the naive candle loop for the scan).
- **Real evals** — `tinybit eval` reports perplexity and runs generation sanity
  prompts so quality is measured, not guessed.

---

## Quick start

```bash
# Prerequisites: Rust stable (see rust-toolchain.toml)
git clone <this-repo> && cd tinybit
cargo build --release --workspace
# GPU build — add --features cuda (needs the CUDA 12.8 toolkit; on Windows build
# from a VS dev shell so nvcc finds cl.exe — see CLAUDE.md decision 12). Big win
# for TRAINING and `eval` (batched: ~30x on an A2000); eval/chat/train auto-detect
# the GPU. Interactive `chat` of a tiny model is launch-bound — often FASTER on CPU
# (force CPU on a cuda build with CUDA_VISIBLE_DEVICES=-1).
#   cargo build --release --workspace --features cuda

# 1. Get the tokenizer
./target/release/tinybit download --output .

# 2. Verify the training pipeline locally (CPU, a few minutes)
./target/release/tinybit train \
  --model-config configs/micro.toml \
  --train-config configs/train-micro-l4.toml --smoke-test

# 3. After you have a checkpoint, chat with it
./target/release/tinybit chat \
  --config configs/micro.toml \
  --model  checkpoints/step_0025000.safetensors

# 4. Measure quality
./target/release/tinybit eval \
  --config configs/micro.toml \
  --model  checkpoints/step_0025000.safetensors \
  --data   data/val.bin
```

> tinybit V1.0 ships **no pretrained weights** — you train your own. The 50M
> `micro` run is the documented target (see [TRAINING.md](TRAINING.md)).

> **Windows:** the same commands work in PowerShell — use
> `.\target\release\tinybit.exe` and `.\scripts\*.ps1` (e.g.
> `.\scripts\prepare_data.ps1`, `.\scripts\gcp_launch.ps1`). Every `scripts/*.sh`
> has a `.ps1` sibling; only the in-VM `scripts/cloud/startup.sh` stays bash
> (it runs on the Linux training VM). No WSL or Git Bash required.

---

## Commands

| Command | What it does |
|---------|--------------|
| `tinybit chat` | Interactive local REPL. Streams tokens, supports tools, slash-commands (`/help`, `/reset`, `/system`, `/save`). |
| `tinybit eval` | Perplexity over a token file + greedy generation sanity prompts (with tok/s). |
| `tinybit train` | Train from a model+train config. `--smoke-test` runs a short sanity loop; `--resume` continues from the latest checkpoint. |
| `tinybit dream` | Offline "sleep" consolidation: replay saved conversations into the model with an anti-forgetting anchor; writes a consolidated checkpoint. |
| `tinybit convert` | Export a checkpoint; `--quantize` packs 2D matrices to ternary (smaller on disk). |
| `tinybit download` | Fetch `tokenizer.json` from HuggingFace. |

Run `tinybit <command> --help` for all flags.

---

## Brain-inspired mode (`nano`)

tinybit's RWKV-7 core is already brain-like — recurrent, attention-free, O(1)
memory with a persistent state. The `nano` model (`configs/nano.toml`, ~17M)
turns on opt-in, biologically-motivated mechanisms (all default **off** on
micro/bit/qbit, so existing checkpoints are unchanged):

- **Spiking sparsity** (`spike_threshold`) — neurons fire only when activated
  past a threshold (event-driven sparse coding). *Efficient.*
- **Hebbian fast-weights** (`fast_weights`) — a decaying associative weight delta
  updated *online during a conversation* and added to the value path. *It rewires
  itself.* (Inference-only; training is unchanged.)
- **Pondering** (`ponder_steps`) — latent recurrence steps over a learned thought
  embedding before answering. *It thinks before it speaks.*
- **Fetch, don't memorize** — `tinybit chat --rag N` searches a local knowledge
  store (`KnowledgeStore`, facts + definitions) and grounds the answer in what it
  retrieves, so a tiny model spends its parameters on language, not trivia.
- **Dreaming** — `tinybit dream` consolidates saved conversations offline (replay
  + pseudo-rehearsal with a frozen-base KL anchor), like sleep.

```bash
# train the brain-native nano model
tinybit train --model-config configs/nano.toml --train-config configs/train-nano-l4.toml
# chat with local retrieval grounding on
tinybit chat --config configs/nano.toml --model models/tinybit-nano.safetensors --rag 3
# consolidate today's saved conversations ("sleep")
tinybit dream --config configs/nano.toml --model models/tinybit-nano.safetensors --sessions sessions
```

**Honesty:** these mechanisms add efficiency, in-context adaptivity, and memory —
they do not magically make a 17M model "smart." Whether `nano` beats `micro` is a
*measured* question (`tinybit eval` against the micro baseline), not an assumption.
At this size, raw capability is bounded by parameters and data; the brain features
are about *how* it uses them. `nano` is not yet trained at scale.

---

## Model variants

Three sizes × two families. Architecture is shared; the family is the
training-data mix + default persona. Full details in [MODELS.md](MODELS.md).

| Preset | Params | Layers | d_model | d_ffn | L4-trainable |
|--------|--------|--------|---------|-------|--------------|
| micro  | ~50M   | 16     | 384     | 1344  | ✅ (main target, batch 11) |
| bit    | ~100M  | 12     | 640     | 2240  | ✅ (train-bit-l4.toml) |
| qbit   | ~150M  | 13     | 768     | 2688  | ✅ (train-qbit-l4.toml) |

Each has a `-coding` sibling (`configs/<size>-coding.toml`) trained on a
code-heavy data mix. All use a 3.5× FFN expansion (d_ffn = 3.5 × d_model).

---

## Tool system

The model calls tools with a stable token protocol:

```
<|tool_call|>{"tool":"calculator","args":{"expr":"2^10"}}<|end_tool_call|>
```

The runtime executes the tool and injects:

```
<|tool_result|>1024<|end_tool_result|>
```

| Tool | Description |
|------|-------------|
| `time` | Current date, time, timezone |
| `calculator` | Math via `evalexpr` (e.g. `2+2`, `12^2`, `sqrt(144)`) |
| `todos` | Add / list / complete / delete tasks (SQLite) |
| `notes` | Save and full-text search notes (SQLite FTS5) |
| `calendar` | Add / list / delete events (SQLite) |

Add your own by implementing the `Tool` trait in `tinybit-tools` and registering
it. Note: reliable tool *calling* depends on instruction/tool fine-tuning — the
plumbing (detect → execute → inject → continue) is complete and tested, but a
base-pretrained model emits tool calls only loosely. See **Known limitations**.

---

## Training

Full guide: **[TRAINING.md](TRAINING.md)**. In short:

```bash
# Prepare data (general or coding mix)
DATA_PROFILE=general TOTAL_TOKENS=1500000000 ./scripts/prepare_data.sh data/

# Train locally or launch an L4 on GCP
DATA_TOKENS=1500000000 TRAIN_CONFIG=configs/train-micro-l4.toml \
  ./scripts/gcp_launch.sh micro
```

```powershell
# Windows / PowerShell equivalents
$env:DATA_PROFILE = "general"; $env:TOTAL_TOKENS = "1500000000"
.\scripts\prepare_data.ps1 data

$env:DATA_TOKENS = "1500000000"; $env:TRAIN_CONFIG = "configs/train-micro-l4.toml"
.\scripts\gcp_launch.ps1 micro
```

- **Optimizer**: AdamW by default (validated). Muon is opt-in (`optimizer =
  "muon"`) for 2D weight matrices and is experimental.
- **LR schedule**: WSD (warmup → stable → cosine decay).
- **Checkpoints**: safetensors + JSON meta; the trainer keeps the best-3 and
  most-recent-3.

---

## Model export

```bash
# Re-save as safetensors (identity copy)
tinybit convert --input ckpt.safetensors --output model.safetensors

# Pack 2D weight matrices to ternary (smaller on disk; ~5 weights/byte, base-3)
tinybit convert --input ckpt.safetensors --output model.q.safetensors --quantize
```

`--quantize` packs every 2D weight matrix to ternary {−1, 0, +1} at ~1.6
bits/weight (five values per byte, base-3); the tied embedding/LM-head stays f32.
On the 50M `micro` checkpoint that is **~3.2× smaller on disk** (the f32 embedding
is the floor). Loading **dequantizes back to f32** and runs the normal path — a
true ternary-matmul runtime is future work, so it's a storage/format win, not an
inference-speed win.

**Caveat — this is *post-training* quantization.** For an f32-trained model (e.g.
`micro`, `ternary_ffn = false`) ternarizing the weights is **lossy**: measured
perplexity rises ~83 → ~590 on the micro checkpoint. Near-lossless ternary needs
quantization-aware training — the `bit`/`qbit` configs with `ternary_ffn = true`,
trained from scratch so the STE learns ternary-friendly weights.

**When to use it (rarely, at these sizes):** at 50–150M params the f32 file is
already small and quantizing gives **no speed win** while costing quality — so for
deployment, don't bother. For *throughput* (training, batched `eval`) build
`--features cuda` and use a GPU (measured ~30× over CPU on an A2000); keep weights
**full-precision**. Note interactive `chat` (single-token decode) of a tiny model
is launch-bound and is often *faster on CPU* — see the speed note below. Reach for
`--quantize` only to ship a `bit`/`qbit` model trained ternary-aware.

---

## Architecture

```
Token IDs (B, T)
    │  EmbeddingHead.embed()
Hidden (B, T, D)
    │  × num_layers:  LN → TimeMix (WKV scan) → +res → LN → ChannelMix → +res
    │  EmbeddingHead.lm_head()   (tied weights, scaled by 1/√d_model)
Logits (B, T, vocab_size)
```

- **TimeMix (RWKV-7 WKV)**: token-shifted r/k/v/gate; recurrent state
  `S_t = S_{t-1}·exp(-exp(time_decay)) + kᵀv`; readout `y = r·S`, group-normed,
  gated. On CUDA this scan is a fused kernel; on CPU it's a sequential loop
  (numerically equivalent).
- **ChannelMix (FFN)**: `k = SiLU(W_k·x)`, `v = W_v·k²`, `r = σ(W_r·x)`,
  `out = r·v`.
- **BitLinear**: RMSNorm + linear; full-precision master weights in training
  (STE), ternary on export.

LayerNorms are a hand-rolled differentiable implementation (candle's fused
LayerNorm has no backward — using it silently froze training before this was
fixed; `tests/grad_flow.rs` guards against regressing it).

---

## Tests

```bash
cargo test --workspace
```

Gate tests (must pass before any real run):
- `test_forward_shapes_nano` — logits shape + finite check
- `test_inference_step_matches_train` — step-by-step matches full-sequence forward
- `smoke_train_nano_100_steps` — initial loss ≤ 2× ln(vocab)
- `grad_flow` (in `tinybit-core`) — every parameter receives a finite gradient
  and the model overfits a fixed batch

---

## Known limitations (honest)

- **Small models are small.** Expect coherent short text and simple instruction
  following — *not* reliable factuality or multi-step reasoning. `tinybit eval`
  prints perplexity so you can see where a checkpoint actually lands.
- **No pretrained weights ship with V1.0.** You train your own.
- **Tool calling needs fine-tuning to be reliable.** The detect/execute/inject
  loop is complete and tested; a base-pretrained model still emits tool calls
  only loosely.
- **Quantization is storage-only and usually not worth it at these sizes.** The
  packed file loads back as f32 (no speed gain), and post-hoc ternary on an
  f32-trained model is lossy. A true ternary-*matmul* runtime is future work.
- **GPU helps throughput, not tiny-model decode (measured).** On an A2000, batched
  `eval`/training is ~30× faster on GPU, but single-token `chat` is ~3.5× *slower*
  than CPU (~21 vs ~75 tok/s) — launch-latency bound at 50M. Use GPU for
  train/`eval`; CPU is fine (often faster) for interactive chat.
- **No GGUF / llama.cpp export.** tinybit's custom RWKV-7 + BitLinear architecture
  isn't a llama.cpp architecture, so a GGUF file couldn't be loaded there — the
  stub was removed rather than left as a dead option.
- **Muon optimizer is experimental** — mechanically correct, quality-at-scale
  unverified. AdamW is the validated default.

---

## Workspace layout

```
crates/
  tinybit-core/    — model, config, tokenizer, state, quantize
  tinybit-tools/   — tool trait, registry, built-in tools
  tinybit-infer/   — inference engine, sampler, session, tool processor
  tinybit-train/   — trainer, optimizer, scheduler, loss, checkpoint, data
  tinybit-cli/     — CLI (chat, eval, train, convert, download)
tests/             — integration tests (workspace member)
configs/           — model + training TOML configs (6 model variants)
scripts/           — data prep + GCP L4 launcher (Bash *.sh + PowerShell *.ps1)
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
