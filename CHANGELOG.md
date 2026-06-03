# Changelog

All notable changes to tinybit are documented here.

## [Unreleased]

### Added
- **Full Windows support for the helper scripts.** Every user-facing `scripts/*.sh`
  now has a PowerShell sibling (`prepare_data.ps1`, `eval.ps1`, `preflight.ps1`,
  and `gcp_{launch,status,tail_logs,sync_now,stop_vm,delete_vm}.ps1`) so the
  project can be built, trained, and launched on Windows without WSL or Git Bash.
  The Rust CLI was already cross-platform (the `x86_64-pc-windows-msvc` target is
  in `rust-toolchain.toml`); these scripts close the tooling gap. GCP PowerShell
  scripts read settings from `$env:` vars or an optional git-ignored
  `.tinybit.env.ps1` (mirrors the bash `.tinybit.env`).

### Changed
- **Data-prep Python extracted to `scripts/prepare_data.py`.** It was previously
  a heredoc inside `prepare_data.sh`; both `prepare_data.sh` and the new
  `prepare_data.ps1` now invoke the shared file, so the two platforms can never
  drift. `scripts/cloud/startup.sh` (which runs only on the Linux VM) is
  unchanged and still bash.
- **General data mix retuned for small models.** Dropped raw **Wikipedia** (was
  30%) — at ~50M params the model could not store encyclopedic facts and only
  learned Wikipedia's proper-noun register, hallucinating band/album/biography
  trivia. Replaced with **Cosmopedia v2** (synthetic textbooks, 30%) and
  **TinyStories** (coherent narratives, 12%); FineWeb-Edu stays the backbone and
  val-set head. Net: cleaner, more coherent generation from a tiny model
  (cf. SmolLM / TinyStories / Phi). Requires a data re-prep + fresh training run
  to take effect. See `scripts/prepare_data.sh`.

## 1.0.0

First serious release. tinybit is now a coherent, local-first Rust AI assistant:
small, fast, tool-aware, measurable, and honest about its limits.

### Added
- **`tinybit eval`** — measures perplexity over a token file and runs greedy
  generation-sanity prompts (with tok/s), so model quality is *measured*, not
  guessed.
- **Two model families** — *general* (`micro`/`bit`/`qbit`) and
  *coding* (`*-coding`). New `configs/*-coding.toml` for all three sizes and a
  `DATA_PROFILE=general|coding` data mix in `scripts/prepare_data.sh`. See
  [MODELS.md](MODELS.md).
- **`--profile general|coding`** on `chat`/`eval` to select the default system
  prompt (auto-detected from a config filename containing "coding").
- **Coding system prompt** and a `Profile` type in `tinybit-core`.
- **MODELS.md** (variant matrix) and this **CHANGELOG.md**.
- `chat` now prints a startup banner and supports `/help`; `/system <text>`
  resets state so the new prompt takes effect.

### Changed
- **Prompt format is now consistent between training and inference.** Conversation
  datasets (OpenHermes, dolphin) are formatted with the exact
  `system:/user:/assistant:` template `tinybit chat` uses, instead of being
  flattened to raw text.
- Version is **1.0.0** across the workspace; the CLI reports it via `--version`.
- `tinybit download` simplified to fetch the tokenizer with clear next steps.
- CLI logging respects `RUST_LOG` and defaults to quiet (warnings only).
- Documentation rewritten to be local-first and honest: new README, consolidated
  TRAINING.md (absorbed the old RUN.md), explicit **Known limitations**.

### Performance
- **Single-GEMM linear projections.** `BitLinear`, `linear_autocast`, and the
  LM head no longer use candle's `broadcast_left` + batched matmul (one small
  GEMM per batch element, weight replicated). They flatten `(B,T,D)→(B*T,D)` and
  do one large GEMM (`linear_flat`). Numerically identical (proven by a tight
  deterministic unit test; checkpoints unchanged). Measured on CPU at the micro
  shape: **1.5–1.7× per projection**, **1.30× on a full micro fwd+bwd step**
  (21.8 s → 16.7 s, b4×t256). The same structure helps on CUDA (one big cuBLAS
  call vs B small ones); confirm the GPU magnitude on an L4 with `TINYBIT_PROFILE=1`.

### Removed
- **The HTTP server (`tinybit serve`) and its OpenAI-compatible API.** tinybit
  V1.0 is local CLI inference first; `axum`/`tower`/`tower-http` dependencies are
  gone and `tokio` is trimmed to what the download path needs.
- **PLAN.md** (a 1.9k-line implementation-plan dev dump) and **RUN.md** (folded
  into TRAINING.md).

### Notes / known limitations
- No pretrained weights ship — you train your own (see TRAINING.md).
- Tool *calling* needs instruction/tool fine-tuning to be reliable; the
  detect→execute→inject loop is complete and tested.
- Quantized export packs ternary at ~1.6 bits/weight (5/byte, base-3) → ~3.2×
  smaller on disk for micro. Loads back as f32 (storage win, NOT a speed win) and
  is *post-training*, so it is lossy on f32-trained models (micro perplexity
  ~83 → ~590) — use `ternary_ffn` quantization-aware training for near-lossless
  ternary. At tinybit's sizes (50–150M) quantizing is usually the wrong trade:
  the f32 file is already small and there's no speed gain — for speed, run
  full-precision on a GPU (`--features cuda`). No GGUF/llama.cpp export
  (incompatible architecture).
- The Muon optimizer is experimental; AdamW is the validated default.
- Measured throughput: 50M `micro` trains at ~6.5 s/step on an L4 (~1.9 days /
  ~$32 for 25k steps), 2.33× faster than before the WKV-backward fix.
