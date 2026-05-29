# Changelog

All notable changes to tinybit are documented here.

## 1.0.0

First serious release. tinybit is now a coherent, local-first Rust AI assistant:
small, fast, tool-aware, measurable, and honest about its limits.

### Added
- **`tinybit eval`** — measures perplexity over a token file and runs greedy
  generation-sanity prompts (with tok/s), so model quality is *measured*, not
  guessed.
- **Two model families** — *general* (`nano`/`micro`/`small`/`base`) and
  *coding* (`*-coding`). New `configs/*-coding.toml` for all four sizes and a
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
- Quantized export is a disk-size win, not yet a speed win (loads as f32). GGUF
  export is not implemented.
- The Muon optimizer is experimental; AdamW is the validated default.
- Measured throughput: 50M `micro` trains at ~6.5 s/step on an L4 (~1.9 days /
  ~$32 for 25k steps), 2.33× faster than before the WKV-backward fix.
