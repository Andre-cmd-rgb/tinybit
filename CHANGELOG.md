# Changelog

All notable changes to tinybit are documented here.

## [Unreleased]

### Docs
- **README**: the tools table now lists `lookup`; the commands table documents
  the chat tool gate (`--tools auto|always|never`); the user-extensible
  knowledge base path (`data/knowledge.json`) is documented.
- **TRAINING.md**: new "Next-run checklist" — custom-data balance
  (`CUSTOM_CHAT_EPOCHS=8`, ~15% tool-call ratio), `DATA_TOKENS` reminder, the
  pending `fused_ce`/batch-size live A/Bs, opt-in scheduler/optimizer knobs,
  and an eval gate (perplexity + tool-emission sanity) before promoting a
  checkpoint.

### Added
- **`lookup` tool — local fact retrieval (RAG-lite for a tiny model).** A 50M model
  can't reliably *store* facts, so it can learn to *fetch* them. `lookup` answers
  factual questions (capitals, geography, space, science, definitions) from a local,
  editable knowledge base — `crates/tinybit-tools/data/knowledge.json` (bundled) plus
  an optional `data/knowledge.json` you can extend. Matching is IDF-weighted token
  overlap with light prefix-stemming, so a generic shared word ("capital") can't
  wrong-match (france≠italy) and misses return a clear "No local entry" instead of a
  bluff. Registered in `with_builtins`; armed by the tool gate for factual (non-self)
  questions; taught by `datasets/chat-lookup-05.jsonl`. Goes live after a retrain (the
  current weights were never trained to emit it).
- **Tool gate for `tinybit chat` (`--tools auto|always|never`, default `auto`).**
  The current weights over-fire tools (they fired `todos` on "hi" — a data-balance
  artifact baked into training). `auto` suppresses tool emission unless the user's
  message plausibly needs one (`processor::message_needs_tools`) by banning the
  token that begins `<|tool_call|>` at sampling time; `always` is the raw model;
  `never` is pure conversation. A stopgap until a retrain on the rebalanced data
  fixes emission at the source. `eval` uses `always` (raw quality gate).

### Fixed
- **Tool-result injection no longer derails generation (train/inference tokenizer
  mismatch).** Data prep tokenizes the `<|tool_*|>` markers as ordinary BPE
  pieces, but the inference tokenizer was calling `add_special_tokens` to mint
  single ids (32000-32003) whose embedding rows the model never trained. So every
  injected `<|tool_result|>…<|end_tool_result|>` fed the model untrained
  embeddings, tipping it into HTML/code garbage right after any tool call.
  `Tokenizer` now resolves a marker to a single id ONLY if the tokenizer file
  already defines one (`resolve_marker`), matching training. Visible effect:
  `1+324` → `[calculator … -> 64] 64.` instead of a wall of `</time><ul>…`.
- **Tool-call detection scans every token.** `ToolProcessor` checked for
  `<|end_tool_call|>` only every 4 tokens, so 1-3 of the model's own post-call
  tokens were stepped into the recurrent state before the real result was
  injected. It now detects per token (the streaming path already decodes per
  token, so no added cost).

### Changed
- **Chat default temperature lowered 0.7 → 0.4.** At tinybit's size a hot
  temperature samples into low-probability tails and derails (incoherent text,
  spurious tool calls); 0.3-0.4 stays coherent. Affects `tinybit chat` and the
  `SamplingParams` default.
- **General data mix retuned to language-first (no code).** Goal: the best small
  *assistant* — English fluency, comprehension, summarising, and instruction
  following — with facts delegated to the `lookup` tool instead of memorised.
  Removed the-stack (Python code) entirely; bumped OpenHermes (0.15→0.20) and
  dolphin (0.07→0.08) for instruction/summarising/Q&A; bumped TinyStories
  (0.12→0.15) for clean simple English; trimmed Cosmopedia (0.30→0.25, still the
  reasoning/English source but not optimised for fact recall); FineWeb-Edu stays
  the backbone (0.33→0.32). Added `datasets/chat-summary-06.jsonl` (41
  summarise/explain/paraphrase examples) to reinforce the headline skill. Takes
  effect on the next re-train.
- **Curated chat data rebalanced to stop tool over-firing.** The
  `identity-tools-*` set is ~54% tool calls and the first definitive run repeated
  it at `CUSTOM_CHAT_EPOCHS=50`, teaching the model that a tool call is the
  default reply (it fired tools on greetings and plain questions). Added
  `datasets/chat-notools-04.jsonl` (90 no-tool examples), rewrote
  `prompts/tinybit-identity-tools-dataset.md` to generate ~15% tools (was ~45%)
  with a new "tool-shaped but answer directly" category, and lowered the
  recommended `CUSTOM_CHAT_EPOCHS` to ~8. Takes effect on the next re-train.

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
  the f32 file is already small and there's no speed gain. No GGUF/llama.cpp
  export (incompatible architecture).
- GPU helps throughput, not tiny-model decode (measured on an A2000): batched
  `eval`/training ~31× faster on GPU (15-batch perplexity 330 s → 10.6 s), but
  single-token `chat` ~3.5× *slower* than CPU (~21 vs ~75 tok/s) — launch-bound at
  50M. Build `--features cuda` for the GPU; on Windows build from a VS dev shell so
  nvcc finds `cl.exe` (CLAUDE.md decision 12). Force CPU with `CUDA_VISIBLE_DEVICES=-1`.
- The Muon optimizer is experimental; AdamW is the validated default.
- Measured throughput: 50M `micro` trains at ~6.5 s/step on an L4 (~1.9 days /
  ~$32 for 25k steps), 2.33× faster than before the WKV-backward fix.
