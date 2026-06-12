# tinybit — notes for Claude Code

## Build
cargo build --release --workspace

## Test
cargo test --workspace

## Key design decisions (do not change without updating this file)

1. RWKV-7 NOT transformer — no attention, no KV cache, O(1) inference memory.
   State is InferenceState with LayerState per layer.

2. BitLinear uses STE (straight-through estimator) — during training, gradients
   flow through as if no quantization happened. Quantize only for export.

3. The Muon optimizer (`crates/tinybit-train/src/optimizer/muon.rs`) is wired
   into the trainer as an OPT-IN. Set `optimizer = "muon"` in the train TOML to
   drive the 2D hidden weight matrices with Muon (Newton-Schulz orthogonalized
   updates, LR = `muon_lr`, default 0.02) while candle's `AdamW` handles the
   tied embedding/LM-head, norms, and biases. The DEFAULT (field absent) is
   AdamW for ALL parameters — the documented L4 runs and their loss targets in
   TRAINING.md assume AdamW, so do NOT change the default without re-validating a
   full run. Muon's quality benefit is unverified at scale; only its mechanical
   correctness (runs, no NaNs, loss decreases) has been smoke-tested. See
   `apply_muon` and the param split in `Trainer::run`.

4. Tool calls use the marker protocol <|tool_call|>JSON<|end_tool_call|>, NOT a
   separate classifier. `parse_tool_call` (tools/parser.rs) detects them in the
   DECODED text, so they work regardless of how many tokens a marker spans. The
   detect→execute→inject→continue loop (infer/processor.rs) scans for the
   markers EVERY token (not every Nth): the markers are multi-token BPE here, so
   a periodic scan detected `<|end_tool_call|>` up to N-1 tokens late and
   forward-stepped the model's own half-written `<|tool_result|>` into the
   recurrent state before the real result was injected — corrupting state and
   tipping the model into post-call garbage. The base-pretraining data contains
   no tool-call demonstrations, so reliable EMISSION needs instruction/tool
   fine-tuning — treat tool calling as experimental. NOTE: even with the
   custom chat data, tool EMISSION over-fires unless the data is balanced and
   `CUSTOM_CHAT_EPOCHS` is low (see datasets/README.md / decision 17). As an
   inference-side stopgap, `chat` has a TOOL GATE (`--tools auto|always|never`,
   default auto): `ToolMode`/`message_needs_tools` (processor.rs) decide per turn
   whether a tool is plausibly needed, and if not the sampler bans the token(s)
   that begin `<|tool_call|>` (`InferenceEngine::tool_start_ban`) so the model
   can't emit one. The ban set must include BOTH the bare `<` and space-prefixed
   `▁<` ids — SentencePiece encodes `<` differently by preceding char and the
   model emits the bare form after `assistant:\n`. `eval` runs `always` (raw).
   Built-in tools (tinybit-tools/src/builtin/): time, calculator, lookup, todos,
   notes, calendar. `lookup` is local fact retrieval over a knowledge base
   (`tinybit-tools/data/knowledge.json` bundled via include_str! + an optional
   `data/knowledge.json` user extension) with IDF-weighted token matching — the
   tiny-model answer to factual recall (fetch, don't memorize). The gate arms it
   for factual non-self questions. It only becomes useful after a retrain that
   includes `datasets/chat-lookup-05.jsonl`; the base + current micro checkpoint
   never saw lookup, so they don't emit it.

5. Tokenizer is LLaMA format: 32000 BPE tokens; the model embedding is 32008 rows
   (8 spare slots). The <|tool_*|> markers are NOT installed as special tokens —
   data prep (prepare_data.py) tokenizes them as ordinary BPE pieces, so the
   model trains on that multi-token spelling. Inference MUST match: `resolve_marker`
   (core/tokenizer.rs) only treats a marker as a single id if the tokenizer file
   ALREADY defines one (the shipped LLaMA tokenizer does not, so `tool_*_id` are
   all None and `supports_tool_tokens()` is false). Do NOT call `add_special_tokens`
   to mint them — minting ids 32000-32003 feeds the model UNTRAINED embedding rows
   when a tool result is encoded back into context, which derailed generation (the
   2026-06-07 fix). The 8 spare embedding rows are currently untrained. A future
   checkpoint that trains marker tokens (data prep would have to add them too)
   picks up the single-token fast path automatically. IDs are deterministic.

6. All configs are in configs/*.toml. No magic numbers in model code.
   Everything reads from ModelConfig.

7. Training data is binary u32 (little-endian), memory-mapped.
   scripts/prepare_data.sh produces data/train.bin and data/val.bin.

8. Checkpoints are safetensors + JSON meta. Never pickle.

9. candle-core is the tensor framework. No PyTorch bindings.
   Metal support is via candle's "metal" feature (auto on macOS aarch64).
   CUDA support is via candle's "cuda" feature (enabled on GCP).

10. tokenizers crate uses fancy-regex (pure Rust) instead of onig (needs C++).
    This avoids a C++ compiler dependency on Linux.

11. GCP training is launched only via `scripts/gcp_launch.sh`. L4 is the
    ONLY supported hardware — no T4/G4/A100/H100 fallbacks. The launcher
    uploads the repo, generates a RUN_ID, and tries each zone in order
    until an L4 VM is created. Stage failures upload FAILED.json and shut
    the VM down (unless KEEP_VM_ON_FAILURE=1).

12. cudarc 0.13 needs CUDA <= 12.8. The startup script installs cuda-toolkit-12-8
    via NVIDIA's apt repo and exports CUDA_ROOT/PATH=/usr/local/cuda-12.8 before
    `cargo build`. Do not assume the image's `/usr/local/cuda` symlink is right.

    Local CUDA build (Fedora / newer GCC): CUDA 12.8's nvcc only accepts host GCC
    <= 14. The GCP L4 image (Ubuntu 22.04, GCC 11) is fine, but a modern Fedora
    (GCC 15/16) fails the `candle-kernels` build — `-allow-unsupported-compiler`
    bypasses the version assert but nvcc then can't parse the new libstdc++
    headers (`char8_t undefined`, …). Fix: install a supported compiler and point
    nvcc at it, e.g.
        sudo dnf install gcc14 gcc14-c++
        export CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
        export NVCC_CCBIN=/usr/bin/g++-14
        cargo build --release -p tinybit-cli --features cuda
    (A stale `target/.../candle-kernels-*` from an older GCC can make `cargo build`
    appear to work in debug while a fresh/release build fails — `cargo clean -p`
    candle-kernels if the cache lies.)

    Local CUDA build (Windows / MSVC): nvcc needs the MSVC host compiler `cl.exe`
    on PATH, which a plain PowerShell/cmd does NOT have (rustup drives the MSVC
    *linker* directly but never puts cl.exe on PATH) → `nvcc fatal: Cannot find
    compiler 'cl.exe' in PATH` and the candle-kernels build dies. Fix: build from
    the "x64 Native Tools Command Prompt for VS", OR load the dev env into
    PowerShell first:
        $vs = "C:\Program Files\Microsoft Visual Studio\2022\Community"
        Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
        Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
        cargo build --release -p tinybit-cli --features cuda
    At RUN time the toolkit's bin (e.g. `C:\Program Files\NVIDIA GPU Computing
    Toolkit\CUDA\v12.8\bin`) must be on PATH so cudart/cublas DLLs load. Verified
    on an RTX A2000 (sm_86): builds clean and `eval`/`chat` report `device: cuda`.

13. prepare_data.sh streams tokens directly to disk with numpy uint32 (4 bytes/token).
    Never store tokens in a Python list — at 1B tokens that is ~28 GB of RAM.
    The script writes to a temp file, then copies the tail as val.bin and the head as
    train.bin without ever loading the full dataset into memory.

14. Gradient accumulation uses per-microbatch backward: each microbatch is forward-passed,
    immediately backpropped (freeing its graph), and gradients are accumulated in a
    GradStore. The full computation graph for only ONE microbatch is in VRAM at a time.
    See trainer.rs for the merge-GradStore pattern.

15. Startup script adds 32 GB swap early (before data prep and cargo build) so that
    the L4 VM (16 GB RAM) cannot OOM during large downloads or compilation.

16. The RWKV-7 WKV scan has two implementations in
    `crates/tinybit-core/src/model/`: a fused CUDA kernel (`wkv.rs`) and the
    sequential candle loop (`time_mix.rs`). `time_mix::forward_train` selects via
    `wkv::fused_wkv_enabled(device)`: DEFAULT is the fused kernel on CUDA and the
    loop on CPU; `TINYBIT_FUSED_WKV=on|off` overrides either way.

    Fused kernel (`wkv.rs`): candle `CustomOp2` (`WkvScan`, forward) + `CustomOp3`
    (`WkvBackwardOp`, backward); CUDA kernels in `WKV_CUDA_SRC`, compiled per
    `head_dim` via nvrtc (`-D DH=<dh>`), one block per (batch,head), DH threads.
    The backward is chunk-checkpointed (chunk C≈√T): it stores only per-chunk entry
    states and recomputes within-chunk states, so its scratch is O(B·H·√T·dh²)
    (~28 MB at the L4 micro shape) instead of O(B·H·T·dh²), freed after each layer's
    backward. The backward's `dv` cross-thread reduction (sum over rows) uses a
    padded shared buffer + column sum — NOT `atomicAdd` (see the profiling note in
    "Common mistakes"; the atomic form was ~9x slower). The forward retains only
    O(T·dh) autograd state per layer (vs the loop's O(T·dh²)). Validated on GPU
    against the gradient-checked CPU references (parity ~1e-7 incl. T=512; CPU-vs-loop
    4.5e-8) → numerically equivalent, so checkpoints are compatible (resume across the
    switch, don't restart). Tests:
    `cargo test -p tinybit-core --features cuda -- --test-threads=1` (cuda_* parity)
    and `-- --ignored --nocapture bench` (speedup; shape overridable via
    `TINYBIT_BENCH_B`/`TINYBIT_BENCH_T`/`TINYBIT_BENCH_SKIP_LOOP`).

    Speedup (RTX A2000, debug, after the 2026-05-29 backward fix): fused WKV
    fwd+bwd is ~25x the candle loop and ~8x the pre-fix fused kernel; a full micro
    step (16L, d384) is ~6x the loop / ~3x the pre-fix kernel. The forward is ~2%
    of the scan; the backward is the rest. At the real micro batch (b=11, t=512)
    one layer's fwd+bwd is ~33 ms (≈0.5 s across 16 layers), down from ~4.4 s.
    Confirmed on a live L4 (resumed from a checkpoint with the new code):
    **6.5 s/step at b=11, vs 15.2 before — 2.33x — so the 25k-step micro run is
    ~1.9 days (~$32 on-demand), down from ~4.4 days.**

    VRAM budget: with the fused kernel the old "micro fits only at batch_size=2,
    max_seq_len=512 or it OOMs" limit (caused by the loop's O(T·dh²)×layers retained
    graph) no longer governs CUDA — there is now headroom to raise batch_size. The
    L4 configs in `configs/*.toml` are NOT yet re-tuned for this; raise batch_size
    only after a live L4 run confirms the new headroom. The loop path (CPU, or
    `TINYBIT_FUSED_WKV=off`) still carries the old budget.

17. Two model families, ONE architecture. `general` (micro/bit/qbit) and
    `coding` (`*-coding`) share byte-identical arch configs; they differ only in
    (a) the training-data mix and (b) the default system prompt. The mix is
    chosen by `DATA_PROFILE=general|coding` in prepare_data.sh; the persona by
    `Profile` (core/tokenizer.rs) — `--profile`, or inferred from a config name
    containing "coding". A checkpoint loads under either config. Do NOT diverge
    the `*-coding.toml` shapes from their siblings. The shipped lineup is
    micro≈50M / bit≈100M / qbit≈150M (`ModelConfig::{micro,bit,qbit}` +
    `configs/{micro,bit,qbit}{,-coding}.toml`), each with an L4 train config
    (`configs/train-{micro,bit,qbit}-l4.toml`; micro is the validated batch-11
    target, bit/qbit use smaller batches to fit 24 GB). Custom curated
    chat data (e.g. identity/tool JSONL) lives in `datasets/` and is mixed in by
    prepare_data.sh (tokenized last, repeated `CUSTOM_CHAT_EPOCHS`×; see
    decision 18 / datasets/README.md).

18. Prompt format is SHARED between training and inference and must stay that
    way. The canonical template is the `ROLE_*_PREFIX` constants in
    core/tokenizer.rs (`system:\n…\nuser:\n…\nassistant:\n…`). prepare_data.sh
    formats conversation datasets (OpenHermes/dolphin) with the SAME strings, and
    `STOP_STRING_USER_TURN` ("\nuser:") is how generation stops. If you change
    the template, change BOTH places or chat and training silently disagree.

19. V1.0 is local-CLI-first: subcommands are chat / eval / train / convert /
    download. There is NO `serve`/HTTP server (axum/tower were removed). `eval`
    (cli/commands/eval.rs) reports val perplexity + greedy generation sanity and
    is the measure-don't-guess quality gate. Don't reintroduce a server without
    a deliberate decision — it was removed to keep the project local-first.

20. Linear projections run as a SINGLE GEMM over flattened batch×time rows
    (`linear_flat` in model/bitlinear.rs), used by BitLinear, `linear_autocast`,
    and the LM head. This replaced candle's `broadcast_left` + batched matmul (B
    small GEMMs with the weight replicated). It is numerically EXACT (matmul rows
    are independent) — pinned by a deterministic unit test in bitlinear.rs and by
    the cuda_* / grad_flow parity tests — so checkpoints are unchanged. Do NOT
    revert to the broadcast/bmm form. Measured: 1.5–1.7× per projection, 1.30× on
    a full micro CPU step. Single-token inference (2D input) passes straight
    through unchanged.

21. Data prep (`scripts/prepare_data.py`) MUST stay hang-proof. HF streaming
    deterministically wedges in a C-level read (~60% into fineweb-edu); the ONLY
    reliable recovery is a killable child PROCESS (not a thread — a thread
    blocked in C can't be killed). The contract, do not weaken any part:
    (a) each dataset streams in a `multiprocessing` child the parent kills on a
    queue-`get` timeout; (b) restarts resume via HF `IterableDataset.state_dict()`
    checkpoints (offset re-scan fallback), base = `global_idx+1`; (c)
    `MAX_RESTARTS_NO_PROGRESS` consecutive zero-progress restarts → write
    `data/prepare_FAILED.json` + exit non-zero (NEVER loop forever, NEVER silently
    skip); (d) `heartbeat_unix`/`phase`/`total_written` are persisted to
    `data/prepare_progress.json` and logs are one-line `EVENT {json}`. The module
    is import-safe (no network at import) so it runs under spawn AND is testable.
    `startup.sh::run_data_prep_watched` runs it in its own process group, syncs
    progress to the bucket every 30s, and kills+FAILS the run if the heartbeat is
    stale (>15m) or tokens flatline (>45m) — a silent multi-hour hang is the bug
    this prevents (it froze run 20260602-074307). Guarded by
    `scripts/test_prepare_data.py` (offline, `TINYBIT_FAKE_STREAM=1`): run it after
    touching prepare_data.py.

22. Quantization is a STORAGE format, not a speed feature — and at tinybit's sizes
    (50–150M) it's usually the wrong trade. `convert --quantize` packs 2D matrices
    to ternary (5/byte, base-3, ~1.6 bit/weight; tied embedding kept f32) and
    `TinyBit::load` DEQUANTIZES back to f32 and runs the SAME path — so there is NO
    inference-speed win, only smaller on-disk size, and post-training ternary on an
    f32-trained model (ternary_ffn=false, e.g. micro) is LOSSY (measured micro
    perplexity ~83 → ~590, val.bin). For SPEED prefer full-precision weights + the
    GPU (build `--features cuda`; eval/chat/train auto-select cuda via
    `InferenceEngine::auto_device` / `Trainer::auto_device`) — RWKV-7 is O(1)
    inference (no KV cache), so spending more RAM/VRAM buys throughput cleanly.
    Only `--quantize` a model trained quantization-aware (bit/qbit,
    ternary_ffn=true), where ternary is near-lossless. The packer is base-3
    (`quantize.rs`, 5 trits/byte) — a true ternary-MATMUL runtime (the real speed
    payoff of BitLinear) is unbuilt; do NOT claim a speed win until it exists.

23. MEASURED GPU vs CPU for inference — do NOT assume the GPU is always faster.
    On an RTX A2000 with the 50M micro model: BATCHED work (training, `eval`
    perplexity) is ~31x faster on GPU (15-batch perplexity 330 s CPU → 10.6 s GPU),
    but single-token interactive DECODE (`chat`) is ~3.5x SLOWER on GPU (~21 tok/s
    vs ~75 tok/s CPU) — at 50M the per-token work is tiny so GPU kernel-launch
    latency dominates. Guidance: use the GPU for training and `eval`; for
    interactive `chat` of a tiny model CPU is fine or better. `auto_device` picks
    CUDA whenever a cuda-built binary sees a GPU — force CPU with
    `CUDA_VISIBLE_DEVICES=-1`. (Bigger models or batched decode tilt back to GPU.)
    This is why quantization-for-speed (decision 22) is doubly pointless here: the
    bottleneck for small-model decode is launch latency, not weight bandwidth.

24. SEQUENCE PREFILL: prompts and injected tool results are prefilled with
    `TinyBit::forward_prefill` (model/mod.rs) — chunks of `PREFILL_CHUNK=128`
    through `forward_seq` (TimeMix/ChannelMix/Rwkv7Block), which runs every
    projection as ONE GEMM over the chunk rows and seeds/writes the
    `InferenceState` (token shift from `state.time_shift`/`ffn_shift`, WKV scan
    init from `state.wkv_state`). The fused WKV CustomOp is TRAINING-ONLY and
    returns no state — do NOT extend it for prefill (single-return-tensor
    contract; the scan is ~1% of prefill FLOPs anyway). The scan loop is shared
    (`wkv_loop`) between `forward_train` (zero init) and `forward_seq` (state
    init) — `forward_train`'s output must stay bit-identical. Invariants:
    `time_shift` = last token's post-ln1 input, `ffn_shift` = last token's
    post-ln2 input; logits + EVERY state tensor parity vs `forward_step` is
    pinned by `test_prefill_matches_step` / `test_prefill_then_decode_matches`
    (tests/model_correctness.rs). NEVER feed the last token of a prefilled
    span again in decode (the prompt prefill feeds head-only; tool-result
    injection feeds all-but-last — the old all+last double-feed was a state
    corruption bug, fixed 2026-06-12). Escape hatch: `TINYBIT_SEQ_PREFILL=off`
    falls back to per-token stepping.

25. INTEGRATIONS ARE FILES, NOT SOCKETS. The "tinybit data API" (INTEGRATIONS.md)
    is `data/integrations/<source>/{events.jsonl,latest.json,meta.json}` +
    `tinybit ingest` + the `user_data` tool — reaffirming decision 19 (no
    server). Shared implementation: `tinybit_tools::integrations`
    (`IntegrationsStore`), used by both the CLI and the tool. Contract:
    third-party DIRECT appends of complete JSON lines are allowed (that's the
    language-agnostic point); readers must SKIP AND COUNT malformed lines,
    never fail; `latest.json` writes are temp-file + atomic rename; JSONL not
    SQLite so any language can write. `parse_time_spec` accepts
    today/yesterday/now/<N>d/<N>h — trainable forms for a 50M model. Keep tool
    OUTPUT strings stable: `datasets/chat-userdata-08.jsonl` mirrors them
    byte-for-byte (training/inference agreement, same rule as decision 18).

26. ONE RETRIEVAL TOOL. Document search lives INSIDE `lookup` (lookup_tool.rs +
    doc_index.rs): curated KB first, then BM25 over paragraph chunks of
    `data/docs/*.{md,txt}` (heading folded into match tokens, ≥2-matched-terms
    evidence threshold), result rendered `From <file>: <chunk>` trimmed to
    ~400 chars (tool results feed a 50M recurrent state — keep them short).
    The docs dir is fingerprinted (mtime+size) per query, index rebuilt only
    on change (live pickup, no watcher). The `No local entry for "…"` phrasing
    is LOAD-BEARING — training data teaches the honest-miss behavior against
    exactly that string. Do NOT split doc search into a second tool: a second
    tool name needs its own gate rules, its own training data, and a 50M model
    that can discriminate between two retrieval tools — all cost, no benefit.

## Common mistakes to avoid

- Do NOT use .unwrap() in library code — propagate with anyhow::Result + ?
- The sampler's repetition penalty is PER OCCURRENCE (`penalty^count` per
  token, sign-preserving) — pinned by tests in sampler.rs. Do NOT "fix" it to
  HF's once-per-distinct-token semantics without revalidating generation
  quality. Top-p's tiered partial selection uses a strict total order
  (prob desc, index asc) precisely so it stays bit-identical to the old
  stable full sort — keep any comparator change total.
- The generation loop's per-token text comes from `Tokenizer::decode_stream`
  (IncrementalDecoder). Its chunk concatenation must stay byte-identical to a
  one-shot decode (pinned in tests/tokenizer.rs incl. a mini LLaMA-style
  tokenizer that runs without tokenizer.json). Do NOT reintroduce a
  full-buffer decode per token (O(n²) per turn), and do NOT scan markers less
  than every token (decision 4).
- Do NOT load entire dataset into RAM — use memory-mapped TokenDataset
- Do NOT mix training mode and inference mode forward passes
- Do NOT forget to call .detach() on state tensors to stop gradient tracking through state
- Time-mix token shift: during training use rolled x (shift by 1 along T dim);
  during inference use state.time_shift (previous actual token embedding)
- candle_nn::Linear::forward() requires `use candle_nn::Module;` in scope
- Use candle_nn::Init::Const(val) not candle_nn::init::Const(val)
- Do NOT store tokens in a Python list — use numpy arrays or stream-write to disk
- Always pass DATA_TOKENS=1500000000 (not the default 20M) when training micro on L4
- GradStore::new() is pub(crate) in candle — cannot be created externally.
  Initialize from the first microbatch's backward() result, then merge subsequent ones.
- Do NOT raise `max_seq_len` or `batch_size` in the L4 configs without first
  verifying the WKV scan still fits in VRAM. See design decision 16.
- Do NOT use `candle_nn::LayerNorm` (or `candle_nn::ops::layer_norm`) for the
  pre-norms or head norm. candle dispatches contiguous affine inputs to a fused
  op registered with `apply_op3_no_bwd` — it has NO backward and silently drops
  all gradient, which froze the entire stack (loss stuck near ln(vocab), gnorm
  ~0.007, unigram-gibberish output) before 2026-05-28. Use the hand-rolled
  differentiable LayerNorm (primitive ops) in the model. `tests/grad_flow.rs`
  guards this — it asserts every parameter gets a finite gradient and the model
  overfits a fixed batch. Run it after touching any norm.
- WKV time-decay is `w = exp(-exp(time_decay))` (w ∈ (0,1)), NOT
  `softplus(-exp(td))`. The softplus form caps state retention at ln2≈0.69,
  limiting memory to ~2 tokens. Do not "simplify" it back.
- Do NOT reintroduce a per-parameter (or per-microbatch) `.to_scalar()` in the
  training loop. `global_grad_norm`/`clip_grad_norm` accumulate each param's
  f32 sum-of-squares on-device and pull the whole vector to the host in ONE
  sync; the per-microbatch loss is accumulated detached on-device and synced
  once per step. Each host read forces a CUDA stream sync — doing one per param
  (~150-200/step) serialized the GPU behind launch+sync latency. Keep reads at
  one-per-step. The per-microbatch loss accumulator MUST stay `.detach()`ed or
  it retains every microbatch graph and defeats the VRAM-bounded accumulation.
- To see where a step's wall time goes, set `TINYBIT_PROFILE=1`: the trainer
  logs forward/backward/optimizer time per step (device synced at phase
  boundaries). The fused WKV scan's FORWARD is already cheap (~2% of the scan);
  the cost was the BACKWARD. Until 2026-05-29 the backward reduced `dv` with a
  per-timestep storm of shared-memory `atomicAdd`s — all `dh` threads contending
  on the same `dh` addresses — which was ~98% of the scan's fwd+bwd wall time.
  It now writes each thread's row into a padded `contrib[dh][dh+1]` shared buffer
  and sums down columns (conflict-free, no atomics): the backward dropped ~9x
  (110→12 ms at b4×t512 on an A2000) and the whole scan ~8x. Do NOT reintroduce
  the per-step `atomicAdd` reduction. Any WKV kernel change must keep
  `w = exp(-exp(td))` and pass the `cuda_*` parity tests at T=512
  (`cargo test -p tinybit-core --features cuda -- --test-threads=1 cuda_`).
  The kernel launches `B*H` blocks of `dh` threads; occupancy is fine at the
  micro batch (b=11 → 66 blocks) — a chunked parallel scan over time would add
  more, but is unneeded now that the atomic storm is gone.
