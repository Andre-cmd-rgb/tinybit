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

3. The Muon optimizer module exists at `crates/tinybit-train/src/optimizer/muon.rs`
   for the Newton-Schulz orthogonalization but is NOT currently wired into the
   trainer — `Trainer::run` uses candle's `AdamW` for all parameters. Wiring
   Muon back in (for 2D weight matrices, AdamW for everything else) is a
   future optimization, not a current invariant.

4. Tool calls use special tokens, not a separate classifier.
   The model is trained to output <|tool_call|>JSON<|end_tool_call|>.
   See tools/parser.rs for detection logic.

5. Tokenizer is LLaMA format (32k vocab + 8 special tokens = 32008).
   The 8 extras are tool-call markers. IDs are deterministic — do not change.

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

16. The RWKV-7 WKV scan in `crates/tinybit-core/src/model/time_mix.rs` is a
    sequential candle loop, NOT a fused CUDA kernel. Each timestep allocates
    a fresh state/outer tensor that the autograd graph keeps for backward,
    so peak training VRAM scales linearly with
    `batch_size × max_seq_len × num_layers`. On the L4's 22.5 GB free VRAM,
    the 16-layer micro model fits at `batch_size = 2, max_seq_len = 512`;
    larger combinations OOM. Both `configs/micro.toml` and the L4 train
    configs are tuned to this budget — don't bump them without changing
    the scan implementation.

## Common mistakes to avoid

- Do NOT use .unwrap() in library code — propagate with anyhow::Result + ?
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
