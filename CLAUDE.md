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

3. Muon optimizer ONLY for 2D weight matrices. AdamW for everything else.
   See optimizer/muon.rs for the Newton-Schulz implementation.

4. Tool calls use special tokens, not a separate classifier.
   The model is trained to output <|tool_call|>JSON<|end_tool_call|>.
   See tools/parser.rs for detection logic.

5. Tokenizer is LLaMA format (32k vocab, SentencePiece BPE).
   Special tokens are added on top. IDs are deterministic — do not change.

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

11. GCP training is launched only via `scripts/gcp_launch.sh`. It uploads the
    repo, generates a RUN_ID, and tries (zone × profile) combinations until one
    VM is created — then stops. Stage failures upload FAILED.json and shut the
    VM down (unless KEEP_VM_ON_FAILURE=1).

12. cudarc 0.13 needs CUDA <= 12.8. The startup script installs cuda-toolkit-12-8
    via NVIDIA's apt repo and exports CUDA_ROOT/PATH=/usr/local/cuda-12.8 before
    `cargo build`. Do not assume the image's `/usr/local/cuda` symlink is right.

## Common mistakes to avoid

- Do NOT use .unwrap() in library code — propagate with anyhow::Result + ?
- Do NOT load entire dataset into RAM — use memory-mapped TokenDataset
- Do NOT mix training mode and inference mode forward passes
- Do NOT forget to call .detach() on state tensors to stop gradient tracking through state
- Time-mix token shift: during training use rolled x (shift by 1 along T dim);
  during inference use state.time_shift (previous actual token embedding)
- candle_nn::Linear::forward() requires `use candle_nn::Module;` in scope
- Use candle_nn::Init::Const(val) not candle_nn::init::Const(val)
