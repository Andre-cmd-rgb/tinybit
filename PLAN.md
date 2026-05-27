# tinybit v0 — Full Implementation Plan for Claude Code

> Apache 2.0 | Rust | RWKV-7 + BitLinear | Configurable sizes (10M → 400M)
> Train on Google Cloud (free credits) | Run on Linux, macOS (M-series), Windows

---

## 0. What you are building

A Rust workspace called `tinybit` that produces:

1. A **local AI assistant** model (RWKV-7 + ternary BitLinear, configurable size)
2. A **tool system** (time, calculator, todos, notes, calendar — user-extensible)
3. A **training pipeline** that runs on Google Cloud with zero external API cost
4. A **CLI** (`tinybit`) that works on Linux, macOS (native ARM/Metal), Windows (AVX2/AVX-512)
5. An **HTTP inference server** (`tinybit serve`) for Oracle Cloud / local hosting
6. Full tests for every module before any training run

The model is a personal assistant that knows about science, maths, economics, history, and general knowledge — trained on free open datasets (FineWeb-Edu, Wikipedia, The Stack Smol).

---

## 1. Repository layout (create every file listed)

```
tinybit/
├── Cargo.toml                        # workspace root
├── CLAUDE.md                         # notes for Claude Code
├── README.md                         # user guide (write last)
├── LICENSE                           # Apache 2.0
├── .gitignore
├── rust-toolchain.toml               # pin to stable 1.82
│
├── crates/
│   ├── tinybit-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs             # ModelConfig + size presets
│   │       ├── tokenizer.rs          # wraps HF tokenizers crate
│   │       ├── state.rs              # InferenceState (fixed-size RWKV state)
│   │       ├── quantize.rs           # BitLinear quant utils
│   │       └── model/
│   │           ├── mod.rs            # TinyBit top-level model
│   │           ├── bitlinear.rs      # ternary BitLinear layer
│   │           ├── time_mix.rs       # RWKV-7 time-mix block
│   │           ├── channel_mix.rs    # RWKV-7 channel-mix (FFN) block
│   │           ├── block.rs          # RWKV7Block = time_mix + channel_mix
│   │           └── embedding.rs      # token embedding + tied LM head
│   │
│   ├── tinybit-tools/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tool.rs               # Tool trait + ToolOutput
│   │       ├── registry.rs           # ToolRegistry
│   │       ├── parser.rs             # parse <|tool_call|>…<|end_tool_call|>
│   │       └── builtin/
│   │           ├── mod.rs
│   │           ├── time_tool.rs      # current time/date/timezone
│   │           ├── calc_tool.rs      # math expression evaluator
│   │           ├── todos_tool.rs     # add/list/complete todos (SQLite)
│   │           ├── notes_tool.rs     # save/search notes (SQLite)
│   │           └── calendar_tool.rs  # events (SQLite)
│   │
│   ├── tinybit-infer/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── engine.rs             # InferenceEngine (owns model + state)
│   │       ├── sampler.rs            # greedy, top-p, top-k, temperature
│   │       ├── session.rs            # Session (conversation history + state)
│   │       └── processor.rs         # detect tool calls, execute, inject result
│   │
│   ├── tinybit-train/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── trainer.rs            # main training loop
│   │       ├── checkpoint.rs         # save/load mid-training
│   │       ├── scheduler.rs          # WSD learning rate schedule
│   │       ├── loss.rs               # cross-entropy + optional KL distillation
│   │       ├── optimizer/
│   │       │   ├── mod.rs
│   │       │   ├── muon.rs           # Muon (momentum orthogonalized)
│   │       │   └── adamw.rs          # AdamW for embeddings/biases/1D params
│   │       └── data/
│   │           ├── mod.rs
│   │           ├── loader.rs         # streaming data loader (memory-mapped)
│   │           ├── dataset.rs        # Dataset trait + TokenizedDataset
│   │           └── pack.rs           # bin-pack tokens into fixed-length chunks
│   │
│   └── tinybit-cli/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs               # clap root
│           └── commands/
│               ├── mod.rs
│               ├── chat.rs           # interactive REPL
│               ├── serve.rs          # axum HTTP server
│               ├── train.rs          # kick off training
│               ├── convert.rs        # export to safetensors / GGUF
│               └── download.rs       # pull tokenizer + pretrained weights
│
├── configs/
│   ├── nano.toml                     # 10M params
│   ├── micro.toml                    # 50M params
│   ├── small.toml                    # 150M params  ← default
│   └── base.toml                     # 400M params
│
├── scripts/
│   ├── prepare_data.sh               # download + tokenize datasets on GCP
│   ├── gcp_train.sh                  # provision GCP VM + launch training
│   ├── gcp_spot_train.sh             # preemptible spot version (cheapest)
│   └── eval.sh                       # run evaluation suite
│
├── data/
│   └── .gitkeep                      # datasets downloaded here, not committed
│
└── tests/
    ├── model_correctness.rs          # forward pass shapes, loss decreases
    ├── tool_system.rs                # each tool call round-trip
    ├── tokenizer.rs                  # encode/decode roundtrip
    ├── quantize.rs                   # ternary packing/unpacking
    └── training_smoke.rs             # 100-step smoke train, loss must fall
```

---

## 2. Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "crates/tinybit-core",
    "crates/tinybit-tools",
    "crates/tinybit-infer",
    "crates/tinybit-train",
    "crates/tinybit-cli",
]

[workspace.package]
version    = "0.1.0"
edition    = "2021"
license    = "Apache-2.0"
repository = "https://github.com/Andre-cmd-rgb/tinybit"
authors    = ["Andre"]

[workspace.dependencies]
# Tensor framework — pure Rust, CPU + Metal + CUDA
candle-core    = { version = "0.8", features = ["metal"] }
candle-nn      = { version = "0.8" }

# Tokenizer
tokenizers     = { version = "0.21", features = ["http"] }

# Serialization
serde          = { version = "1", features = ["derive"] }
serde_json     = "1"
toml           = "0.8"
safetensors    = "0.4"

# CLI
clap           = { version = "4", features = ["derive", "env"] }

# Async + HTTP
tokio          = { version = "1", features = ["full"] }
axum           = { version = "0.7", features = ["json"] }
tower          = "0.4"
tower-http     = { version = "0.5", features = ["cors", "trace"] }

# Errors + logging
anyhow         = "1"
thiserror      = "1"
tracing        = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Utilities
rayon          = "1.10"
indicatif      = { version = "0.17", features = ["rayon"] }
memmap2        = "0.9"
half           = { version = "2", features = ["num-traits"] }
rand           = "0.8"
chrono         = { version = "0.4", features = ["serde"] }

# Tools — built-in
evalexpr       = "11"
rusqlite       = { version = "0.31", features = ["bundled"] }

# HuggingFace Hub for downloading tokenizers
hf-hub         = { version = "0.3", features = ["tokio"] }

[profile.release]
opt-level      = 3
lto            = "thin"
codegen-units  = 1
panic          = "abort"

[profile.release.build-override]
opt-level = 3
```

---

## 3. rust-toolchain.toml

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu", "aarch64-apple-darwin", "x86_64-pc-windows-msvc"]
```

---

## 4. tinybit-core

### 4.1 `config.rs` — ModelConfig + presets

```rust
/// All hyperparameters for a model variant.
/// Every architectural decision lives here — no magic numbers in model code.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    // Architecture
    pub vocab_size:   usize,   // default 32000 (use LLaMA tokenizer)
    pub num_layers:   usize,   // transformer-equivalent depth
    pub d_model:      usize,   // hidden dimension (must be divisible by num_heads)
    pub d_ffn:        usize,   // channel-mix hidden dim (typically 3.5 * d_model)
    pub num_heads:    usize,   // RWKV-7 heads (state shape: num_heads × head_dim × head_dim)
    pub head_dim:     usize,   // = d_model / num_heads (computed, not set)

    // Quantization (set both false during training, true for inference export)
    pub ternary_ffn:  bool,    // apply ternary quantization to channel-mix weights
    pub int8_time:    bool,    // quantize time-mix R/K/V/O projections to INT8

    // Training
    pub max_seq_len:  usize,   // training context length (512 for small configs)
    pub dropout:      f64,     // 0.0 for inference, 0.05 for training

    // Speculative decoding heads
    pub spec_heads:   usize,   // 0 = disabled, 3 = predict t+2,t+3,t+4
}
```

Add a `fn param_count(&self) -> usize` method that estimates parameter count.

Add these four constructor functions:

```rust
impl ModelConfig {
    /// ~10M params — fastest, edge devices
    pub fn nano() -> Self {
        Self {
            vocab_size: 32000, num_layers: 6, d_model: 256,
            d_ffn: 512, num_heads: 4, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 512, dropout: 0.05, spec_heads: 0,
        }
    }

    /// ~50M params
    pub fn micro() -> Self {
        Self {
            vocab_size: 32000, num_layers: 12, d_model: 512,
            d_ffn: 1024, num_heads: 8, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 1024, dropout: 0.05, spec_heads: 0,
        }
    }

    /// ~150M params — default
    pub fn small() -> Self {
        Self {
            vocab_size: 32000, num_layers: 18, d_model: 768,
            d_ffn: 2048, num_heads: 12, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 1024, dropout: 0.05, spec_heads: 3,
        }
    }

    /// ~400M params
    pub fn base() -> Self {
        Self {
            vocab_size: 32000, num_layers: 32, d_model: 1024,
            d_ffn: 2048, num_heads: 16, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 2048, dropout: 0.05, spec_heads: 3,
        }
    }

    /// Load from a TOML config file
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self>

    /// Save to TOML
    pub fn to_file(&self, path: &std::path::Path) -> anyhow::Result<()>
}
```

### 4.2 `state.rs` — InferenceState

The RWKV state is a fixed-size matrix per layer per head. 
At inference, this replaces the KV cache. It is O(1) memory regardless of sequence length.

```rust
/// Per-layer inference state for RWKV-7.
/// Shape: [num_heads, head_dim, head_dim] — a matrix per head.
/// This accumulates a weighted sum of outer products k_t^T * v_t.
#[derive(Debug, Clone)]
pub struct LayerState {
    /// The recurrent state matrix W.
    /// Shape: (num_heads, head_dim, head_dim)
    /// dtype: f32 always (never quantized — state precision matters)
    pub wkv_state: candle_core::Tensor,

    /// Shift state for time-mix (previous token's embedding).
    /// Shape: (d_model,)
    pub time_shift: candle_core::Tensor,

    /// Shift state for channel-mix.
    /// Shape: (d_model,)
    pub ffn_shift: candle_core::Tensor,
}

/// Complete model inference state (one per active session).
#[derive(Debug, Clone)]
pub struct InferenceState {
    pub layers: Vec<LayerState>,
    pub device: candle_core::Device,
}

impl InferenceState {
    /// Allocate zeroed state for the given config on the given device.
    pub fn zeros(config: &ModelConfig, device: &candle_core::Device) -> anyhow::Result<Self>

    /// Save state to disk (for session persistence).
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()>

    /// Load state from disk.
    pub fn load(path: &std::path::Path, device: &candle_core::Device) -> anyhow::Result<Self>

    /// Clone the state (used for speculative decoding rollback).
    pub fn detach_clone(&self) -> anyhow::Result<Self>
}
```

### 4.3 `quantize.rs` — BitLinear quantization utilities

```rust
/// Quantize a weight matrix to ternary {-1, 0, +1}.
/// Uses the mean absolute value as the threshold (BitNet b1.58 method).
/// Returns (quantized_weights: Tensor[i8], scale: f32).
/// During training use with Straight-Through Estimator — gradients pass through unchanged.
pub fn quantize_ternary(w: &candle_core::Tensor) -> anyhow::Result<(candle_core::Tensor, f32)>

/// Quantize activations to INT8 with per-tensor scaling.
/// scale = max(|x|) / 127
/// Returns (quantized: Tensor[i8], scale: f32)
pub fn quantize_int8(x: &candle_core::Tensor) -> anyhow::Result<(candle_core::Tensor, f32)>

/// Dequantize: (w_ternary * scale_w * x_int8 * scale_x) / 127
pub fn dequantize(
    result: &candle_core::Tensor,
    scale_w: f32,
    scale_x: f32,
) -> anyhow::Result<candle_core::Tensor>

/// Pack two ternary values into one byte (4 bits each: 0b0000_00xx where xx in {0,1,2} = {-1,0,+1}).
/// Used for export only — during training keep as i8.
pub fn pack_ternary(weights: &[i8]) -> Vec<u8>

/// Unpack from packed ternary bytes back to i8 slice.
pub fn unpack_ternary(packed: &[u8], count: usize) -> Vec<i8>
```

### 4.4 `model/bitlinear.rs` — BitLinear layer

This is the core innovation layer. During training it uses full f32 weights with a
quantize-dequantize pass (STE). At inference it uses ternary weights for add/sub-only compute.

```rust
/// BitLinear: a Linear layer with ternary weights.
/// Replaces all standard Linear layers in the channel-mix (FFN) path.
///
/// Forward pass (training mode):
///   1. RMSNorm the input x → x_norm
///   2. Quantize weights W to ternary W_t using mean-abs threshold; STE keeps gradient flowing
///   3. Quantize x_norm to INT8 x_q with scale β = mean(|x_norm|)
///   4. Compute y = x_q @ W_t.T  (integer dot products)
///   5. Rescale: y = y * α * β / 127  where α = mean(|W|) before quantization
///
/// The STE is implicit: in backward(), treat W_t as if it were W (no quantization gradient).
/// Implement this by using candle's .detach() + adding back the full-precision residual.
pub struct BitLinear {
    /// Full-precision master weights (trained in f32, quantized on forward)
    weight: candle_nn::Linear,  // or raw Tensor if you want manual control
    /// Input RMSNorm (fused — saves one memory pass)
    norm: RmsNorm,
    /// Whether to actually apply ternary quantization (false during early training)
    pub quantized: bool,
    in_features:  usize,
    out_features: usize,
}

impl BitLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        vb: candle_nn::VarBuilder,
    ) -> anyhow::Result<Self>

    pub fn forward(&self, x: &candle_core::Tensor) -> anyhow::Result<candle_core::Tensor>
}

/// Simple RMSNorm — fused into BitLinear.
pub struct RmsNorm {
    weight: candle_core::Tensor,  // learnable scale γ
    eps: f64,
    d_model: usize,
}
impl RmsNorm {
    pub fn new(d_model: usize, vb: candle_nn::VarBuilder) -> anyhow::Result<Self>
    pub fn forward(&self, x: &candle_core::Tensor) -> anyhow::Result<candle_core::Tensor>
}
```

### 4.5 `model/time_mix.rs` — RWKV-7 time-mix

This is the recurrent "attention replacement". Implement exactly as follows:

```rust
/// RWKV-7 Time-Mix block.
///
/// Math (per timestep t, batch B, d_model D, num_heads H, head_dim dh = D/H):
///
///   # Linear projections (BitLinear for r,k,v,o — INT8 for time-sensitive paths)
///   r_t = x_t @ W_r      # receptance  (B, D)
///   k_t = x_t @ W_k      # key         (B, D)
///   v_t = x_t @ W_v      # value       (B, D)
///   g_t = silu(x_t @ W_g1) * (x_t @ W_g2)   # gating
///
///   # Time decay (data-independent, learned per head per channel)
///   w_t = softplus(-exp(time_decay))  # shape (H, dh), broadcast
///
///   # Reshape to heads
///   r_t = r_t.reshape(B, H, dh)   → apply group_norm per head
///   k_t = k_t.reshape(B, H, dh)
///   v_t = v_t.reshape(B, H, dh)
///
///   # State update (the recurrence — O(1) memory, O(1) per step)
///   # State shape: (B, H, dh, dh) — a matrix per head
///   state = w_t * state + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
///   #         ^^decay      ^^outer product accumulator
///
///   # Read out
///   y_t = (r_t.unsqueeze(-2) @ state).squeeze(-2)  # (B, H, dh)
///   y_t = group_norm(y_t)                           # stabilize
///   y_t = y_t.reshape(B, D) * g_t                  # apply gate
///   out = y_t @ W_o                                 # output projection
///
/// For a full sequence (training): parallelize with a scan or chunked computation.
/// For single-token inference: use the sequential update above with stored state.
pub struct TimeMix {
    w_r: BitLinear,     // receptance projection
    w_k: candle_nn::Linear,  // key (INT8, more sensitive — use standard linear with careful init)
    w_v: candle_nn::Linear,  // value (INT8)
    w_g1: candle_nn::Linear, // gate pre-activation
    w_g2: candle_nn::Linear, // gate second branch
    w_o: candle_nn::Linear,  // output projection

    time_decay: candle_core::Tensor,  // shape (num_heads, head_dim), learned
    group_norm: candle_nn::GroupNorm, // normalize per-head output

    // Token shift mixing (lerp with previous token — RWKV signature trick)
    time_maa_x: candle_core::Tensor,  // shape (d_model,)
    time_maa_r: candle_core::Tensor,
    time_maa_k: candle_core::Tensor,
    time_maa_v: candle_core::Tensor,

    num_heads: usize,
    head_dim:  usize,
    d_model:   usize,
}

impl TimeMix {
    pub fn new(config: &ModelConfig, vb: candle_nn::VarBuilder) -> anyhow::Result<Self>

    /// Training forward: process full sequence (B, T, D) → (B, T, D).
    /// Does NOT require external state — builds state internally via scan.
    pub fn forward_train(
        &self,
        x: &candle_core::Tensor,  // (B, T, D)
    ) -> anyhow::Result<candle_core::Tensor>

    /// Inference forward: process one token at a time.
    /// Reads and writes the LayerState in-place.
    pub fn forward_step(
        &self,
        x: &candle_core::Tensor,  // (B, D)
        state: &mut crate::state::LayerState,
    ) -> anyhow::Result<candle_core::Tensor>  // (B, D)
}
```

**Important note on the token-shift trick:**
Before projecting to r/k/v, RWKV mixes the current token with the previous token:
```
x_shifted = lerp(prev_x, x, time_maa_x)  # element-wise interpolation
r_input   = lerp(prev_x, x, time_maa_r)
k_input   = lerp(prev_x, x, time_maa_k)
v_input   = lerp(prev_x, x, time_maa_v)
```
The `prev_x` during training is `x` shifted by one position (roll along T dim).
During inference it comes from `state.time_shift`.

### 4.6 `model/channel_mix.rs` — RWKV-7 channel-mix (FFN)

```rust
/// RWKV-7 Channel-Mix (the FFN equivalent).
///
/// Math:
///   x_shifted = lerp(prev_x, x, time_maa_ffn)
///   k = silu(x_shifted @ W_k)         # key gate (BitLinear)
///   v = (k * k) @ W_v                 # squared activation → expand
///   r = sigmoid(x_shifted @ W_r)      # receptance gate (BitLinear)
///   out = r * v
///
/// The squared key activation (k*k before W_v) is the RWKV FFN secret —
/// it approximates a softmax nonlinearity cheaply.
pub struct ChannelMix {
    w_k: BitLinear,
    w_v: BitLinear,
    w_r: BitLinear,
    time_maa_k: candle_core::Tensor,  // (d_model,)
    time_maa_r: candle_core::Tensor,
    d_model: usize,
}

impl ChannelMix {
    pub fn new(config: &ModelConfig, vb: candle_nn::VarBuilder) -> anyhow::Result<Self>

    /// Training forward: (B, T, D) → (B, T, D)
    pub fn forward_train(
        &self,
        x: &candle_core::Tensor,
    ) -> anyhow::Result<candle_core::Tensor>

    /// Inference step: (B, D) → (B, D), updates ffn_shift in state
    pub fn forward_step(
        &self,
        x: &candle_core::Tensor,
        state: &mut crate::state::LayerState,
    ) -> anyhow::Result<candle_core::Tensor>
}
```

### 4.7 `model/block.rs` — Full RWKV7Block

```rust
/// One complete RWKV-7 layer = LN(x) → TimeMix → residual → LN(x) → ChannelMix → residual
pub struct Rwkv7Block {
    ln1: candle_nn::LayerNorm,
    ln2: candle_nn::LayerNorm,
    time_mix: TimeMix,
    channel_mix: ChannelMix,
}

impl Rwkv7Block {
    pub fn new(config: &ModelConfig, layer_idx: usize, vb: candle_nn::VarBuilder) -> anyhow::Result<Self>

    /// Training: (B, T, D) → (B, T, D)
    pub fn forward_train(&self, x: &candle_core::Tensor) -> anyhow::Result<candle_core::Tensor>

    /// Inference: (B, D), LayerState → (B, D)
    pub fn forward_step(
        &self,
        x: &candle_core::Tensor,
        state: &mut crate::state::LayerState,
    ) -> anyhow::Result<candle_core::Tensor>
}
```

### 4.8 `model/embedding.rs`

```rust
/// Token embedding table + final LayerNorm + LM head (tied weights).
pub struct EmbeddingHead {
    embed: candle_nn::Embedding,
    ln_out: candle_nn::LayerNorm,
    /// LM head weight is TIED to embed.weight (same Tensor, transposed on forward)
    /// This halves the parameter count for vocab projections.
    tied: bool,
}

impl EmbeddingHead {
    pub fn new(config: &ModelConfig, vb: candle_nn::VarBuilder) -> anyhow::Result<Self>

    /// Embed token IDs to vectors: (B, T) → (B, T, D)
    pub fn embed(&self, token_ids: &candle_core::Tensor) -> anyhow::Result<candle_core::Tensor>

    /// Project hidden states to logits: (B, T, D) → (B, T, vocab_size)
    pub fn lm_head(&self, hidden: &candle_core::Tensor) -> anyhow::Result<candle_core::Tensor>
}
```

### 4.9 `model/mod.rs` — TinyBit (top-level model)

```rust
/// The complete tinybit model.
pub struct TinyBit {
    pub config:    ModelConfig,
    embed:         EmbeddingHead,
    blocks:        Vec<Rwkv7Block>,
    /// Optional: speculative decoding auxiliary heads (None if config.spec_heads == 0)
    spec_heads:    Option<Vec<candle_nn::Linear>>,
}

impl TinyBit {
    pub fn new(config: ModelConfig, vb: candle_nn::VarBuilder) -> anyhow::Result<Self>

    /// Training forward pass. Returns logits (B, T, vocab_size).
    /// Also returns spec logits if spec_heads > 0: Vec<Tensor> each (B, T, vocab_size).
    pub fn forward_train(
        &self,
        token_ids: &candle_core::Tensor,  // (B, T) i64
    ) -> anyhow::Result<(candle_core::Tensor, Vec<candle_core::Tensor>)>

    /// Single-token inference step. Returns logits (B, vocab_size).
    pub fn forward_step(
        &self,
        token_id: &candle_core::Tensor,  // (B,) i64
        state: &mut InferenceState,
    ) -> anyhow::Result<candle_core::Tensor>

    /// Save weights to safetensors file.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()>

    /// Load weights from safetensors file.
    pub fn load(
        path: &std::path::Path,
        config: ModelConfig,
        device: &candle_core::Device,
    ) -> anyhow::Result<Self>

    /// Count total parameters.
    pub fn num_parameters(&self) -> usize

    /// Enable/disable ternary quantization for export.
    pub fn set_quantized(&mut self, quantized: bool)
}
```

### 4.10 `tokenizer.rs`

Use the HuggingFace `tokenizers` crate. We use the **LLaMA tokenizer** (32k vocab, SentencePiece BPE) because it is open, high quality, and already trained. Download it from HF.

```rust
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    // Special tokens for tool use (added to vocabulary)
    pub tool_call_start_id: u32,    // <|tool_call|>
    pub tool_call_end_id:   u32,    // <|end_tool_call|>
    pub tool_result_start_id: u32,  // <|tool_result|>
    pub tool_result_end_id:   u32,  // <|end_tool_result|>
    pub assistant_token_id:   u32,  // <|assistant|>
    pub user_token_id:        u32,  // <|user|>
    pub system_token_id:      u32,  // <|system|>
}

impl Tokenizer {
    /// Load from a saved tokenizer.json (downloaded from HF hub)
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self>

    /// Download from HuggingFace (model_id: "hf-internal-testing/llama-tokenizer" or similar)
    pub async fn from_hub(model_id: &str) -> anyhow::Result<Self>

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>>
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String>
    pub fn vocab_size(&self) -> usize

    /// Format a chat message into the prompt template.
    /// Template: <|system|>{system}<|user|>{user}<|assistant|>
    pub fn apply_chat_template(
        &self,
        system: Option<&str>,
        user: &str,
    ) -> anyhow::Result<Vec<u32>>
}
```

---

## 5. tinybit-tools

### 5.1 `tool.rs` — The Tool trait

```rust
/// Every tool implements this trait.
/// Tools are called synchronously from the inference loop.
pub trait Tool: Send + Sync {
    /// Short machine-readable name, e.g. "calculator".
    fn name(&self) -> &str;

    /// One-line description shown to the model in the system prompt.
    fn description(&self) -> &str;

    /// JSON schema describing the args field of a tool call.
    /// Model is trained to produce valid JSON matching this schema.
    fn args_schema(&self) -> &str;  // JSON string

    /// Execute the tool with the given JSON args string.
    /// Returns the result as a plain string (injected back into context).
    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput>;
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,    // the result text
    pub is_error: bool,     // if true, model is told the tool failed
}
```

### 5.2 `parser.rs` — Tool call detection and parsing

The model outputs tool calls using special tokens. Define the protocol:

**Tool call format in model output:**
```
<|tool_call|>{"tool":"calculator","args":{"expr":"2+2*3"}}<|end_tool_call|>
```

**Injected result:**
```
<|tool_result|>8<|end_tool_result|>
```

```rust
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ToolCall {
    pub tool: String,
    pub args: serde_json::Value,
}

/// Scan a string (partial or complete model output) for tool calls.
/// Returns None if no complete call found yet (still streaming),
/// Returns Some(call, before_text, after_marker) if a complete call found.
pub fn parse_tool_call(text: &str) -> Option<(ToolCall, &str, &str)>

/// Format a tool result for injection into context.
pub fn format_tool_result(output: &ToolOutput) -> String
```

### 5.3 `registry.rs` — ToolRegistry

```rust
pub struct ToolRegistry {
    tools: std::collections::HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self
    pub fn register(&mut self, tool: Box<dyn Tool>)
    pub fn get(&self, name: &str) -> Option<&dyn Tool>
    pub fn execute(&self, call: &ToolCall) -> anyhow::Result<ToolOutput>

    /// Build the tools section of the system prompt.
    /// Lists each tool: name, description, schema.
    pub fn system_prompt_section(&self) -> String

    /// Register all built-in tools with default data directory.
    pub fn with_builtins(data_dir: &std::path::Path) -> anyhow::Result<Self>
}
```

### 5.4 Built-in tools

**`builtin/time_tool.rs`**
```rust
/// Returns current date, time, day of week, and timezone.
/// Args JSON: {} (no args needed)
/// Output example: "2025-01-15 14:32:05 Wednesday UTC+0"
pub struct TimeTool;
```

**`builtin/calc_tool.rs`**
```rust
/// Evaluates a math expression using the evalexpr crate.
/// Supports: +,-,*,/,^,sqrt(),sin(),cos(),log(),pi,e
/// Args JSON: {"expr": "sqrt(2) * pi"}
/// Output: "4.442882938158366"
pub struct CalcTool;
```

**`builtin/todos_tool.rs`**
```rust
/// SQLite-backed todo list.
/// Args JSON: 
///   {"action":"add", "text":"Buy milk"}
///   {"action":"list"}
///   {"action":"complete", "id":1}
///   {"action":"delete", "id":1}
/// Uses rusqlite with a db file at data_dir/todos.db
pub struct TodosTool { db_path: std::path::PathBuf }
```

**`builtin/notes_tool.rs`**
```rust
/// SQLite-backed notes with FTS5 full-text search.
/// Args JSON:
///   {"action":"save", "title":"Meeting notes", "content":"..."}
///   {"action":"search", "query":"meeting"}
///   {"action":"get", "id":1}
///   {"action":"list"}
/// Uses FTS5 for fast search.
pub struct NotesTool { db_path: std::path::PathBuf }
```

**`builtin/calendar_tool.rs`**
```rust
/// SQLite-backed calendar events.
/// Args JSON:
///   {"action":"add", "title":"Doctor", "date":"2025-01-20", "time":"14:00", "notes":"..."}
///   {"action":"today"}
///   {"action":"week"}
///   {"action":"list", "from":"2025-01-01", "to":"2025-01-31"}
///   {"action":"delete", "id":1}
pub struct CalendarTool { db_path: std::path::PathBuf }
```

---

## 6. tinybit-infer

### 6.1 `sampler.rs`

```rust
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SamplingParams {
    pub temperature:  f64,   // 0.0 = greedy, 1.0 = normal. default 0.7
    pub top_p:        f64,   // nucleus sampling threshold. default 0.9
    pub top_k:        usize, // 0 = disabled. default 0
    pub max_new_tokens: usize,  // default 512
    pub repetition_penalty: f64,  // 1.0 = none, 1.3 = moderate. default 1.1
}

/// Sample the next token from logits (1, vocab_size).
pub fn sample(
    logits: &candle_core::Tensor,
    params: &SamplingParams,
    token_history: &[u32],  // for repetition penalty
) -> anyhow::Result<u32>
```

### 6.2 `engine.rs`

```rust
pub struct InferenceEngine {
    model:   TinyBit,
    tokenizer: Tokenizer,
    tools:   ToolRegistry,
    device:  candle_core::Device,
    pub params: SamplingParams,
}

impl InferenceEngine {
    pub fn new(
        model_path: &std::path::Path,
        config: ModelConfig,
        data_dir: &std::path::Path,
        device: candle_core::Device,
    ) -> anyhow::Result<Self>

    /// Auto-detect best device: Metal on Apple Silicon, CUDA if available, else CPU.
    pub fn auto_device() -> candle_core::Device

    /// Generate a response token by token.
    /// Returns the final complete response string.
    pub fn generate(
        &self,
        prompt: &str,
        state: &mut InferenceState,
        on_token: Option<&mut dyn FnMut(&str)>,  // streaming callback
    ) -> anyhow::Result<String>

    /// Process a single chat turn. Handles system prompt injection,
    /// tool call detection/execution, and multi-turn state management.
    pub fn chat_turn(
        &self,
        user_message: &str,
        session: &mut Session,
        on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<String>
}
```

### 6.3 `session.rs`

```rust
pub struct Session {
    pub id: String,
    pub state: InferenceState,
    pub history: Vec<ChatMessage>,
    pub system_prompt: String,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: Role,    // User | Assistant | ToolCall | ToolResult
    pub content: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Role { User, Assistant, ToolCall, ToolResult, System }

impl Session {
    pub fn new(config: &ModelConfig, device: &candle_core::Device) -> anyhow::Result<Self>
    pub fn reset_state(&mut self) -> anyhow::Result<()>
    /// Save to disk as JSON (history) + binary (state)
    pub fn save(&self, dir: &std::path::Path) -> anyhow::Result<()>
    pub fn load(dir: &std::path::Path, device: &candle_core::Device) -> anyhow::Result<Self>
}
```

### 6.4 `processor.rs`

```rust
/// Handles the tool call loop during generation.
///
/// Algorithm:
///   1. Generate tokens until EOS, max_tokens, or tool_call_start token detected
///   2. If tool_call_start: buffer until tool_call_end
///   3. Parse the JSON tool call
///   4. Execute via ToolRegistry
///   5. Inject result tokens back into context
///   6. Continue generation from step 1
///   7. Repeat up to MAX_TOOL_ROUNDS = 8 times
pub struct ToolProcessor<'a> {
    engine: &'a InferenceEngine,
    max_rounds: usize,  // default 8
}

impl<'a> ToolProcessor<'a> {
    pub fn new(engine: &'a InferenceEngine) -> Self

    pub fn run(
        &self,
        encoded_prompt: &[u32],
        state: &mut InferenceState,
        on_token: Option<&mut dyn FnMut(&str)>,
    ) -> anyhow::Result<String>
}
```

---

## 7. tinybit-train

### 7.1 `data/pack.rs` — token packing

```rust
/// Pack a stream of token IDs into fixed-length training chunks.
/// Each chunk is exactly `seq_len` tokens.
/// Documents are separated by EOS token; chunks may span document boundaries.
/// This maximizes training efficiency — no padding wasted.
pub fn pack_tokens(
    token_stream: &[u32],
    seq_len: usize,
    eos_id: u32,
) -> Vec<Vec<u32>>  // each inner Vec is exactly seq_len long
```

### 7.2 `data/dataset.rs`

```rust
/// Memory-mapped token dataset for efficient training without loading all data into RAM.
/// Format: binary file of u32 token IDs, little-endian.
/// The file is generated by scripts/prepare_data.sh from raw text.
pub struct TokenDataset {
    mmap: memmap2::Mmap,
    seq_len: usize,
    num_chunks: usize,
}

impl TokenDataset {
    pub fn open(path: &std::path::Path, seq_len: usize) -> anyhow::Result<Self>
    pub fn num_chunks(&self) -> usize
    /// Get the i-th chunk as input tokens (B=1, T=seq_len).
    pub fn get(&self, idx: usize) -> anyhow::Result<Vec<u32>>
    /// Get input and target (target = input shifted left by 1).
    pub fn get_pair(&self, idx: usize) -> anyhow::Result<(Vec<u32>, Vec<u32>)>
}

pub struct DataLoader {
    dataset: TokenDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current: usize,
}

impl DataLoader {
    pub fn new(dataset: TokenDataset, batch_size: usize, shuffle: bool) -> Self
    /// Returns (input_ids, target_ids) both (B, T)
    pub fn next_batch(&mut self) -> anyhow::Result<Option<(Vec<Vec<u32>>, Vec<Vec<u32>>)>>
    pub fn reset(&mut self)
    pub fn num_batches(&self) -> usize
}
```

### 7.3 `optimizer/muon.rs` — Muon optimizer

Muon orthogonalizes gradients using Newton-Schulz iteration before applying momentum.
This gives ~2× compute efficiency vs AdamW for 2D weight matrices.

```rust
/// Muon optimizer for 2D weight matrices (Linear layers).
/// Uses momentum + Newton-Schulz gradient orthogonalization.
///
/// Update rule:
///   G = gradient of weight W (shape: out × in)
///   M = β * M + (1-β) * G              # momentum
///   O = newton_schulz(M, steps=5)       # orthogonalize
///   W = W - lr * O
///
/// Newton-Schulz iteration (quintic Zhu-Schulz):
///   X = M / ||M||_F                     # normalize
///   for _ in range(5):
///       X = 1.5*X - 0.5 * X @ X.T @ X  # contract toward orthogonal matrix
///
/// Use Muon ONLY for 2D weight matrices of Linear layers.
/// Use AdamW for: embeddings, biases, LayerNorm params, 1D tensors.
pub struct Muon {
    lr:          f64,
    momentum:    f64,   // default 0.95
    nesterov:    bool,  // default true
    ns_steps:    usize, // Newton-Schulz iterations, default 5
    state:       std::collections::HashMap<String, candle_core::Tensor>,  // momentum buffers
}

impl Muon {
    pub fn new(lr: f64, momentum: f64) -> Self

    /// Update a list of (param_name, weight_tensor, grad_tensor) triples.
    pub fn step(
        &mut self,
        params: &[(String, &mut candle_core::Tensor, &candle_core::Tensor)],
    ) -> anyhow::Result<()>

    fn newton_schulz(x: &candle_core::Tensor, steps: usize) -> anyhow::Result<candle_core::Tensor>
}
```

### 7.4 `optimizer/adamw.rs`

Standard AdamW for 1D params. Use candle_nn's built-in or implement:

```rust
pub struct AdamW {
    lr:      f64,
    beta1:   f64,  // 0.9
    beta2:   f64,  // 0.95
    eps:     f64,  // 1e-8
    weight_decay: f64,  // 0.01 for embeddings, 0.0 for biases/norms
    step_count: usize,
    m: std::collections::HashMap<String, candle_core::Tensor>,
    v: std::collections::HashMap<String, candle_core::Tensor>,
}
```

### 7.5 `scheduler.rs` — WSD learning rate schedule

WSD = Warmup → Stable → Decay

```rust
/// Warmup-Stable-Decay learning rate schedule.
///   Steps 0..warmup_steps:  linear warmup from 0 to peak_lr
///   Steps warmup..stable:   constant at peak_lr
///   Steps stable..total:    cosine decay from peak_lr to min_lr (0.1 * peak_lr)
pub struct WsdScheduler {
    peak_lr:       f64,
    min_lr:        f64,
    warmup_steps:  usize,
    stable_steps:  usize,
    total_steps:   usize,
}

impl WsdScheduler {
    pub fn new(peak_lr: f64, total_steps: usize) -> Self {
        // warmup = 2% of total, decay = last 20% of total
    }

    pub fn get_lr(&self, step: usize) -> f64
}
```

### 7.6 `loss.rs`

```rust
/// Cross-entropy loss for next-token prediction.
/// logits: (B, T, vocab_size), targets: (B, T) as i64
/// Ignores positions where target == -100 (padding).
pub fn cross_entropy_loss(
    logits: &candle_core::Tensor,
    targets: &candle_core::Tensor,
) -> anyhow::Result<candle_core::Tensor>

/// Optional KL divergence loss for distillation.
/// student_logits: (B, T, vocab_size)
/// teacher_log_probs: (B, T, K) — top-K log probabilities from teacher
/// teacher_indices: (B, T, K) — vocabulary indices for those log probs
pub fn distillation_loss(
    student_logits: &candle_core::Tensor,
    teacher_log_probs: &candle_core::Tensor,
    teacher_indices: &candle_core::Tensor,
    alpha: f64,  // blend factor: alpha * KL + (1-alpha) * CE
) -> anyhow::Result<candle_core::Tensor>
```

### 7.7 `checkpoint.rs`

```rust
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub step:         usize,
    pub train_loss:   f64,
    pub val_loss:     f64,
    pub tokens_seen:  usize,
    pub config:       ModelConfig,
    pub timestamp:    String,
}

pub fn save_checkpoint(
    model: &TinyBit,
    meta: &CheckpointMeta,
    dir: &std::path::Path,
) -> anyhow::Result<()>

pub fn load_checkpoint(
    dir: &std::path::Path,
    device: &candle_core::Device,
) -> anyhow::Result<(TinyBit, CheckpointMeta)>

/// Keep only the 3 best and 3 latest checkpoints, delete the rest.
pub fn prune_checkpoints(dir: &std::path::Path, keep_best: usize, keep_recent: usize)
    -> anyhow::Result<()>
```

### 7.8 `trainer.rs` — Main training loop

```rust
#[derive(Debug, serde::Deserialize)]
pub struct TrainingConfig {
    // Data
    pub train_data:     std::path::PathBuf,
    pub val_data:       std::path::PathBuf,
    pub checkpoint_dir: std::path::PathBuf,

    // Training
    pub batch_size:     usize,   // e.g. 4
    pub grad_accum:     usize,   // accumulate this many batches before stepping (e.g. 8)
    pub total_steps:    usize,   // e.g. 100000
    pub peak_lr:        f64,     // e.g. 3e-4
    pub weight_decay:   f64,     // e.g. 0.01
    pub grad_clip:      f64,     // e.g. 1.0

    // Checkpointing
    pub save_every:     usize,   // steps between saves, e.g. 500
    pub eval_every:     usize,   // steps between val eval, e.g. 100
    pub eval_batches:   usize,   // how many batches to eval on, e.g. 50

    // Smoke test
    pub smoke_test_steps: usize, // run this many steps, check loss drops, then exit (0 = disabled)
}

pub struct Trainer {
    config: TrainingConfig,
    model_config: ModelConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig, model_config: ModelConfig) -> Self

    /// Run the full training loop.
    /// Saves checkpoints to config.checkpoint_dir.
    /// Prints loss to stdout compatible with GCP Cloud Logging.
    pub fn run(&self) -> anyhow::Result<()>

    fn eval_loss(&self, model: &TinyBit, loader: &mut DataLoader) -> anyhow::Result<f64>
}
```

---

## 8. tinybit-cli

### 8.1 `main.rs` — clap root

```rust
#[derive(clap::Parser)]
#[command(name = "tinybit", version = "0.1.0", about = "Your local AI assistant")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Start interactive chat
    Chat(chat::ChatArgs),
    /// Start HTTP inference server
    Serve(serve::ServeArgs),
    /// Train the model
    Train(train::TrainArgs),
    /// Export model to different formats
    Convert(convert::ConvertArgs),
    /// Download tokenizer and optional pretrained weights
    Download(download::DownloadArgs),
    /// Run evaluation suite
    Eval(eval::EvalArgs),
}
```

### 8.2 `commands/chat.rs`

```rust
#[derive(clap::Args)]
pub struct ChatArgs {
    #[arg(long, default_value = "configs/small.toml")]
    pub config: std::path::PathBuf,

    #[arg(long, default_value = "models/tinybit-small.safetensors")]
    pub model: std::path::PathBuf,

    #[arg(long, default_value = "data/")]
    pub data_dir: std::path::PathBuf,

    #[arg(long, default_value = "sessions/default")]
    pub session: std::path::PathBuf,

    /// Load a saved session (resume conversation)
    #[arg(long)]
    pub resume: bool,

    /// System prompt to use
    #[arg(long)]
    pub system: Option<String>,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,
}

/// Interactive REPL with readline-style input.
/// Prints tokens as they stream.
/// Commands: /quit, /reset, /save, /system <text>, /params
pub fn run(args: ChatArgs) -> anyhow::Result<()>
```

### 8.3 `commands/serve.rs`

HTTP server with OpenAI-compatible API (so it works with existing clients):

```rust
#[derive(clap::Args)]
pub struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    #[arg(long, default_value = "configs/small.toml")]
    pub config: std::path::PathBuf,

    #[arg(long, default_value = "models/tinybit-small.safetensors")]
    pub model: std::path::PathBuf,
}

/// Routes to implement:
///   POST /v1/chat/completions   — OpenAI-compatible (streaming + non-streaming)
///   GET  /v1/models             — list models
///   GET  /health                — health check
///   GET  /metrics               — tokens/sec, uptime
```

### 8.4 `commands/train.rs`

```rust
#[derive(clap::Args)]
pub struct TrainArgs {
    #[arg(long, default_value = "configs/small.toml")]
    pub model_config: std::path::PathBuf,

    #[arg(long, default_value = "configs/train.toml")]
    pub train_config: std::path::PathBuf,

    /// Run only smoke_test_steps steps to verify everything works
    #[arg(long)]
    pub smoke_test: bool,

    /// Resume from latest checkpoint
    #[arg(long)]
    pub resume: bool,
}
```

### 8.5 `commands/convert.rs`

```rust
#[derive(clap::Args)]
pub struct ConvertArgs {
    #[arg(long)]
    pub input: std::path::PathBuf,

    #[arg(long, value_enum, default_value = "safetensors")]
    pub format: ExportFormat,

    #[arg(long)]
    pub output: std::path::PathBuf,

    /// Quantize to ternary + INT8 before export
    #[arg(long)]
    pub quantize: bool,
}

#[derive(clap::ValueEnum, Clone)]
pub enum ExportFormat {
    Safetensors,
    /// GGUF format for llama.cpp compatibility
    Gguf,
}
```

---

## 9. configs/ — TOML config files

### `configs/nano.toml` (10M params — fastest)
```toml
vocab_size  = 32000
num_layers  = 6
d_model     = 256
d_ffn       = 512
num_heads   = 4
head_dim    = 64
max_seq_len = 512
dropout     = 0.05
spec_heads  = 0
ternary_ffn = false
int8_time   = false
```

### `configs/micro.toml` (50M params)
```toml
vocab_size  = 32000
num_layers  = 12
d_model     = 512
d_ffn       = 1024
num_heads   = 8
head_dim    = 64
max_seq_len = 1024
dropout     = 0.05
spec_heads  = 0
ternary_ffn = false
int8_time   = false
```

### `configs/small.toml` (150M params — default)
```toml
vocab_size  = 32000
num_layers  = 18
d_model     = 768
d_ffn       = 2048
num_heads   = 12
head_dim    = 64
max_seq_len = 1024
dropout     = 0.05
spec_heads  = 3
ternary_ffn = false
int8_time   = false
```

### `configs/base.toml` (400M params)
```toml
vocab_size  = 32000
num_layers  = 32
d_model     = 1024
d_ffn       = 2048
num_heads   = 16
head_dim    = 64
max_seq_len = 2048
dropout     = 0.05
spec_heads  = 3
ternary_ffn = false
int8_time   = false
```

### `configs/train.toml` (default training config)
```toml
train_data     = "data/train.bin"
val_data       = "data/val.bin"
checkpoint_dir = "checkpoints/"

batch_size     = 4
grad_accum     = 8          # effective batch = 4*8 = 32
total_steps    = 100000
peak_lr        = 3e-4
weight_decay   = 0.01
grad_clip      = 1.0

save_every     = 500
eval_every     = 100
eval_batches   = 50

smoke_test_steps = 0        # set to 200 via --smoke-test flag
```

---

## 10. scripts/

### `scripts/prepare_data.sh`

This script runs on a GCP VM (or locally). It downloads open datasets and tokenizes them.
No paid APIs needed — everything is freely available.

```bash
#!/bin/bash
# Usage: ./scripts/prepare_data.sh [output_dir]
# Datasets used (all free, no API needed):
#   - FineWeb-Edu (HuggingFace): high-quality educational web text
#   - Wikipedia English (HuggingFace): world knowledge (science, history, math)
#   - OpenHermes-2.5 (HuggingFace): instruction following + conversations
#   - The Stack Smol (HuggingFace): programming knowledge
#   - DolphinR2 (HuggingFace): reasoning + general QA
#
# Install: pip install datasets transformers tqdm
# Total size: ~30-50GB raw, ~8GB tokenized
#
# Steps:
#   1. Download each dataset using HuggingFace datasets (streaming, no full download)
#   2. Mix: 40% FineWeb-Edu + 30% Wikipedia + 15% OpenHermes + 10% Stack + 5% Dolphin
#   3. Tokenize using the LLaMA tokenizer
#   4. Pack into fixed-length chunks (seq_len = config.max_seq_len)
#   5. Save as binary u32 files: data/train.bin, data/val.bin
#   6. 98% train, 2% val split

OUTPUT_DIR="${1:-data}"
mkdir -p "$OUTPUT_DIR"

python3 - <<'PYTHON'
import sys
from pathlib import Path
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
import struct
import random

# Load tokenizer (LLaMA 3.1 tokenizer, open on HF)
tokenizer = Tokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# Load datasets (streaming — never loads full dataset into RAM)
datasets_config = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.40),
    ("wikimedia/wikipedia", "20231101.en", 0.30),
    ("teknium/OpenHermes-2.5", None, 0.15),
    ("bigcode/the-stack-smol", "data/python", 0.10),
    ("cognitivecomputations/dolphin-r1", None, 0.05),
]
# ... streaming tokenization loop
# Outputs: data/train.bin and data/val.bin as raw u32 binary
PYTHON
```

### `scripts/gcp_train.sh`

```bash
#!/bin/bash
# GCP training script — uses preemptible L4 GPU (cheapest GPU on GCP)
# Cost: ~$0.40/hr preemptible. 100 hours = ~$40 for small model.
# Budget: €200 total — leaves buffer for reruns.
#
# Requirements: gcloud CLI installed and authenticated
#
# Usage: ./scripts/gcp_train.sh [nano|micro|small|base]
MODEL_SIZE="${1:-small}"
PROJECT="your-gcp-project-id"      # EDIT THIS
REGION="us-central1"
ZONE="${REGION}-a"
INSTANCE_NAME="tinybit-train-$(date +%s)"
BUCKET="gs://your-bucket-tinybit"  # EDIT THIS

# Create preemptible L4 GPU instance
gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="g2-standard-8" \   # 8 vCPU, 32GB RAM, 1x L4 GPU
  --accelerator="type=nvidia-l4,count=1" \
  --image-family="common-cu121" \
  --image-project="deeplearning-platform-release" \
  --boot-disk-size="200GB" \
  --boot-disk-type="pd-ssd" \
  --provisioning-model="SPOT" \       # preemptible = cheapest
  --instance-termination-action="STOP" \
  --maintenance-policy="TERMINATE" \
  --scopes="storage-full"

# Startup script runs on the VM:
STARTUP="
apt-get update -q && apt-get install -y cargo rustup screen
rustup default stable

# Copy code from GCS bucket (or git clone)
gsutil -m cp -r ${BUCKET}/tinybit/ /home/user/
cd /home/user/tinybit

# Build release binary
cargo build --release -p tinybit-cli

# Prepare data (if not already done)
./scripts/prepare_data.sh data/

# Run smoke test first!
./target/release/tinybit train --model-config configs/${MODEL_SIZE}.toml --smoke-test
echo 'Smoke test passed, starting full training...'

# Start training in screen (survives SSH disconnect)
screen -dm -S train ./target/release/tinybit train \
  --model-config configs/${MODEL_SIZE}.toml \
  --train-config configs/train.toml \
  --resume

# Sync checkpoints to GCS every 10 minutes
while true; do
  sleep 600
  gsutil -m rsync -r checkpoints/ ${BUCKET}/checkpoints/
done
"
```

### `scripts/gcp_spot_train.sh`

CPU-only version (cheapest possible, using n2-highcpu-32):
```bash
#!/bin/bash
# CPU training on n2-highcpu-32 spot instance.
# ~€0.05/hr. Slower but free to restart. Use for nano/micro models.
# n2-highcpu-32: 32 vCPUs, 32GB RAM.
# RWKV-7 parallel training + Muon makes this viable.
# nano model (10M): ~2-3 days. micro (50M): ~1-2 weeks.
```

---

## 11. Tests (write these BEFORE implementing, run BEFORE training)

### `tests/model_correctness.rs`

```rust
#[test]
fn test_forward_shapes_nano() {
    // Create a nano model
    // Run forward pass with batch=2, seq=16
    // Assert output shape == (2, 16, vocab_size)
    // Assert no NaN in logits
}

#[test]
fn test_inference_step_matches_train() {
    // Run forward_train on a 4-token sequence
    // Run 4 forward_step calls with accumulated state
    // Assert outputs match (up to floating point tolerance)
    // This validates that training and inference use the same computation
}

#[test]
fn test_state_is_fixed_size() {
    // Create InferenceState
    // Run 10 inference steps
    // Run 100 inference steps
    // Assert state tensor shapes are identical — O(1) memory
}

#[test]
fn test_loss_decreases_on_trivial_data() {
    // Create a nano model
    // Create a trivial dataset: repeating "1 2 3 4 5"
    // Train for 50 steps
    // Assert final loss < initial loss (model learns)
}

#[test]
fn test_all_config_presets_build() {
    // Build nano, micro, small, base models
    // Assert param_count is approximately correct for each
}
```

### `tests/tool_system.rs`

```rust
#[test]
fn test_calc_tool_basic() {
    let tool = CalcTool;
    let result = tool.execute(r#"{"expr":"2+2*3"}"#).unwrap();
    assert_eq!(result.content, "8");
}

#[test]
fn test_parse_tool_call() {
    let text = r#"Let me calculate that. <|tool_call|>{"tool":"calculator","args":{"expr":"sqrt(144)"}}<|end_tool_call|>"#;
    let parsed = parse_tool_call(text).unwrap();
    assert_eq!(parsed.0.tool, "calculator");
}

#[test]
fn test_todos_add_and_list() {
    // Use temp dir for db
    // Add 2 todos, list them, complete one
    // Assert correct state
}

#[test]
fn test_time_tool_returns_date() {
    let tool = TimeTool;
    let result = tool.execute("{}").unwrap();
    assert!(!result.content.is_empty());
    // Assert it contains a year like "2025" or "2026"
    assert!(result.content.contains("202"));
}
```

### `tests/tokenizer.rs`

```rust
#[test]
fn test_encode_decode_roundtrip() {
    // "Hello world" → encode → decode → "Hello world"
}

#[test]
fn test_special_tokens_present() {
    // Assert tool_call_start_id etc. are in vocab and unique
}

#[test]
fn test_chat_template() {
    // Assert apply_chat_template produces expected token sequence
    // with <|user|>, <|assistant|> in correct positions
}
```

### `tests/training_smoke.rs`

```rust
#[test]
fn smoke_train_nano_100_steps() {
    // This is the GATE TEST — must pass before any real training run.
    // Create nano model
    // Generate synthetic data: random token IDs (any sequence will do for smoke test)
    // Train for 100 steps
    // Assert: final_loss < initial_loss * 0.95  (loss dropped at least 5%)
    // Assert: no NaN anywhere in weights
    // Assert: checkpoint saves and loads correctly
    // Assert: loaded model produces identical output to original
}
```

---

## 12. System prompt for the assistant

Build this into the default system prompt (injected by the Session struct):

```
You are tinybit, a helpful personal AI assistant. You are knowledgeable about 
science, mathematics, economics, history, philosophy, and everyday tasks.

You have access to the following tools:
{tool_descriptions}

To use a tool, output exactly:
<|tool_call|>{"tool":"<tool_name>","args":<json_args>}<|end_tool_call|>

Guidelines:
- Be concise and direct. Prefer short clear answers over long explanations.
- Use tools when they provide accurate real-time data (time, calculations).
- For maths: use the calculator tool for non-trivial computations.
- For todos/notes/calendar: always confirm what you added or found.
- If unsure, say so. Never hallucinate facts or tool results.
- You run entirely locally — the user's data stays on their device.
```

---

## 13. Implementation order for Claude Code

Follow this exact order. Do not skip ahead.

1. **Workspace + Cargo.toml** — get it compiling with all deps
2. **tinybit-core: config.rs** — ModelConfig, presets, TOML load/save
3. **tinybit-core: state.rs** — InferenceState, zeros, save/load
4. **tinybit-core: quantize.rs** — ternary quant, INT8 quant, pack/unpack
5. **tinybit-core: model/bitlinear.rs + model/embedding.rs** — leaf layers
6. **tinybit-core: model/channel_mix.rs** — channel-mix block
7. **tinybit-core: model/time_mix.rs** — time-mix block (hardest)
8. **tinybit-core: model/block.rs** — assemble Rwkv7Block
9. **tinybit-core: model/mod.rs** — TinyBit top-level
10. **tinybit-core: tokenizer.rs** — tokenizer wrapper
11. **→ RUN TEST: test_forward_shapes_nano** — must pass before continuing
12. **→ RUN TEST: test_inference_step_matches_train** — must pass
13. **tinybit-tools** — all tools + registry + parser
14. **→ RUN TEST: tests/tool_system.rs** — all tool tests must pass
15. **tinybit-infer** — engine, sampler, session, processor
16. **tinybit-train: data** — loader, dataset, pack
17. **tinybit-train: optimizer** — Muon, AdamW
18. **tinybit-train: trainer, scheduler, loss, checkpoint**
19. **→ RUN TEST: tests/training_smoke.rs** — GATE: loss must drop before GCP
20. **tinybit-cli** — all commands
21. **scripts/** — prepare_data.sh, gcp_train.sh
22. **README.md** — the user guide (see section 14 below)

---

## 14. README.md outline (write as the final step)

The README must cover these sections:

### Installation
```bash
# Prerequisites: Rust 1.82+, Git
git clone https://github.com/Andre-cmd-rgb/tinybit
cd tinybit
cargo build --release
```

### Quick start (local inference with pretrained weights)
```bash
# Download tokenizer and small pretrained weights
./target/release/tinybit download --model small

# Chat
./target/release/tinybit chat
```

### Training on Google Cloud (step by step)
1. Set up GCP project + billing alerts
2. Create GCS bucket for data + checkpoints
3. Run `prepare_data.sh` (can run locally or on a cheap CPU VM)
4. Run smoke test locally: `tinybit train --smoke-test`
5. Launch training: `./scripts/gcp_train.sh small`
6. Monitor: `gcloud compute instances get-serial-port-output ...`
7. Download checkpoint: `gsutil cp gs://your-bucket/checkpoints/best/ ./checkpoints/`

### Local inference performance expectations

| Hardware | Model | Tok/s |
|---|---|---|
| Apple M5 Pro (MLX) | small 150M | 200-400 |
| Apple M2 Air | small 150M | 80-150 |
| Linux x86 AVX-512 (Ryzen 9) | small 150M | 150-250 |
| Oracle Cloud Ampere A1 (4 OCPU) | nano 10M | 40-80 |
| Any modern laptop | nano 10M | 60-120 |

### Adding custom tools
```rust
// Implement the Tool trait
struct WeatherTool;
impl tinybit_tools::Tool for WeatherTool {
    fn name(&self) -> &str { "weather" }
    fn description(&self) -> &str { "Get current weather for a city" }
    fn args_schema(&self) -> &str { r#"{"city": "string"}"# }
    fn execute(&self, args: &str) -> anyhow::Result<ToolOutput> {
        // your implementation
    }
}

// Register it
registry.register(Box::new(WeatherTool));
```

### HTTP API (OpenAI-compatible)
```bash
# Start server
tinybit serve --port 8080

# Use with any OpenAI client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tinybit-small","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## 15. CLAUDE.md (file for Claude Code to read first)

```markdown
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

5. Tokenizer is LLaMA 3.1 format (32k vocab, SentencePiece BPE).
   Special tokens are added on top. IDs are deterministic — do not change.

6. All configs are in configs/*.toml. No magic numbers in model code.
   Everything reads from ModelConfig.

7. Training data is binary u32 (little-endian), memory-mapped.
   scripts/prepare_data.sh produces data/train.bin and data/val.bin.

8. Checkpoints are safetensors + JSON meta. Never pickle.

9. candle-core is the tensor framework. No PyTorch bindings.
   Metal support is via candle's "metal" feature (auto on macOS aarch64).
   CUDA support is via candle's "cuda" feature (enabled on GCP).

## Common mistakes to avoid

- Do NOT use .unwrap() in library code — propagate with anyhow::Result + ?
- Do NOT load entire dataset into RAM — use memory-mapped TokenDataset
- Do NOT mix training mode and inference mode forward passes
- Do NOT forget to call .detach() on state tensors to stop gradient tracking through state
- Time-mix token shift: during training use rolled x (shift by 1 along T dim);
  during inference use state.time_shift (previous actual token embedding)
```

---

## 16. Dependency versions (Cargo.toml exact)

```toml
candle-core   = { version = "0.8.0", features = ["metal"] }
candle-nn     = "0.8.0"
tokenizers    = "0.21.0"
safetensors   = "0.4.5"
serde         = { version = "1.0", features = ["derive"] }
serde_json    = "1.0"
toml          = "0.8"
clap          = { version = "4.5", features = ["derive", "env"] }
tokio         = { version = "1.40", features = ["full"] }
axum          = { version = "0.7", features = ["json"] }
tower-http    = { version = "0.5", features = ["cors", "trace"] }
anyhow        = "1.0"
thiserror     = "1.0"
tracing       = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
rayon         = "1.10"
indicatif     = { version = "0.17", features = ["rayon"] }
memmap2       = "0.9"
half          = { version = "2.4", features = ["num-traits"] }
rand          = "0.8"
chrono        = { version = "0.4", features = ["serde"] }
evalexpr      = "11.3"
rusqlite      = { version = "0.31", features = ["bundled"] }
hf-hub        = { version = "0.3", features = ["tokio"] }
```

---

## 17. Platform-specific build notes

### macOS Apple Silicon (M1/M2/M3/M4/M5)
candle's `metal` feature routes through Apple's Metal Performance Shaders automatically.
No extra code needed — `Device::Metal(0)` in the engine selects it.
The InferenceEngine::auto_device() function should check:
```rust
#[cfg(target_os = "macos")]
#[cfg(target_arch = "aarch64")]
return candle_core::Device::Metal(0);  // Apple Silicon — use Metal
```

### Windows x86_64
candle uses BLAS (OpenBLAS on Windows) for CPU matmuls automatically.
The release build will use AVX2 by default. For AVX-512 (newer Intel Xeon/Core):
Add `RUSTFLAGS="-C target-cpu=native"` to the build command.
No code changes needed.

### Linux x86_64
Same as Windows. `RUSTFLAGS="-C target-cpu=native"` for maximum SIMD.

### Oracle Cloud Ampere A1 (ARM64)
This is aarch64-unknown-linux-gnu.
candle's NEON auto-vectorization handles this.
Cross-compile from Linux: `cargo build --release --target aarch64-unknown-linux-gnu`
Or build directly on the Ampere instance (recommended).

### GCP L4 GPU (training only)
Enable candle's CUDA feature:
```toml
candle-core = { version = "0.8.0", features = ["cuda"] }
```
InferenceEngine::auto_device() should detect CUDA:
```rust
if candle_core::utils::cuda_is_available() {
    return candle_core::Device::Cuda(0);
}
```

---

End of plan. Total estimated implementation: 3,000-4,000 lines of Rust.
Implement in the order specified in section 13. Run all tests before training.
```
