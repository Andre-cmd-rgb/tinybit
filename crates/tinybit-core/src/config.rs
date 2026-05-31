/// All hyperparameters for a model variant.
/// Every architectural decision lives here — no magic numbers in model code.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    // Architecture
    pub vocab_size:  usize,
    pub num_layers:  usize,
    pub d_model:     usize,
    pub d_ffn:       usize,
    pub num_heads:   usize,
    pub head_dim:    usize,

    // Quantization
    /// Export the FFN/projection matrices as ternary weights (see `quantize.rs`).
    /// Consumed by the `convert` command's export path, not the training forward.
    pub ternary_ffn: bool,
    /// Reserved: int8 time-mix quantization. Not applied by the current forward;
    /// kept so existing configs/checkpoints deserialize unchanged.
    pub int8_time:   bool,

    // Training
    pub max_seq_len: usize,
    /// Reserved: dropout probability. The forward pass applies NO dropout — at
    /// ~1 epoch over the data (data-limited, not overfitting-limited) it is
    /// unnecessary. Kept for config/checkpoint compatibility; any nonzero value
    /// is currently a no-op.
    pub dropout:     f64,

    /// Number of extra speculative-decoding LM heads to allocate. The heads are
    /// built and counted in `param_count`, but the training loss does not yet
    /// supervise them, so leave at 0 unless experimenting. (See `forward_train`.)
    pub spec_heads:  usize,
}

impl ModelConfig {
    pub fn param_count(&self) -> usize {
        // Embedding table
        let embed = self.vocab_size * self.d_model;
        // Per block: time_mix (r,k,v,g1,g2,o projections + decay + maa) + channel_mix (k,v,r)
        let time_mix_per_layer = 6 * self.d_model * self.d_model
            + self.num_heads * self.head_dim  // time_decay
            + 4 * self.d_model;              // time_maa_x/r/k/v
        let channel_mix_per_layer = 3 * self.d_model * self.d_ffn
            + 2 * self.d_model;              // time_maa_k/r
        let ln_per_layer = 4 * self.d_model; // ln1 + ln2 (weight + bias each)
        let per_layer = time_mix_per_layer + channel_mix_per_layer + ln_per_layer;
        // Final LN
        let ln_out = 2 * self.d_model;
        // Spec heads (linear projections)
        let spec = self.spec_heads * self.d_model * self.vocab_size;
        embed + self.num_layers * per_layer + ln_out + spec
    }

    /// ~50M params (3.5× FFN, 16 layers) — the smallest shipped model and the
    /// only one validated end-to-end on a single L4 (batch 11). See
    /// configs/micro.toml.
    pub fn micro() -> Self {
        Self {
            vocab_size: 32008, num_layers: 16, d_model: 384,
            d_ffn: 1344, num_heads: 6, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 512, dropout: 0.05, spec_heads: 0,
        }
    }

    /// ~100M params (3.5× FFN, 12 layers). Larger than an L4 comfortably trains
    /// at the micro batch — treat as needing batch/seq tuning or bigger hardware.
    pub fn small() -> Self {
        Self {
            vocab_size: 32008, num_layers: 12, d_model: 640,
            d_ffn: 2240, num_heads: 10, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 1024, dropout: 0.05, spec_heads: 0,
        }
    }

    /// ~150M params (3.5× FFN, 13 layers) — the largest shipped model.
    pub fn medium() -> Self {
        Self {
            vocab_size: 32008, num_layers: 13, d_model: 768,
            d_ffn: 2688, num_heads: 12, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 1024, dropout: 0.05, spec_heads: 0,
        }
    }

    /// Validate config consistency. Call after loading from TOML.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.d_model == self.num_heads * self.head_dim,
            "d_model ({}) must equal num_heads ({}) * head_dim ({})",
            self.d_model, self.num_heads, self.head_dim
        );
        anyhow::ensure!(self.vocab_size > 0, "vocab_size must be > 0");
        anyhow::ensure!(self.num_layers > 0, "num_layers must be > 0");
        anyhow::ensure!(self.d_model > 0, "d_model must be > 0");
        anyhow::ensure!(self.d_ffn > 0, "d_ffn must be > 0");
        anyhow::ensure!(self.max_seq_len > 0, "max_seq_len must be > 0");
        Ok(())
    }

    /// Load from a TOML config file.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    /// Save to TOML.
    pub fn to_file(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let text = toml::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }
}
