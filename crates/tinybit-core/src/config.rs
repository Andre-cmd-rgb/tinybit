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
    pub ternary_ffn: bool,
    pub int8_time:   bool,

    // Training
    pub max_seq_len: usize,
    pub dropout:     f64,

    // Speculative decoding heads
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

    /// ~25M params — smallest useful model (3.5× FFN)
    pub fn nano() -> Self {
        Self {
            vocab_size: 32000, num_layers: 9, d_model: 320,
            d_ffn: 1120, num_heads: 5, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 1024, dropout: 0.05, spec_heads: 0,
        }
    }

    /// ~50M params (3.5× FFN, 16 layers)
    pub fn micro() -> Self {
        Self {
            vocab_size: 32000, num_layers: 16, d_model: 384,
            d_ffn: 1344, num_heads: 6, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 1024, dropout: 0.05, spec_heads: 0,
        }
    }

    /// ~258M params (3.5× FFN)
    pub fn small() -> Self {
        Self {
            vocab_size: 32000, num_layers: 13, d_model: 1024,
            d_ffn: 3584, num_heads: 16, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 2048, dropout: 0.05, spec_heads: 0,
        }
    }

    /// ~501M params (3.5× FFN)
    pub fn base() -> Self {
        Self {
            vocab_size: 32000, num_layers: 17, d_model: 1280,
            d_ffn: 4480, num_heads: 20, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 2048, dropout: 0.05, spec_heads: 0,
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
