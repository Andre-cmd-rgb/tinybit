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
