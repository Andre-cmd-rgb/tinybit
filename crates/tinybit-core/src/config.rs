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

    // --- Brain-inspired extensions ---
    // All default to off/zero via `#[serde(default)]`, so existing configs and
    // checkpoints deserialize and behave byte-identically. The `nano` brain
    // model turns them on; `micro`/`bit`/`qbit` leave them off.

    /// Spiking activation sparsity ("efficient"): in channel-mix, post-activation
    /// values whose magnitude is below this threshold are zeroed — the neuron
    /// "doesn't fire". 0.0 = dense (no gate), the legacy behavior. Event-driven
    /// sparse coding; the firing fraction is data/threshold dependent. The real
    /// FLOPs win needs a sparse kernel (future); for now this is sparsity-ready
    /// and trains the model to use a sparse code.
    #[serde(default)]
    pub spike_threshold: f64,

    /// Hebbian fast-weights ("rewires itself"): maintain a per-layer, decaying
    /// associative weight delta `ΔW` updated online during inference (no
    /// gradients) and added to the time-mix value path, so the network adapts
    /// within a conversation. Training is unchanged. false = off (legacy).
    #[serde(default)]
    pub fast_weights: bool,
    /// Hebbian learning rate η for the fast-weight update `ΔW += η·(post⊗pre)`.
    #[serde(default)]
    pub fw_eta: f64,
    /// Per-step multiplicative forgetting for the fast-weight trace, in [0,1).
    /// Bounds `‖ΔW‖` and gives the adaptation a finite memory horizon.
    #[serde(default)]
    pub fw_decay: f64,

    /// Pondering ("thinks"): number of latent recurrence steps run over a learned
    /// "thought" embedding at the start of each assistant turn before emitting a
    /// token — internal deliberation that evolves the recurrent state without
    /// sampling. 0 = no pondering (legacy). Inference-time.
    #[serde(default)]
    pub ponder_steps: usize,
}

impl ModelConfig {
    /// Whether any brain-inspired extension is active.
    pub fn brain_enabled(&self) -> bool {
        self.spike_threshold > 0.0 || self.fast_weights || self.ponder_steps > 0
    }
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
        // Learned "thought" embedding for pondering (one d_model vector).
        let thought = if self.ponder_steps > 0 { self.d_model } else { 0 };
        embed + self.num_layers * per_layer + ln_out + spec + thought
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
            spike_threshold: 0.0, fast_weights: false, fw_eta: 0.0,
            fw_decay: 0.0, ponder_steps: 0,
        }
    }

    /// ~17M params — `nano`, the brain-native model: smaller and faster to train
    /// and run, with the brain-inspired mechanisms ON (spiking sparsity, Hebbian
    /// fast-weights, pondering). Designed to lean on the local knowledge store for
    /// facts rather than memorizing them. See configs/nano.toml.
    pub fn nano() -> Self {
        Self {
            vocab_size: 32008, num_layers: 8, d_model: 256,
            d_ffn: 896, num_heads: 4, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 512, dropout: 0.05, spec_heads: 0,
            // Brain mechanisms on (this model is trained from scratch with them).
            spike_threshold: 0.02,
            fast_weights: true, fw_eta: 0.02, fw_decay: 0.97,
            ponder_steps: 2,
        }
    }

    /// ~100M params (3.5× FFN, 12 layers) — `bit`. Larger than an L4 trains at
    /// the micro batch; see configs/train-bit-l4.toml for its tuned (smaller) batch.
    pub fn bit() -> Self {
        Self {
            vocab_size: 32008, num_layers: 12, d_model: 640,
            d_ffn: 2240, num_heads: 10, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 1024, dropout: 0.05, spec_heads: 0,
            spike_threshold: 0.0, fast_weights: false, fw_eta: 0.0,
            fw_decay: 0.0, ponder_steps: 0,
        }
    }

    /// ~150M params (3.5× FFN, 13 layers) — `qbit`, the largest shipped model.
    /// See configs/train-qbit-l4.toml for its tuned (smaller) L4 batch.
    pub fn qbit() -> Self {
        Self {
            vocab_size: 32008, num_layers: 13, d_model: 768,
            d_ffn: 2688, num_heads: 12, head_dim: 64,
            ternary_ffn: false, int8_time: false,
            max_seq_len: 1024, dropout: 0.05, spec_heads: 0,
            spike_threshold: 0.0, fast_weights: false, fw_eta: 0.0,
            fw_decay: 0.0, ponder_steps: 0,
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
        anyhow::ensure!(self.spike_threshold >= 0.0, "spike_threshold must be >= 0");
        if self.fast_weights {
            anyhow::ensure!(self.fw_eta >= 0.0, "fw_eta must be >= 0");
            anyhow::ensure!(
                self.fw_decay >= 0.0 && self.fw_decay < 1.0,
                "fw_decay must be in [0, 1) when fast_weights is enabled (got {})",
                self.fw_decay
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A legacy TOML with no brain fields must still deserialize (serde defaults)
    /// and report the brain extensions as OFF — so old configs are unchanged.
    #[test]
    fn legacy_toml_without_brain_fields_defaults_off() {
        let toml = r#"
            vocab_size = 32008
            num_layers = 16
            d_model = 384
            d_ffn = 1344
            num_heads = 6
            head_dim = 64
            max_seq_len = 512
            dropout = 0.05
            spec_heads = 0
            ternary_ffn = false
            int8_time = false
        "#;
        let cfg: ModelConfig = toml::from_str(toml).unwrap();
        cfg.validate().unwrap();
        assert!(!cfg.brain_enabled());
        assert_eq!(cfg.spike_threshold, 0.0);
        assert!(!cfg.fast_weights);
        assert_eq!(cfg.ponder_steps, 0);
        // micro's documented ~50M param count is unchanged by the new fields.
        assert_eq!(cfg.param_count(), ModelConfig::micro().param_count());
    }

    /// The nano brain config validates, turns the mechanisms on, and lands in
    /// its ~17M size band.
    #[test]
    fn nano_is_a_valid_brain_config() {
        let cfg = ModelConfig::nano();
        cfg.validate().unwrap();
        assert!(cfg.brain_enabled());
        assert!(cfg.fast_weights && cfg.spike_threshold > 0.0 && cfg.ponder_steps > 0);
        let p = cfg.param_count();
        assert!((14_000_000..=20_000_000).contains(&p), "nano param count {p} out of band");
    }

    /// fast_weights with an out-of-range decay is rejected.
    #[test]
    fn invalid_fast_weight_decay_rejected() {
        let mut cfg = ModelConfig::nano();
        cfg.fw_decay = 1.0;
        assert!(cfg.validate().is_err());
    }
}
