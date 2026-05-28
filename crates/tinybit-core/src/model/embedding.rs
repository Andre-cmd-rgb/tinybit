use crate::config::ModelConfig;
use crate::model::bitlinear::LayerNorm;
use candle_core::{DType, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

/// Token embedding table + final LayerNorm + LM head (tied weights).
pub struct EmbeddingHead {
    pub embed: Embedding,
    pub ln_out: LayerNorm,
    pub tied: bool,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl EmbeddingHead {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let embed = candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("embed"))?;
        let ln_out = LayerNorm::new(config.d_model, 1e-5, vb.pp("ln_out"))?;
        Ok(Self {
            embed,
            ln_out,
            tied: true,
            vocab_size: config.vocab_size,
            d_model: config.d_model,
        })
    }

    /// Embed token IDs to vectors: (B, T) → (B, T, D)
    pub fn embed(&self, token_ids: &Tensor) -> anyhow::Result<Tensor> {
        Ok(self.embed.forward(token_ids)?)
    }

    /// Project hidden states to logits: (..., D) → (..., vocab_size)
    /// Handles both 2D (B, D) and 3D (B, T, D) inputs.
    pub fn lm_head(&self, hidden: &Tensor) -> anyhow::Result<Tensor> {
        let normed = self.ln_out.forward(hidden)?;
        let normed_f32 = normed.to_dtype(DType::F32)?;
        let w = self.embed.embeddings().to_dtype(DType::F32)?; // (vocab_size, d_model)
        // Expand weight for batched matmul — same trick as candle's Linear::forward
        let w_exp = match normed_f32.dims() {
            &[bsize, _, _] => w.broadcast_left(bsize)?, // (bsize, vocab_size, d_model)
            _ => w,
        };
        // Divide by sqrt(d_model) to counteract the N(0,1) embedding init scale.
        // Without this, logit std = sqrt(d_model) * std_embed ≈ 16, giving CE ≈ 87 instead of ~ln(V).
        let scale = 1.0 / (self.d_model as f64).sqrt();
        Ok((normed_f32.matmul(&w_exp.t()?)? * scale)?)
    }
}
