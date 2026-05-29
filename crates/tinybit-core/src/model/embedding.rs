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
        // LayerNorm preserves the input dtype; run the (large) vocab projection in
        // that dtype so bf16 hidden states hit tensor cores, then return f32 logits
        // for a stable softmax/loss. With f32 hidden (inference) this is unchanged.
        let normed = self.ln_out.forward(hidden)?;
        let cdt = normed.dtype();
        let w = self.embed.embeddings().to_dtype(cdt)?; // (vocab_size, d_model)
        // Single GEMM over the flattened (B*T, d_model) rows rather than a
        // broadcast batched matmul — this is the largest matmul in the model
        // (d_model × vocab_size), so it benefits most. See `linear_flat`.
        let logits = crate::model::bitlinear::linear_flat(&normed, &w.t()?)?.to_dtype(DType::F32)?;
        // Divide by sqrt(d_model) to counteract the N(0,1) embedding init scale.
        // Without this, logit std = sqrt(d_model) * std_embed ≈ 16, giving CE ≈ 87 instead of ~ln(V).
        let scale = 1.0 / (self.d_model as f64).sqrt();
        Ok((logits * scale)?)
    }
}
