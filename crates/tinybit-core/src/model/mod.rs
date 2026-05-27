pub mod bitlinear;
pub mod block;
pub mod channel_mix;
pub mod embedding;
pub mod time_mix;

use crate::config::ModelConfig;
use crate::state::InferenceState;
use block::Rwkv7Block;
use embedding::EmbeddingHead;
use candle_core::{Device, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// The complete tinybit model.
pub struct TinyBit {
    pub config: ModelConfig,
    embed: EmbeddingHead,
    blocks: Vec<Rwkv7Block>,
    spec_heads: Option<Vec<Linear>>,
}

impl TinyBit {
    pub fn new(config: ModelConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let embed = EmbeddingHead::new(&config, vb.pp("embed"))?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(Rwkv7Block::new(&config, i, vb.pp(format!("block_{i}")))?);
        }
        let spec_heads = if config.spec_heads > 0 {
            let mut heads = Vec::with_capacity(config.spec_heads);
            for i in 0..config.spec_heads {
                heads.push(candle_nn::linear_no_bias(
                    config.d_model,
                    config.vocab_size,
                    vb.pp(format!("spec_head_{i}")),
                )?);
            }
            Some(heads)
        } else {
            None
        };
        Ok(Self { config, embed, blocks, spec_heads })
    }

    /// Training forward pass. Returns logits (B, T, vocab_size) + optional spec logits.
    pub fn forward_train(
        &self,
        token_ids: &Tensor,
    ) -> anyhow::Result<(Tensor, Vec<Tensor>)> {
        let mut x = self.embed.embed(token_ids)?;
        for block in &self.blocks {
            x = block.forward_train(&x)?;
        }
        let logits = self.embed.lm_head(&x)?;
        let spec = match &self.spec_heads {
            Some(heads) => heads
                .iter()
                .map(|h| {
                    h.forward(&x)
                        .map_err(anyhow::Error::from)
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
            None => vec![],
        };
        Ok((logits, spec))
    }

    /// Single-token inference step. Returns logits (B, vocab_size).
    pub fn forward_step(
        &self,
        token_id: &Tensor,
        state: &mut InferenceState,
    ) -> anyhow::Result<Tensor> {
        let mut x = self.embed.embed(token_id)?.squeeze(1)?; // (B, D)
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_step(&x, &mut state.layers[i])?;
        }
        let x_t = x.unsqueeze(1)?; // (B, 1, D)
        let logits = self.embed.lm_head(&x_t)?.squeeze(1)?; // (B, vocab_size)
        Ok(logits)
    }

    /// Load weights from a safetensors file.
    ///
    /// Before constructing the model we peek at the file to discover the
    /// actual `embed.embed.weight` row count and, if it differs from the
    /// supplied `config.vocab_size`, override the config so the embedding
    /// shape matches what's on disk. This keeps older checkpoints (trained
    /// with a smaller vocab) loadable against newer configs that reserve
    /// extra slots for special tokens.
    pub fn load(
        path: &std::path::Path,
        mut config: ModelConfig,
        device: &Device,
    ) -> anyhow::Result<Self> {
        if let Some(actual) = inspect_embed_vocab(path) {
            if actual != config.vocab_size {
                tracing::warn!(
                    checkpoint = %path.display(),
                    cfg_vocab = config.vocab_size,
                    ckpt_vocab = actual,
                    "checkpoint embedding vocab differs from config — using checkpoint value"
                );
                config.vocab_size = actual;
            }
        }
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?
        };
        Self::new(config, vb)
    }

    /// Count total parameters.
    pub fn num_parameters(&self) -> usize {
        self.config.param_count()
    }

    /// Enable/disable ternary quantization (sets config flag for export decisions).
    pub fn set_quantized(&mut self, quantized: bool) {
        self.config.ternary_ffn = quantized;
    }
}

/// Peek at a safetensors file and return the embedding-table row count
/// (`vocab_size`) if it can be determined. Returns `None` on any error so the
/// caller can fall back to the supplied config.
fn inspect_embed_vocab(path: &std::path::Path) -> Option<usize> {
    let bytes = std::fs::read(path).ok()?;
    let st = safetensors::SafeTensors::deserialize(&bytes).ok()?;
    // Prefer the expected tinybit name; fall back to anything ending in
    // ".weight" with two dims at the embedding head namespace.
    for candidate in ["embed.embed.weight", "embed.weight"] {
        if let Ok(t) = st.tensor(candidate) {
            return t.shape().first().copied();
        }
    }
    None
}
