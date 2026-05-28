pub mod bitlinear;
pub mod block;
pub mod channel_mix;
pub mod embedding;
pub mod time_mix;
pub mod wkv;

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
        let (vocab_override, is_quantized) = inspect_file(path);
        if let Some(actual) = vocab_override {
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

        if is_quantized {
            // Quantized export: rebuild full-precision tensors in memory and run
            // the normal f32 path. The win is on-disk size (~16x smaller for the
            // quantized matrices); a true ternary-matmul runtime is future work.
            let tensors = load_quantized_tensors(path, device)?;
            let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device);
            return Self::new(config, vb);
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

/// Peek at a safetensors file's header once. Returns the embedding-table row
/// count (`vocab_size`) if determinable, and whether the file is a tinybit
/// quantized export (carries the [`crate::quantize::QUANT_MARKER`] tensor).
/// Returns conservative defaults on any error so the caller falls back to its
/// config and the standard (non-quantized) load path.
fn inspect_file(path: &std::path::Path) -> (Option<usize>, bool) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return (None, false),
    };
    let st = match safetensors::SafeTensors::deserialize(&bytes) {
        Ok(s) => s,
        Err(_) => return (None, false),
    };
    let is_quant = st.tensor(crate::quantize::QUANT_MARKER).is_ok();
    // The embedding stays full precision (under its original name) in both the
    // plain and quantized formats, so this lookup works for either.
    let mut vocab = None;
    for candidate in ["embed.embed.weight", "embed.weight"] {
        if let Ok(t) = st.tensor(candidate) {
            vocab = t.shape().first().copied();
            break;
        }
    }
    (vocab, is_quant)
}

/// Load a tinybit quantized export and reconstruct a full-precision tensor map.
/// Each `<name>.qweight` (packed ternary) is unpacked and rescaled back to an
/// f32 `[rows, cols]` matrix; non-quantized tensors pass through unchanged.
fn load_quantized_tensors(
    path: &std::path::Path,
    device: &Device,
) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
    use std::collections::HashMap;
    let raw = candle_core::safetensors::load(path, device)?;
    let mut out: HashMap<String, Tensor> = HashMap::new();
    for (name, t) in &raw {
        if name.as_str() == crate::quantize::QUANT_MARKER {
            continue;
        }
        if let Some(base) = name.strip_suffix(".qweight") {
            let scale = scalar_f32(&raw, &format!("{base}.qscale"))?;
            let rows = scalar_i64(&raw, &format!("{base}.qrows"))? as usize;
            let cols = scalar_i64(&raw, &format!("{base}.qcols"))? as usize;
            let packed: Vec<u8> = t.flatten_all()?.to_vec1::<u8>()?;
            let w = crate::quantize::dequantize_unpack_2d(&packed, scale, rows, cols, device)?;
            out.insert(base.to_string(), w);
        } else if name.ends_with(".qscale")
            || name.ends_with(".qrows")
            || name.ends_with(".qcols")
        {
            // sidecar metadata — consumed with its `.qweight` above
        } else {
            out.insert(name.clone(), t.clone());
        }
    }
    Ok(out)
}

fn scalar_f32(
    raw: &std::collections::HashMap<String, Tensor>,
    name: &str,
) -> anyhow::Result<f32> {
    let t = raw.get(name).ok_or_else(|| anyhow::anyhow!("quantized file missing {name}"))?;
    Ok(t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?[0])
}

fn scalar_i64(
    raw: &std::collections::HashMap<String, Tensor>,
    name: &str,
) -> anyhow::Result<i64> {
    let t = raw.get(name).ok_or_else(|| anyhow::anyhow!("quantized file missing {name}"))?;
    Ok(t.flatten_all()?.to_dtype(DType::I64)?.to_vec1::<i64>()?[0])
}
