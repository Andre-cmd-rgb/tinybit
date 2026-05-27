use crate::config::ModelConfig;
use crate::model::channel_mix::ChannelMix;
use crate::model::time_mix::TimeMix;
use crate::state::LayerState;
use candle_core::Tensor;
use candle_nn::{LayerNorm, Module, VarBuilder};

/// One complete RWKV-7 layer = LN(x) → TimeMix → residual → LN(x) → ChannelMix → residual
pub struct Rwkv7Block {
    ln1: LayerNorm,
    ln2: LayerNorm,
    time_mix: TimeMix,
    channel_mix: ChannelMix,
}

impl Rwkv7Block {
    pub fn new(config: &ModelConfig, _layer_idx: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        let ln1 = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("ln1"))?;
        let ln2 = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("ln2"))?;
        let time_mix = TimeMix::new(config, vb.pp("time_mix"))?;
        let channel_mix = ChannelMix::new(config, vb.pp("channel_mix"))?;
        Ok(Self { ln1, ln2, time_mix, channel_mix })
    }

    /// Training: (B, T, D) → (B, T, D)
    pub fn forward_train(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let h = self.time_mix.forward_train(&self.ln1.forward(x)?)?;
        let x = (x + h)?;
        let h = self.channel_mix.forward_train(&self.ln2.forward(&x)?)?;
        Ok((x + h)?)
    }

    /// Inference: (B, D), LayerState → (B, D)
    pub fn forward_step(
        &self,
        x: &Tensor,
        state: &mut LayerState,
    ) -> anyhow::Result<Tensor> {
        let h = self.time_mix.forward_step(&self.ln1.forward(x)?, state)?;
        let x = (x + h)?;
        let h = self.channel_mix.forward_step(&self.ln2.forward(&x)?, state)?;
        Ok((x + h)?)
    }
}
