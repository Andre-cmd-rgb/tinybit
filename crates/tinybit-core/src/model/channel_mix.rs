use crate::config::ModelConfig;
use crate::model::bitlinear::BitLinear;
use crate::state::LayerState;
use candle_core::{DType, Tensor};
use candle_nn::{ops::sigmoid, ops::silu, VarBuilder};

/// RWKV-7 Channel-Mix (the FFN equivalent).
pub struct ChannelMix {
    w_k: BitLinear,
    w_v: BitLinear,
    w_r: BitLinear,
    time_maa_k: Tensor,
    time_maa_r: Tensor,
    d_model: usize,
}

impl ChannelMix {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let w_k = BitLinear::new(config.d_model, config.d_ffn, vb.pp("w_k"))?;
        let w_v = BitLinear::new(config.d_ffn, config.d_model, vb.pp("w_v"))?;
        let w_r = BitLinear::new(config.d_model, config.d_model, vb.pp("w_r"))?;
        // Token-shift mix (see TimeMix): 0.5 = balanced blend of current/previous
        // token. 0.0 degenerately fed the projections the previous token only.
        let time_maa_k = vb.get_with_hints(
            config.d_model,
            "time_maa_k",
            candle_nn::Init::Const(0.5),
        )?;
        let time_maa_r = vb.get_with_hints(
            config.d_model,
            "time_maa_r",
            candle_nn::Init::Const(0.5),
        )?;
        Ok(Self { w_k, w_v, w_r, time_maa_k, time_maa_r, d_model: config.d_model })
    }

    /// Training forward: (B, T, D) → (B, T, D)
    pub fn forward_train(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let t = x.dim(1)?;
        let prev_x = if t == 1 {
            Tensor::zeros_like(x)?
        } else {
            let zeros = Tensor::zeros(
                (x.dim(0)?, 1, self.d_model),
                DType::F32,
                x.device(),
            )?;
            let sliced = x.narrow(1, 0, t - 1)?;
            Tensor::cat(&[&zeros.to_dtype(x.dtype())?, &sliced], 1)?
        };

        let maa_k = self.time_maa_k.to_dtype(x.dtype())?;
        let maa_r = self.time_maa_r.to_dtype(x.dtype())?;
        let diff = x.broadcast_sub(&prev_x)?;
        let x_k = prev_x.broadcast_add(&maa_k.broadcast_mul(&diff)?)?;
        let x_r = prev_x.broadcast_add(&maa_r.broadcast_mul(&diff)?)?;

        let k_pre = self.w_k.forward(&x_k)?;
        let k = silu(&k_pre)?;
        let k_sq = k.sqr()?;
        let v = self.w_v.forward(&k_sq)?;
        let r_pre = self.w_r.forward(&x_r)?;
        let r = sigmoid(&r_pre)?;
        Ok(r.broadcast_mul(&v)?)
    }

    /// Sequence inference (prefill): (1, T, D) → (1, T, D), seeding the token
    /// shift from `state.ffn_shift` and leaving it exactly as T successive
    /// `forward_step` calls would (the last token's post-ln2 input).
    pub fn forward_seq(
        &self,
        x: &Tensor, // (1, T, D)
        state: &mut LayerState,
    ) -> anyhow::Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        anyhow::ensure!(b == 1, "forward_seq expects batch size 1, got {b}");

        let prev0 = state.ffn_shift.to_dtype(x.dtype())?.reshape((1, 1, d))?;
        let prev_x = if t == 1 {
            prev0
        } else {
            Tensor::cat(&[&prev0, &x.narrow(1, 0, t - 1)?], 1)?
        };

        let maa_k = self.time_maa_k.to_dtype(x.dtype())?;
        let maa_r = self.time_maa_r.to_dtype(x.dtype())?;
        let diff = x.broadcast_sub(&prev_x)?;
        let x_k = prev_x.broadcast_add(&maa_k.broadcast_mul(&diff)?)?;
        let x_r = prev_x.broadcast_add(&maa_r.broadcast_mul(&diff)?)?;

        state.ffn_shift = x
            .narrow(1, t - 1, 1)?
            .reshape(d)?
            .to_dtype(DType::F32)?
            .detach();

        let k_pre = self.w_k.forward(&x_k)?;
        let k = silu(&k_pre)?;
        let k_sq = k.sqr()?;
        let v = self.w_v.forward(&k_sq)?;
        let r_pre = self.w_r.forward(&x_r)?;
        let r = sigmoid(&r_pre)?;
        Ok(r.broadcast_mul(&v)?)
    }

    /// Inference step: (B, D) → (B, D), updates ffn_shift in state
    pub fn forward_step(
        &self,
        x: &Tensor,
        state: &mut LayerState,
    ) -> anyhow::Result<Tensor> {
        let prev_x = state.ffn_shift.to_dtype(x.dtype())?;
        let prev_x = if prev_x.dims().len() == 1 {
            prev_x.unsqueeze(0)?
        } else {
            prev_x
        };

        let maa_k = self.time_maa_k.to_dtype(x.dtype())?;
        let maa_r = self.time_maa_r.to_dtype(x.dtype())?;
        let diff = x.broadcast_sub(&prev_x)?;
        let x_k = prev_x.broadcast_add(&maa_k.broadcast_mul(&diff)?)?;
        let x_r = prev_x.broadcast_add(&maa_r.broadcast_mul(&diff)?)?;

        let x_for_state = if x.dims().len() > 1 { x.get(0)? } else { x.clone() };
        state.ffn_shift = x_for_state.to_dtype(DType::F32)?.detach();

        let k_pre = self.w_k.forward(&x_k)?;
        let k = silu(&k_pre)?;
        let k_sq = k.sqr()?;
        let v = self.w_v.forward(&k_sq)?;
        let r_pre = self.w_r.forward(&x_r)?;
        let r = sigmoid(&r_pre)?;
        Ok(r.broadcast_mul(&v)?)
    }
}
