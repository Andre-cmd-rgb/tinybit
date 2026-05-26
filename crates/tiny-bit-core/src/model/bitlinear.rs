use crate::quantize::{quantize_int8, quantize_ternary};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

/// Simple RMSNorm — fused into BitLinear.
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
    pub d_model: usize,
}

impl RmsNorm {
    pub fn new(d_model: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        let weight = vb.get_with_hints(d_model, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps: 1e-6, d_model })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_norm = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        Ok(x_norm.broadcast_mul(&w)?)
    }
}

/// BitLinear: a Linear layer with ternary weights.
pub struct BitLinear {
    weight: Tensor,
    norm: RmsNorm,
    pub quantized: bool,
    pub in_features: usize,
    pub out_features: usize,
}

impl BitLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_UNIFORM;
        let weight = vb.get_with_hints((out_features, in_features), "weight", init)?;
        let norm = RmsNorm::new(in_features, vb.pp("norm"))?;
        Ok(Self { weight, norm, quantized: false, in_features, out_features })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x_norm = self.norm.forward(x)?;
        // Expand weight for batched matmul (mirrors candle's Linear::forward logic)
        let w = match x_norm.dims() {
            &[bsize, _, _] => self.weight.broadcast_left(bsize)?,
            _ => self.weight.clone(),
        };
        if !self.quantized {
            let w_f32 = w.to_dtype(DType::F32)?;
            Ok(x_norm.matmul(&w_f32.t()?)?)
        } else {
            let (w_ternary, scale_w) = quantize_ternary(&w)?;
            let (x_q, scale_x) = quantize_int8(&x_norm)?;
            let result = x_q.matmul(&w_ternary.t()?)?;
            let factor = (scale_w * scale_x / 127.0) as f64;
            Ok((result * factor)?)
        }
    }
}
