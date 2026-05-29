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
        // Reduction in f32 for stability, output restored to the input dtype so a
        // bf16 activation stays bf16 (and the downstream matmul runs in bf16).
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_norm = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        Ok(x_norm.broadcast_mul(&w)?.to_dtype(x_dtype)?)
    }
}

/// LayerNorm built from primitive ops (mean-subtracted, affine).
///
/// We deliberately do NOT use `candle_nn::LayerNorm`: its `forward` dispatches to
/// the fused `candle_nn::ops::layer_norm` kernel, which candle registers via
/// `apply_op3_no_bwd` — it has NO backward and silently drops all gradient. Used
/// for the pre-norm layers it froze the entire transformer stack (only the tied
/// embedding, reached through the lm-head matmul, ever received a gradient).
/// Composing primitive ops keeps the normalization differentiable. Param names
/// ("weight"/"bias") match candle's so existing checkpoints load unchanged.
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn new(d_model: usize, eps: f64, vb: VarBuilder) -> anyhow::Result<Self> {
        let weight = vb.get_with_hints(d_model, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(d_model, "bias", candle_nn::Init::Const(0.0))?;
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let hidden = x.dim(candle_core::D::Minus1)?;
        let mean = (x.sum_keepdim(candle_core::D::Minus1)? / hidden as f64)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = (xc.sqr()?.sum_keepdim(candle_core::D::Minus1)? / hidden as f64)?;
        let x_norm = xc.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        let b = self.bias.to_dtype(DType::F32)?;
        let out = x_norm.broadcast_mul(&w)?.broadcast_add(&b)?;
        Ok(out.to_dtype(x_dtype)?)
    }
}

/// Apply a linear projection `(..., in) · w_tᵀ` as a SINGLE GEMM.
///
/// candle's `Linear`/`broadcast_left` path turns a `(B, T, in)` activation into
/// `B` batched GEMMs with the weight replicated across the batch. Flattening the
/// leading dims into one row axis runs a single `(B*T, in) × (in, out)` GEMM
/// instead — one large cuBLAS call (better GPU utilization, fewer kernel
/// launches, no weight broadcast). matmul rows are independent, so the result is
/// numerically identical and gradients/checkpoints are unchanged.
///
/// `w_t` is the already-transposed weight, shape `(in, out)` (pass `w.t()`).
/// Inputs with ≤2 dims pass straight through (e.g. single-token inference).
pub(crate) fn linear_flat(x: &Tensor, w_t: &Tensor) -> anyhow::Result<Tensor> {
    let dims = x.dims();
    if dims.len() <= 2 {
        return Ok(x.matmul(w_t)?);
    }
    let k = dims[dims.len() - 1];
    let out = w_t.dim(1)?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_dims = dims[..dims.len() - 1].to_vec();
    out_dims.push(out);
    // reshape needs a contiguous source; the activations feeding the projections
    // are elementwise-op outputs (already contiguous), so this is a view in the
    // common case — guard anyway so a strided input can't panic.
    let x2 = if x.is_contiguous() {
        x.reshape((rows, k))?
    } else {
        x.contiguous()?.reshape((rows, k))?
    };
    Ok(x2.matmul(w_t)?.reshape(out_dims)?)
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
        if !self.quantized {
            // Match the activation dtype: bf16 activations → bf16 matmul (tensor
            // cores on CUDA) while `self.weight` stays the f32 master copy.
            // `linear_flat` runs one GEMM over the flattened batch×time rows.
            let w = self.weight.to_dtype(x_norm.dtype())?;
            return linear_flat(&x_norm, &w.t()?);
        }
        // Quantized export/inference path (not the training hot path). Single-
        // token inference feeds a 2D activation, so this rarely sees a batch dim;
        // keep the simple broadcast form here.
        let w = match x_norm.dims() {
            &[bsize, _, _] => self.weight.broadcast_left(bsize)?,
            _ => self.weight.clone(),
        };
        let (w_ternary, scale_w) = quantize_ternary(&w)?;
        let (x_q, scale_x) = quantize_int8(&x_norm)?;
        let result = x_q.matmul(&w_ternary.t()?)?;
        let factor = (scale_w * scale_x / 127.0) as f64;
        Ok((result * factor)?)
    }
}

/// Forward a candle `Linear` honoring the activation's dtype: the f32 master
/// weight (and bias, if any) is cast to the input dtype so the matmul runs in
/// bf16 on CUDA (tensor cores) while gradients still accumulate into the f32
/// master via the differentiable cast. Mirrors `Linear::forward`'s batched
/// broadcasting. With an f32 input this is identical to `Linear::forward`.
pub fn linear_autocast(lin: &candle_nn::Linear, x: &Tensor) -> anyhow::Result<Tensor> {
    let w = lin.weight().to_dtype(x.dtype())?; // (out, in), f32 master cast to activation dtype
    let mut y = linear_flat(x, &w.t()?)?; // single GEMM over flattened batch×time rows
    if let Some(b) = lin.bias() {
        y = y.broadcast_add(&b.to_dtype(x.dtype())?)?;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::linear_flat;
    use candle_core::{Device, Tensor};

    /// `linear_flat` (flattened single GEMM) must equal the broadcast batched
    /// matmul it replaces. Deterministic input (no RNG) and a tight tolerance —
    /// this is pure float-summation reordering, so the results agree to ~1e-5.
    #[test]
    fn linear_flat_matches_broadcast_bmm() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let (b, t, d, out) = (3usize, 5usize, 8usize, 7usize);
        let x = (Tensor::arange(0f32, (b * t * d) as f32, &dev)?.reshape((b, t, d))? * 0.01)?;
        let w = (Tensor::arange(0f32, (out * d) as f32, &dev)?.reshape((out, d))? * 0.013)?;

        // old path: replicate weight across batch, batched matmul
        let y_old = x.matmul(&w.broadcast_left(b)?.t()?)?;
        // new path: single GEMM over flattened (b*t, d) rows
        let y_new = linear_flat(&x, &w.t()?)?;

        assert_eq!(y_old.dims(), y_new.dims());
        let max_diff = (y_old - y_new)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(max_diff < 1e-4, "linear_flat diverged from broadcast bmm: {max_diff}");
        Ok(())
    }

    /// A 2D input (single-token inference shape) must pass straight through.
    #[test]
    fn linear_flat_2d_passthrough() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let x = (Tensor::arange(0f32, 12f32, &dev)?.reshape((2, 6))? * 0.1)?;
        let w = (Tensor::arange(0f32, 18f32, &dev)?.reshape((3, 6))? * 0.1)?;
        let y_direct = x.matmul(&w.t()?)?;
        let y_flat = linear_flat(&x, &w.t()?)?;
        let max_diff = (y_direct - y_flat)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(max_diff < 1e-6, "2D path changed: {max_diff}");
        Ok(())
    }
}
