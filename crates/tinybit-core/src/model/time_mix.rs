use crate::config::ModelConfig;
use crate::model::bitlinear::BitLinear;
use crate::state::LayerState;
use candle_core::{DType, Tensor};
use candle_nn::{ops::silu, Linear, Module, VarBuilder};

pub struct TimeMix {
    w_r: BitLinear,
    w_k: Linear,
    w_v: Linear,
    w_g1: Linear,
    w_g2: Linear,
    w_o: Linear,

    time_decay: Tensor,
    group_norm_weight: Tensor,
    group_norm_bias: Tensor,

    time_maa_x: Tensor,
    time_maa_r: Tensor,
    time_maa_k: Tensor,
    time_maa_v: Tensor,

    pub num_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
}

impl TimeMix {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let d = config.d_model;
        let h = config.num_heads;
        let dh = config.head_dim;

        let w_r  = BitLinear::new(d, d, vb.pp("w_r"))?;
        let w_k  = candle_nn::linear_no_bias(d, d, vb.pp("w_k"))?;
        let w_v  = candle_nn::linear_no_bias(d, d, vb.pp("w_v"))?;
        let w_g1 = candle_nn::linear_no_bias(d, d, vb.pp("w_g1"))?;
        let w_g2 = candle_nn::linear_no_bias(d, d, vb.pp("w_g2"))?;
        let w_o  = candle_nn::linear_no_bias(d, d, vb.pp("w_o"))?;

        let time_decay =
            vb.get_with_hints((h, dh), "time_decay", candle_nn::Init::Const(-0.5))?;
        let group_norm_weight =
            vb.get_with_hints(d, "gn_weight", candle_nn::Init::Const(1.0))?;
        let group_norm_bias =
            vb.get_with_hints(d, "gn_bias", candle_nn::Init::Const(0.0))?;
        let time_maa_x = vb.get_with_hints(d, "time_maa_x", candle_nn::Init::Const(0.0))?;
        let time_maa_r = vb.get_with_hints(d, "time_maa_r", candle_nn::Init::Const(0.0))?;
        let time_maa_k = vb.get_with_hints(d, "time_maa_k", candle_nn::Init::Const(0.0))?;
        let time_maa_v = vb.get_with_hints(d, "time_maa_v", candle_nn::Init::Const(0.0))?;

        Ok(Self {
            w_r, w_k, w_v, w_g1, w_g2, w_o,
            time_decay, group_norm_weight, group_norm_bias,
            time_maa_x, time_maa_r, time_maa_k, time_maa_v,
            num_heads: h, head_dim: dh, d_model: d,
        })
    }

    fn group_norm(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let shape = x.dims().to_vec();
        let last = *shape.last().ok_or_else(|| anyhow::anyhow!("group_norm: empty tensor shape"))?;
        let prefix: Vec<usize> = shape[..shape.len() - 1].to_vec();

        let mut nh_shape = prefix.clone();
        nh_shape.push(self.num_heads);
        nh_shape.push(self.head_dim);

        let xr = x.reshape(nh_shape.as_slice())?;
        let mean = xr.mean_keepdim(candle_core::D::Minus1)?;
        let var  = xr.var_keepdim(candle_core::D::Minus1)?;
        let xn   = xr.broadcast_sub(&mean)?.broadcast_div(&(var + 1e-5_f64)?.sqrt()?)?;

        let mut flat_shape = prefix.clone();
        flat_shape.push(last);
        let xf = xn.reshape(flat_shape.as_slice())?;

        let w = self.group_norm_weight.to_dtype(x.dtype())?;
        let b = self.group_norm_bias.to_dtype(x.dtype())?;
        Ok(xf.broadcast_mul(&w)?.broadcast_add(&b)?)
    }

    fn compute_decay(&self) -> anyhow::Result<Tensor> {
        let td = self.time_decay.to_dtype(DType::F32)?;
        // softplus(-exp(td)) ≈ small negative decay
        let neg_exp = td.exp()?.neg()?;
        let sp = (neg_exp.exp()? + 1.0_f64)?.log()?;
        Ok(sp)
    }

    fn token_shift(x: &Tensor, d_model: usize) -> anyhow::Result<Tensor> {
        let t = x.dim(1)?;
        if t == 1 {
            return Ok(Tensor::zeros_like(x)?);
        }
        let zeros = Tensor::zeros((x.dim(0)?, 1, d_model), DType::F32, x.device())?;
        let sliced = x.narrow(1, 0, t - 1)?;
        Ok(Tensor::cat(&[&zeros.to_dtype(x.dtype())?, &sliced], 1)?)
    }

    /// Training forward: (B, T, D) → (B, T, D)
    pub fn forward_train(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let prev_x = Self::token_shift(x, self.d_model)?;

        let maa_x = self.time_maa_x.to_dtype(x.dtype())?;
        let maa_r = self.time_maa_r.to_dtype(x.dtype())?;
        let maa_k = self.time_maa_k.to_dtype(x.dtype())?;
        let maa_v = self.time_maa_v.to_dtype(x.dtype())?;

        let diff = x.broadcast_sub(&prev_x)?;
        let x_x = prev_x.broadcast_add(&maa_x.broadcast_mul(&diff)?)?;
        let x_r = prev_x.broadcast_add(&maa_r.broadcast_mul(&diff)?)?;
        let x_k = prev_x.broadcast_add(&maa_k.broadcast_mul(&diff)?)?;
        let x_v = prev_x.broadcast_add(&maa_v.broadcast_mul(&diff)?)?;

        let r = self.w_r.forward(&x_r)?;
        let k = self.w_k.forward(&x_k)?;
        let v = self.w_v.forward(&x_v)?;
        let g = silu(&self.w_g1.forward(&x_x)?)?.broadcast_mul(&self.w_g2.forward(&x_x)?)?;

        let w = self.compute_decay()?; // (H, dh)

        // Reshape to (B, T, H, dh).
        let r = r.reshape((b, t, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, t, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, t, self.num_heads, self.head_dim))?;

        let y = if crate::model::wkv::fused_wkv_enabled(x.device()) {
            // Fused scan (default on CUDA): a single autograd node, so candle retains
            // O(T·dh) instead of O(T·dh²) per layer. Numerically equal to the loop below.
            crate::model::wkv::fused_wkv(&r, &k, &v, &w)?.reshape((b, t, d))?
        } else {
            // Sequential candle scan (CPU default). decay broadcast (1,H,dh,1),
            // hoisted out of the per-timestep loop.
            let w_b = w.unsqueeze(0)?.unsqueeze(candle_core::D::Minus1)?.contiguous()?;
            let mut state = Tensor::zeros(
                (b, self.num_heads, self.head_dim, self.head_dim),
                DType::F32,
                x.device(),
            )?;
            let mut outputs: Vec<Tensor> = Vec::with_capacity(t);
            for ti in 0..t {
                let k_t = k.narrow(1, ti, 1)?.squeeze(1)?; // (B, H, dh)
                let v_t = v.narrow(1, ti, 1)?.squeeze(1)?;
                let r_t = r.narrow(1, ti, 1)?.squeeze(1)?;

                let k_f = k_t.to_dtype(DType::F32)?;
                let v_f = v_t.to_dtype(DType::F32)?;

                // outer product: (B, H, dh, 1) × (B, H, 1, dh) → (B, H, dh, dh)
                let k_unsq = k_f.unsqueeze(candle_core::D::Minus1)?.contiguous()?;
                let v_unsq = v_f.unsqueeze(candle_core::D::Minus2)?.contiguous()?;
                let outer = k_unsq.broadcast_mul(&v_unsq)?;

                state = state.broadcast_mul(&w_b)?.add(&outer)?;

                // readout: r_t → (B, H, 1, dh)
                let r_unsq = r_t.to_dtype(DType::F32)?
                    .unsqueeze(candle_core::D::Minus2)?
                    .contiguous()?;
                let y_t = r_unsq.matmul(&state.contiguous()?)?; // (B, H, 1, dh)
                let y_t = y_t.squeeze(candle_core::D::Minus2)?;  // (B, H, dh)
                let y_t = y_t.reshape((b, d))?;
                outputs.push(y_t.unsqueeze(1)?); // (B, 1, D)
            }
            Tensor::cat(&outputs, 1)? // (B, T, D)
        };

        let y = self.group_norm(&y.to_dtype(x.dtype())?)?;
        let out = y.broadcast_mul(&g)?;
        Ok(self.w_o.forward(&out)?)
    }

    /// Inference step: (B, D) → (B, D), reads/writes LayerState
    pub fn forward_step(
        &self,
        x: &Tensor, // (B, D)
        state: &mut LayerState,
    ) -> anyhow::Result<Tensor> {
        let b = x.dim(0)?;
        let prev_x = state.time_shift.to_dtype(x.dtype())?;
        // prev_x is (D,) — unsqueeze to (1, D) to broadcast with (B, D)
        let prev_x = prev_x.unsqueeze(0)?;

        let maa_x = self.time_maa_x.to_dtype(x.dtype())?;
        let maa_r = self.time_maa_r.to_dtype(x.dtype())?;
        let maa_k = self.time_maa_k.to_dtype(x.dtype())?;
        let maa_v = self.time_maa_v.to_dtype(x.dtype())?;

        let diff = x.broadcast_sub(&prev_x)?;
        let x_x = prev_x.broadcast_add(&maa_x.broadcast_mul(&diff)?)?;
        let x_r = prev_x.broadcast_add(&maa_r.broadcast_mul(&diff)?)?;
        let x_k = prev_x.broadcast_add(&maa_k.broadcast_mul(&diff)?)?;
        let x_v = prev_x.broadcast_add(&maa_v.broadcast_mul(&diff)?)?;

        // Update time shift — store first sample in batch (for B=1 inference)
        state.time_shift = x.get(0)?.to_dtype(DType::F32)?.detach();

        let r = self.w_r.forward(&x_r)?; // (B, D)
        let k = self.w_k.forward(&x_k)?;
        let v = self.w_v.forward(&x_v)?;
        let g = silu(&self.w_g1.forward(&x_x)?)?.broadcast_mul(&self.w_g2.forward(&x_x)?)?;

        let w = self.compute_decay()?; // (H, dh)

        let r = r.reshape((b, self.num_heads, self.head_dim))?; // (B, H, dh)
        let k = k.reshape((b, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, self.num_heads, self.head_dim))?;

        let k_f = k.to_dtype(DType::F32)?;
        let v_f = v.to_dtype(DType::F32)?;

        let k_unsq = k_f.unsqueeze(candle_core::D::Minus1)?.contiguous()?;
        let v_unsq = v_f.unsqueeze(candle_core::D::Minus2)?.contiguous()?;
        let outer = k_unsq.broadcast_mul(&v_unsq)?; // (B, H, dh, dh)

        // wkv_state is (H, dh, dh) — unsqueeze to (1, H, dh, dh) for batch
        let wkv = state.wkv_state.to_dtype(DType::F32)?.unsqueeze(0)?; // (1, H, dh, dh)
        let w_b = w.unsqueeze(0)?.unsqueeze(candle_core::D::Minus1)?.contiguous()?;
        let new_state_batched = wkv.broadcast_mul(&w_b)?.add(&outer)?; // (B, H, dh, dh)

        // Store without batch dim (take first sample for session state)
        state.wkv_state = new_state_batched.get(0)?.detach();

        let r_unsq = r.to_dtype(DType::F32)?
            .unsqueeze(candle_core::D::Minus2)?
            .contiguous()?; // (B, H, 1, dh)
        let y = r_unsq.matmul(&new_state_batched.contiguous()?)?; // (B, H, 1, dh)
        let y = y.squeeze(candle_core::D::Minus2)?; // (B, H, dh)
        let y = y.reshape((b, self.d_model))?;      // (B, D)

        let y = self.group_norm(&y.to_dtype(x.dtype())?)?;
        let out = y.broadcast_mul(&g)?;
        Ok(self.w_o.forward(&out)?)
    }
}
