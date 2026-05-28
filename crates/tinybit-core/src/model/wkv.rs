//! Fused WKV scan (gated linear attention) — the RWKV-7 time-mixing recurrence.
//!
//! Recurrence (per batch `b`, head `h`, independent):
//!
//! ```text
//!   S_t[i,j] = w[i] * S_{t-1}[i,j] + k_t[i] * v_t[j]     (S_0 = 0)
//!   y_t[j]   = sum_i r_t[i] * S_t[i,j]
//! ```
//!
//! `w` is a per-(head, key-dim) decay in (0, 1), constant across time. This is
//! exactly the math the candle op loop in `time_mix.rs` computes — the kernel
//! and the reference below are numerically-equivalent reorganizations, so a
//! checkpoint trained with the old loop stays valid.
//!
//! The reference implementations here are plain Rust on flat `f32` slices with
//! the same indexing the CUDA kernel uses: `r,k,v,y` are `[B,T,H,dh]` and `w`
//! is `[H,dh]`, all row-major. They exist to (a) lock the forward/backward math
//! with a finite-difference gradient check that needs no GPU, and (b) back the
//! CPU path of the candle custom op so the whole thing runs (slower) without
//! CUDA.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

/// Runtime toggle for the fused scan. Off by default so existing training and
/// the documented L4 runs are unchanged until the kernel is validated on GPU.
/// Enable with `TINYBIT_FUSED_WKV=1`.
pub fn fused_wkv_enabled() -> bool {
    matches!(
        std::env::var("TINYBIT_FUSED_WKV").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Forward scan. Returns `(y, states)` where `y` is `[B,T,H,dh]` and `states`
/// is `[B,T,H,dh,dh]` holding `S_t` after each step (needed by the backward).
///
/// Storing every `S_t` is fine for the CPU reference and tests (it has RAM);
/// the CUDA kernel will instead checkpoint at chunk boundaries to stay within
/// VRAM. Both must agree numerically.
pub fn wkv_forward_ref(
    r: &[f32],
    k: &[f32],
    v: &[f32],
    w: &[f32],
    b: usize,
    t: usize,
    h: usize,
    dh: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0f32; b * t * h * dh];
    let mut states = vec![0f32; b * t * h * dh * dh];
    let mut s = vec![0f32; dh * dh];

    for bi in 0..b {
        for hi in 0..h {
            s.iter_mut().for_each(|x| *x = 0.0);
            let wbase = hi * dh;
            for ti in 0..t {
                let base = ((bi * t + ti) * h + hi) * dh;
                // S[i,j] = w[i]*S[i,j] + k[i]*v[j]
                for i in 0..dh {
                    let wi = w[wbase + i];
                    let ki = k[base + i];
                    let row = i * dh;
                    for j in 0..dh {
                        s[row + j] = wi * s[row + j] + ki * v[base + j];
                    }
                }
                // y[j] = sum_i r[i]*S[i,j]
                for j in 0..dh {
                    let mut acc = 0f32;
                    for i in 0..dh {
                        acc += r[base + i] * s[i * dh + j];
                    }
                    y[base + j] = acc;
                }
                let sbase = (((bi * t + ti) * h + hi) * dh) * dh;
                states[sbase..sbase + dh * dh].copy_from_slice(&s);
            }
        }
    }
    (y, states)
}

/// Backward scan. Given upstream `dy` (`[B,T,H,dh]`) and the `states` from the
/// forward pass, returns `(dr, dk, dv, dw)`. `dr/dk/dv` are `[B,T,H,dh]`; `dw`
/// is `[H,dh]` (summed over batch and time).
pub fn wkv_backward_ref(
    r: &[f32],
    k: &[f32],
    v: &[f32],
    w: &[f32],
    states: &[f32],
    dy: &[f32],
    b: usize,
    t: usize,
    h: usize,
    dh: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dr = vec![0f32; b * t * h * dh];
    let mut dk = vec![0f32; b * t * h * dh];
    let mut dv = vec![0f32; b * t * h * dh];
    let mut dw = vec![0f32; h * dh];
    let mut ds = vec![0f32; dh * dh];

    for bi in 0..b {
        for hi in 0..h {
            let wbase = hi * dh;

            // dr_t[i] = sum_j dy_t[j] * S_t[i,j]
            for ti in 0..t {
                let base = ((bi * t + ti) * h + hi) * dh;
                let sbase = (((bi * t + ti) * h + hi) * dh) * dh;
                for i in 0..dh {
                    let mut acc = 0f32;
                    let row = i * dh;
                    for j in 0..dh {
                        acc += dy[base + j] * states[sbase + row + j];
                    }
                    dr[base + i] = acc;
                }
            }

            // Reverse scan for dS, then dk/dv/dw.
            ds.iter_mut().for_each(|x| *x = 0.0);
            for ti in (0..t).rev() {
                let base = ((bi * t + ti) * h + hi) * dh;
                // dS_t = r_t (x) dy_t + diag(w) * dS_{t+1}
                for i in 0..dh {
                    let wi = w[wbase + i];
                    let ri = r[base + i];
                    let row = i * dh;
                    for j in 0..dh {
                        ds[row + j] = ri * dy[base + j] + wi * ds[row + j];
                    }
                }
                // dk_t[i] = sum_j dS_t[i,j]*v_t[j]
                for i in 0..dh {
                    let mut acc = 0f32;
                    let row = i * dh;
                    for j in 0..dh {
                        acc += ds[row + j] * v[base + j];
                    }
                    dk[base + i] = acc;
                }
                // dv_t[j] = sum_i dS_t[i,j]*k_t[i]
                for j in 0..dh {
                    let mut acc = 0f32;
                    for i in 0..dh {
                        acc += ds[i * dh + j] * k[base + i];
                    }
                    dv[base + j] = acc;
                }
                // dw[i] += sum_j dS_t[i,j]*S_{t-1}[i,j]   (S_{-1}=0)
                if ti > 0 {
                    let sprev = (((bi * t + (ti - 1)) * h + hi) * dh) * dh;
                    for i in 0..dh {
                        let mut acc = 0f32;
                        let row = i * dh;
                        for j in 0..dh {
                            acc += ds[row + j] * states[sprev + row + j];
                        }
                        dw[wbase + i] += acc;
                    }
                }
            }
        }
    }
    (dr, dk, dv, dw)
}

// ---------------------------------------------------------------------------
// candle custom op
// ---------------------------------------------------------------------------
//
// Wrapping the scan in a CustomOp2 collapses the autograd graph: candle retains
// only the inputs (packed rkv, w) and the output y, not the T intermediate
// states. That cuts training memory from O(T·dh²) to O(T·dh) per layer — the
// thing that caps batch_size on the L4 — independent of whether the forward
// runs on CPU loops or a fused CUDA kernel.

fn unpack_rkv(rkv: &[f32], b: usize, t: usize, h: usize, dh: usize)
    -> (Vec<f32>, Vec<f32>, Vec<f32>)
{
    let nbth = b * t * h;
    let mut r = vec![0f32; nbth * dh];
    let mut k = vec![0f32; nbth * dh];
    let mut v = vec![0f32; nbth * dh];
    for idx in 0..nbth {
        let pb = idx * 3 * dh;
        let ob = idx * dh;
        r[ob..ob + dh].copy_from_slice(&rkv[pb..pb + dh]);
        k[ob..ob + dh].copy_from_slice(&rkv[pb + dh..pb + 2 * dh]);
        v[ob..ob + dh].copy_from_slice(&rkv[pb + 2 * dh..pb + 3 * dh]);
    }
    (r, k, v)
}

fn pack_drkv(dr: &[f32], dk: &[f32], dv: &[f32], b: usize, t: usize, h: usize, dh: usize)
    -> Vec<f32>
{
    let nbth = b * t * h;
    let mut out = vec![0f32; nbth * 3 * dh];
    for idx in 0..nbth {
        let pb = idx * 3 * dh;
        let ob = idx * dh;
        out[pb..pb + dh].copy_from_slice(&dr[ob..ob + dh]);
        out[pb + dh..pb + 2 * dh].copy_from_slice(&dk[ob..ob + dh]);
        out[pb + 2 * dh..pb + 3 * dh].copy_from_slice(&dv[ob..ob + dh]);
    }
    out
}

/// candle custom op for the WKV scan over packed `(B,T,H,3,dh)` rkv and `(H,dh)`
/// decay `w`. CPU path uses the verified reference above; the CUDA path (the
/// fused kernel in [`WKV_CUDA_SRC`]) is wired in `cuda_fwd`/a cuda `bwd` once
/// validated on a GPU. Until then `cuda_fwd` falls back to the trait default
/// (errors), so CUDA callers must keep `TINYBIT_FUSED_WKV` off.
pub struct WkvScan;

impl CustomOp2 for WkvScan {
    fn name(&self) -> &'static str {
        "wkv-scan"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let d = l1.dims();
        if d.len() != 5 || d[3] != 3 {
            candle_core::bail!("wkv-scan: arg1 must be (B,T,H,3,dh), got {d:?}");
        }
        let (b, t, h, dh) = (d[0], d[1], d[2], d[4]);
        if !l1.is_contiguous()
            || l1.start_offset() != 0
            || !l2.is_contiguous()
            || l2.start_offset() != 0
        {
            candle_core::bail!("wkv-scan cpu_fwd: inputs must be contiguous with zero offset");
        }
        let rkv = s1.as_slice::<f32>()?;
        let w = s2.as_slice::<f32>()?;
        let (r, k, v) = unpack_rkv(rkv, b, t, h, dh);
        let (y, _states) = wkv_forward_ref(&r, &k, &v, w, b, t, h, dh);
        Ok((CpuStorage::F32(y), Shape::from((b, t, h, dh))))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let d = arg1.dims();
        let (b, t, h, dh) = (d[0], d[1], d[2], d[4]);
        let dev = arg1.device().clone();
        // Recompute on CPU (correct on any device; an optimized CUDA bwd kernel
        // replaces this once validated).
        let rkv = arg1.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let w = arg2.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let dy = grad_res.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let (r, k, v) = unpack_rkv(&rkv, b, t, h, dh);
        let (_, states) = wkv_forward_ref(&r, &k, &v, &w, b, t, h, dh);
        let (dr, dk, dv, dw) = wkv_backward_ref(&r, &k, &v, &w, &states, &dy, b, t, h, dh);
        let drkv = pack_drkv(&dr, &dk, &dv, b, t, h, dh);
        let d_rkv = Tensor::from_vec(drkv, (b, t, h, 3, dh), &dev)?;
        let d_w = Tensor::from_vec(dw, (h, dh), &dev)?;
        Ok((Some(d_rkv), Some(d_w)))
    }
}

/// Run the fused WKV scan. `r,k,v` are `(B,T,H,dh)`, `w` is `(H,dh)`; returns
/// `y` of `(B,T,H,dh)`. Equivalent to the sequential candle loop in
/// `time_mix.rs`, but as a single graph node (see module note).
pub fn fused_wkv(r: &Tensor, k: &Tensor, v: &Tensor, w: &Tensor) -> Result<Tensor> {
    let rkv = Tensor::stack(&[r, k, v], 3)?.contiguous()?; // (B,T,H,3,dh)
    let w = w.contiguous()?;
    rkv.apply_op2(&w, WkvScan)
}

/// Fused WKV CUDA kernels (forward + backward), compiled per `head_dim` via
/// `-D DH=<dh>` and launched with one block per `(batch, head)`. Implements the
/// exact math in [`wkv_forward_ref`]/[`wkv_backward_ref`]; must be validated on
/// a GPU against those references before use. Loaded at runtime via cudarc nvrtc
/// from `cuda_fwd` (wired in the GPU session).
pub const WKV_CUDA_SRC: &str = r#"
// One block per (batch, head); blockDim.x == DH.
extern "C" __global__ void wkv_forward_f32(
    const float* __restrict__ rkv,  // [B,T,H,3,DH]
    const float* __restrict__ w,    // [H,DH]
    float* __restrict__ y,          // [B,T,H,DH]
    const int B, const int T, const int H)
{
    const int bh = blockIdx.x;
    const int b = bh / H, h = bh % H;
    const int j = threadIdx.x;            // column index
    if (j >= DH) return;
    __shared__ float sr[DH], sk[DH], sw[DH];
    sw[j] = w[h*DH + j];
    float scol[DH];                       // scol[i] == S[i,j]
    for (int i = 0; i < DH; i++) scol[i] = 0.f;
    for (int t = 0; t < T; t++) {
        const long blk = (long)(b*T + t)*H + h;
        const float* base = rkv + blk*3*DH;   // r at +0, k at +DH, v at +2DH
        sr[j] = base[j];
        sk[j] = base[DH + j];
        __syncthreads();
        const float vj = base[2*DH + j];
        for (int i = 0; i < DH; i++) scol[i] = sw[i]*scol[i] + sk[i]*vj;
        float acc = 0.f;
        for (int i = 0; i < DH; i++) acc += sr[i]*scol[i];
        y[blk*DH + j] = acc;
        __syncthreads();
    }
}

// Backward. thread == row i. `scratch` is [B,H,T,DH,DH] (S_t per (b,h,t)).
// dw is accumulated across batch with atomicAdd. Two sub-passes share the block.
extern "C" __global__ void wkv_backward_f32(
    const float* __restrict__ rkv,
    const float* __restrict__ w,
    const float* __restrict__ dy,
    float* __restrict__ drkv,       // [B,T,H,3,DH]: dr at +0, dk +DH, dv +2DH
    float* __restrict__ dw,         // [H,DH], pre-zeroed
    float* __restrict__ scratch,    // [B,H,T,DH,DH]
    const int B, const int T, const int H)
{
    const int bh = blockIdx.x;
    const int b = bh / H, h = bh % H;
    const int i = threadIdx.x;            // row index
    if (i >= DH) return;
    const float wi = w[h*DH + i];
    __shared__ float sv[DH], sdy[DH], sr[DH], sk[DH], sdv[DH];

    // Sub-pass A: forward recompute, emit dr, store states.
    float srow[DH];
    for (int j = 0; j < DH; j++) srow[j] = 0.f;
    for (int t = 0; t < T; t++) {
        const long blk = (long)(b*T + t)*H + h;
        const float* base = rkv + blk*3*DH;
        sv[i]  = base[2*DH + i];
        sdy[i] = dy[blk*DH + i];
        __syncthreads();
        const float ki = base[DH + i];
        for (int j = 0; j < DH; j++) srow[j] = wi*srow[j] + ki*sv[j];
        float dri = 0.f;
        for (int j = 0; j < DH; j++) dri += sdy[j]*srow[j];
        drkv[blk*3*DH + i] = dri;          // dr slot
        float* st = scratch + (((long)(b*H + h)*T + t)*DH + i)*DH;
        for (int j = 0; j < DH; j++) st[j] = srow[j];
        __syncthreads();
    }

    // Sub-pass B: reverse scan -> dk, dv, dw.
    float dsrow[DH];
    for (int j = 0; j < DH; j++) dsrow[j] = 0.f;
    float dwi = 0.f;
    for (int t = T - 1; t >= 0; t--) {
        const long blk = (long)(b*T + t)*H + h;
        const float* base = rkv + blk*3*DH;
        sr[i]  = base[i];
        sdy[i] = dy[blk*DH + i];
        sk[i]  = base[DH + i];
        sv[i]  = base[2*DH + i];
        sdv[i] = 0.f;
        __syncthreads();
        const float ri = sr[i];
        for (int j = 0; j < DH; j++) dsrow[j] = ri*sdy[j] + wi*dsrow[j];
        float dki = 0.f;
        for (int j = 0; j < DH; j++) dki += dsrow[j]*sv[j];
        drkv[blk*3*DH + DH + i] = dki;     // dk slot
        for (int j = 0; j < DH; j++) atomicAdd(&sdv[j], sk[i]*dsrow[j]);
        __syncthreads();
        drkv[blk*3*DH + 2*DH + i] = sdv[i]; // dv slot
        if (t > 0) {
            const float* sp = scratch + (((long)(b*H + h)*T + (t-1))*DH + i)*DH;
            for (int j = 0; j < DH; j++) dwi += dsrow[j]*sp[j];
        }
        __syncthreads();
    }
    atomicAdd(&dw[h*DH + i], dwi);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic LCG so the test needs no rng dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            // returns ~uniform in [-0.5, 0.5]
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        }
    }

    /// Finite-difference gradient check: with loss L = sum(y * dy), the analytic
    /// grads from `wkv_backward_ref` must match central differences.
    #[test]
    fn wkv_gradient_check() {
        let (b, t, h, dh) = (2usize, 5usize, 2usize, 3usize);
        let n = b * t * h * dh;
        let mut rng = Lcg(0x1234_5678_9abc_def0);

        let r: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.8).collect();
        let k: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.8).collect();
        let v: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.8).collect();
        // w in (0,1): ~[0.2, 0.7], mimicking softplus(-exp(.)).
        let w: Vec<f32> = (0..h * dh).map(|_| 0.45 + rng.next_f32() * 0.5).collect();
        let dy: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

        let loss = |r: &[f32], k: &[f32], v: &[f32], w: &[f32]| -> f32 {
            let (y, _) = wkv_forward_ref(r, k, v, w, b, t, h, dh);
            y.iter().zip(dy.iter()).map(|(a, c)| a * c).sum()
        };

        let (_, states) = wkv_forward_ref(&r, &k, &v, &w, b, t, h, dh);
        let (dr, dk, dv, dw) = wkv_backward_ref(&r, &k, &v, &w, &states, &dy, b, t, h, dh);

        let eps = 1e-3f32;
        // which: 0=r,1=k,2=v,3=w. Fresh clones per call avoids any aliasing.
        let perturb_loss = |which: u8, idx: usize, delta: f32| -> f32 {
            let (mut rr, mut kk, mut vv, mut ww) = (r.clone(), k.clone(), v.clone(), w.clone());
            match which {
                0 => rr[idx] += delta,
                1 => kk[idx] += delta,
                2 => vv[idx] += delta,
                _ => ww[idx] += delta,
            }
            loss(&rr, &kk, &vv, &ww)
        };

        let mut max_rel = 0f32;
        let mut check = |which: u8, idx: usize, analytic: f32| {
            let fd = (perturb_loss(which, idx, eps) - perturb_loss(which, idx, -eps)) / (2.0 * eps);
            let rel = (fd - analytic).abs() / analytic.abs().max(fd.abs()).max(1e-3);
            max_rel = max_rel.max(rel);
            assert!(
                rel < 2e-2,
                "grad mismatch which={which} idx={idx}: fd={fd:.5} analytic={analytic:.5} rel={rel:.4}"
            );
        };

        let probe = [0usize, 1, 7, 11, n / 2, n - 1];
        for &idx in &probe {
            check(0, idx, dr[idx]);
            check(1, idx, dk[idx]);
            check(2, idx, dv[idx]);
        }
        for idx in 0..h * dh {
            check(3, idx, dw[idx]);
        }

        eprintln!("wkv_gradient_check ok, max relative error = {max_rel:.5}");
    }

    /// The fused custom-op forward must match the sequential candle loop in
    /// `time_mix.rs` (the thing it replaces) to within f32 noise.
    #[test]
    fn fused_matches_candle_loop() {
        use candle_core::{Device, D};
        let (b, t, h, dh) = (2usize, 6, 2, 4);
        let n = b * t * h * dh;
        let mut rng = Lcg(0x0bad_c0de_dead_beef);
        let rv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let kv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let vv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let wv: Vec<f32> = (0..h * dh).map(|_| 0.4 + rng.next_f32() * 0.4).collect();
        let dev = Device::Cpu;
        let r = Tensor::from_vec(rv, (b, t, h, dh), &dev).unwrap();
        let k = Tensor::from_vec(kv, (b, t, h, dh), &dev).unwrap();
        let v = Tensor::from_vec(vv, (b, t, h, dh), &dev).unwrap();
        let w = Tensor::from_vec(wv, (h, dh), &dev).unwrap();

        let y_fused = super::fused_wkv(&r, &k, &v, &w).unwrap();

        // Replicate the time_mix sequential scan.
        let w_b = w.unsqueeze(0).unwrap().unsqueeze(D::Minus1).unwrap();
        let mut state = Tensor::zeros((b, h, dh, dh), DType::F32, &dev).unwrap();
        let mut outs = Vec::new();
        for ti in 0..t {
            let k_t = k.narrow(1, ti, 1).unwrap().squeeze(1).unwrap();
            let v_t = v.narrow(1, ti, 1).unwrap().squeeze(1).unwrap();
            let r_t = r.narrow(1, ti, 1).unwrap().squeeze(1).unwrap();
            let outer = k_t
                .unsqueeze(D::Minus1)
                .unwrap()
                .broadcast_mul(&v_t.unsqueeze(D::Minus2).unwrap())
                .unwrap();
            state = state.broadcast_mul(&w_b).unwrap().add(&outer).unwrap();
            let y_t = r_t
                .unsqueeze(D::Minus2)
                .unwrap()
                .contiguous()
                .unwrap()
                .matmul(&state.contiguous().unwrap())
                .unwrap()
                .squeeze(D::Minus2)
                .unwrap();
            outs.push(y_t.unsqueeze(1).unwrap());
        }
        let y_loop = Tensor::cat(&outs, 1).unwrap(); // (b,t,h,dh)

        let diff = (y_fused - y_loop)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("fused vs candle-loop max abs diff = {diff:.3e}");
        assert!(diff < 1e-4, "fused scan diverged from candle loop: {diff}");
    }
}
