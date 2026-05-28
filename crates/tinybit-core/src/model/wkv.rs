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

use candle_core::{CpuStorage, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor};

/// Whether to use the fused scan for a given device. Defaults to ON for CUDA
/// (the kernel is faster and the autograd-graph collapse cuts VRAM) and OFF for
/// CPU (the candle loop is the well-trodden path). `TINYBIT_FUSED_WKV` overrides
/// either way: `1`/`true`/`on` forces it on, `0`/`false`/`off` forces it off —
/// used by the parity tests and benchmarks, and as an escape hatch.
pub fn fused_wkv_enabled(device: &candle_core::Device) -> bool {
    match std::env::var("TINYBIT_FUSED_WKV").as_deref() {
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("on") => true,
        Ok("0") | Ok("false") | Ok("FALSE") | Ok("off") => false,
        _ => device.is_cuda(),
    }
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
/// decay `w`. The CPU path uses the verified reference above; the CUDA path runs
/// the fused kernel in [`WKV_CUDA_SRC`] (forward in [`WkvScan::cuda_fwd`], backward
/// via [`WkvBackwardOp`]). Without the `cuda` feature `cuda_fwd` falls back to the
/// trait default (errors), so non-CUDA callers must keep `TINYBIT_FUSED_WKV` off.
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

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_impl::forward(s1, l1, s2, l2)
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

        // CUDA: run the fused backward kernel via WkvBackwardOp, which returns the
        // grads packed into one tensor [drkv | dw]; split it back out. Inputs are
        // detached so candle does not try to build a (nonexistent) 2nd-order graph.
        #[cfg(feature = "cuda")]
        if arg1.device().is_cuda() {
            let rkv = arg1.detach().contiguous()?;
            let w = arg2.detach().contiguous()?;
            let dy = grad_res.detach().contiguous()?;
            let packed = rkv.apply_op3(&w, &dy, WkvBackwardOp)?;
            let nbth = b * t * h;
            let d_rkv = packed.narrow(0, 0, nbth * 3 * dh)?.reshape((b, t, h, 3, dh))?;
            let d_w = packed.narrow(0, nbth * 3 * dh, h * dh)?.reshape((h, dh))?;
            return Ok((Some(d_rkv), Some(d_w)));
        }

        let dev = arg1.device().clone();
        // CPU recompute: correct on any device, and the reference the CUDA path is
        // validated against. (On CUDA this is bypassed by the branch above.)
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

/// Backward of [`WkvScan`], as its own op so the CUDA kernel can run on-device
/// (candle's `bwd` hands back `Tensor`s, not storage, so we re-enter the op
/// machinery to reach `cuda_fwd`). Inputs are the saved `rkv` `(B,T,H,3,dh)`,
/// decay `w` `(H,dh)`, and upstream `dy` `(B,T,H,dh)`. The single output packs
/// the grads as a flat `[B*T*H*3*dh + H*dh]` tensor — `drkv` in the head, `dw`
/// in the tail — which [`WkvScan::bwd`] slices back apart. No `bwd` of its own:
/// callers detach the inputs, so no second-order graph is built.
pub struct WkvBackwardOp;

impl CustomOp3 for WkvBackwardOp {
    fn name(&self) -> &'static str {
        "wkv-backward"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        _l2: &Layout,
        s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let d = l1.dims();
        if d.len() != 5 || d[3] != 3 {
            candle_core::bail!("wkv-backward: arg1 must be (B,T,H,3,dh), got {d:?}");
        }
        let (b, t, h, dh) = (d[0], d[1], d[2], d[4]);
        if !l1.is_contiguous() || l1.start_offset() != 0 {
            candle_core::bail!("wkv-backward cpu_fwd: inputs must be contiguous with zero offset");
        }
        let rkv = s1.as_slice::<f32>()?;
        let w = s2.as_slice::<f32>()?;
        let dy = s3.as_slice::<f32>()?;
        let (r, k, v) = unpack_rkv(rkv, b, t, h, dh);
        let (_, states) = wkv_forward_ref(&r, &k, &v, w, b, t, h, dh);
        let (dr, dk, dv, dw) = wkv_backward_ref(&r, &k, &v, w, &states, dy, b, t, h, dh);
        let drkv = pack_drkv(&dr, &dk, &dv, b, t, h, dh);
        let nbth = b * t * h;
        let mut out = vec![0f32; nbth * 3 * dh + h * dh];
        out[..nbth * 3 * dh].copy_from_slice(&drkv);
        out[nbth * 3 * dh..].copy_from_slice(&dw);
        Ok((CpuStorage::F32(out), Shape::from((nbth * 3 * dh + h * dh,))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_impl::backward(s1, l1, s2, l2, s3, l3)
    }
}

/// CUDA launchers for the fused WKV kernels. [`WKV_CUDA_SRC`] is compiled once per
/// `head_dim` (via `-D DH=<dh>`) with nvrtc and cached on the device; each launch
/// uses one block per `(batch, head)` and `DH` threads. Numerically equivalent to
/// the CPU references in this module (validated by the `cuda` tests below).
#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::WKV_CUDA_SRC;
    use candle_core::cuda_backend::cudarc;
    use candle_core::cuda_backend::cudarc::driver::{
        CudaFunction, CudaSlice, LaunchAsync, LaunchConfig,
    };
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use candle_core::{Layout, Result, Shape};
    use std::sync::Arc;

    fn wrap<E: std::error::Error + Send + Sync + 'static>(e: E) -> candle_core::Error {
        candle_core::Error::Cuda(Box::new(e))
    }

    fn f32_slice(s: &CudaStorage) -> Result<&CudaSlice<f32>> {
        match &s.slice {
            CudaStorageSlice::F32(sl) => Ok(sl),
            _ => candle_core::bail!("wkv cuda: expected f32 storage"),
        }
    }

    fn require_contig(l: &Layout) -> Result<()> {
        if !l.is_contiguous() || l.start_offset() != 0 {
            candle_core::bail!("wkv cuda: inputs must be contiguous with zero offset");
        }
        Ok(())
    }

    // Compile (once per head_dim) and fetch a kernel by name. Both kernels share
    // one module, so a single nvrtc compile serves forward and backward.
    fn get_func(
        dev: &Arc<cudarc::driver::CudaDevice>,
        dh: usize,
        name: &'static str,
    ) -> Result<CudaFunction> {
        let module = format!("wkv_dh{dh}");
        if dev.get_func(&module, name).is_none() {
            let opts = cudarc::nvrtc::CompileOptions {
                options: vec![format!("--define-macro=DH={dh}")],
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(WKV_CUDA_SRC, opts).map_err(wrap)?;
            dev.load_ptx(ptx, &module, &["wkv_forward_f32", "wkv_backward_f32"])
                .map_err(wrap)?;
        }
        dev.get_func(&module, name)
            .ok_or_else(|| candle_core::Error::Cuda(format!("wkv: missing fn {name}").into()))
    }

    fn launch_cfg(b: usize, h: usize, dh: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((b * h) as u32, 1, 1),
            block_dim: (dh as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub fn forward(
        s_rkv: &CudaStorage,
        l_rkv: &Layout,
        s_w: &CudaStorage,
        l_w: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let d = l_rkv.dims();
        if d.len() != 5 || d[3] != 3 {
            candle_core::bail!("wkv cuda fwd: arg1 must be (B,T,H,3,dh), got {d:?}");
        }
        let (b, t, h, dh) = (d[0], d[1], d[2], d[4]);
        require_contig(l_rkv)?;
        require_contig(l_w)?;
        let dev = s_rkv.device.clone();
        let cu = dev.cuda_device();
        let rkv = f32_slice(s_rkv)?;
        let w = f32_slice(s_w)?;
        let out = cu.alloc_zeros::<f32>(b * t * h * dh).map_err(wrap)?;
        let f = get_func(&cu, dh, "wkv_forward_f32")?;
        let cfg = launch_cfg(b, h, dh);
        unsafe { f.launch(cfg, (rkv, w, &out, b as i32, t as i32, h as i32)) }.map_err(wrap)?;
        Ok((
            CudaStorage { slice: CudaStorageSlice::F32(out), device: dev },
            Shape::from((b, t, h, dh)),
        ))
    }

    pub fn backward(
        s_rkv: &CudaStorage,
        l_rkv: &Layout,
        s_w: &CudaStorage,
        l_w: &Layout,
        s_dy: &CudaStorage,
        l_dy: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let d = l_rkv.dims();
        if d.len() != 5 || d[3] != 3 {
            candle_core::bail!("wkv cuda bwd: arg1 must be (B,T,H,3,dh), got {d:?}");
        }
        let (b, t, h, dh) = (d[0], d[1], d[2], d[4]);
        require_contig(l_rkv)?;
        require_contig(l_w)?;
        require_contig(l_dy)?;
        let dev = s_rkv.device.clone();
        let cu = dev.cuda_device();
        let rkv = f32_slice(s_rkv)?;
        let w = f32_slice(s_w)?;
        let dy = f32_slice(s_dy)?;
        let nbth = b * t * h;
        // grads is pre-zeroed: dw is built with atomicAdd across the batch blocks.
        let grads = cu.alloc_zeros::<f32>(nbth * 3 * dh + h * dh).map_err(wrap)?;
        // Chunk-checkpointed backward: store NC chunk-entry states + one C-step
        // recompute buffer per (b,h) instead of all T states. C ≈ sqrt(T) balances
        // checkpoint count (NC) against per-chunk recompute (C). scratch shrinks
        // from B·H·T·dh² to B·H·(NC+C)·dh² (~10× at T=512).
        let c = ((t as f64).sqrt().round() as usize).clamp(1, t.max(1));
        let nc = (t + c - 1) / c;
        let scratch = cu.alloc_zeros::<f32>(b * h * (nc + c) * dh * dh).map_err(wrap)?;
        let f = get_func(&cu, dh, "wkv_backward_f32")?;
        let cfg = launch_cfg(b, h, dh);
        unsafe {
            f.launch(
                cfg,
                (rkv, w, dy, &grads, &scratch, b as i32, t as i32, h as i32, c as i32),
            )
        }
        .map_err(wrap)?;
        Ok((
            CudaStorage { slice: CudaStorageSlice::F32(grads), device: dev },
            Shape::from((nbth * 3 * dh + h * dh,)),
        ))
    }
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

// Backward, chunk-checkpointed. thread == row i; one block per (b,h).
// Instead of storing all T forward states (O(T·dh²)), checkpoint the state
// entering each chunk of C steps and recompute within-chunk states on the fly —
// cutting `scratch` to O((NC+C)·dh²) per (b,h), NC=ceil(T/C) (≈2·sqrt(T) at
// C≈sqrt(T)). The math is identical to the all-states version, just recomputed.
// All grads land in one packed buffer `grads`: drkv in [0, B*T*H*3*DH), dw in the
// tail (accumulated with atomicAdd, so `grads` must be pre-zeroed). `scratch` per
// block = [ckpt: NC·dh²][cbuf: C·dh²]:
//   ckpt[c] = S_{c*C-1} (state entering chunk c; c=0 is the zero state, unused)
//   cbuf[m] = S_{lo+m-1} for the chunk being reversed (reused across chunks).
extern "C" __global__ void wkv_backward_f32(
    const float* __restrict__ rkv,
    const float* __restrict__ w,
    const float* __restrict__ dy,
    float* __restrict__ grads,      // [B*T*H*3*DH + H*DH]: drkv head, dw tail
    float* __restrict__ scratch,    // [B*H*(NC+C)*DH*DH], internal
    const int B, const int T, const int H, const int C)
{
    const int bh = blockIdx.x;
    const int b = bh / H, h = bh % H;
    const int i = threadIdx.x;            // row index
    if (i >= DH) return;
    float* drkv = grads;                          // dr +0, dk +DH, dv +2DH per (b,t,h)
    float* dw   = grads + (long)B*T*H*3*DH;        // [H,DH]
    const float wi = w[h*DH + i];
    const int NC = (T + C - 1) / C;
    const long sbase = (long)(b*H + h) * (NC + C) * DH * DH; // this block's scratch
    float* ckpt = scratch + sbase;                          // [NC][DH][DH]
    float* cbuf = scratch + sbase + (long)NC*DH*DH;          // [C][DH][DH]
    __shared__ float sv[DH], sdy[DH], sr[DH], sk[DH], sdv[DH];

    // Sub-pass A: forward, emit dr, checkpoint chunk-entry states.
    {
        float srow[DH];
        for (int j = 0; j < DH; j++) srow[j] = 0.f;
        for (int t = 0; t < T; t++) {
            const long blk = (long)(b*T + t)*H + h;
            const float* base = rkv + blk*3*DH;
            sv[i]  = base[2*DH + i];
            sdy[i] = dy[blk*DH + i];
            __syncthreads();
            const float ki = base[DH + i];
            for (int j = 0; j < DH; j++) srow[j] = wi*srow[j] + ki*sv[j]; // S_t row i
            float dri = 0.f;
            for (int j = 0; j < DH; j++) dri += sdy[j]*srow[j];
            drkv[blk*3*DH + i] = dri;
            if ((t + 1) % C == 0) {              // t = c*C-1 -> entering chunk c
                int c = (t + 1) / C;
                if (c < NC) {
                    float* ck = ckpt + (long)c*DH*DH + (long)i*DH;
                    for (int j = 0; j < DH; j++) ck[j] = srow[j];
                }
            }
            __syncthreads();
        }
    }

    // Sub-pass B: reverse over chunks; recompute chunk states, emit dk/dv/dw.
    float dsrow[DH];
    for (int j = 0; j < DH; j++) dsrow[j] = 0.f;
    float dwi = 0.f;
    for (int c = NC - 1; c >= 0; c--) {
        const int lo = c*C;
        int hi = lo + C; if (hi > T) hi = T;
        const int clen = hi - lo;
        // Recompute cbuf[m] = S_{lo+m-1}, m = 0..clen-1, starting from ckpt[c].
        {
            float srow[DH];
            if (c == 0) { for (int j = 0; j < DH; j++) srow[j] = 0.f; }
            else {
                float* ck = ckpt + (long)c*DH*DH + (long)i*DH;
                for (int j = 0; j < DH; j++) srow[j] = ck[j];
            }
            float* cb0 = cbuf + (long)i*DH;       // cbuf[0] row i = S_{lo-1}
            for (int j = 0; j < DH; j++) cb0[j] = srow[j];
            for (int m = 1; m < clen; m++) {
                const int t = lo + m - 1;
                const long blk = (long)(b*T + t)*H + h;
                const float* base = rkv + blk*3*DH;
                sv[i] = base[2*DH + i];
                __syncthreads();
                const float ki = base[DH + i];
                for (int j = 0; j < DH; j++) srow[j] = wi*srow[j] + ki*sv[j];
                float* cb = cbuf + (long)m*DH*DH + (long)i*DH;
                for (int j = 0; j < DH; j++) cb[j] = srow[j];
                __syncthreads();
            }
        }
        // Reverse within chunk: m = clen-1..0, t = lo+m. S_{t-1} = cbuf[m].
        for (int m = clen - 1; m >= 0; m--) {
            const int t = lo + m;
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
            float* cb = cbuf + (long)m*DH*DH + (long)i*DH;  // S_{t-1} row i
            for (int j = 0; j < DH; j++) dwi += dsrow[j]*cb[j];
            __syncthreads();
        }
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

    /// `cuda_fwd` must match the verified CPU forward at the real `head_dim` (64).
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_forward_matches_cpu() {
        use candle_core::Device;
        let (b, t, h, dh) = (2usize, 7, 3, 64);
        let n = b * t * h * dh;
        let mut rng = Lcg(0xfeed_face_cafe_babe);
        let rv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let kv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let vv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let wv: Vec<f32> = (0..h * dh).map(|_| 0.3 + rng.next_f32() * 0.5).collect();

        let mk = |dev: &Device| -> (Tensor, Tensor, Tensor, Tensor) {
            (
                Tensor::from_vec(rv.clone(), (b, t, h, dh), dev).unwrap(),
                Tensor::from_vec(kv.clone(), (b, t, h, dh), dev).unwrap(),
                Tensor::from_vec(vv.clone(), (b, t, h, dh), dev).unwrap(),
                Tensor::from_vec(wv.clone(), (h, dh), dev).unwrap(),
            )
        };
        let cpu = Device::Cpu;
        let cuda = Device::new_cuda(0).unwrap();
        let (rc, kc, vc, wc) = mk(&cpu);
        let (rg, kg, vg, wg) = mk(&cuda);

        let y_cpu = super::fused_wkv(&rc, &kc, &vc, &wc).unwrap();
        let y_cuda = super::fused_wkv(&rg, &kg, &vg, &wg)
            .unwrap()
            .to_device(&cpu)
            .unwrap();
        let diff = (y_cpu - y_cuda)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        eprintln!("cuda fwd vs cpu max abs diff = {diff:.3e}");
        assert!(diff < 1e-4, "cuda forward diverged from cpu: {diff}");
    }

    /// The fused CUDA backward (via `WkvBackwardOp`) must match the CPU reference
    /// grads. Compares the packed `[drkv | dw]` output element-wise across several
    /// T that exercise the chunk-checkpointing: single step, multi-chunk, a
    /// non-divisible tail, and the production seq len 512.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backward_matches_cpu() {
        use candle_core::Device;
        let cuda = Device::new_cuda(0).unwrap();
        for &(b, t, h, dh) in &[
            (2usize, 1usize, 2usize, 64usize),
            (2, 8, 3, 64),
            (1, 33, 2, 64),
            (2, 64, 3, 64),
            (2, 512, 2, 64),
        ] {
            let n = b * t * h * dh;
            let mut rng = Lcg(0x5151_2323_9090_aaaa ^ t as u64);
            let rkvv: Vec<f32> = (0..n * 3).map(|_| rng.next_f32()).collect();
            let wv: Vec<f32> = (0..h * dh).map(|_| 0.3 + rng.next_f32() * 0.5).collect();
            let dyv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

            let pack = |dev: &Device| -> Vec<f32> {
                let rkv = Tensor::from_vec(rkvv.clone(), (b, t, h, 3, dh), dev).unwrap();
                let w = Tensor::from_vec(wv.clone(), (h, dh), dev).unwrap();
                let dy = Tensor::from_vec(dyv.clone(), (b, t, h, dh), dev).unwrap();
                rkv.apply_op3(&w, &dy, super::WkvBackwardOp)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let g_cpu = pack(&Device::Cpu);
            let g_cuda = pack(&cuda);
            let diff = g_cpu
                .iter()
                .zip(&g_cuda)
                .map(|(a, c)| (a - c).abs())
                .fold(0f32, f32::max);
            eprintln!("cuda bwd T={t}: max abs diff = {diff:.3e}");
            assert!(diff < 1e-4, "cuda backward diverged at T={t}: {diff}");
        }
    }

    /// End-to-end: gradients from `loss = sum(fused_wkv(r,k,v,w) * dy)` must agree
    /// between CUDA and CPU, exercising the full `WkvScan::bwd` plumbing across a
    /// short, a multi-chunk, and the production (512) seq len.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_autograd_matches_cpu() {
        use candle_core::{Device, Var};
        let cuda = Device::new_cuda(0).unwrap();
        for &(b, t, h, dh) in &[
            (2usize, 5usize, 2usize, 64usize),
            (2, 64, 3, 64),
            (2, 512, 2, 64),
        ] {
            let n = b * t * h * dh;
            let mut rng = Lcg(0xc0ff_eeee_1234_5678 ^ t as u64);
            let rv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let kv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let vv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let wv: Vec<f32> = (0..h * dh).map(|_| 0.3 + rng.next_f32() * 0.5).collect();
            let dyv: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

            let run = |dev: &Device| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
                let r = Var::from_vec(rv.clone(), (b, t, h, dh), dev).unwrap();
                let k = Var::from_vec(kv.clone(), (b, t, h, dh), dev).unwrap();
                let v = Var::from_vec(vv.clone(), (b, t, h, dh), dev).unwrap();
                let w = Var::from_vec(wv.clone(), (h, dh), dev).unwrap();
                let dy = Tensor::from_vec(dyv.clone(), (b, t, h, dh), dev).unwrap();
                let y =
                    super::fused_wkv(r.as_tensor(), k.as_tensor(), v.as_tensor(), w.as_tensor())
                        .unwrap();
                let loss = y.mul(&dy).unwrap().sum_all().unwrap();
                let grads = loss.backward().unwrap();
                let g = |x: &Var| {
                    grads
                        .get(x.as_tensor())
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_device(&Device::Cpu)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                };
                (g(&r), g(&k), g(&v), g(&w))
            };

            let (rc, kc, vc, wc) = run(&Device::Cpu);
            let (rg, kg, vg, wg) = run(&cuda);
            let md = |a: &[f32], c: &[f32]| {
                a.iter().zip(c).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
            };
            let (dr, dk, dv, dw) = (md(&rc, &rg), md(&kc, &kg), md(&vc, &vg), md(&wc, &wg));
            eprintln!("cuda autograd T={t}: dr={dr:.2e} dk={dk:.2e} dv={dv:.2e} dw={dw:.2e}");
            assert!(
                dr < 1e-4 && dk < 1e-4 && dv < 1e-4 && dw < 1e-4,
                "grad mismatch T={t}: dr={dr} dk={dk} dv={dv} dw={dw}"
            );
        }
    }

    /// Benchmark (ignored): fused op vs the sequential candle loop, fwd+bwd, at a
    /// representative training shape. Run with:
    ///   cargo test -p tinybit-core --features cuda -- --ignored --nocapture bench_wkv
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_wkv_fused_vs_loop() {
        use candle_core::{Device, Var, D};
        use std::time::Instant;
        let dev = Device::new_cuda(0).unwrap();
        let (b, t, h, dh) = (4usize, 512usize, 6usize, 64usize);
        let n = b * t * h * dh;
        let mut rng = Lcg(0xbeef_0bad_f00d_1234);
        let mut mk = || -> Vec<f32> { (0..n).map(|_| rng.next_f32()).collect() };
        let (rv, kv, vv) = (mk(), mk(), mk());
        let wv: Vec<f32> = (0..h * dh).map(|_| 0.3 + rng.next_f32() * 0.5).collect();

        // Fresh Vars each call so the autograd graph does not accumulate.
        let make = || -> (Var, Var, Var, Var) {
            (
                Var::from_vec(rv.clone(), (b, t, h, dh), &dev).unwrap(),
                Var::from_vec(kv.clone(), (b, t, h, dh), &dev).unwrap(),
                Var::from_vec(vv.clone(), (b, t, h, dh), &dev).unwrap(),
                Var::from_vec(wv.clone(), (h, dh), &dev).unwrap(),
            )
        };

        // Sequential candle loop (mirrors the non-fused branch in time_mix).
        let loop_scan = |r: &Tensor, k: &Tensor, v: &Tensor, w: &Tensor| -> Tensor {
            let w_b = w.unsqueeze(0).unwrap().unsqueeze(D::Minus1).unwrap();
            let mut state = Tensor::zeros((b, h, dh, dh), DType::F32, &dev).unwrap();
            let mut outs = Vec::with_capacity(t);
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
            Tensor::cat(&outs, 1).unwrap()
        };

        let iters = 20;
        let warmup = 5;
        let bench = |fused: bool| -> f64 {
            let mut total = 0f64;
            for it in 0..(iters + warmup) {
                let (r, k, v, w) = make();
                let start = Instant::now();
                let y = if fused {
                    super::fused_wkv(r.as_tensor(), k.as_tensor(), v.as_tensor(), w.as_tensor())
                        .unwrap()
                } else {
                    loop_scan(r.as_tensor(), k.as_tensor(), v.as_tensor(), w.as_tensor())
                };
                let loss = y.sum_all().unwrap();
                let _g = loss.backward().unwrap();
                // Force completion: pull a scalar to the host (syncs the stream).
                let _ = loss.to_scalar::<f32>().unwrap();
                if it >= warmup {
                    total += start.elapsed().as_secs_f64();
                }
            }
            total / iters as f64
        };

        let t_loop = bench(false);
        let t_fused = bench(true);
        let toks = (b * t) as f64;
        eprintln!(
            "WKV fwd+bwd @ b={b} t={t} h={h} dh={dh}: loop={:.2} ms, fused={:.2} ms, speedup={:.2}x ({:.0} vs {:.0} tok/s, one layer)",
            t_loop * 1e3,
            t_fused * 1e3,
            t_loop / t_fused,
            toks / t_fused,
            toks / t_loop,
        );
    }
}
