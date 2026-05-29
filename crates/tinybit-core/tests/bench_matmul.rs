//! Microbenchmarks for the linear-projection matmul pattern.
//!
//! Compares candle's `broadcast_left` + batched matmul (one small GEMM per batch
//! element, weight replicated across the batch) against flattening `(B,T,D)` to
//! `(B*T,D)` and doing a SINGLE GEMM.
//!
//! CPU:  cargo test -p tinybit-core --release --test bench_matmul -- --ignored --nocapture
//! CUDA: cargo test -p tinybit-core --release --features cuda --test bench_matmul -- --ignored --nocapture
//!
//! Env overrides: TINYBIT_BENCH_B, TINYBIT_BENCH_T, TINYBIT_BENCH_CPU=1 (force CPU).
//! `bench_micro_step` times the whole micro model fwd+bwd (uses whichever matmul
//! path the model is compiled with — a clean before/after A/B).

use candle_core::{Device, Tensor, Var};
use std::time::Instant;

fn dev() -> Device {
    if std::env::var("TINYBIT_BENCH_CPU").is_ok() {
        return Device::Cpu;
    }
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

fn dev_name(d: &Device) -> &'static str {
    if d.is_cuda() {
        "cuda"
    } else {
        "cpu"
    }
}

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Average ms per call, synchronizing the device each iter so CUDA's async
/// queue is fully drained before the timer stops.
fn avg_ms<F: FnMut()>(d: &Device, mut f: F, iters: usize, warmup: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let _ = d.synchronize();
    let s = Instant::now();
    for _ in 0..iters {
        f();
    }
    let _ = d.synchronize();
    s.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[test]
#[ignore]
fn bench_proj_broadcast_bmm_vs_flat_gemm() {
    let d = dev();
    // Default to a 4GB-safe shape on GPU; override for the real L4 microbatch.
    let b = env_usize("TINYBIT_BENCH_B", 8);
    let t = env_usize("TINYBIT_BENCH_T", 256);
    let dm = 384usize;
    let dffn = 1344usize;
    let v = 32008usize;
    let x = Tensor::randn(0f32, 1.0, (b, t, dm), &d).unwrap();

    eprintln!("projection fwd+bwd, b={b} t={t} d={dm} ({}):", dev_name(&d));
    for (label, out) in [("d->d   ", dm), ("d->dffn", dffn), ("d->V    ", v)] {
        let w = Var::from_tensor(&Tensor::randn(0f32, 0.02, (out, dm), &d).unwrap()).unwrap();
        let wt = w.as_tensor();
        let old = || {
            let wb = wt.broadcast_left(b).unwrap();
            let y = x.matmul(&wb.t().unwrap()).unwrap();
            let loss = y.sqr().unwrap().sum_all().unwrap();
            let _ = loss.backward().unwrap();
            let _ = loss.to_scalar::<f32>().unwrap();
        };
        let new = || {
            let x2 = x.reshape((b * t, dm)).unwrap();
            let y = x2.matmul(&wt.t().unwrap()).unwrap();
            let loss = y.sqr().unwrap().sum_all().unwrap();
            let _ = loss.backward().unwrap();
            let _ = loss.to_scalar::<f32>().unwrap();
        };
        let to = avg_ms(&d, old, 12, 4);
        let tn = avg_ms(&d, new, 12, 4);
        eprintln!("  {label}: bmm={to:7.2}ms  flat-gemm={tn:7.2}ms  speedup={:.2}x", to / tn);
    }
}

/// Real cross-entropy (log_softmax over the full vocab + gather), matching
/// tinybit-train's loss, so the profile includes the vocab-softmax cost.
fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    let (b, t, v) = logits.dims3().unwrap();
    let lf = logits.reshape((b * t, v)).unwrap();
    let lp = candle_nn::ops::log_softmax(&lf.to_dtype(candle_core::DType::F32).unwrap(), 1).unwrap();
    let ti = targets.reshape(b * t).unwrap().to_dtype(candle_core::DType::I64).unwrap();
    let g = lp.gather(&ti.unsqueeze(1).unwrap(), 1).unwrap().squeeze(1).unwrap();
    g.neg().unwrap().mean_all().unwrap()
}

#[test]
#[ignore]
fn bench_micro_step() {
    use candle_nn::{VarBuilder, VarMap};
    use tinybit_core::config::ModelConfig;
    use tinybit_core::model::TinyBit;

    let d = dev();
    let cfg = ModelConfig::micro();
    let b = env_usize("TINYBIT_BENCH_B", 4);
    let t = env_usize("TINYBIT_BENCH_T", 256);
    let bf16 = std::env::var("TINYBIT_BENCH_BF16").is_ok();
    let vocab = cfg.vocab_size as u32;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &d);
    let mut model = TinyBit::new(cfg, vb).unwrap();
    if bf16 {
        model.set_compute_dtype(candle_core::DType::BF16);
    }
    let ids: Vec<u32> = (0..b * t).map(|i| (i as u32).wrapping_mul(2654435761) % vocab).collect();
    let input = Tensor::from_vec(ids.clone(), (b, t), &d).unwrap();
    let tgts: Vec<u32> = ids.iter().map(|i| i.wrapping_add(1) % vocab).collect();
    let targets = Tensor::from_vec(tgts, (b, t), &d).unwrap();

    // Phase split: forward (block stack + lm head) / loss (vocab softmax) / backward.
    let (mut tf, mut tl, mut tb) = (0f64, 0f64, 0f64);
    let (iters, warm) = (10usize, 3usize);
    for it in 0..(iters + warm) {
        let _ = d.synchronize();
        let s = Instant::now();
        let (logits, _) = model.forward_train(&input).unwrap();
        let _ = d.synchronize();
        let f = s.elapsed().as_secs_f64() * 1e3;
        let s = Instant::now();
        let loss = cross_entropy(&logits, &targets);
        let _ = loss.to_scalar::<f32>().unwrap();
        let l = s.elapsed().as_secs_f64() * 1e3;
        let s = Instant::now();
        let _ = loss.backward().unwrap();
        let _ = d.synchronize();
        let bw = s.elapsed().as_secs_f64() * 1e3;
        if it >= warm {
            tf += f;
            tl += l;
            tb += bw;
        }
    }
    let n = iters as f64;
    eprintln!(
        "micro step (16L d384) b={b} t={t} {} bf16={bf16}: fwd={:.1} loss={:.1} bwd={:.1} total={:.1} ms/step",
        dev_name(&d),
        tf / n,
        tl / n,
        tb / n,
        (tf + tl + tb) / n,
    );
}
