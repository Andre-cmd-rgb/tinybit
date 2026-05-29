//! Microbenchmarks for the linear-projection matmul pattern.
//!
//! Compares candle's `broadcast_left` + batched matmul (one small GEMM per batch
//! element, weight replicated across the batch) against flattening `(B,T,D)` to
//! `(B*T,D)` and doing a SINGLE GEMM. Run on CPU here; the win is larger on CUDA
//! (one big cuBLAS call vs B small ones).
//!
//!   cargo test -p tinybit-core --release --test bench_matmul -- --ignored --nocapture
//!
//! `bench_micro_step` times the whole micro model fwd+bwd, so running it before
//! and after the model change is a clean A/B (it uses whichever path the model
//! is currently compiled with).

use candle_core::{Device, Tensor, Var};
use std::time::Instant;

fn avg_ms<F: FnMut()>(mut f: F, iters: usize, warmup: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let s = Instant::now();
    for _ in 0..iters {
        f();
    }
    s.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[test]
#[ignore]
fn bench_proj_broadcast_bmm_vs_flat_gemm() {
    let dev = Device::Cpu;
    let (b, t, d) = (11usize, 512usize, 384usize); // micro at the real L4 microbatch
    let dffn = 1344usize;
    let v = 32008usize;
    let x = Tensor::randn(0f32, 1.0, (b, t, d), &dev).unwrap();

    eprintln!("micro projection fwd+bwd, b={b} t={t} d={d} (CPU):");
    for (label, out) in [("d->d   ", d), ("d->dffn", dffn), ("d->V    ", v)] {
        let w = Var::from_tensor(&Tensor::randn(0f32, 0.02, (out, d), &dev).unwrap()).unwrap();
        let wt = w.as_tensor();
        let old = || {
            let wb = wt.broadcast_left(b).unwrap();
            let y = x.matmul(&wb.t().unwrap()).unwrap();
            let loss = y.sqr().unwrap().sum_all().unwrap();
            let _ = loss.backward().unwrap();
            let _ = loss.to_scalar::<f32>().unwrap();
        };
        let new = || {
            let x2 = x.reshape((b * t, d)).unwrap();
            let y = x2.matmul(&wt.t().unwrap()).unwrap();
            let loss = y.sqr().unwrap().sum_all().unwrap();
            let _ = loss.backward().unwrap();
            let _ = loss.to_scalar::<f32>().unwrap();
        };
        let to = avg_ms(old, 8, 3);
        let tn = avg_ms(new, 8, 3);
        eprintln!("  {label}: bmm={to:7.1}ms  flat-gemm={tn:7.1}ms  speedup={:.2}x", to / tn);
    }
}

#[test]
#[ignore]
fn bench_micro_step() {
    use candle_nn::{VarBuilder, VarMap};
    use tinybit_core::config::ModelConfig;
    use tinybit_core::model::TinyBit;

    let dev = Device::Cpu;
    let cfg = ModelConfig::micro();
    let (b, t) = (4usize, 256usize); // smaller than the real microbatch so CPU iters are quick
    let vocab = cfg.vocab_size as u32;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
    let model = TinyBit::new(cfg, vb).unwrap();
    let ids: Vec<u32> = (0..b * t).map(|i| (i as u32).wrapping_mul(2654435761) % vocab).collect();
    let input = Tensor::from_vec(ids, (b, t), &dev).unwrap();

    let run = || {
        let (logits, _) = model.forward_train(&input).unwrap();
        let loss = logits.sum_all().unwrap();
        let _ = loss.backward().unwrap();
        let _ = loss.to_scalar::<f32>().unwrap();
    };
    let ms = avg_ms(run, 6, 2);
    eprintln!("micro fwd+bwd (16L d384) b={b} t={t} CPU: {ms:.1} ms/step");
}
