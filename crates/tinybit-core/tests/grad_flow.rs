// Gradient-flow regression tests.
//
// These guard the class of bug where a layer silently drops gradient (candle's
// fused `ops::layer_norm` is registered with `apply_op3_no_bwd` and has no
// backward — using `candle_nn::LayerNorm` froze the entire transformer stack,
// so only the tied embedding ever trained and the model produced gibberish).
//
// The fast tests run by default. `diagnose` (ignored) dumps per-layer norms on
// the real micro model for manual inspection:
//   cargo test -p tinybit-core --test grad_flow -- --ignored --nocapture diagnose

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use tinybit_core::config::ModelConfig;
use tinybit_core::model::bitlinear::LayerNorm;
use tinybit_core::model::TinyBit;

fn ce_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    let (b, t, v) = logits.dims3().unwrap();
    let lf = logits.reshape((b * t, v)).unwrap();
    let tf = targets.reshape(b * t).unwrap().to_dtype(DType::I64).unwrap();
    let lp = candle_nn::ops::log_softmax(&lf.to_dtype(DType::F32).unwrap(), 1).unwrap();
    let g = lp.gather(&tf.unsqueeze(1).unwrap(), 1).unwrap().squeeze(1).unwrap();
    g.neg().unwrap().mean_all().unwrap()
}

fn norm(t: &Tensor) -> f64 {
    t.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt() as f64
}

/// Small model that builds the full architecture but runs fast on CPU.
fn tiny_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 128,
        num_layers: 3,
        d_model: 64,
        d_ffn: 128,
        num_heads: 2,
        head_dim: 32,
        ternary_ffn: false,
        int8_time: false,
        max_seq_len: 64,
        dropout: 0.0,
        spec_heads: 0,
    }
}

/// The model's hand-rolled LayerNorm must backprop to its input AND its affine
/// params. (candle_nn::LayerNorm's fused path does not — that was the bug.)
#[test]
fn our_layernorm_is_differentiable() {
    let dev = Device::Cpu;
    let (b, t, d) = (2usize, 4usize, 16usize);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let ln = LayerNorm::new(d, 1e-5, vb.pp("ln")).unwrap();
    let x = Var::from_tensor(&Tensor::randn(0f32, 1.0, (b, t, d), &dev).unwrap()).unwrap();

    let y = ln.forward(x.as_tensor()).unwrap();
    let loss = y.sqr().unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let data = varmap.data().lock().unwrap();

    let xg = grads.get(x.as_tensor()).expect("LayerNorm dropped gradient to its input");
    assert!(norm(&xg) > 0.0, "LayerNorm input gradient is zero");
    for p in ["ln.weight", "ln.bias"] {
        let v = data.get(p).unwrap();
        let g = grads.get(v.as_tensor()).unwrap_or_else(|| panic!("no gradient for {p}"));
        assert!(norm(&g) > 0.0, "{p} gradient is zero");
    }
}

/// Every trainable parameter in the full architecture must receive a finite,
/// non-zero gradient from a single forward+backward. A regression here means a
/// sublayer is silently severing the autograd graph.
#[test]
fn all_params_receive_finite_gradient() {
    let dev = Device::Cpu;
    let cfg = tiny_config();
    let vocab = cfg.vocab_size as u32;
    let (b, t) = (2usize, 16usize);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = TinyBit::new(cfg, vb).unwrap();

    let ids: Vec<u32> = (0..b * t).map(|i| (i as u32).wrapping_mul(2654435761) % vocab).collect();
    let tgt: Vec<u32> = (0..b * t).map(|i| ((i + 7) as u32).wrapping_mul(40503) % vocab).collect();
    let input = Tensor::from_vec(ids, (b, t), &dev).unwrap();
    let target = Tensor::from_vec(tgt, (b, t), &dev).unwrap();

    let (logits, _) = model.forward_train(&input).unwrap();
    let grads = ce_loss(&logits, &target).backward().unwrap();
    let data = varmap.data().lock().unwrap();

    let mut missing = Vec::new();
    for (name, v) in data.iter() {
        match grads.get(v.as_tensor()) {
            None => missing.push(format!("{name}: NONE")),
            Some(g) => {
                let n = norm(&g);
                if !n.is_finite() || n == 0.0 {
                    missing.push(format!("{name}: {n}"));
                }
            }
        }
    }
    assert!(missing.is_empty(), "params with no/zero/non-finite gradient:\n  {}", missing.join("\n  "));
}

/// End-to-end learning check: the model must be able to drive a fixed batch's
/// loss far below the random baseline within a handful of steps.
#[test]
fn overfits_a_fixed_batch() {
    let dev = Device::Cpu;
    let cfg = tiny_config();
    let vocab = cfg.vocab_size as u32;
    let (b, t) = (2usize, 16usize);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = TinyBit::new(cfg, vb).unwrap();

    let ids: Vec<u32> = (0..b * t).map(|i| (i as u32).wrapping_mul(2654435761) % vocab).collect();
    let tgt: Vec<u32> = (0..b * t).map(|i| ((i + 7) as u32).wrapping_mul(40503) % vocab).collect();
    let input = Tensor::from_vec(ids, (b, t), &dev).unwrap();
    let target = Tensor::from_vec(tgt, (b, t), &dev).unwrap();

    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() }).unwrap();
    let mut last = f32::INFINITY;
    for _ in 0..120 {
        let (logits, _) = model.forward_train(&input).unwrap();
        let loss = ce_loss(&logits, &target);
        last = loss.to_scalar::<f32>().unwrap();
        opt.backward_step(&loss).unwrap();
    }
    assert!(last < 0.5, "loss did not drop (final {last:.4}, baseline ln(V)={:.2})", (vocab as f64).ln());
}

/// bf16 mixed precision (CUDA only): the bf16 forward must track the f32 forward
/// on identical weights, and bf16 training must overfit with finite gradients.
/// Exercises the real production path (bf16 matmuls + f32 fused WKV scan + f32
/// norms/loss + f32 master weights). Run with:
///   cargo test -p tinybit-core --features cuda --test grad_flow -- --test-threads=1 bf16
#[cfg(feature = "cuda")]
#[test]
fn bf16_tracks_f32_and_overfits_cuda() {
    let dev = Device::new_cuda(0).unwrap();
    let cfg = tiny_config();
    let vocab = cfg.vocab_size as u32;
    let (b, t) = (2usize, 16usize);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let mut model = TinyBit::new(cfg, vb).unwrap();

    let ids: Vec<u32> = (0..b * t).map(|i| (i as u32).wrapping_mul(2654435761) % vocab).collect();
    let tgt: Vec<u32> = (0..b * t).map(|i| ((i + 7) as u32).wrapping_mul(40503) % vocab).collect();
    let input = Tensor::from_vec(ids, (b, t), &dev).unwrap();
    let target = Tensor::from_vec(tgt, (b, t), &dev).unwrap();

    // (1) bf16 forward tracks f32 on the SAME weights.
    let (lf32, _) = model.forward_train(&input).unwrap();
    let loss_f32 = ce_loss(&lf32, &target).to_scalar::<f32>().unwrap();
    model.set_compute_dtype(DType::BF16);
    let (lbf16, _) = model.forward_train(&input).unwrap();
    let loss_bf16 = ce_loss(&lbf16, &target).to_scalar::<f32>().unwrap();
    assert!(loss_bf16.is_finite(), "bf16 forward produced non-finite loss");
    assert!(
        (loss_bf16 - loss_f32).abs() < 0.15,
        "bf16 loss {loss_bf16:.4} diverged from f32 {loss_f32:.4}"
    );

    // (2) bf16 training overfits a fixed batch (grads finite, master weights f32).
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() }).unwrap();
    let mut last = f32::INFINITY;
    for _ in 0..150 {
        let (logits, _) = model.forward_train(&input).unwrap();
        let loss = ce_loss(&logits, &target);
        last = loss.to_scalar::<f32>().unwrap();
        assert!(last.is_finite(), "bf16 loss went non-finite during training");
        opt.backward_step(&loss).unwrap();
    }
    assert!(
        last < 1.0,
        "bf16 did not overfit (final {last:.4}, baseline ln(V)={:.2})",
        (vocab as f64).ln()
    );
}

#[test]
#[ignore]
fn diagnose() {
    let dev = Device::Cpu;
    let cfg = ModelConfig::micro();
    let nl = cfg.num_layers;
    let vocab = cfg.vocab_size as u32;
    let (b, t) = (2usize, 32usize);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = TinyBit::new(cfg, vb).unwrap();

    let ids: Vec<u32> = (0..b * t).map(|i| (i as u32).wrapping_mul(2654435761) % vocab).collect();
    let tgt: Vec<u32> = (0..b * t).map(|i| ((i + 7) as u32).wrapping_mul(40503) % vocab).collect();
    let input = Tensor::from_vec(ids, (b, t), &dev).unwrap();
    let target = Tensor::from_vec(tgt, (b, t), &dev).unwrap();

    let (logits, _) = model.forward_train(&input).unwrap();
    let loss = ce_loss(&logits, &target);
    eprintln!("loss = {:.4}  (random baseline ln(V) = {:.4})", loss.to_scalar::<f32>().unwrap(), (vocab as f64).ln());

    let grads = loss.backward().unwrap();
    let data = varmap.data().lock().unwrap();
    for key in ["embed.embed.weight", "embed.ln_out.weight", "embed.ln_out.bias"] {
        if let Some(v) = data.get(key) {
            let g = grads.get(v.as_tensor());
            eprintln!("{key:32} gnorm={}", g.map(|g| format!("{:.4e}", norm(&g))).unwrap_or_else(|| "NONE".into()));
        }
    }
    eprintln!("\n{:>3}  {:>14}  {:>14}  {:>14}", "blk", "ln1.w", "tmix.w_o", "cmix.w_v");
    for i in 0..nl {
        let gn = |suffix: &str| -> String {
            let key = format!("block_{i}.{suffix}");
            data.get(&key)
                .and_then(|v| grads.get(v.as_tensor()).map(|g| norm(&g)))
                .map(|n| format!("{:.4e}", n))
                .unwrap_or_else(|| "NONE".into())
        };
        eprintln!("{:>3}  {:>14}  {:>14}  {:>14}", i,
            gn("ln1.weight"), gn("time_mix.w_o.weight"), gn("channel_mix.w_v.weight"));
    }
}
