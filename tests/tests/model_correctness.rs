use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tinybit_core::{config::ModelConfig, model::TinyBit, state::InferenceState};

/// Tiny throwaway config for fast correctness tests — decoupled from the shipped
/// model lineup (micro/small/medium) so those can change without touching these.
fn tiny_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 256, num_layers: 3, d_model: 64, d_ffn: 224,
        num_heads: 1, head_dim: 64, ternary_ffn: false, int8_time: false,
        max_seq_len: 64, dropout: 0.0, spec_heads: 0,
    }
}

fn nano_model() -> anyhow::Result<(TinyBit, ModelConfig)> {
    let config = tiny_config();
    let device = Device::Cpu;
    let vmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&vmap, DType::F32, &device);
    let model = TinyBit::new(config.clone(), vb)?;
    Ok((model, config))
}

#[test]
fn test_forward_shapes_nano() -> anyhow::Result<()> {
    let (model, config) = nano_model()?;
    let device = Device::Cpu;
    let b = 2usize;
    let t = 16usize;
    // Random token IDs in range [0, vocab_size)
    let ids: Vec<u32> = (0..b * t).map(|i| (i % config.vocab_size) as u32).collect();
    let token_ids = Tensor::from_vec(ids, (b, t), &device)?.to_dtype(DType::U32)?;
    let (logits, _spec) = model.forward_train(&token_ids)?;
    assert_eq!(logits.dims(), &[b, t, config.vocab_size], "logits shape mismatch");
    // Check no NaN
    let flat = logits.flatten_all()?.to_vec1::<f32>()?;
    assert!(flat.iter().all(|v| v.is_finite()), "logits contain NaN/inf");
    Ok(())
}

#[test]
fn test_inference_step_matches_train() -> anyhow::Result<()> {
    let (model, config) = nano_model()?;
    let device = Device::Cpu;
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let t = ids.len();
    let token_ids = Tensor::from_vec(ids.clone(), (1, t), &device)?.to_dtype(DType::U32)?;
    let (train_logits, _) = model.forward_train(&token_ids)?; // (1, T, vocab)

    // Now run step by step
    let mut state = InferenceState::zeros(&config, &device)?;
    let mut step_logits: Vec<Vec<f32>> = Vec::new();
    for &id in &ids {
        let tid = Tensor::from_vec(vec![id], (1, 1), &device)?.to_dtype(DType::U32)?;
        let logits_step = model.forward_step(&tid, &mut state)?; // (1, vocab)
        step_logits.push(logits_step.to_vec2::<f32>()?[0].clone());
    }

    // Compare last token's output (index t-1)
    let train_last = train_logits.narrow(1, t - 1, 1)?.squeeze(1)?.to_vec2::<f32>()?[0].clone();
    let step_last = &step_logits[t - 1];
    // The training path (parallel WKV scan, token-shift via roll, flattened-GEMM
    // projections) and the inference path (sequential per-token state) are
    // mathematically equivalent but accumulate floats in different orders, so
    // they agree only approximately. With UNSEEDED random init the gap varies
    // per run and occasionally grazes 1e-3; 3e-3 is robust while still catching a
    // real path bug (e.g. the old LayerNorm-freeze diverged by orders of
    // magnitude). `linear_flat`'s exactness is pinned separately and tightly in
    // tinybit-core's bitlinear unit tests.
    let tol = 3e-3_f32;
    for (a, b) in train_last.iter().zip(step_last.iter()) {
        assert!(
            (a - b).abs() < tol,
            "train/step mismatch: {a} vs {b} (diff {})",
            (a - b).abs()
        );
    }
    Ok(())
}

#[test]
fn test_state_is_fixed_size() -> anyhow::Result<()> {
    let config = tiny_config();
    let device = Device::Cpu;
    let (model, _) = nano_model()?;

    let mut state = InferenceState::zeros(&config, &device)?;
    let shapes_before: Vec<Vec<usize>> = state
        .layers
        .iter()
        .map(|l| l.wkv_state.dims().to_vec())
        .collect();

    // Run 100 steps
    for step in 0..100 {
        let id = (step % config.vocab_size) as u32;
        let tid = Tensor::from_vec(vec![id], (1, 1), &device)?.to_dtype(DType::U32)?;
        model.forward_step(&tid, &mut state)?;
    }

    let shapes_after: Vec<Vec<usize>> = state
        .layers
        .iter()
        .map(|l| l.wkv_state.dims().to_vec())
        .collect();

    assert_eq!(shapes_before, shapes_after, "state shape changed — not O(1)!");
    Ok(())
}

#[test]
fn test_all_config_presets_build() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let configs = [
        // Shipped lineup: micro 50M / small 100M / medium 150M.
        ("micro",  ModelConfig::micro(),  30_000_000usize, 80_000_000),
        ("small",  ModelConfig::small(),  80_000_000,      130_000_000),
        ("medium", ModelConfig::medium(), 120_000_000,     200_000_000),
    ];
    for (name, cfg, lo, hi) in configs {
        let vmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&vmap, DType::F32, &device);
        let model = TinyBit::new(cfg.clone(), vb)?;
        let p = model.num_parameters();
        assert!(p >= lo && p <= hi, "{name}: param_count {p} not in [{lo}, {hi}]");
    }
    Ok(())
}
