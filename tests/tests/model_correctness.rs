use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tinybit_core::{config::ModelConfig, model::TinyBit, state::InferenceState};

fn nano_model() -> anyhow::Result<(TinyBit, ModelConfig)> {
    let config = ModelConfig::nano();
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
    let tol = 1e-3_f32;
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
    let config = ModelConfig::nano();
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
        // Upper bounds include spec_heads (each = d_model × vocab_size params)
        ("nano",  ModelConfig::nano(),  5_000_000usize,  30_000_000),
        ("micro", ModelConfig::micro(), 30_000_000,     120_000_000),
        ("small", ModelConfig::small(), 100_000_000,    280_000_000),
        ("base",  ModelConfig::base(),  250_000_000,    700_000_000),
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
