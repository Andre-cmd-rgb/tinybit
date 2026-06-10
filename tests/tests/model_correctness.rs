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

/// Max-abs-diff between two equal-shape tensors.
fn max_abs_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch");
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max))
}

/// Assert every LayerState tensor of two InferenceStates matches within tol.
fn assert_states_match(a: &InferenceState, b: &InferenceState, tol: f32) -> anyhow::Result<()> {
    assert_eq!(a.layers.len(), b.layers.len());
    for (i, (la, lb)) in a.layers.iter().zip(b.layers.iter()).enumerate() {
        for (name, ta, tb) in [
            ("wkv_state", &la.wkv_state, &lb.wkv_state),
            ("time_shift", &la.time_shift, &lb.time_shift),
            ("ffn_shift", &la.ffn_shift, &lb.ffn_shift),
        ] {
            let d = max_abs_diff(ta, tb)?;
            assert!(d < tol, "layer {i} {name} diverged: max diff {d}");
        }
    }
    Ok(())
}

/// Run `forward_step` over each id in turn, returning the last logits.
fn step_all(
    model: &TinyBit,
    ids: &[u32],
    state: &mut InferenceState,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mut logits = None;
    for &id in ids {
        let tid = Tensor::from_vec(vec![id], (1, 1), device)?.to_dtype(DType::U32)?;
        logits = Some(model.forward_step(&tid, state)?);
    }
    logits.ok_or_else(|| anyhow::anyhow!("empty ids"))
}

/// The chunked sequence prefill must leave the recurrent state — every
/// wkv_state / time_shift / ffn_shift tensor — and the final logits exactly
/// where token-by-token `forward_step` leaves them. T=150 crosses the
/// PREFILL_CHUNK=128 boundary mid-sequence.
#[test]
fn test_prefill_matches_step() -> anyhow::Result<()> {
    let (model, config) = nano_model()?;
    let device = Device::Cpu;
    let t = 150usize;
    let ids: Vec<u32> = (0..t)
        .map(|i| ((i * 2654435761) % config.vocab_size) as u32)
        .collect();

    let mut step_state = InferenceState::zeros(&config, &device)?;
    let step_logits = step_all(&model, &ids, &mut step_state, &device)?;

    let mut prefill_state = InferenceState::zeros(&config, &device)?;
    let ids_t = Tensor::from_vec(ids.clone(), (1, t), &device)?.to_dtype(DType::U32)?;
    let prefill_logits = model.forward_prefill(&ids_t, &mut prefill_state)?;

    let tol = 1e-4_f32;
    let d = max_abs_diff(&step_logits, &prefill_logits)?;
    assert!(d < tol, "prefill/step logits diverged: max diff {d}");
    assert_states_match(&step_state, &prefill_state, tol)?;
    Ok(())
}

/// Edge cases: T=1 (shift comes purely from state) and T=PREFILL_CHUNK (exact
/// chunk boundary), both seeded from a NON-zero state (run a few tokens first)
/// so the state-carry path is exercised, not just the zero init.
#[test]
fn test_prefill_edge_lengths_match_step() -> anyhow::Result<()> {
    let (model, config) = nano_model()?;
    let device = Device::Cpu;
    let warmup: Vec<u32> = vec![5, 9, 13];

    for t in [1usize, tinybit_core::model::PREFILL_CHUNK] {
        let ids: Vec<u32> = (0..t)
            .map(|i| ((i * 48271 + 7) % config.vocab_size) as u32)
            .collect();

        let mut step_state = InferenceState::zeros(&config, &device)?;
        step_all(&model, &warmup, &mut step_state, &device)?;
        let step_logits = step_all(&model, &ids, &mut step_state, &device)?;

        let mut prefill_state = InferenceState::zeros(&config, &device)?;
        step_all(&model, &warmup, &mut prefill_state, &device)?;
        let ids_t = Tensor::from_vec(ids.clone(), (1, t), &device)?.to_dtype(DType::U32)?;
        let prefill_logits = model.forward_prefill(&ids_t, &mut prefill_state)?;

        let tol = 1e-4_f32;
        let d = max_abs_diff(&step_logits, &prefill_logits)?;
        assert!(d < tol, "T={t}: prefill/step logits diverged: max diff {d}");
        assert_states_match(&step_state, &prefill_state, tol)?;
    }
    Ok(())
}

/// End-to-end guarantee: greedy decoding after a sequence prefill produces the
/// SAME tokens as greedy decoding after token-by-token prefill — chat output
/// is unchanged by the speedup.
#[test]
fn test_prefill_then_decode_matches() -> anyhow::Result<()> {
    let (model, config) = nano_model()?;
    let device = Device::Cpu;
    let t = 140usize; // crosses the chunk boundary
    let prompt: Vec<u32> = (0..t)
        .map(|i| ((i * 69621 + 3) % config.vocab_size) as u32)
        .collect();

    let greedy_decode = |state: &mut InferenceState, first: u32| -> anyhow::Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut prev = first;
        for _ in 0..8 {
            let tid = Tensor::from_vec(vec![prev], (1, 1), &device)?.to_dtype(DType::U32)?;
            let logits = model.forward_step(&tid, state)?;
            let v = logits.flatten_all()?.to_vec1::<f32>()?;
            let best = v
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            out.push(best);
            prev = best;
        }
        Ok(out)
    };

    // Token-by-token: prefill head, decode from the last prompt token.
    let (last, head) = prompt.split_last().expect("non-empty");
    let mut step_state = InferenceState::zeros(&config, &device)?;
    step_all(&model, head, &mut step_state, &device)?;
    let step_tokens = greedy_decode(&mut step_state, *last)?;

    // Sequence prefill of the head, then the same greedy decode.
    let mut prefill_state = InferenceState::zeros(&config, &device)?;
    let head_t = Tensor::from_vec(head.to_vec(), (1, head.len()), &device)?.to_dtype(DType::U32)?;
    model.forward_prefill(&head_t, &mut prefill_state)?;
    let prefill_tokens = greedy_decode(&mut prefill_state, *last)?;

    assert_eq!(step_tokens, prefill_tokens, "greedy continuation diverged after prefill");
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
        // Shipped lineup: micro 50M / bit 100M / qbit 150M.
        ("micro", ModelConfig::micro(), 30_000_000usize, 80_000_000),
        ("bit",   ModelConfig::bit(),   80_000_000,      130_000_000),
        ("qbit",  ModelConfig::qbit(),  120_000_000,     200_000_000),
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
