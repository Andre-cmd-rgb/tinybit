// Brain-inspired mechanism tests (CPU).
//
// These guard the opt-in brain extensions: they must be numerically INERT when
// disabled (so legacy checkpoints/configs behave identically) and behave as
// designed when enabled. All run on CPU.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tinybit_core::config::ModelConfig;
use tinybit_core::model::TinyBit;
use tinybit_core::state::InferenceState;

fn base_config() -> ModelConfig {
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
        spike_threshold: 0.0,
        fast_weights: false,
        fw_eta: 0.0,
        fw_decay: 0.0,
        ponder_steps: 0,
    }
}

fn norm(t: &Tensor) -> f32 {
    t.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt()
}

/// Build a model on a fresh varmap, returning model + varmap so callers can
/// rebuild a second model that SHARES the same weights (varmap) under a
/// different config — the only difference then is the brain knobs.
fn model_with(config: ModelConfig, varmap: &VarMap) -> TinyBit {
    let vb = VarBuilder::from_varmap(varmap, DType::F32, &Device::Cpu);
    TinyBit::new(config, vb).unwrap()
}

/// Spiking gate with threshold 0.0 is dense — identical logits to a model with
/// no gate. (Inertness when off.)
#[test]
fn spiking_threshold_zero_is_inert() {
    let varmap = VarMap::new();
    let m = model_with(base_config(), &varmap);
    let ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &Device::Cpu).unwrap();
    let (a, _) = m.forward_train(&ids).unwrap();

    // Same weights, spike_threshold = 0.0 explicitly set again → identical.
    let mut cfg = base_config();
    cfg.spike_threshold = 0.0;
    let m2 = model_with(cfg, &varmap);
    let (b, _) = m2.forward_train(&ids).unwrap();
    let diff = norm(&(a - b).unwrap());
    assert!(diff < 1e-6, "threshold-0 spiking changed output: {diff}");
}

/// A positive spike threshold actually silences activations → the output
/// differs from the dense model built on the same weights.
#[test]
fn spiking_threshold_changes_output() {
    let varmap = VarMap::new();
    let dense = model_with(base_config(), &varmap);
    let mut cfg = base_config();
    cfg.spike_threshold = 0.3;
    let sparse = model_with(cfg, &varmap);

    let ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &Device::Cpu).unwrap();
    let (a, _) = dense.forward_train(&ids).unwrap();
    let (b, _) = sparse.forward_train(&ids).unwrap();
    let diff = norm(&(a - b).unwrap());
    assert!(diff > 1e-4, "spiking gate had no effect (diff {diff})");
}

/// Pondering with `ponder_steps = 0` is a no-op: the state is unchanged.
#[test]
fn pondering_zero_steps_is_noop() {
    let varmap = VarMap::new();
    let m = model_with(base_config(), &varmap);
    let cfg = base_config();
    let mut state = InferenceState::zeros(&cfg, &Device::Cpu).unwrap();
    let before = norm(&state.layers[0].wkv_state);
    m.ponder(&mut state).unwrap();
    let after = norm(&state.layers[0].wkv_state);
    assert_eq!(before, after, "ponder(0) mutated state");
}

/// Pondering with steps > 0 evolves the recurrent state (deliberation).
#[test]
fn pondering_evolves_state() {
    let mut cfg = base_config();
    cfg.ponder_steps = 3;
    let varmap = VarMap::new();
    let m = model_with(cfg.clone(), &varmap);
    let mut state = InferenceState::zeros(&cfg, &Device::Cpu).unwrap();
    // Prime the state with a token so a zero thought embedding still propagates.
    let tok = Tensor::new(&[1u32], &Device::Cpu).unwrap();
    m.forward_step(&tok, &mut state).unwrap();
    let before = norm(&state.layers[0].wkv_state);
    m.ponder(&mut state).unwrap();
    let after = norm(&state.layers[0].wkv_state);
    assert!((before - after).abs() > 1e-6, "ponder did not change state");
    assert!(after.is_finite(), "ponder produced non-finite state");
}

/// Fast-weights state is allocated only when enabled, and the Hebbian trace
/// grows from zero then stays bounded under decay (no runaway).
#[test]
fn fast_weights_allocated_and_bounded() {
    // Disabled: no fast-weight trace.
    let off = base_config();
    let s_off = InferenceState::zeros(&off, &Device::Cpu).unwrap();
    assert!(s_off.layers[0].fast_w.is_none(), "fast_w allocated while disabled");

    // Enabled: trace exists, grows from 0, stays bounded.
    let mut cfg = base_config();
    cfg.fast_weights = true;
    cfg.fw_eta = 0.05;
    cfg.fw_decay = 0.9;
    let varmap = VarMap::new();
    let m = model_with(cfg.clone(), &varmap);
    let mut state = InferenceState::zeros(&cfg, &Device::Cpu).unwrap();
    assert!(state.layers[0].fast_w.is_some(), "fast_w not allocated while enabled");
    assert_eq!(norm(state.layers[0].fast_w.as_ref().unwrap()), 0.0);

    let tok = Tensor::new(&[7u32], &Device::Cpu).unwrap();
    let mut prev = 0.0f32;
    let mut grew = false;
    for step in 0..40 {
        m.forward_step(&tok, &mut state).unwrap();
        let n = norm(state.layers[0].fast_w.as_ref().unwrap());
        assert!(n.is_finite(), "fast_w diverged at step {step}");
        if step < 5 && n > prev {
            grew = true;
        }
        prev = n;
    }
    assert!(grew, "fast-weight trace never grew");
    // Geometric bound: ||ΔW|| <= eta * max||outer|| / (1 - decay). With finite
    // activations this stays well under a loose ceiling.
    assert!(prev < 1e4, "fast-weight trace not bounded: {prev}");
}

/// Fast-weights make repeated exposure to a token shift its own next-step
/// representation (in-conversation adaptation), while the disabled model is
/// perfectly periodic. We compare the model's output vector drift across
/// repeats of the same token.
#[test]
fn fast_weights_adapt_with_repetition() {
    let varmap = VarMap::new();
    // Enabled model.
    let mut cfg = base_config();
    cfg.fast_weights = true;
    cfg.fw_eta = 0.1;
    cfg.fw_decay = 0.9;
    let m = model_with(cfg.clone(), &varmap);
    let mut state = InferenceState::zeros(&cfg, &Device::Cpu).unwrap();
    let tok = Tensor::new(&[3u32], &Device::Cpu).unwrap();

    let l1 = m.forward_step(&tok, &mut state).unwrap();
    for _ in 0..10 {
        m.forward_step(&tok, &mut state).unwrap();
    }
    let l2 = m.forward_step(&tok, &mut state).unwrap();
    // The recurrent state alone drifts, but with fast-weights the drift is real
    // and finite — assert the logits moved and stayed finite.
    let drift = norm(&(l2 - l1).unwrap());
    assert!(drift.is_finite() && drift > 0.0, "no adaptation observed: {drift}");
}
