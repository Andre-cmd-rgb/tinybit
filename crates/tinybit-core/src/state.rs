use crate::config::ModelConfig;
use anyhow::Context;
use candle_core::{Device, Tensor};
use std::path::Path;

/// Per-layer inference state for RWKV-7.
#[derive(Debug, Clone)]
pub struct LayerState {
    /// The recurrent state matrix W. Shape: (num_heads, head_dim, head_dim).
    pub wkv_state: Tensor,
    /// Shift state for time-mix (previous token's embedding). Shape: (d_model,).
    pub time_shift: Tensor,
    /// Shift state for channel-mix. Shape: (d_model,).
    pub ffn_shift: Tensor,
    /// Hebbian fast-weight trace ΔW for the time-mix value path, shape
    /// (d_model, d_model). `None` unless `fast_weights` is enabled. Updated
    /// online during inference (no gradients); this is what "rewires itself".
    pub fast_w: Option<Tensor>,
}

/// Complete model inference state (one per active session).
#[derive(Debug, Clone)]
pub struct InferenceState {
    pub layers: Vec<LayerState>,
    pub device: Device,
}

impl InferenceState {
    /// Allocate zeroed state for the given config on the given device.
    pub fn zeros(config: &ModelConfig, device: &Device) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let wkv_state = Tensor::zeros(
                (config.num_heads, config.head_dim, config.head_dim),
                candle_core::DType::F32,
                device,
            )?;
            let time_shift = Tensor::zeros(config.d_model, candle_core::DType::F32, device)?;
            let ffn_shift = Tensor::zeros(config.d_model, candle_core::DType::F32, device)?;
            let fast_w = if config.fast_weights {
                Some(Tensor::zeros(
                    (config.d_model, config.d_model),
                    candle_core::DType::F32,
                    device,
                )?)
            } else {
                None
            };
            layers.push(LayerState { wkv_state, time_shift, ffn_shift, fast_w });
        }
        Ok(Self { layers, device: device.clone() })
    }

    /// Save state to disk (for session persistence).
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut map: std::collections::HashMap<String, candle_core::Tensor> =
            std::collections::HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            map.insert(format!("layer_{i}_wkv"), layer.wkv_state.clone());
            map.insert(format!("layer_{i}_time"), layer.time_shift.clone());
            map.insert(format!("layer_{i}_ffn"), layer.ffn_shift.clone());
            if let Some(fw) = &layer.fast_w {
                map.insert(format!("layer_{i}_fast"), fw.clone());
            }
        }
        candle_core::safetensors::save(&map, path)?;
        Ok(())
    }

    /// Load state from disk. Layers are discovered by probing `layer_{i}_wkv`
    /// keys (not by a fixed tensor-per-layer count), so state files written with
    /// or without the optional fast-weight trace both load correctly.
    pub fn load(path: &Path, device: &Device) -> anyhow::Result<Self> {
        let loaded = candle_core::safetensors::load(path, device)?;
        let mut layers = Vec::new();
        let mut i = 0;
        while let Some(wkv_state) = loaded.get(&format!("layer_{i}_wkv")) {
            let time_shift = loaded
                .get(&format!("layer_{i}_time"))
                .with_context(|| format!("missing layer_{i}_time"))?
                .clone();
            let ffn_shift = loaded
                .get(&format!("layer_{i}_ffn"))
                .with_context(|| format!("missing layer_{i}_ffn"))?
                .clone();
            let fast_w = loaded.get(&format!("layer_{i}_fast")).cloned();
            layers.push(LayerState {
                wkv_state: wkv_state.clone(),
                time_shift,
                ffn_shift,
                fast_w,
            });
            i += 1;
        }
        Ok(Self { layers, device: device.clone() })
    }

    /// Clone the state (used for speculative decoding rollback).
    pub fn detach_clone(&self) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(LayerState {
                wkv_state: layer.wkv_state.detach(),
                time_shift: layer.time_shift.detach(),
                ffn_shift: layer.ffn_shift.detach(),
                fast_w: layer.fast_w.as_ref().map(|t| t.detach()),
            });
        }
        Ok(Self { layers, device: self.device.clone() })
    }
}
