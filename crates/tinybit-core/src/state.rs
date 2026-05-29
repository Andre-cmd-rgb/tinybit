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
            layers.push(LayerState { wkv_state, time_shift, ffn_shift });
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
        }
        candle_core::safetensors::save(&map, path)?;
        Ok(())
    }

    /// Load state from disk.
    pub fn load(path: &Path, device: &Device) -> anyhow::Result<Self> {
        let loaded = candle_core::safetensors::load(path, device)?;
        let num_layers = loaded.len() / 3;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let wkv_state = loaded
                .get(&format!("layer_{i}_wkv"))
                .with_context(|| format!("missing layer_{i}_wkv"))?
                .clone();
            let time_shift = loaded
                .get(&format!("layer_{i}_time"))
                .with_context(|| format!("missing layer_{i}_time"))?
                .clone();
            let ffn_shift = loaded
                .get(&format!("layer_{i}_ffn"))
                .with_context(|| format!("missing layer_{i}_ffn"))?
                .clone();
            layers.push(LayerState { wkv_state, time_shift, ffn_shift });
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
            });
        }
        Ok(Self { layers, device: self.device.clone() })
    }
}
