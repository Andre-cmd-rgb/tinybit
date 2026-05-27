use tinybit_core::{config::ModelConfig, model::TinyBit};
use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub step:        usize,
    pub train_loss:  f64,
    pub val_loss:    f64,
    pub tokens_seen: usize,
    pub config:      ModelConfig,
    pub timestamp:   String,
}

pub fn save_checkpoint(
    varmap: &VarMap,
    meta: &CheckpointMeta,
    dir: &Path,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let weights_path = dir.join(format!("step_{:07}.safetensors", meta.step));
    varmap.save(&weights_path)?;
    let meta_path = dir.join(format!("step_{:07}.json", meta.step));
    std::fs::write(meta_path, serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

pub fn load_checkpoint(
    dir: &Path,
    device: &Device,
) -> anyhow::Result<(TinyBit, CheckpointMeta, VarMap)> {
    // Find latest checkpoint by step number
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .collect();
    entries.sort_by_key(|e| e.path());
    let meta_path = entries
        .last()
        .ok_or_else(|| anyhow::anyhow!("no checkpoints in {}", dir.display()))?
        .path();
    let meta: CheckpointMeta = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;

    let weights_path = meta_path.with_extension("safetensors");
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = TinyBit::new(meta.config.clone(), vb)?;
    varmap.load(&weights_path)?;
    Ok((model, meta, varmap))
}

/// Keep only best `keep_best` and latest `keep_recent` checkpoints.
pub fn prune_checkpoints(dir: &Path, keep_best: usize, keep_recent: usize) -> anyhow::Result<()> {
    let mut metas: Vec<(std::path::PathBuf, CheckpointMeta)> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| {
            let p = e.path();
            let meta: CheckpointMeta = serde_json::from_str(&std::fs::read_to_string(&p).ok()?).ok()?;
            Some((p, meta))
        })
        .collect();

    metas.sort_by_key(|(_, m)| m.step);

    let n = metas.len();
    if n <= keep_best + keep_recent {
        return Ok(());
    }

    // Best by val_loss
    let mut by_loss = metas.clone();
    by_loss.sort_by(|(_, a), (_, b)| a.val_loss.partial_cmp(&b.val_loss).unwrap_or(std::cmp::Ordering::Equal));
    let keep_paths: std::collections::HashSet<std::path::PathBuf> = by_loss
        .iter()
        .take(keep_best)
        .chain(metas.iter().rev().take(keep_recent))
        .map(|(p, _)| p.clone())
        .collect();

    for (meta_path, _) in &metas {
        if !keep_paths.contains(meta_path) {
            let _ = std::fs::remove_file(meta_path);
            let weights = meta_path.with_extension("safetensors");
            let _ = std::fs::remove_file(weights);
        }
    }
    Ok(())
}
