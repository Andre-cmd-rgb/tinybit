//! Dream consolidation ("dreams") — offline replay + pseudo-rehearsal.
//!
//! Biological sleep consolidates the day's experience into long-term memory while
//! replaying it to avoid overwriting what was already learned. This module does
//! the same for tinybit: it replays recent conversation tokens (CONSOLIDATE,
//! cross-entropy) while distilling toward a FROZEN copy of the base model on a
//! pseudo-rehearsal set (ANTI-FORGETTING, KL) so consolidation doesn't wreck
//! general ability. It reuses the project's loss + optimizer machinery.
//!
//! It is deliberately small and CPU-friendly: a handful of gradient steps at a
//! low learning rate, producing a consolidated checkpoint. (LoRA adapters that
//! keep the update reversible are a future optimization; this consolidates into a
//! full checkpoint, which is simple and robust.)

use crate::loss::cross_entropy_loss;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::path::Path;
use tinybit_core::{config::ModelConfig, model::TinyBit};

/// Knobs for a dream/consolidation run. None are architecture — they live here,
/// not in `ModelConfig`.
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Number of gradient steps.
    pub steps: usize,
    /// Learning rate (kept small — consolidation, not training from scratch).
    pub lr: f64,
    /// Weight on the KL-to-frozen-base anti-forgetting term.
    pub kl_weight: f64,
    /// Max tokens per replay/anchor sequence (longer sequences are chunked).
    pub seq_len: usize,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self { steps: 40, lr: 5e-5, kl_weight: 1.0, seq_len: 256 }
    }
}

/// What a dream run did, for the CLI to report.
#[derive(Debug, Clone)]
pub struct DreamReport {
    pub steps: usize,
    pub first_loss: f64,
    pub last_loss: f64,
    pub first_ce: f64,
    pub last_ce: f64,
}

/// KL(teacher ‖ student) averaged over all positions. `teacher_logits` must be
/// detached (the frozen base provides the target distribution).
fn kl_to_teacher(student_logits: &Tensor, teacher_logits: &Tensor) -> anyhow::Result<Tensor> {
    let (b, t, v) = student_logits.dims3()?;
    let s = candle_nn::ops::log_softmax(&student_logits.reshape((b * t, v))?.to_dtype(DType::F32)?, 1)?;
    let tt = candle_nn::ops::log_softmax(&teacher_logits.reshape((b * t, v))?.to_dtype(DType::F32)?, 1)?;
    let pt = tt.exp()?;
    // sum_v p_t * (log p_t - log p_s), then mean over rows.
    let kl = (pt * (tt - s)?)?.sum(1)?.mean_all()?;
    Ok(kl)
}

/// Build a (input, target) next-token pair from a token sequence, truncated to
/// `seq_len`. Returns `None` for sequences too short to form a pair.
fn pair(seq: &[u32], seq_len: usize, device: &Device) -> anyhow::Result<Option<(Tensor, Tensor)>> {
    if seq.len() < 2 {
        return Ok(None);
    }
    let end = seq.len().min(seq_len + 1);
    let inp = &seq[..end - 1];
    let tgt = &seq[1..end];
    let t = inp.len();
    let input = Tensor::from_vec(inp.to_vec(), (1, t), device)?.to_dtype(DType::U32)?;
    let target = Tensor::from_vec(tgt.to_vec(), (1, t), device)?.to_dtype(DType::U32)?;
    Ok(Some((input, target)))
}

/// Consolidate `replay_seqs` (recent experience, cross-entropy) into the model
/// while anchoring to the frozen base on `anchor_seqs` (pseudo-rehearsal, KL).
/// Writes the consolidated weights to `out_path`. Token-level so it has no
/// dependency on the inference/session crate.
pub fn consolidate(
    model_path: &Path,
    config: ModelConfig,
    device: &Device,
    replay_seqs: &[Vec<u32>],
    anchor_seqs: &[Vec<u32>],
    out_path: &Path,
    cfg: &DreamConfig,
) -> anyhow::Result<DreamReport> {
    anyhow::ensure!(!replay_seqs.is_empty(), "dream: no replay sequences (need saved sessions to consolidate)");
    // Frozen teacher (read-only) for the anti-forgetting KL term.
    let teacher = TinyBit::load(model_path, config.clone(), device)?;
    // Trainable student: same weights loaded into a VarMap so gradients flow.
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let student = TinyBit::new(config.clone(), vb)?;
    varmap
        .load(model_path)
        .map_err(|e| anyhow::anyhow!("dream needs an f32 checkpoint matching the config ({e})"))?;

    // Anchor on the replay tokens themselves if no separate pseudo-rehearsal set
    // was supplied (KL-to-base on conversation tokens still resists drift).
    let anchors: &[Vec<u32>] = if anchor_seqs.is_empty() { replay_seqs } else { anchor_seqs };

    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: cfg.lr, ..Default::default() })?;

    let mut report = DreamReport {
        steps: cfg.steps,
        first_loss: 0.0,
        last_loss: 0.0,
        first_ce: 0.0,
        last_ce: 0.0,
    };

    for step in 0..cfg.steps {
        let replay = &replay_seqs[step % replay_seqs.len()];
        let (input, target) = match pair(replay, cfg.seq_len, device)? {
            Some(p) => p,
            None => continue,
        };
        let (logits, _) = student.forward_train(&input)?;
        let ce = cross_entropy_loss(&logits, &target)?;

        // Anti-forgetting KL on the anchor distribution.
        let anchor = &anchors[step % anchors.len()];
        let total = if let Some((a_in, _)) = pair(anchor, cfg.seq_len, device)? {
            let (s_logits, _) = student.forward_train(&a_in)?;
            let (t_logits, _) = teacher.forward_train(&a_in)?;
            let kl = kl_to_teacher(&s_logits, &t_logits.detach())?;
            (ce.clone() + (kl * cfg.kl_weight)?)?
        } else {
            ce.clone()
        };

        let loss_val = total.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        let ce_val = ce.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        opt.backward_step(&total)?;

        if step == 0 {
            report.first_loss = loss_val;
            report.first_ce = ce_val;
        }
        report.last_loss = loss_val;
        report.last_ce = ce_val;
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    varmap.save(out_path)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 64,
            num_layers: 2,
            d_model: 32,
            d_ffn: 64,
            num_heads: 2,
            head_dim: 16,
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

    /// A dream run reduces cross-entropy on the replayed sequence (consolidation
    /// works) and writes a loadable checkpoint.
    #[test]
    fn dream_consolidates_replayed_sequence() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config();

        // Build and save a random base checkpoint.
        let tmp = std::env::temp_dir().join(format!("tinybit_dream_{}", std::process::id()));
        std::fs::create_dir_all(&tmp)?;
        let base_path = tmp.join("base.safetensors");
        {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let _m = TinyBit::new(cfg.clone(), vb)?;
            varmap.save(&base_path)?;
        }

        // Replay a fixed, learnable pattern.
        let replay: Vec<Vec<u32>> = vec![(0..32u32).map(|i| 1 + (i % 5)).collect()];
        let out = tmp.join("dreamed.safetensors");
        let dcfg = DreamConfig { steps: 20, lr: 1e-3, kl_weight: 0.5, seq_len: 32 };

        let report = consolidate(&base_path, cfg.clone(), &device, &replay, &[], &out, &dcfg)?;
        assert!(out.exists(), "dream did not write a checkpoint");
        assert!(
            report.last_ce <= report.first_ce + 1e-4,
            "consolidation did not reduce CE: {} -> {}",
            report.first_ce,
            report.last_ce
        );

        // The consolidated checkpoint loads back into the same architecture.
        let _reloaded = TinyBit::load(&out, cfg, &device)?;
        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }
}
