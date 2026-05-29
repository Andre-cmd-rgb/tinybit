use crate::{
    checkpoint::{load_checkpoint, prune_checkpoints, save_checkpoint, CheckpointMeta},
    data::{DataLoader, TokenDataset},
    loss::cross_entropy_loss,
    optimizer::Muon,
    scheduler::WsdScheduler,
};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};
use tinybit_core::{config::ModelConfig, model::TinyBit};
use tracing::{info, warn};

/// Which optimizer the trainer drives. Defaults to AdamW for everything;
/// `muon` uses Muon for 2D hidden weight matrices and AdamW for the rest
/// (embeddings, norms, biases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerKind {
    #[default]
    Adamw,
    Muon,
}

#[derive(Debug, serde::Deserialize)]
pub struct TrainingConfig {
    pub train_data: std::path::PathBuf,
    pub val_data: std::path::PathBuf,
    pub checkpoint_dir: std::path::PathBuf,

    pub batch_size: usize,
    pub grad_accum: usize,
    pub total_steps: usize,
    pub peak_lr: f64,
    pub weight_decay: f64,
    pub grad_clip: f64,

    pub save_every: usize,
    pub eval_every: usize,
    pub eval_batches: usize,

    pub smoke_test_steps: usize,

    /// Mixed-precision training. When true and the device is CUDA, the block-stack
    /// matmuls run in bf16 (tensor cores) while master weights, the WKV scan, the
    /// norms, and the loss stay f32. Absent/false → full f32 (unchanged behavior).
    #[serde(default)]
    pub bf16: bool,

    /// Optimizer selection. Absent in older configs → AdamW (unchanged behavior).
    #[serde(default)]
    pub optimizer: OptimizerKind,
    /// Peak LR for Muon-updated matrices (orthogonalized updates need a larger
    /// LR than AdamW). Follows the same WSD shape as `peak_lr`. Ignored unless
    /// `optimizer = "muon"`.
    #[serde(default = "default_muon_lr")]
    pub muon_lr: f64,
}

fn default_muon_lr() -> f64 {
    0.02
}

impl TrainingConfig {
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }
}

pub struct Trainer {
    pub config: TrainingConfig,
    pub model_config: ModelConfig,
    pub resume: bool,
}

impl Trainer {
    pub fn new(config: TrainingConfig, model_config: ModelConfig, resume: bool) -> Self {
        Self { config, model_config, resume }
    }

    fn auto_device() -> anyhow::Result<Device> {
        if candle_core::utils::cuda_is_available() {
            let dev = Device::new_cuda(0)?;
            // Pre-flight: verify actual allocation works before committing.
            // CUDA_ERROR_OUT_OF_MEMORY here means MPS/context issue, not real OOM.
            candle_core::Tensor::zeros((1,), candle_core::DType::F32, &dev)
                .map_err(|e| anyhow::anyhow!(
                    "CUDA pre-flight failed on device 0 — \
                     check CUDA_VISIBLE_DEVICES and that MPS is not running: {e}"
                ))?;
            return Ok(dev);
        }
        Ok(Device::Cpu)
    }

    pub fn run(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.config.batch_size > 0, "batch_size must be > 0");
        anyhow::ensure!(self.config.grad_accum > 0, "grad_accum must be > 0");
        anyhow::ensure!(self.config.total_steps > 0, "total_steps must be > 0");
        anyhow::ensure!(self.config.eval_every > 0, "eval_every must be > 0");
        anyhow::ensure!(self.config.save_every > 0, "save_every must be > 0");
        self.model_config.validate()?;

        let device = Self::auto_device()?;
        info!("training device: {device:?}");

        let (mut model, varmap, mut step, mut tokens_seen) =
            if self.resume && self.config.checkpoint_dir.exists() {
                match load_checkpoint(&self.config.checkpoint_dir, &device) {
                    Ok((model, meta, varmap)) => {
                        info!("resumed checkpoint at step={}", meta.step);
                        (model, varmap, meta.step, meta.tokens_seen)
                    }
                    Err(err) => {
                        warn!("could not resume checkpoint, starting from scratch: {err}");
                        let varmap = VarMap::new();
                        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
                        let model = TinyBit::new(self.model_config.clone(), vb)?;
                        (model, varmap, 0, 0)
                    }
                }
            } else {
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
                let model = TinyBit::new(self.model_config.clone(), vb)?;
                (model, varmap, 0, 0)
            };

        if self.config.bf16 {
            if device.is_cuda() {
                model.set_compute_dtype(DType::BF16);
                info!("bf16 mixed precision: ON (compute bf16; master weights, WKV scan, norms, loss stay f32)");
            } else {
                warn!("bf16 requested but training device is CPU — staying f32");
            }
        }

        let train_ds = TokenDataset::open(&self.config.train_data, self.model_config.max_seq_len)?;
        let val_ds = TokenDataset::open(&self.config.val_data, self.model_config.max_seq_len)?;

        let mut train_loader = DataLoader::new(train_ds, self.config.batch_size, true);
        let mut val_loader = DataLoader::new(val_ds, self.config.batch_size, false);

        let scheduler = WsdScheduler::new(self.config.peak_lr, self.config.total_steps);
        let params = ParamsAdamW {
            lr: self.config.peak_lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: self.config.weight_decay,
        };
        let all_vars = varmap.all_vars();

        // Optimizer split. For Muon mode, 2D hidden weight matrices are driven
        // by Muon; the tied embedding/LM-head table, norms, and biases stay on
        // AdamW. For AdamW mode, everything is on AdamW (unchanged behavior).
        let use_muon = self.config.optimizer == OptimizerKind::Muon;
        let mut muon_params: Vec<(String, Var)> = Vec::new();
        let adamw_vars: Vec<Var> = if use_muon {
            let mut adamw = Vec::new();
            let data = varmap.data().lock().expect("varmap mutex poisoned");
            for (name, var) in data.iter() {
                let is_matrix = var.as_tensor().dims().len() == 2;
                let is_embed = name.contains("embed.embed");
                if is_matrix && !is_embed {
                    muon_params.push((name.clone(), var.clone()));
                } else {
                    adamw.push(var.clone());
                }
            }
            adamw
        } else {
            all_vars.clone()
        };

        let mut optimizer = AdamW::new(adamw_vars, params)?;
        let mut muon = if use_muon {
            info!(
                "optimizer=muon: {} matrices on Muon (lr={}), {} tensors on AdamW",
                muon_params.len(),
                self.config.muon_lr,
                all_vars.len() - muon_params.len()
            );
            Some(Muon::new(self.config.muon_lr, 0.95))
        } else {
            None
        };
        let muon_scheduler = WsdScheduler::new(self.config.muon_lr, self.config.total_steps);
        let mut skipped_steps = 0usize;

        let total_steps = if self.config.smoke_test_steps > 0 {
            self.config.smoke_test_steps
        } else {
            self.config.total_steps
        };

        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{elapsed_precise} [{bar:40}] {pos}/{len} loss={msg}")
                .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb.set_position(step as u64);

        let mut last_val_loss = f64::INFINITY;

        // Coarse per-phase profiler (opt-in via TINYBIT_PROFILE=1). Attributes a
        // step's wall time to forward+loss / backward / optimizer, synchronizing
        // the device at phase boundaries so the numbers reflect real GPU time.
        // Off by default → zero overhead on the normal training path.
        let profile = std::env::var("TINYBIT_PROFILE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);

        while step < total_steps {
            // ---- gradient accumulation loop ----------------------------------
            // Each microbatch is backpropped immediately so its computation graph
            // is freed before the next microbatch is allocated. This keeps VRAM
            // usage proportional to one microbatch rather than grad_accum batches.
            let mut merged_grads: Option<GradStore> = None;
            // Running sum of the per-microbatch losses, kept on-device and synced
            // to the host ONCE after the accumulation loop (was one
            // `.to_scalar()` per microbatch). Stored detached so it never retains
            // a microbatch's computation graph — that retention would defeat the
            // per-microbatch graph-freeing this loop relies on for VRAM.
            let mut loss_acc: Option<Tensor> = None;
            let mut actual_microbatches = 0usize;
            let (mut t_fwd, mut t_bwd) = (Duration::ZERO, Duration::ZERO);

            for _ in 0..self.config.grad_accum {
                let batch = train_loader.next_batch()?;
                let (inputs, targets) = match batch {
                    Some(b) => b,
                    None => {
                        train_loader.reset();
                        break;
                    }
                };

                let b = inputs.len();
                let t = inputs[0].len();
                let input_flat: Vec<u32> = inputs.into_iter().flatten().collect();
                let target_flat: Vec<u32> = targets.into_iter().flatten().collect();
                let input_t =
                    Tensor::from_vec(input_flat, (b, t), &device)?.to_dtype(DType::U32)?;
                let target_t =
                    Tensor::from_vec(target_flat, (b, t), &device)?.to_dtype(DType::U32)?;

                let fwd_start = profile.then(Instant::now);
                let (logits, _) = model.forward_train(&input_t)?;
                let loss = cross_entropy_loss(&logits, &target_t)?;

                // Accumulate the (detached) loss on-device; synced once below.
                let detached = loss.detach();
                loss_acc = Some(match loss_acc {
                    Some(acc) => (acc + &detached)?,
                    None => detached,
                });

                // Scale and immediately backprop — frees this graph before next microbatch.
                let scaled = (loss / self.config.grad_accum as f64)?;
                if let Some(s) = fwd_start {
                    device.synchronize()?;
                    t_fwd += s.elapsed();
                }
                let bwd_start = profile.then(Instant::now);
                let step_grads = scaled.backward()?;
                if let Some(s) = bwd_start {
                    device.synchronize()?;
                    t_bwd += s.elapsed();
                }

                // Accumulate into merged_grads.
                match merged_grads {
                    None => merged_grads = Some(step_grads),
                    Some(ref mut merged) => {
                        let mut sg = step_grads;
                        for v in &all_vars {
                            if let Some(g) = sg.remove(v.as_tensor()) {
                                let prev = merged.remove(v.as_tensor());
                                let new_g = match prev {
                                    Some(p) => (p + &g)?,
                                    None => g,
                                };
                                merged.insert(v.as_tensor(), new_g);
                            }
                        }
                    }
                }

                tokens_seen += b * t;
                actual_microbatches += 1;
            }

            if actual_microbatches == 0 {
                continue;
            }

            // If the dataset ended mid-accumulation, each microbatch loss was
            // scaled by 1/grad_accum, but only actual_microbatches contributed.
            // Rescale so the effective divisor is actual_microbatches, not grad_accum.
            if actual_microbatches < self.config.grad_accum {
                let rescale = self.config.grad_accum as f64 / actual_microbatches as f64;
                if let Some(ref mut merged) = merged_grads {
                    for v in &all_vars {
                        if let Some(g) = merged.remove(v.as_tensor()) {
                            merged.insert(v.as_tensor(), (g * rescale)?);
                        }
                    }
                }
            }

            // Single GPU->CPU sync for the whole step's mean loss.
            let train_loss = match loss_acc {
                Some(acc) => acc.to_scalar::<f32>()? as f64 / actual_microbatches as f64,
                None => continue,
            };

            // ---- guard: skip non-finite loss ---------------------------------
            if !train_loss.is_finite() {
                warn!(
                    "step={step} skipping update — non-finite loss ({:.4})",
                    train_loss
                );
                skipped_steps += 1;
                continue;
            }

            let mut grads = match merged_grads {
                Some(g) => g,
                None => continue,
            };

            let lr = scheduler.get_lr(step);
            optimizer.set_learning_rate(lr);

            let opt_start = profile.then(Instant::now);
            let grad_norm = if self.config.grad_clip > 0.0 {
                clip_grad_norm(&mut grads, &all_vars, self.config.grad_clip)?
            } else {
                global_grad_norm(&grads, &all_vars)?
            };

            if !grad_norm.is_finite() {
                warn!(
                    "step={step} skipping update — non-finite grad norm ({grad_norm:.4})"
                );
                skipped_steps += 1;
                continue;
            }

            optimizer.step(&grads)?;
            if let Some(ref mut muon) = muon {
                muon.lr = muon_scheduler.get_lr(step);
                apply_muon(muon, &muon_params, &grads)?;
            }
            if let Some(s) = opt_start {
                device.synchronize()?;
                let t_opt = s.elapsed();
                info!(
                    "profile step={step} fwd={:.0}ms bwd={:.0}ms opt={:.0}ms (sum over {} microbatches)",
                    t_fwd.as_secs_f64() * 1e3,
                    t_bwd.as_secs_f64() * 1e3,
                    t_opt.as_secs_f64() * 1e3,
                    actual_microbatches,
                );
            }

            info!(
                "step={step} lr={lr:.2e} loss={:.4} gnorm={:.3}",
                train_loss, grad_norm
            );
            pb.set_message(format!("loss={:.4} gnorm={:.3}", train_loss, grad_norm));
            pb.inc(1);
            step += 1;

            if step % self.config.eval_every == 0 {
                let val_loss = self.eval_loss(&model, &mut val_loader, &device)?;
                last_val_loss = val_loss;
                info!("step={step} val_loss={val_loss:.4}");
            }

            if step % self.config.save_every == 0 || step == total_steps {
                let meta = CheckpointMeta {
                    step,
                    train_loss,
                    val_loss: last_val_loss,
                    tokens_seen,
                    config: self.model_config.clone(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };
                save_checkpoint(&varmap, &meta, &self.config.checkpoint_dir)?;
                if let Err(e) = prune_checkpoints(
                    &self.config.checkpoint_dir,
                    CKPT_KEEP_BEST,
                    CKPT_KEEP_RECENT,
                ) {
                    warn!("checkpoint prune failed: {e}");
                }
            }
        }

        pb.finish();
        if skipped_steps > 0 {
            warn!(
                "training finished with {skipped_steps} skipped step(s) due to non-finite loss/grad"
            );
        }
        Ok(())
    }

    pub fn eval_loss(
        &self,
        model: &TinyBit,
        loader: &mut DataLoader,
        device: &Device,
    ) -> anyhow::Result<f64> {
        loader.reset();
        let mut total = 0.0f64;
        let mut count = 0usize;
        for _ in 0..self.config.eval_batches {
            if let Some((inputs, targets)) = loader.next_batch()? {
                let b = inputs.len();
                let t = inputs[0].len();
                let input_flat: Vec<u32> = inputs.into_iter().flatten().collect();
                let target_flat: Vec<u32> = targets.into_iter().flatten().collect();
                let input_t =
                    Tensor::from_vec(input_flat, (b, t), device)?.to_dtype(DType::U32)?;
                let target_t =
                    Tensor::from_vec(target_flat, (b, t), device)?.to_dtype(DType::U32)?;
                let (logits, _) = model.forward_train(&input_t)?;
                let loss = cross_entropy_loss(&logits, &target_t)?;
                total += loss.to_scalar::<f32>()? as f64;
                count += 1;
            } else {
                break;
            }
        }
        Ok(if count == 0 { f64::INFINITY } else { total / count as f64 })
    }
}

const CKPT_KEEP_BEST: usize = 3;
const CKPT_KEEP_RECENT: usize = 3;

/// Apply a Muon update to each 2D matrix var using its gradient from `grads`.
/// Reads the current weight, runs Muon's Newton-Schulz orthogonalized update,
/// and writes the result back into the `Var` (which carries interior
/// mutability, so this updates the live model weights).
fn apply_muon(
    muon: &mut Muon,
    params: &[(String, Var)],
    grads: &GradStore,
) -> anyhow::Result<()> {
    for (name, var) in params {
        if let Some(g) = grads.get(var.as_tensor()) {
            let mut w = var.as_tensor().clone();
            let g = g.clone();
            {
                let mut slice: [(&str, &mut Tensor, &Tensor); 1] = [(name.as_str(), &mut w, &g)];
                muon.step(&mut slice)?;
            }
            var.set(&w)?;
        }
    }
    Ok(())
}

fn global_grad_norm(grads: &GradStore, vars: &[Var]) -> anyhow::Result<f64> {
    // Accumulate each parameter's f32 sum-of-squares as a 1-element device
    // tensor, concatenate them, and pull the whole vector to the host in a
    // SINGLE GPU->CPU sync. The previous version did one `.to_scalar()` per
    // parameter (~150-200 forced stream syncs per step, each serializing a tiny
    // reduction kernel behind sync latency). The per-parameter f32 square-sums
    // and the f64 host accumulation order are unchanged, so the resulting norm
    // is bit-identical — this is purely a stall-removal. (Grads w.r.t. the f32
    // master weights are f32; the explicit `to_dtype(F32)` is a no-op clone in
    // practice but guards against a bf16 grad squaring/overflowing in bf16.)
    let mut sqs: Vec<Tensor> = Vec::with_capacity(vars.len());
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            sqs.push(g.to_dtype(DType::F32)?.sqr()?.sum_all()?.reshape(1)?);
        }
    }
    if sqs.is_empty() {
        return Ok(0.0);
    }
    let all = Tensor::cat(&sqs, 0)?.to_vec1::<f32>()?;
    let sq_sum: f64 = all.iter().map(|&x| x as f64).sum();
    Ok(sq_sum.sqrt())
}

fn clip_grad_norm(
    grads: &mut GradStore,
    vars: &[Var],
    max_norm: f64,
) -> anyhow::Result<f64> {
    let total_norm = global_grad_norm(grads, vars)?;
    if total_norm.is_finite() && max_norm > 0.0 && total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for v in vars {
            if let Some(g) = grads.remove(v.as_tensor()) {
                let clipped = (g * scale)?;
                grads.insert(v.as_tensor(), clipped);
            }
        }
    }
    Ok(total_norm)
}
