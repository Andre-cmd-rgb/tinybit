use crate::{
    checkpoint::{load_checkpoint, save_checkpoint, CheckpointMeta},
    data::{DataLoader, TokenDataset},
    loss::cross_entropy_loss,
    scheduler::WsdScheduler,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use indicatif::{ProgressBar, ProgressStyle};
use tinybit_core::{config::ModelConfig, model::TinyBit};
use tracing::{info, warn};

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
        Self {
            config,
            model_config,
            resume,
        }
    }

    fn auto_device() -> anyhow::Result<Device> {
        if candle_core::utils::cuda_is_available() {
            return Ok(Device::new_cuda(0)?);
        }
        Ok(Device::Cpu)
    }

    pub fn run(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.config.batch_size > 0, "batch_size must be > 0");
        anyhow::ensure!(self.config.grad_accum > 0, "grad_accum must be > 0");
        anyhow::ensure!(self.config.eval_every > 0, "eval_every must be > 0");
        anyhow::ensure!(self.config.save_every > 0, "save_every must be > 0");

        let device = Self::auto_device()?;
        info!("training device: {device:?}");

        let (model, varmap, mut step, mut tokens_seen) =
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
        let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

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

        let mut grad_accum_count = 0usize;
        let mut accum_loss = 0.0f64;
        let mut accum_loss_tensor: Option<Tensor> = None;
        let mut last_val_loss = f64::INFINITY;

        while step < total_steps {
            if let Some((inputs, targets)) = train_loader.next_batch()? {
                let b = inputs.len();
                let t = inputs[0].len();
                // Stack to (B, T)
                let input_flat: Vec<u32> = inputs.into_iter().flatten().collect();
                let target_flat: Vec<u32> = targets.into_iter().flatten().collect();
                let input_t =
                    Tensor::from_vec(input_flat, (b, t), &device)?.to_dtype(DType::U32)?;
                let target_t =
                    Tensor::from_vec(target_flat, (b, t), &device)?.to_dtype(DType::U32)?;

                let (logits, _) = model.forward_train(&input_t)?;
                let loss = cross_entropy_loss(&logits, &target_t)?;
                let loss_val = loss.to_scalar::<f32>()? as f64;
                accum_loss += loss_val;
                let scaled_loss = (loss / self.config.grad_accum as f64)?;
                accum_loss_tensor = Some(match accum_loss_tensor.take() {
                    Some(accum) => (accum + scaled_loss)?,
                    None => scaled_loss,
                });
                tokens_seen += b * t;
                grad_accum_count += 1;

                if grad_accum_count >= self.config.grad_accum {
                    let lr = scheduler.get_lr(step);
                    let train_loss = accum_loss / grad_accum_count as f64;
                    optimizer.set_learning_rate(lr);
                    if let Some(loss) = accum_loss_tensor.take() {
                        optimizer.backward_step(&loss)?;
                    }
                    info!(
                        "step={step} lr={lr:.2e} loss={:.4}",
                        accum_loss / grad_accum_count as f64
                    );
                    pb.set_message(format!("{:.4}", accum_loss / grad_accum_count as f64));
                    pb.inc(1);
                    grad_accum_count = 0;
                    accum_loss = 0.0;
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
                    }
                }
            } else {
                train_loader.reset();
            }
        }
        pb.finish();
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
                let input_t = Tensor::from_vec(input_flat, (b, t), device)?.to_dtype(DType::U32)?;
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
        Ok(if count == 0 {
            f64::INFINITY
        } else {
            total / count as f64
        })
    }
}
