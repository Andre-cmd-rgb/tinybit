use clap::Args;
use tinybit_core::config::ModelConfig;
use tinybit_train::trainer::{Trainer, TrainingConfig};

#[derive(Args)]
pub struct TrainArgs {
    #[arg(long, default_value = "configs/small.toml")]
    pub model_config: std::path::PathBuf,

    #[arg(long, default_value = "configs/train.toml")]
    pub train_config: std::path::PathBuf,

    /// Run only smoke_test_steps steps
    #[arg(long)]
    pub smoke_test: bool,

    /// Resume from latest checkpoint
    #[arg(long)]
    pub resume: bool,
}

pub fn run(args: TrainArgs) -> anyhow::Result<()> {
    let model_config = ModelConfig::from_file(&args.model_config)?;
    let mut train_config = TrainingConfig::from_file(&args.train_config)?;

    if args.smoke_test && train_config.smoke_test_steps == 0 {
        train_config.smoke_test_steps = 200;
    }

    let trainer = Trainer::new(train_config, model_config, args.resume);
    trainer.run()
}
