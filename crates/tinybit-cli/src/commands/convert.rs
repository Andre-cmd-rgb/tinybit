use clap::{Args, ValueEnum};
use tinybit_core::config::ModelConfig;
use tinybit_core::model::TinyBit;
use candle_core::Device;

#[derive(Args)]
pub struct ConvertArgs {
    #[arg(long)]
    pub input: std::path::PathBuf,

    #[arg(long, value_enum, default_value = "safetensors")]
    pub format: ExportFormat,

    #[arg(long)]
    pub output: std::path::PathBuf,

    #[arg(long)]
    pub config: std::path::PathBuf,

    /// Quantize to ternary + INT8 before export
    #[arg(long)]
    pub quantize: bool,
}

#[derive(ValueEnum, Clone)]
pub enum ExportFormat {
    Safetensors,
    /// GGUF format for llama.cpp compatibility
    Gguf,
}

pub fn run(args: ConvertArgs) -> anyhow::Result<()> {
    let config = ModelConfig::from_file(&args.config)?;
    let device = Device::Cpu;
    let mut model = TinyBit::load(&args.input, config, &device)?;

    if args.quantize {
        model.set_quantized(true);
    }

    match args.format {
        ExportFormat::Safetensors => {
            println!("Note: safetensors export requires VarMap — load via trainer for full export.");
            println!("Model loaded with {} parameters.", model.num_parameters());
        }
        ExportFormat::Gguf => {
            anyhow::bail!("GGUF export not yet implemented");
        }
    }
    Ok(())
}
