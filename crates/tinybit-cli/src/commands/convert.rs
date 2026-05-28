use clap::{Args, ValueEnum};
use std::collections::HashMap;
use candle_core::{Device, Tensor};
use tinybit_core::quantize::{quantize_pack_2d, QUANT_MARKER};

#[derive(Args)]
pub struct ConvertArgs {
    #[arg(long)]
    pub input: std::path::PathBuf,

    #[arg(long, value_enum, default_value = "safetensors")]
    pub format: ExportFormat,

    #[arg(long)]
    pub output: std::path::PathBuf,

    /// Quantize 2D weight matrices to packed ternary before export.
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
    let device = Device::Cpu;

    match args.format {
        ExportFormat::Safetensors => {
            let raw = candle_core::safetensors::load(&args.input, &device)?;
            let out = if args.quantize {
                build_quantized(&raw, &device)?
            } else {
                raw
            };
            candle_core::safetensors::save(&out, &args.output)?;

            let in_sz = std::fs::metadata(&args.input)?.len();
            let out_sz = std::fs::metadata(&args.output)?.len();
            println!(
                "wrote {} — {:.1} MiB (input {:.1} MiB, {:.2}x smaller)",
                args.output.display(),
                mib(out_sz),
                mib(in_sz),
                in_sz as f64 / out_sz.max(1) as f64,
            );
        }
        ExportFormat::Gguf => anyhow::bail!("GGUF export not yet implemented"),
    }
    Ok(())
}

/// Quantize every 2D weight matrix (except the full-precision tied embedding /
/// LM-head table) to packed ternary, leaving norms, biases, and the embedding
/// untouched. Emits `<name>.qweight` (packed u8) plus `<name>.qscale/.qrows/
/// .qcols` sidecars and a marker tensor so `TinyBit::load` can rebuild it.
fn build_quantized(
    raw: &HashMap<String, Tensor>,
    device: &Device,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let mut out: HashMap<String, Tensor> = HashMap::new();
    let mut quantized = 0usize;

    for (name, t) in raw {
        let is_matrix = t.dims().len() == 2;
        let is_embed = name == "embed.embed.weight";
        if is_matrix && !is_embed {
            let (packed, scale, rows, cols) = quantize_pack_2d(t)?;
            let n = packed.len();
            out.insert(format!("{name}.qweight"), Tensor::from_vec(packed, (n,), device)?);
            out.insert(format!("{name}.qscale"), Tensor::from_vec(vec![scale], (1,), device)?);
            out.insert(format!("{name}.qrows"), Tensor::from_vec(vec![rows as i64], (1,), device)?);
            out.insert(format!("{name}.qcols"), Tensor::from_vec(vec![cols as i64], (1,), device)?);
            quantized += 1;
        } else {
            out.insert(name.clone(), t.clone());
        }
    }
    out.insert(QUANT_MARKER.to_string(), Tensor::from_vec(vec![1i64], (1,), device)?);

    println!("quantized {quantized} matrices to packed ternary (2 weights/byte)");
    Ok(out)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
