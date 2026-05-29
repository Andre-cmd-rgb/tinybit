use candle_core::{DType, Tensor};
use clap::Args;
use std::io::IsTerminal;
use tinybit_core::config::ModelConfig;
use tinybit_core::tokenizer::Profile;
use tinybit_infer::engine::InferenceEngine;
use tinybit_infer::session::Session;
use tinybit_train::data::{DataLoader, TokenDataset};
use tinybit_train::loss::cross_entropy_loss;

/// A handful of fixed prompts used for the generation sanity check. They are
/// deliberately simple — the point is to confirm the model produces coherent,
/// on-topic, non-repeating text, not to measure factual accuracy.
const GENERAL_PROMPTS: &[&str] = &[
    "What is the capital of France?",
    "Explain what a prime number is in one sentence.",
    "Write a short to-do list for moving house.",
];
const CODING_PROMPTS: &[&str] = &[
    "Write a Rust function that returns the nth Fibonacci number.",
    "What does the `?` operator do in Rust?",
    "How do I list files by size in a Linux shell?",
];

#[derive(Args)]
pub struct EvalArgs {
    /// Model architecture config (TOML). For coding checkpoints use a *-coding config.
    #[arg(long, default_value = "configs/micro.toml")]
    pub config: std::path::PathBuf,

    /// Model checkpoint (safetensors).
    #[arg(long, default_value = "models/tinybit-micro.safetensors")]
    pub model: std::path::PathBuf,

    /// Tokenizer JSON.
    #[arg(long, default_value = "tokenizer.json")]
    pub tokenizer: std::path::PathBuf,

    /// Validation token file (u32, little-endian — e.g. data/val.bin). When
    /// omitted, the perplexity measurement is skipped.
    #[arg(long)]
    pub data: Option<std::path::PathBuf>,

    /// Data dir for the built-in tools (chat sanity uses them).
    #[arg(long, default_value = "data/")]
    pub data_dir: std::path::PathBuf,

    /// Max number of batches to measure perplexity over.
    #[arg(long, default_value_t = 20)]
    pub max_batches: usize,

    /// Sequences per batch for perplexity.
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Model family for the generation sanity prompts and default system prompt.
    /// Defaults to coding when the config filename contains "coding".
    #[arg(long)]
    pub profile: Option<String>,

    /// Skip the generation sanity check (perplexity only).
    #[arg(long)]
    pub no_generate: bool,

    /// Max tokens to generate per sanity prompt.
    #[arg(long, default_value_t = 64)]
    pub gen_tokens: usize,
}

pub fn run(args: EvalArgs) -> anyhow::Result<()> {
    let config = ModelConfig::from_file(&args.config)?;
    config.validate()?;
    let profile = match &args.profile {
        Some(p) => Profile::parse(p)?,
        None => Profile::from_config_path(&args.config),
    };

    let device = InferenceEngine::auto_device();
    let dev_name = device_name(&device);
    if !args.model.exists() {
        anyhow::bail!(
            "model checkpoint not found: {}\n\
             Train one first (see TRAINING.md) or pass --model <path>.",
            args.model.display()
        );
    }
    let mut engine = InferenceEngine::new(
        &args.model,
        config.clone(),
        &args.tokenizer,
        &args.data_dir,
        device.clone(),
    )?;

    let dim = std::cmp::min;
    println!("{}", bold("tinybit eval — quality sanity check"));
    println!(
        "  model    : {} ({} params, vocab {})",
        args.model.display(),
        human(engine.model.config.param_count()),
        engine.model.config.vocab_size,
    );
    println!("  device   : {dev_name}");
    println!("  profile  : {profile:?}");

    // ---- perplexity ---------------------------------------------------------
    if let Some(data_path) = &args.data {
        println!("\n{}", bold("[perplexity]"));
        let seq_len = engine.model.config.max_seq_len;
        let ds = TokenDataset::open(data_path, seq_len)?;
        let total_tokens = ds.num_chunks * seq_len;
        let mut loader = DataLoader::new(ds, args.batch_size, false);
        let want = dim(args.max_batches, loader.num_batches());

        let mut sum_ce = 0.0f64;
        let mut n = 0usize;
        for _ in 0..want {
            let Some((inputs, targets)) = loader.next_batch()? else { break };
            let b = inputs.len();
            let t = inputs[0].len();
            let inp: Vec<u32> = inputs.into_iter().flatten().collect();
            let tgt: Vec<u32> = targets.into_iter().flatten().collect();
            let input_t = Tensor::from_vec(inp, (b, t), &device)?.to_dtype(DType::U32)?;
            let target_t = Tensor::from_vec(tgt, (b, t), &device)?.to_dtype(DType::U32)?;
            let (logits, _) = engine.model.forward_train(&input_t)?;
            let loss = cross_entropy_loss(&logits, &target_t)?;
            sum_ce += loss.to_scalar::<f32>()? as f64;
            n += 1;
        }
        if n == 0 {
            println!("  (no full batches available in {})", data_path.display());
        } else {
            let mean_ce = sum_ce / n as f64;
            let ppl = mean_ce.exp();
            let random = (engine.model.config.vocab_size as f64).ln();
            println!(
                "  data       : {} ({} tokens, seq_len {seq_len})",
                data_path.display(),
                human(total_tokens),
            );
            println!("  measured   : {n} batches × {} seq = {} sequences", args.batch_size, n * args.batch_size);
            println!("  mean CE    : {mean_ce:.4} nats/token");
            println!("  perplexity : {ppl:.2}");
            println!(
                "  context    : random-init baseline is CE≈{random:.2} (ppl≈{:.0}); lower is better.",
                random.exp()
            );
        }
    }

    // ---- generation sanity --------------------------------------------------
    if !args.no_generate {
        println!("\n{}", bold("[generation sanity]"));
        println!("  greedy, up to {} tokens per prompt\n", args.gen_tokens);
        engine.params.temperature = 0.0; // deterministic
        engine.params.max_new_tokens = args.gen_tokens;
        let prompts = match profile {
            Profile::Coding => CODING_PROMPTS,
            Profile::General => GENERAL_PROMPTS,
        };
        for prompt in prompts {
            let mut session = Session::new(&engine.model.config, &device)?;
            session.system_prompt = profile.default_system_prompt().to_string();
            let (response, stats) = engine.chat_turn(prompt, &mut session, None)?;
            let one_line: String = response.split_whitespace().collect::<Vec<_>>().join(" ");
            println!("  > {prompt}");
            println!("    {}", if one_line.is_empty() { "(empty response)" } else { &one_line });
            println!(
                "    \x1b[2m{} tok · {:.1} tok/s · {dev_name}\x1b[0m",
                stats.gen_tokens,
                stats.tokens_per_sec(),
            );
            println!();
        }
    }

    println!(
        "{}",
        dimmed(
            "Note: tinybit models are small (25M–500M). Expect coherent short text and \
             simple instruction following, not reliable factuality or reasoning."
        )
    );
    Ok(())
}

fn device_name(device: &candle_core::Device) -> &'static str {
    if device.is_cuda() {
        "cuda"
    } else if device.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

/// Human-readable count: 1234567 -> "1.2M".
fn human(n: usize) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.1}B", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.1}M", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.1}K", n / 1e3)
    } else {
        format!("{n:.0}")
    }
}

fn bold(s: &str) -> String {
    if std::io::stdout().is_terminal() {
        format!("\x1b[1m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

fn dimmed(s: &str) -> String {
    if std::io::stdout().is_terminal() {
        format!("\x1b[2m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
