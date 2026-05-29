mod commands;

use clap::Parser;

/// tinybit — a small, fast, local-first AI assistant (RWKV-7 + ternary BitLinear).
///
/// Local CLI inference first: chat, evaluate, train, and export — no server, no
/// cloud calls at inference time.
#[derive(Parser)]
#[command(
    name = "tinybit",
    version,
    about = "A small, fast, local-first AI assistant (RWKV-7 + ternary BitLinear)",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Interactive local chat with a model checkpoint.
    Chat(commands::chat::ChatArgs),
    /// Measure perplexity on a token file and run generation sanity prompts.
    Eval(commands::eval::EvalArgs),
    /// Train a model from a config (local smoke test or full run).
    Train(commands::train::TrainArgs),
    /// Export / quantize a checkpoint (safetensors; ternary packing optional).
    Convert(commands::convert::ConvertArgs),
    /// Download the tokenizer from HuggingFace.
    Download(commands::download::DownloadArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Honor RUST_LOG; default to warnings so the CLI output stays clean.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat(args) => commands::chat::run(args),
        Commands::Eval(args) => commands::eval::run(args),
        Commands::Train(args) => commands::train::run(args),
        Commands::Convert(args) => commands::convert::run(args),
        Commands::Download(args) => commands::download::run(args).await,
    }
}
