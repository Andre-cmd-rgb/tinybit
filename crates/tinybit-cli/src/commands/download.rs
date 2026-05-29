use clap::Args;

#[derive(Args)]
pub struct DownloadArgs {
    /// HuggingFace repo to pull `tokenizer.json` from. The default is the LLaMA
    /// tokenizer tinybit is built around (32k vocab; tinybit reserves 8 extra
    /// slots for tool markers → vocab_size 32008).
    #[arg(long, default_value = "hf-internal-testing/llama-tokenizer")]
    pub tokenizer_id: String,

    /// Directory to write `tokenizer.json` into.
    #[arg(long, default_value = ".")]
    pub output: std::path::PathBuf,
}

pub async fn run(args: DownloadArgs) -> anyhow::Result<()> {
    println!("Downloading tokenizer from HuggingFace: {}", args.tokenizer_id);
    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.model(args.tokenizer_id.clone());
    let tokenizer_path = repo.get("tokenizer.json").await?;
    std::fs::create_dir_all(&args.output)?;
    let dest = args.output.join("tokenizer.json");
    std::fs::copy(&tokenizer_path, &dest)?;
    println!("Saved {}", dest.display());
    println!();
    println!("tinybit V1.0 does not ship pretrained weights — train your own:");
    println!("  • local smoke test:  tinybit train --smoke-test");
    println!("  • full run on an L4:  see TRAINING.md");
    Ok(())
}
