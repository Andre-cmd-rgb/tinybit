use clap::Args;

#[derive(Args)]
pub struct DownloadArgs {
    /// Model size to download: nano, micro, small, base
    #[arg(long, default_value = "small")]
    pub model: String,

    /// HuggingFace model ID for tokenizer
    #[arg(long, default_value = "hf-internal-testing/llama-tokenizer")]
    pub tokenizer_id: String,

    /// Output directory
    #[arg(long, default_value = ".")]
    pub output: std::path::PathBuf,
}

pub async fn run(args: DownloadArgs) -> anyhow::Result<()> {
    println!("Downloading tokenizer from HuggingFace: {}", args.tokenizer_id);
    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.model(args.tokenizer_id.clone());
    let tokenizer_path = repo.get("tokenizer.json").await?;
    let dest = args.output.join("tokenizer.json");
    std::fs::copy(&tokenizer_path, &dest)?;
    println!("Tokenizer saved to {}", dest.display());
    println!("Note: pretrained weights for tiny-bit-{} are not yet available for download.", args.model);
    println!("To train from scratch, run: tiny-bit train --smoke-test");
    Ok(())
}
