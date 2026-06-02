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
    // Direct HTTPS GET from the HF "resolve" endpoint. We used to go through
    // hf-hub, but its tokio Api built a malformed URL in some environments
    // ("relative URL without a base"), which broke `tinybit download` outright.
    // A plain request is simpler, robust, and removes the dependency.
    let url = format!(
        "https://huggingface.co/{}/resolve/main/tokenizer.json",
        args.tokenizer_id
    );
    println!("Downloading tokenizer.json from {url}");

    let bytes = reqwest::Client::new()
        .get(&url)
        .header(reqwest::header::USER_AGENT, "tinybit-cli")
        .send()
        .await?
        .error_for_status()
        .map_err(|e| anyhow::anyhow!("download failed for {}: {e}", args.tokenizer_id))?
        .bytes()
        .await?;

    // Guard against silently saving an HTML error page as tokenizer.json.
    if bytes.len() < 1024 {
        anyhow::bail!(
            "downloaded tokenizer.json is suspiciously small ({} bytes) — \
             check that --tokenizer-id is a correct, public repo",
            bytes.len()
        );
    }

    std::fs::create_dir_all(&args.output)?;
    let dest = args.output.join("tokenizer.json");
    std::fs::write(&dest, &bytes)?;
    println!("Saved {} ({:.1} KiB)", dest.display(), bytes.len() as f64 / 1024.0);
    println!();
    println!("tinybit V1.0 does not ship pretrained weights — train your own:");
    println!("  • local smoke test:  tinybit train --smoke-test");
    println!("  • full run on an L4:  see TRAINING.md");
    Ok(())
}
