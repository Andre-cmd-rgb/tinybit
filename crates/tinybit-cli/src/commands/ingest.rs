//! `tinybit ingest` — the write side of the tinybit data API
//! (see INTEGRATIONS.md). External apps in any language push user data
//! (smartwatch metrics, health, app counters) by piping JSON here or by
//! appending JSONL to `data/integrations/<source>/events.jsonl` directly;
//! this command is the validating, atomic-snapshot-updating front door.

use clap::Args;
use std::io::Read;
use tinybit_tools::integrations::IntegrationsStore;

#[derive(Args)]
pub struct IngestArgs {
    /// Source name for the events (e.g. "watch", "scale"): lowercase
    /// [a-z0-9_-], max 32 chars. Each source gets its own dir under
    /// data/integrations/.
    #[arg(long)]
    pub source: String,

    /// Read events from this file instead of stdin. Accepts a single JSON
    /// object, a JSON array, or JSONL (one object per line). Event shape:
    /// {"metric":"heart_rate","value":61,"unit":"bpm","ts":"2026-06-10T08:31:00Z"}
    /// — "ts" defaults to now, "unit"/"tags" are optional.
    #[arg(long)]
    pub file: Option<std::path::PathBuf>,

    /// Tools data directory (the same one `chat` uses; the store lives in
    /// <data-dir>/integrations).
    #[arg(long, default_value = "data")]
    pub data_dir: std::path::PathBuf,
}

pub fn run(args: IngestArgs) -> anyhow::Result<()> {
    let raw = match &args.file {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("could not read {}: {e}", path.display()))?,
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| anyhow::anyhow!("could not read stdin: {e}"))?;
            buf
        }
    };
    anyhow::ensure!(
        !raw.trim().is_empty(),
        "no input — pipe JSON to stdin or pass --file (see INTEGRATIONS.md)"
    );

    let store = IntegrationsStore::open(&args.data_dir);
    let report = store.ingest(&args.source, &raw)?;

    let skipped = if report.skipped > 0 {
        format!(", {} skipped", report.skipped)
    } else {
        String::new()
    };
    println!(
        "ingested {} event(s) into {} ({} metric(s){skipped})",
        report.events,
        args.data_dir.join("integrations").join(&report.source).display(),
        report.metrics,
    );
    Ok(())
}
