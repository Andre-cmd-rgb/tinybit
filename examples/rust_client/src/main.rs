//! Example tinybit data-API client (Rust).
//!
//! Pushes a fake smartwatch sample into tinybit's integrations store by
//! piping JSON to `tinybit ingest` — the validating front door of the
//! file-based data API (see INTEGRATIONS.md at the repo root). Any language
//! that can spawn a process (or append a JSON line to a file) can integrate.

use std::io::Write;
use std::process::{Command, Stdio};

fn push_events(source: &str, events: &serde_json::Value) -> std::io::Result<String> {
    let mut child = Command::new("tinybit")
        .args(["ingest", "--source", source, "--data-dir", "data"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(events.to_string().as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn main() -> std::io::Result<()> {
    let sample = serde_json::json!([
        {"metric": "heart_rate", "value": 61, "unit": "bpm"},
        {"metric": "steps", "value": 8204},
        {"metric": "sleep_hours", "value": 7.5, "unit": "h"},
    ]);
    print!("{}", push_events("watch", &sample)?);
    println!(
        "Ask tinybit: `tinybit chat` → \"what's my heart rate?\" (the user_data tool reads this store)"
    );
    Ok(())
}
