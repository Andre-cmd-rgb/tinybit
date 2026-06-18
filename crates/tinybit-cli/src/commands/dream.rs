use clap::Args;
use std::path::{Path, PathBuf};
use tinybit_core::config::ModelConfig;
use tinybit_core::tokenizer::{
    Tokenizer, ROLE_ASSISTANT_PREFIX, ROLE_SYSTEM_PREFIX, ROLE_USER_PREFIX,
};
use tinybit_infer::engine::InferenceEngine;
use tinybit_infer::session::{Role, Session};
use tinybit_train::dream::{consolidate, DreamConfig};

/// Offline "sleep" consolidation: replay saved conversations and consolidate
/// them into the model, while distilling toward the frozen base on the model's
/// own generated pseudo-rehearsal ("dreams") so it doesn't forget.
#[derive(Args)]
pub struct DreamArgs {
    /// Model architecture config (TOML).
    #[arg(long, default_value = "configs/nano.toml")]
    pub config: PathBuf,

    /// Base model checkpoint to consolidate (f32 safetensors).
    #[arg(long, default_value = "models/tinybit-nano.safetensors")]
    pub model: PathBuf,

    /// Tokenizer JSON.
    #[arg(long, default_value = "tokenizer.json")]
    pub tokenizer: PathBuf,

    /// Data dir for the tools/knowledge store.
    #[arg(long, default_value = "data/")]
    pub data_dir: PathBuf,

    /// A saved session dir (with history.json), or a parent dir containing
    /// several such session dirs — the experience to consolidate.
    #[arg(long, default_value = "sessions")]
    pub sessions: PathBuf,

    /// Where to write the consolidated checkpoint.
    #[arg(long, default_value = "models/tinybit-dreamed.safetensors")]
    pub out: PathBuf,

    /// Number of consolidation gradient steps.
    #[arg(long, default_value_t = 40)]
    pub steps: usize,

    /// Learning rate (small — consolidation, not training from scratch).
    #[arg(long, default_value_t = 5e-5)]
    pub lr: f64,

    /// Weight on the KL-to-frozen-base anti-forgetting term.
    #[arg(long, default_value_t = 1.0)]
    pub kl: f64,

    /// Tokens per replay/anchor sequence.
    #[arg(long, default_value_t = 256)]
    pub seq_len: usize,

    /// Number of pseudo-rehearsal sequences to generate from the base model
    /// (its own "dreams") for the anti-forgetting anchor. 0 → anchor on the
    /// replay tokens instead.
    #[arg(long, default_value_t = 4)]
    pub pseudo: usize,
}

pub fn run(args: DreamArgs) -> anyhow::Result<()> {
    let config = ModelConfig::from_file(&args.config)?;
    config.validate()?;
    if !args.model.exists() {
        anyhow::bail!("model checkpoint not found: {}", args.model.display());
    }
    if !args.tokenizer.exists() {
        anyhow::bail!("tokenizer not found: {}\n  run: tinybit download --output .", args.tokenizer.display());
    }
    let tokenizer = Tokenizer::from_file_with_vocab(&args.tokenizer, config.vocab_size)?;

    // 1) Replay set: tokenize saved conversations.
    let device = InferenceEngine::auto_device();
    let session_dirs = find_sessions(&args.sessions);
    if session_dirs.is_empty() {
        anyhow::bail!(
            "no saved sessions found under {} (chat with /save first, so there is experience to consolidate)",
            args.sessions.display()
        );
    }
    let mut replay: Vec<Vec<u32>> = Vec::new();
    for dir in &session_dirs {
        if let Ok(session) = Session::load(dir, &device) {
            let text = format_history(&session);
            if let Ok(ids) = tokenizer.encode(&text, false) {
                if ids.len() >= 2 {
                    replay.push(ids);
                }
            }
        }
    }
    anyhow::ensure!(!replay.is_empty(), "saved sessions had no usable conversation text");
    println!("dream: replaying {} session(s), {} sequence(s)", session_dirs.len(), replay.len());

    // 2) Pseudo-rehearsal: let the base model dream — generate a few sequences
    //    from generic seeds; KL toward the frozen base on these resists forgetting.
    let mut anchors: Vec<Vec<u32>> = Vec::new();
    if args.pseudo > 0 {
        let engine = InferenceEngine::new(&args.model, config.clone(), &args.tokenizer, &args.data_dir, device.clone())?;
        let seeds = ["The", "I think that", "Here is", "In summary,", "A good way to", "Once"];
        for i in 0..args.pseudo {
            let seed = seeds[i % seeds.len()];
            let mut state = tinybit_core::state::InferenceState::zeros(&config, &device)?;
            let text = engine.generate(seed, &mut state, None)?;
            let combined = format!("{seed} {text}");
            if let Ok(ids) = tokenizer.encode(&combined, false) {
                if ids.len() >= 2 {
                    anchors.push(ids);
                }
            }
        }
        println!("dream: generated {} pseudo-rehearsal sequence(s)", anchors.len());
    }

    // 3) Consolidate.
    let dcfg = DreamConfig { steps: args.steps, lr: args.lr, kl_weight: args.kl, seq_len: args.seq_len };
    println!(
        "dream: consolidating ({} steps, lr {}, kl {}) ...",
        dcfg.steps, dcfg.lr, dcfg.kl_weight
    );
    let report = consolidate(&args.model, config, &device, &replay, &anchors, &args.out, &dcfg)?;

    println!(
        "dream: done. loss {:.4} -> {:.4} (CE {:.4} -> {:.4}) over {} steps",
        report.first_loss, report.last_loss, report.first_ce, report.last_ce, report.steps
    );
    println!("dream: consolidated checkpoint written to {}", args.out.display());
    println!("       run it with:  tinybit chat --config {} --model {}", "configs/nano.toml", args.out.display());
    Ok(())
}

/// A session dir is one that contains `history.json`. If `root` is itself such a
/// dir, return just it; otherwise scan its immediate children.
fn find_sessions(root: &Path) -> Vec<PathBuf> {
    if root.join("history.json").exists() {
        return vec![root.to_path_buf()];
    }
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() && p.join("history.json").exists() {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Render a session's history into the shared training/inference template so the
/// consolidated tokens match how the model was trained.
fn format_history(session: &Session) -> String {
    let mut s = String::new();
    s.push_str(ROLE_SYSTEM_PREFIX);
    s.push_str(&session.system_prompt);
    for m in &session.history {
        match m.role {
            Role::User => {
                s.push_str(ROLE_USER_PREFIX);
                s.push_str(&m.content);
            }
            Role::Assistant => {
                s.push_str(ROLE_ASSISTANT_PREFIX);
                s.push_str(&m.content);
            }
            // Tool/system turns are folded in as plain context.
            _ => {
                s.push('\n');
                s.push_str(&m.content);
            }
        }
    }
    s
}
