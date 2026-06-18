use clap::Args;
use std::io::{self, BufRead, IsTerminal, Write};
use tinybit_core::config::ModelConfig;
use tinybit_core::tokenizer::Profile;
use tinybit_infer::{engine::InferenceEngine, session::Session};

#[derive(Args)]
pub struct ChatArgs {
    /// Model architecture config (TOML). For coding checkpoints use a *-coding config.
    #[arg(long, default_value = "configs/micro.toml")]
    pub config: std::path::PathBuf,

    /// Model checkpoint (safetensors).
    #[arg(long, default_value = "models/tinybit-micro.safetensors")]
    pub model: std::path::PathBuf,

    /// Tokenizer JSON.
    #[arg(long, default_value = "tokenizer.json")]
    pub tokenizer: std::path::PathBuf,

    /// Data dir for the built-in tools (todos/notes/calendar SQLite files).
    #[arg(long, default_value = "data/")]
    pub data_dir: std::path::PathBuf,

    /// Where to save/load the conversation when you use /save or --resume.
    #[arg(long, default_value = "sessions/default")]
    pub session: std::path::PathBuf,

    /// Resume a previously saved session.
    #[arg(long)]
    pub resume: bool,

    /// Override the system prompt (otherwise picked from --profile).
    #[arg(long)]
    pub system: Option<String>,

    /// Model family: general | coding. Selects the default system prompt.
    /// Defaults to coding when the config filename contains "coding".
    #[arg(long)]
    pub profile: Option<String>,

    /// Sampling temperature (0.0 = greedy/deterministic). Kept low by default:
    /// at tinybit's size a hot temperature samples into low-probability tails
    /// and derails (incoherent text, spurious tool calls). 0.3–0.4 stays
    /// coherent; raise toward 0.7 only if you want more variety.
    #[arg(long, default_value_t = 0.4)]
    pub temperature: f64,

    /// When the model may call a built-in tool: `auto` (only when the message
    /// looks like it needs one — the default; stops the tiny model firing tools
    /// on greetings), `always` (let the raw model decide — it over-fires), or
    /// `never` (pure conversation, no tools).
    #[arg(long, default_value = "auto")]
    pub tools: String,

    /// Automatic retrieval (RAG): number of passages from the local knowledge
    /// store to prepend to the system prompt for factual turns. The "fetch,
    /// don't memorize" path — a tiny model answers facts by searching local
    /// knowledge. 0 disables it.
    #[arg(long, default_value_t = 3)]
    pub rag: usize,
}

pub fn run(args: ChatArgs) -> anyhow::Result<()> {
    let config = ModelConfig::from_file(&args.config)?;
    config.validate()?;
    let profile = match &args.profile {
        Some(p) => Profile::parse(p)?,
        None => Profile::from_config_path(&args.config),
    };

    if !args.model.exists() {
        anyhow::bail!(
            "model checkpoint not found: {}\n\
             - train one: see TRAINING.md (or `tinybit train --smoke-test`)\n\
             - or point --model at an existing .safetensors checkpoint",
            args.model.display()
        );
    }
    if !args.tokenizer.exists() {
        anyhow::bail!(
            "tokenizer not found: {}\n  run: tinybit download --output .",
            args.tokenizer.display()
        );
    }

    let tool_mode = tinybit_infer::processor::ToolMode::parse(&args.tools)?;

    let device = InferenceEngine::auto_device();
    let mut engine = InferenceEngine::new(
        &args.model,
        config.clone(),
        &args.tokenizer,
        &args.data_dir,
        device.clone(),
    )?;
    engine.params.temperature = args.temperature;
    engine.set_rag(args.rag);

    let mut session = if args.resume && args.session.exists() {
        Session::load(&args.session, &device)?
    } else {
        let mut s = Session::new(&config, &device)?;
        s.system_prompt = profile.default_system_prompt().to_string();
        s
    };
    if let Some(sys) = args.system {
        session.system_prompt = sys;
    }

    print_banner(&engine, &device, profile);
    println!("tools: {} (change with --tools auto|always|never)", args.tools.to_ascii_lowercase());
    if engine.rag_top_k > 0 && engine.knowledge.is_some() {
        println!("retrieval: on (local knowledge, top-{})", engine.rag_top_k);
    } else {
        println!("retrieval: off");
    }

    let stdin = io::stdin();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF (Ctrl-D)
        }
        let input = line.trim();
        match input {
            "/quit" | "/exit" => break,
            "/help" => {
                print_help();
                continue;
            }
            "/reset" => {
                session.reset_state(&config)?;
                println!("State reset (conversation memory cleared).");
                continue;
            }
            "/save" => {
                session.save(&args.session)?;
                println!("Session saved to {}", args.session.display());
                continue;
            }
            "/system" => {
                println!("Current system prompt:\n  {}", session.system_prompt);
                continue;
            }
            s if s.starts_with("/system ") => {
                session.system_prompt = s[8..].to_string();
                session.reset_state(&config)?;
                println!("System prompt updated (state reset).");
                continue;
            }
            "" => continue,
            s if s.starts_with('/') => {
                println!("Unknown command. Type /help for the list.");
                continue;
            }
            _ => {}
        }

        let mut sink = |chunk: &str| {
            print!("{chunk}");
            let _ = io::stdout().flush();
        };
        let (_response, stats) = engine.chat_turn(input, &mut session, tool_mode, Some(&mut sink))?;
        println!(); // end the streamed line
        print_stats(&stats, &device);
    }
    Ok(())
}

fn print_banner(engine: &InferenceEngine, device: &candle_core::Device, profile: Profile) {
    let dev = device_name(device);
    println!("tinybit v{}  ·  {profile:?} profile  ·  {dev}", env!("CARGO_PKG_VERSION"));
    println!(
        "model: {} params, vocab {}, ctx {}",
        human(engine.model.config.param_count()),
        engine.model.config.vocab_size,
        engine.model.config.max_seq_len,
    );
    println!("Type /help for commands, /quit to exit.");
    println!("Note: tinybit is a small local model — expect short coherent text, not reliable facts.");
}

fn print_help() {
    println!("Commands:");
    println!("  /help            show this help");
    println!("  /reset           clear conversation memory (recurrent state)");
    println!("  /system          show the current system prompt");
    println!("  /system <text>   set a new system prompt (resets state)");
    println!("  /save            save the session to disk");
    println!("  /quit, /exit     leave");
}

/// Print a one-line generation summary (tokens, throughput, timing) below the
/// model's answer. Dimmed when stdout is a terminal so it reads as metadata.
fn print_stats(stats: &tinybit_infer::engine::GenStats, device: &candle_core::Device) {
    let line = format!(
        "{} tok · {:.1} tok/s · {} prompt · {:.2}s ({:.2}s prefill + {:.2}s gen) · {}",
        stats.gen_tokens,
        stats.tokens_per_sec(),
        stats.prompt_tokens,
        stats.total_secs(),
        stats.prefill_secs,
        stats.decode_secs,
        device_name(device),
    );
    if std::io::stdout().is_terminal() {
        println!("\x1b[2m{line}\x1b[0m");
    } else {
        println!("{line}");
    }
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

fn human(n: usize) -> String {
    let n = n as f64;
    if n >= 1e9 {
        format!("{:.1}B", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.0}M", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.0}K", n / 1e3)
    } else {
        format!("{n:.0}")
    }
}
