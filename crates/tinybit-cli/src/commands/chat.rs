use clap::Args;
use std::io::{self, BufRead, Write};
use tinybit_core::config::ModelConfig;
use tinybit_infer::{engine::InferenceEngine, session::Session};

#[derive(Args)]
pub struct ChatArgs {
    #[arg(long, default_value = "configs/micro.toml")]
    pub config: std::path::PathBuf,

    #[arg(long, default_value = "models/tinybit-micro.safetensors")]
    pub model: std::path::PathBuf,

    #[arg(long, default_value = "tokenizer.json")]
    pub tokenizer: std::path::PathBuf,

    #[arg(long, default_value = "data/")]
    pub data_dir: std::path::PathBuf,

    #[arg(long, default_value = "sessions/default")]
    pub session: std::path::PathBuf,

    #[arg(long)]
    pub resume: bool,

    #[arg(long)]
    pub system: Option<String>,

    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,
}

pub fn run(args: ChatArgs) -> anyhow::Result<()> {
    let config = ModelConfig::from_file(&args.config)?;
    let device = InferenceEngine::auto_device();
    let engine = InferenceEngine::new(
        &args.model,
        config.clone(),
        &args.tokenizer,
        &args.data_dir,
        device.clone(),
    )?;

    let mut session = if args.resume && args.session.exists() {
        Session::load(&args.session, &device)?
    } else {
        Session::new(&config, &device)?
    };

    if let Some(sys) = args.system {
        session.system_prompt = sys;
    }

    let stdin = io::stdin();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let input = line.trim();
        match input {
            "/quit" | "/exit" => break,
            "/reset" => {
                session.reset_state(&config)?;
                println!("State reset.");
                continue;
            }
            "/save" => {
                session.save(&args.session)?;
                println!("Session saved to {}", args.session.display());
                continue;
            }
            s if s.starts_with("/system ") => {
                session.system_prompt = s[8..].to_string();
                println!("System prompt updated.");
                continue;
            }
            "" => continue,
            _ => {}
        }

        let response = engine.chat_turn(input, &mut session, None)?;
        println!("{response}");
    }
    Ok(())
}
