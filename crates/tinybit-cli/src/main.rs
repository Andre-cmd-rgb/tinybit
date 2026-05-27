mod commands;

use clap::Parser;

#[derive(Parser)]
#[command(name = "tinybit", version = "0.1.0", about = "Your local AI assistant")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    Chat(commands::chat::ChatArgs),
    Serve(commands::serve::ServeArgs),
    Train(commands::train::TrainArgs),
    Convert(commands::convert::ConvertArgs),
    Download(commands::download::DownloadArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Commands::Chat(args)     => commands::chat::run(args),
        Commands::Serve(args)    => commands::serve::run(args).await,
        Commands::Train(args)    => commands::train::run(args),
        Commands::Convert(args)  => commands::convert::run(args),
        Commands::Download(args) => commands::download::run(args).await,
    }
}
