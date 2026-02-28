use std::process;

use clap::{Parser, Subcommand};
use mpipe::commands::ask::{self, AskArgs};

#[derive(Debug, Parser)]
#[command(name = "mpipe", about = "Multi-provider LLM CLI tools")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Ask(AskArgs),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Ask(args) => ask::run(args).await,
    };

    if let Err(err) = result {
        eprintln!("{err}");
        process::exit(1);
    }
}
