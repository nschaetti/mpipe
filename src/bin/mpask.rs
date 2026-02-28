use std::process;

use clap::Parser;
use mpipe::commands::ask::{self, AskArgs};

#[derive(Debug, Parser)]
#[command(
    name = "mpask",
    about = "Ask a question to an LLM provider",
    disable_version_flag = true
)]
struct Cli {
    #[command(flatten)]
    ask: AskArgs,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(err) = ask::run(cli.ask).await {
        eprintln!("{err}");
        process::exit(1);
    }
}
