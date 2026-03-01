use std::process;

use clap::Parser;
use mpipe::commands::ask::{self, AskArgs};

const ASK_HELP_EXAMPLES: &str = "Examples:\n  mpask --provider fireworks --model accounts/fireworks/models/kimi-k2-instruct-0905 \"2+2?\"\n  echo \"2+2?\" | mpask --provider openai --model gpt-4o-mini\n  mpask --provider fireworks --model accounts/fireworks/models/kimi-k2-instruct-0905 --dry-run --json \"Explain retries\"";

#[derive(Debug, Parser)]
#[command(
    name = "mpask",
    about = "Ask a question to an LLM provider",
    disable_version_flag = true,
    after_help = ASK_HELP_EXAMPLES
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
