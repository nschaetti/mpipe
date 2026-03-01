use std::io;
use std::process;

use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, shells};
use mpipe::commands::ask::{self, AskArgs};
use mpipe::commands::config::{self, ConfigArgs};

const ROOT_HELP_EXAMPLES: &str = "Examples:\n  mpipe ask --provider fireworks --model accounts/fireworks/models/kimi-k2-instruct-0905 \"2+2?\"\n  echo \"2+2?\" | mpipe ask --provider openai --model gpt-4o-mini\n  mpipe config check\n  mpipe completion bash > ~/.local/share/bash-completion/completions/mpipe";

const ASK_HELP_EXAMPLES: &str = "Examples:\n  mpipe ask --provider fireworks --model accounts/fireworks/models/kimi-k2-instruct-0905 \"2+2?\"\n  echo \"2+2?\" | mpipe ask --provider openai --model gpt-4o-mini\n  mpipe ask --provider fireworks --model accounts/fireworks/models/kimi-k2-instruct-0905 --dry-run --json \"Explain retries\"";

#[derive(Debug, Parser)]
#[command(
    name = "mpipe",
    about = "Multi-provider LLM CLI tools",
    after_help = ROOT_HELP_EXAMPLES
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(about = "Ask a question to an LLM provider", after_help = ASK_HELP_EXAMPLES)]
    Ask(AskArgs),
    #[command(about = "Manage local config")]
    Config(ConfigArgs),
    #[command(about = "Generate shell completion script")]
    Completion {
        #[arg(value_enum)]
        shell: CompletionShell,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CompletionShell {
    Bash,
    Zsh,
    Fish,
}

fn print_completion(shell: CompletionShell) {
    let mut cmd = Cli::command();
    match shell {
        CompletionShell::Bash => generate(shells::Bash, &mut cmd, "mpipe", &mut io::stdout()),
        CompletionShell::Zsh => generate(shells::Zsh, &mut cmd, "mpipe", &mut io::stdout()),
        CompletionShell::Fish => generate(shells::Fish, &mut cmd, "mpipe", &mut io::stdout()),
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Ask(args) => ask::run(args).await,
        Commands::Config(args) => config::run(args),
        Commands::Completion { shell } => {
            print_completion(shell);
            Ok(())
        }
    };

    if let Err(err) = result {
        eprintln!("{err}");
        process::exit(1);
    }
}
