use std::path::PathBuf;
use clap::{Args, ValueEnum};
use crate::commands::ask::AskArgs;


#[derive(Debug, Args, Clone)]
pub struct AgentArgs {
    #[arg(long)]
    pub profile: Option<String>,

    #[arg(long, value_enum)]
    provider: Option<ProviderArg>,

    #[arg(long)]
    model: Option<String>,

    /// Main prompt
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,

    #[arg(long = "prompt-file")]
    prompt_file: Option<PathBuf>,
}


#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProviderArg {
    Openai,
    Fireworks,
}


pub async fn run(cli: AgentArgs) -> Result<(), String> {
    Ok(())
}
