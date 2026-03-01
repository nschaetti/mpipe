use clap::{Args, Subcommand};
use serde::Serialize;

use crate::commands::prompting::{build_messages, compose_prompt, non_empty, resolve_prompt};
use crate::rchain::provider::ChatMessage;

#[derive(Debug, Args, Clone)]
pub struct PromptArgs {
    #[command(subcommand)]
    command: PromptSubcommand,
}

#[derive(Debug, Subcommand, Clone)]
enum PromptSubcommand {
    #[command(about = "Render the final prompt locally")]
    Render(PromptRenderArgs),
}

#[derive(Debug, Args, Clone)]
pub struct PromptRenderArgs {
    #[arg(long)]
    system: Option<String>,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    postprompt: Option<String>,

    #[arg(long)]
    json: bool,

    input: Option<String>,
}

#[derive(Debug, Serialize)]
struct RenderOutput {
    prompt: String,
    messages: Vec<ChatMessage>,
    prompt_source: String,
}

pub fn run(args: PromptArgs) -> Result<(), String> {
    match args.command {
        PromptSubcommand::Render(args) => run_render(args),
    }
}

fn run_render(args: PromptRenderArgs) -> Result<(), String> {
    let main_prompt = resolve_prompt(args.input)?;
    let prompt = compose_prompt(
        args.prompt.as_deref(),
        &main_prompt.text,
        args.postprompt.as_deref(),
    );
    let messages = build_messages(non_empty(args.system.as_deref()), &prompt);

    if args.json {
        let output = RenderOutput {
            prompt,
            messages,
            prompt_source: main_prompt.source.as_str().to_string(),
        };
        let rendered = serde_json::to_string(&output)
            .map_err(|err| format!("Failed to serialize prompt render output: {err}"))?;
        println!("{rendered}");
        return Ok(());
    }

    println!("{prompt}");
    Ok(())
}
