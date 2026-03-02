use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read};

use clap::{Args, ValueEnum};
use serde::Serialize;

use crate::config::{self, ProfileConfig};
use crate::rchain::embeddings::{
    self, ChunkStrategy, EmbeddingProvider, EmbeddingResult, EmbeddingsConfig,
};

#[derive(Debug, Args, Clone)]
pub struct EmbedArgs {
    #[arg(long)]
    pub profile: Option<String>,

    #[arg(long, value_enum)]
    pub provider: Option<ProviderArg>,

    #[arg(long)]
    pub model: Option<String>,

    #[arg(long)]
    pub chunk_size: Option<usize>,

    #[arg(long)]
    pub chunk_overlap: Option<usize>,

    #[arg(long, value_enum)]
    pub chunk_strategy: Option<ChunkStrategyArg>,

    #[arg(long, value_enum)]
    pub output: Option<OutputFormatArg>,

    #[arg(long)]
    pub json: bool,

    #[arg(long)]
    pub file: Option<std::path::PathBuf>,

    input: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ProviderArg {
    Openai,
    Fireworks,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ChunkStrategyArg {
    Paragraph,
    Sentence,
    Token,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormatArg {
    Text,
    Json,
}

impl From<ChunkStrategyArg> for ChunkStrategy {
    fn from(arg: ChunkStrategyArg) -> Self {
        match arg {
            ChunkStrategyArg::Paragraph => ChunkStrategy::Paragraph,
            ChunkStrategyArg::Sentence => ChunkStrategy::Sentence,
            ChunkStrategyArg::Token => ChunkStrategy::Token,
        }
    }
}

impl From<ProviderArg> for EmbeddingProvider {
    fn from(arg: ProviderArg) -> Self {
        match arg {
            ProviderArg::Openai => EmbeddingProvider::Openai,
            ProviderArg::Fireworks => EmbeddingProvider::Fireworks,
        }
    }
}

#[derive(Debug, Serialize)]
struct JsonOutput {
    provider: String,
    model: String,
    chunks: Vec<String>,
    embeddings: Vec<Vec<f64>>,
}

pub fn run(cli: EmbedArgs) -> Result<(), String> {
    let profile = resolve_profile(cli.profile.as_deref())?;
    let provider = resolve_provider(cli.provider, &profile)?;
    let model = resolve_model(cli.model, &profile)?;
    let chunk_size = resolve_chunk_size(cli.chunk_size, &profile)?;
    let chunk_overlap = resolve_chunk_overlap(cli.chunk_overlap, &profile)?;
    let chunk_strategy = resolve_chunk_strategy(cli.chunk_strategy, &profile)?;
    let output_format = resolve_output_format(cli.output, cli.json)?;

    let input_text = resolve_input(cli.input, cli.file)?;

    let config = EmbeddingsConfig {
        provider,
        model,
        chunk_size,
        chunk_overlap,
        chunk_strategy,
    };

    let result = embeddings::embed_texts(&config, &[input_text]).map_err(|err| err.to_string())?;

    render_output(&result, output_format)?;

    Ok(())
}

fn resolve_profile(profile_name: Option<&str>) -> Result<ProfileConfig, String> {
    match profile_name {
        Some(name) => config::load_profile(name),
        None => Ok(ProfileConfig::default()),
    }
}

fn resolve_provider(
    cli_provider: Option<ProviderArg>,
    profile: &ProfileConfig,
) -> Result<EmbeddingProvider, String> {
    if let Some(provider) = cli_provider {
        return Ok(provider.into());
    }

    if let Ok(raw) = env::var("MP_PROVIDER") {
        return parse_provider_value(&raw, "MP_PROVIDER");
    }

    if let Some(provider) = &profile.provider {
        return parse_provider_value(provider, "profile provider");
    }

    Ok(EmbeddingProvider::Fireworks)
}

fn parse_provider_value(raw: &str, source: &str) -> Result<EmbeddingProvider, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "openai" => Ok(EmbeddingProvider::Openai),
        "fireworks" => Ok(EmbeddingProvider::Fireworks),
        other => Err(format!(
            "Invalid {source} '{other}'. Supported values: openai, fireworks."
        )),
    }
}

fn resolve_model(cli_model: Option<String>, profile: &ProfileConfig) -> Result<String, String> {
    if let Some(model) = cli_model {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    if let Ok(model) = env::var("MP_MODEL") {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    if let Some(model) = &profile.embedding_model {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    Ok(EmbeddingProvider::Fireworks.as_str().to_string())
}

fn resolve_chunk_size(cli_size: Option<usize>, profile: &ProfileConfig) -> Result<usize, String> {
    if let Some(size) = cli_size {
        if size == 0 {
            return Err("Chunk size must be greater than 0.".to_string());
        }
        return Ok(size);
    }

    if let Some(size) = profile.chunk_size {
        if size == 0 {
            return Err("Chunk size in profile must be greater than 0.".to_string());
        }
        return Ok(size);
    }

    Ok(8000)
}

fn resolve_chunk_overlap(
    cli_overlap: Option<usize>,
    profile: &ProfileConfig,
) -> Result<usize, String> {
    if let Some(overlap) = cli_overlap {
        if overlap > 100 {
            return Err("Chunk overlap must be between 0 and 100.".to_string());
        }
        return Ok(overlap);
    }

    if let Some(overlap) = profile.chunk_overlap {
        if overlap > 100 {
            return Err("Chunk overlap in profile must be between 0 and 100.".to_string());
        }
        return Ok(overlap);
    }

    Ok(10)
}

fn resolve_chunk_strategy(
    cli_strategy: Option<ChunkStrategyArg>,
    profile: &ProfileConfig,
) -> Result<ChunkStrategy, String> {
    if let Some(strategy) = cli_strategy {
        return Ok(strategy.into());
    }

    if let Some(raw) = &profile.chunk_strategy {
        if let Some(strategy) = ChunkStrategy::from_str(raw) {
            return Ok(strategy);
        }
        return Err(format!(
            "Invalid chunk strategy '{raw}'. Supported values: paragraph, sentence, token."
        ));
    }

    Ok(ChunkStrategy::Paragraph)
}

fn resolve_output_format(
    output: Option<OutputFormatArg>,
    json: bool,
) -> Result<OutputFormatArg, String> {
    if json {
        return Ok(OutputFormatArg::Json);
    }

    if let Some(output) = output {
        return Ok(output);
    }

    Ok(OutputFormatArg::Text)
}

fn resolve_input(
    cli_input: Option<String>,
    file: Option<std::path::PathBuf>,
) -> Result<String, String> {
    if let Some(text) = cli_input {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    if let Some(path) = file {
        return fs::read_to_string(&path)
            .map_err(|err| format!("Failed to read file '{}': {}", path.display(), err));
    }

    if io::stdin().is_terminal() {
        return Err("No input provided. Pass an argument, --file, or pipe stdin.".to_string());
    }

    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .map_err(|err| format!("Failed to read stdin: {err}"))?;

    let text = buffer.trim().to_string();
    if text.is_empty() {
        return Err("Input is empty.".to_string());
    }

    Ok(text)
}

fn render_output(result: &EmbeddingResult, format: OutputFormatArg) -> Result<(), String> {
    match format {
        OutputFormatArg::Text => render_text(result),
        OutputFormatArg::Json => render_json(result),
    }
}

fn render_text(result: &EmbeddingResult) -> Result<(), String> {
    for embedding in &result.embeddings {
        let line = embedding
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("{line}");
    }
    Ok(())
}

fn render_json(result: &EmbeddingResult) -> Result<(), String> {
    let output = JsonOutput {
        provider: result.provider.clone(),
        model: result.model.clone(),
        chunks: result.chunks.clone(),
        embeddings: result.embeddings.clone(),
    };

    let json =
        serde_json::to_string(&output).map_err(|err| format!("Failed to serialize JSON: {err}"))?;

    println!("{json}");
    Ok(())
}
