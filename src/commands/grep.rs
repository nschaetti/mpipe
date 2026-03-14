use std::env;

use chromadb::collection::QueryOptions;
use clap::{Args, ValueEnum};
use serde::Serialize;

use crate::commands::chroma::ChromaConnectArgs;
use crate::commands::prompting::resolve_prompt;
use crate::rchain::embeddings::{EmbeddingProvider, embed_chunks_with_provider};
use crate::rchain::provider::{self, AskOptions, Provider};

const DEFAULT_COLLECTION: &str = "mpipe";

#[derive(Debug, Args, Clone)]
pub struct GrepArgs {
    #[arg(long)]
    collection: Option<String>,

    #[arg(long = "embedding-model")]
    embedding_model: String,

    #[arg(long, default_value_t = 5)]
    top_k: usize,

    #[arg(long, value_enum)]
    provider: Option<ProviderArg>,

    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    system: Option<String>,

    #[arg(long)]
    temperature: Option<f32>,

    #[arg(long = "max-tokens")]
    max_tokens: Option<u32>,

    #[arg(long)]
    timeout: Option<u64>,

    #[arg(long)]
    json: bool,

    #[command(flatten)]
    chroma: ChromaConnectArgs,

    prompt: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProviderArg {
    Openai,
    Fireworks,
}

#[derive(Debug, Clone, Serialize)]
struct SourceHit {
    rank: usize,
    id: String,
    source: Option<String>,
    chunk_index: Option<usize>,
    distance: Option<f32>,
    document: String,
}

#[derive(Debug, Serialize)]
struct GrepJsonOutput {
    collection: String,
    prompt: String,
    answer: String,
    sources: Vec<SourceHit>,
}

pub async fn run(args: GrepArgs) -> Result<(), String> {
    if args.top_k == 0 {
        return Err("--top-k must be > 0".to_string());
    }

    let prompt = resolve_prompt(args.prompt)?;
    let prompt_text = prompt.text;
    let provider = resolve_provider(args.provider)?;
    let model = resolve_model(args.model)?;
    let collection_name = resolve_collection_name(args.collection.as_deref());

    let query_embedding = embed_prompt(&args.embedding_model, &prompt_text)?;

    let (client, _local_chroma) = crate::commands::chroma::connect(&args.chroma).await?;
    let collection = client
        .get_collection(&collection_name)
        .await
        .map_err(|err| format!("Failed to open collection '{collection_name}': {err}"))?;

    let query_result = collection
        .query(
            QueryOptions {
                query_embeddings: Some(vec![query_embedding]),
                query_texts: None,
                n_results: Some(args.top_k),
                where_metadata: None,
                where_document: None,
                include: Some(vec!["metadatas", "documents", "distances"]),
            },
            None,
        )
        .await
        .map_err(|err| format!("Failed to query collection '{collection_name}': {err}"))?;

    let sources = collect_sources(&query_result)?;
    if sources.is_empty() {
        return Err(format!(
            "No matching chunks found in collection '{collection_name}'."
        ));
    }

    let context = build_context(&sources);
    let user_prompt = format!(
        "Question:\n{prompt_text}\n\nContext:\n{context}\n\nAnswer in the same language as the question. Use the context above and cite sources like [1], [2]. If the context is insufficient, say it clearly."
    );

    let mut messages = Vec::new();
    if let Some(system) = args.system.as_deref().map(str::trim)
        && !system.is_empty()
    {
        messages.push(provider::ChatMessage::system(system));
    }
    messages.push(provider::ChatMessage::user_with_text(user_prompt));

    let response = provider::ask(
        provider,
        &model,
        &messages,
        AskOptions {
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            timeout_secs: args.timeout,
            retries: 0,
            retry_delay_ms: 500,
        },
    )
    .await
    .map_err(|err| err.to_string())?;

    if args.json {
        let payload = GrepJsonOutput {
            collection: collection_name,
            prompt: prompt_text,
            answer: response.content,
            sources,
        };
        let rendered = serde_json::to_string(&payload)
            .map_err(|err| format!("Failed to serialize grep output: {err}"))?;
        println!("{rendered}");
        return Ok(());
    }

    println!("{}", response.content.trim_end());
    println!();
    println!("Sources:");
    for hit in &sources {
        let source = hit.source.as_deref().unwrap_or("unknown");
        let chunk = hit
            .chunk_index
            .map(|idx| format!(" chunk={}", idx + 1))
            .unwrap_or_default();
        let distance = hit
            .distance
            .map(|d| format!(" distance={d:.4}"))
            .unwrap_or_default();
        println!(
            "- [{}] {} (id={}{}{})",
            hit.rank, source, hit.id, chunk, distance
        );
    }

    Ok(())
}

fn embed_prompt(model: &str, prompt: &str) -> Result<Vec<f32>, String> {
    let chunks = vec![prompt.to_string()];
    let mut vectors = embed_chunks_with_provider(EmbeddingProvider::Fireworks, model, &chunks)
        .map_err(|err| format!("Failed to embed prompt: {err}"))?;
    let vector = vectors
        .pop()
        .ok_or_else(|| "Embedding provider returned no vector for prompt.".to_string())?;
    Ok(vector.into_iter().map(|v| v as f32).collect())
}

fn collect_sources(
    query_result: &chromadb::collection::QueryResult,
) -> Result<Vec<SourceHit>, String> {
    let ids = query_result
        .ids
        .first()
        .ok_or_else(|| "Query result did not contain ids.".to_string())?;
    let docs = query_result
        .documents
        .as_ref()
        .and_then(|all| all.first())
        .cloned()
        .unwrap_or_default();
    let metadatas = query_result
        .metadatas
        .as_ref()
        .and_then(|all| all.first())
        .cloned()
        .unwrap_or_default();
    let distances = query_result
        .distances
        .as_ref()
        .and_then(|all| all.first())
        .cloned()
        .unwrap_or_default();

    let mut hits = Vec::with_capacity(ids.len());
    for (idx, id) in ids.iter().enumerate() {
        let document = docs.get(idx).cloned().unwrap_or_default();
        let metadata = metadatas.get(idx).and_then(|m| m.as_ref());
        let source = metadata.and_then(|m| {
            m.get("source")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        });
        let chunk_index = metadata.and_then(|m| {
            m.get("chunk_index")
                .and_then(|value| value.as_u64())
                .map(|v| v as usize)
        });
        let distance = distances.get(idx).cloned();

        hits.push(SourceHit {
            rank: idx + 1,
            id: id.clone(),
            source,
            chunk_index,
            distance,
            document,
        });
    }
    Ok(hits)
}

fn build_context(sources: &[SourceHit]) -> String {
    let mut lines = Vec::with_capacity(sources.len());
    for hit in sources {
        let source = hit.source.as_deref().unwrap_or("unknown");
        let distance = hit
            .distance
            .map(|d| format!("{d:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        lines.push(format!(
            "[{}] source={} id={} distance={}\n{}",
            hit.rank,
            source,
            hit.id,
            distance,
            hit.document.trim()
        ));
    }
    lines.join("\n\n")
}

fn resolve_provider(cli_provider: Option<ProviderArg>) -> Result<Provider, String> {
    if let Some(provider) = cli_provider {
        return Ok(match provider {
            ProviderArg::Openai => Provider::Openai,
            ProviderArg::Fireworks => Provider::Fireworks,
        });
    }

    match env::var("MP_PROVIDER") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "openai" => Ok(Provider::Openai),
            "fireworks" => Ok(Provider::Fireworks),
            other => Err(format!(
                "Invalid MP_PROVIDER '{other}'. Supported values: openai, fireworks."
            )),
        },
        Err(_) => Ok(Provider::Openai),
    }
}

fn resolve_model(cli_model: Option<String>) -> Result<String, String> {
    if let Some(model) = cli_model {
        let trimmed = model.trim();
        if trimmed.is_empty() {
            return Err("--model cannot be empty".to_string());
        }
        return Ok(trimmed.to_string());
    }

    if let Ok(model) = env::var("MP_MODEL") {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    Err("No model provided. Use --model or set MP_MODEL.".to_string())
}

fn resolve_collection_name(cli_collection: Option<&str>) -> String {
    if let Some(collection) = cli_collection {
        let trimmed = collection.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    if let Ok(env_collection) = env::var("CHROMA_COLLECTION") {
        let trimmed = env_collection.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    DEFAULT_COLLECTION.to_string()
}
