use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::{Path, PathBuf};

use chromadb::collection::CollectionEntries;
use clap::Args;
use serde_json::{Map, Value};

use crate::commands::chroma::{self, ChromaConnectArgs};
use crate::rchain::embeddings::{EmbeddingProvider, embed_chunks_with_provider};

const DEFAULT_COLLECTION: &str = "mpipe";
const DEFAULT_CHUNK_SIZE: usize = 1000;
const DEFAULT_CHUNK_OVERLAP: usize = 200;

#[derive(Debug, Args, Clone)]
pub struct IndexArgs {
    #[arg(long)]
    file: Option<PathBuf>,

    #[arg(long)]
    document: Option<String>,

    #[arg(long = "embedding-model")]
    embedding_model: Option<String>,

    #[arg(long, value_name = "SIZE")]
    chunk_size: Option<usize>,

    #[arg(long, value_name = "SIZE")]
    chunk_overlap: Option<usize>,

    #[arg(long)]
    collection: Option<String>,

    #[command(flatten)]
    chroma: ChromaConnectArgs,

    #[arg(long = "source")]
    source: Option<String>,

    #[arg(long = "id-prefix")]
    id_prefix: Option<String>,

    #[arg(long = "metadata", value_name = "KEY=VALUE")]
    metadata: Vec<String>,

    #[arg(long = "metadata-json")]
    metadata_json: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct Chunk {
    text: String,
    char_start: usize,
    char_end: usize,
}

pub async fn run(args: IndexArgs) -> Result<(), String> {
    validate_inputs(&args)?;

    let chunk_size = args.chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);
    let chunk_overlap = args.chunk_overlap.unwrap_or(DEFAULT_CHUNK_OVERLAP);
    if chunk_size == 0 {
        return Err("--chunk-size must be > 0".to_string());
    }
    if chunk_overlap >= chunk_size {
        return Err("--chunk-overlap must be < --chunk-size".to_string());
    }

    let document = read_document(&args)?;
    let chunks = split_text(&document, chunk_size, chunk_overlap);
    if chunks.is_empty() {
        return Err("Document is empty after trimming.".to_string());
    }

    let embeddings_from_stdin = read_embeddings_from_stdin()?;
    if embeddings_from_stdin.is_some() && args.embedding_model.is_some() {
        // Embeddings already provided; model is ignored.
    }

    let embeddings = if let Some(vectors) = embeddings_from_stdin {
        validate_embeddings_count(&vectors, chunks.len())?;
        validate_embeddings_dimensions(&vectors)?;
        vectors
    } else {
        let model = args.embedding_model.as_ref().ok_or_else(|| {
            "Missing --embedding-model (required when stdin embeddings are not provided)."
                .to_string()
        })?;
        embed_chunks(model, &chunks).await?
    };
    let source = resolve_source(&args)?;

    let collection_name = resolve_collection_name(args.collection.as_deref());
    let (client, _local_chroma) = chroma::connect(&args.chroma).await?;

    let collection = client
        .get_or_create_collection(&collection_name, None)
        .await
        .map_err(|err| format!("Failed to open collection '{collection_name}': {err}"))?;

    let mut base_metadata = load_metadata_json(args.metadata_json.as_deref())?;
    let overrides = parse_metadata_overrides(&args.metadata)?;
    apply_metadata_overrides(&mut base_metadata, overrides);

    let chunk_count = chunks.len();
    let ids = build_ids(args.id_prefix.as_deref(), &args, chunk_count)?;
    let documents = chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<Vec<_>>();
    let metadatas = build_chunk_metadatas(&chunks, &base_metadata, &source, chunk_count);

    let collection_entries = CollectionEntries {
        ids: ids.iter().map(|id| id.as_str()).collect(),
        metadatas: Some(metadatas),
        documents: Some(documents),
        embeddings: Some(embeddings),
    };

    collection
        .upsert(collection_entries, None)
        .await
        .map_err(|err| format!("Failed to upsert into collection '{collection_name}': {err}"))?;

    println!(
        "indexed {} chunks into collection '{}'",
        chunk_count, collection_name
    );
    Ok(())
}

fn validate_inputs(args: &IndexArgs) -> Result<(), String> {
    if args.file.is_some() && args.document.is_some() {
        return Err("Use either --file or --document, not both.".to_string());
    }
    if args.file.is_none() && args.document.is_none() {
        return Err("Missing input: provide --file or --document.".to_string());
    }
    if args.document.is_some() && args.source.is_none() {
        return Err("--source is required when using --document.".to_string());
    }
    if let Some(source) = &args.source
        && source.trim().is_empty()
    {
        return Err("--source cannot be empty.".to_string());
    }
    Ok(())
}

fn resolve_source(args: &IndexArgs) -> Result<String, String> {
    if let Some(source) = &args.source {
        return Ok(source.trim().to_string());
    }
    if let Some(path) = &args.file {
        return Ok(path.display().to_string());
    }
    Err("--source is required when input is not a file.".to_string())
}

fn read_document(args: &IndexArgs) -> Result<String, String> {
    if let Some(path) = &args.file {
        return read_file(path);
    }

    let document = args
        .document
        .as_ref()
        .map(|value| value.trim())
        .unwrap_or("");

    if document.is_empty() {
        return Err("Provided --document is empty.".to_string());
    }

    Ok(document.to_string())
}

fn read_file(path: &Path) -> Result<String, String> {
    let contents = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read file '{}': {err}", path.display()))?;
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        return Err(format!("File '{}' is empty.", path.display()));
    }
    Ok(contents)
}

fn read_embeddings_from_stdin() -> Result<Option<Vec<Vec<f32>>>, String> {
    if io::stdin().is_terminal() {
        return Ok(None);
    }

    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .map_err(|err| format!("Failed to read stdin embeddings: {err}"))?;

    if buffer.trim().is_empty() {
        return Err("Stdin embeddings are empty.".to_string());
    }

    let mut embeddings = Vec::new();
    for (line_idx, raw_line) in buffer.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let mut vector = Vec::new();
        for (value_idx, value) in line.split(',').enumerate() {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parsed = trimmed.parse::<f32>().map_err(|_| {
                format!(
                    "Invalid float at line {} position {}: '{}'",
                    line_idx + 1,
                    value_idx + 1,
                    trimmed
                )
            })?;
            vector.push(parsed);
        }
        if vector.is_empty() {
            return Err(format!(
                "No floats parsed for embeddings line {}.",
                line_idx + 1
            ));
        }
        embeddings.push(vector);
    }

    if embeddings.is_empty() {
        return Err("No embeddings parsed from stdin.".to_string());
    }

    Ok(Some(embeddings))
}

fn validate_embeddings_count(embeddings: &[Vec<f32>], chunk_count: usize) -> Result<(), String> {
    if embeddings.len() != chunk_count {
        return Err(format!(
            "Embeddings count ({}) does not match chunk count ({}).",
            embeddings.len(),
            chunk_count
        ));
    }
    Ok(())
}

fn validate_embeddings_dimensions(embeddings: &[Vec<f32>]) -> Result<(), String> {
    if embeddings.is_empty() {
        return Ok(());
    }

    let expected = embeddings[0].len();
    if expected == 0 {
        return Err("Embeddings cannot be empty vectors.".to_string());
    }
    for (idx, vector) in embeddings.iter().enumerate() {
        if vector.len() != expected {
            return Err(format!(
                "Embedding dimension mismatch at index {} (expected {}, got {}).",
                idx,
                expected,
                vector.len()
            ));
        }
    }
    Ok(())
}

async fn embed_chunks(model: &str, chunks: &[Chunk]) -> Result<Vec<Vec<f32>>, String> {
    let model = model.to_string();
    let chunk_texts = chunks
        .iter()
        .map(|chunk| chunk.text.clone())
        .collect::<Vec<_>>();

    tokio::task::spawn_blocking(move || {
        let embeddings =
            embed_chunks_with_provider(EmbeddingProvider::Fireworks, &model, &chunk_texts)
                .map_err(|err| format!("Failed to embed chunks: {err}"))?
                .into_iter()
                .map(|vector| vector.into_iter().map(|value| value as f32).collect())
                .collect();
        Ok::<_, String>(embeddings)
    })
    .await
    .map_err(|err| format!("Embedding task failed: {err}"))?
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

fn load_metadata_json(path: Option<&Path>) -> Result<Map<String, Value>, String> {
    let Some(path) = path else {
        return Ok(Map::new());
    };
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read metadata JSON '{}': {err}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|err| format!("Failed to parse metadata JSON '{}': {err}", path.display()))?;
    let map = value
        .as_object()
        .ok_or_else(|| format!("Metadata JSON '{}' must be a JSON object.", path.display()))?;
    Ok(map.clone())
}

fn parse_metadata_overrides(entries: &[String]) -> Result<Map<String, Value>, String> {
    let mut map = Map::new();
    for entry in entries {
        let (key, value) = parse_metadata_entry(entry)?;
        map.insert(key, Value::String(value));
    }
    Ok(map)
}

fn parse_metadata_entry(entry: &str) -> Result<(String, String), String> {
    let mut parts = entry.splitn(2, '=');
    let key = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("Invalid metadata entry '{entry}'. Expected KEY=VALUE."))?;
    let value = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("Invalid metadata entry '{entry}'. Expected KEY=VALUE."))?;
    Ok((key.to_string(), value.to_string()))
}

fn apply_metadata_overrides(base: &mut Map<String, Value>, overrides: Map<String, Value>) {
    for (key, value) in overrides {
        base.insert(key, value);
    }
}

fn build_ids(
    id_prefix: Option<&str>,
    args: &IndexArgs,
    chunk_count: usize,
) -> Result<Vec<String>, String> {
    let prefix = if let Some(prefix) = id_prefix {
        let trimmed = prefix.trim();
        if trimmed.is_empty() {
            return Err("--id-prefix cannot be empty".to_string());
        }
        trimmed.to_string()
    } else if let Some(path) = &args.file {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.to_string())
            .unwrap_or_else(|| "file".to_string())
    } else {
        "document".to_string()
    };

    let mut ids = Vec::with_capacity(chunk_count);
    for index in 0..chunk_count {
        ids.push(format!("{prefix}-{index}"));
    }
    Ok(ids)
}

fn build_chunk_metadatas(
    chunks: &[Chunk],
    base: &Map<String, Value>,
    source: &str,
    chunk_count: usize,
) -> Vec<Map<String, Value>> {
    chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            let mut metadata = base.clone();
            metadata.insert("source".to_string(), Value::String(source.to_string()));
            metadata.insert("chunk_index".to_string(), Value::from(index));
            metadata.insert("chunk_count".to_string(), Value::from(chunk_count));
            metadata.insert("char_start".to_string(), Value::from(chunk.char_start));
            metadata.insert("char_end".to_string(), Value::from(chunk.char_end));
            metadata
        })
        .collect()
}

fn split_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < len {
        let mut end = (start + chunk_size).min(len);
        if end < len {
            if let Some(split_index) = (start..end)
                .rev()
                .find(|&idx| chars[idx].is_whitespace() && idx > start)
            {
                end = split_index;
            }
        }

        if end == start {
            end = (start + chunk_size).min(len);
        }

        let text = chars[start..end].iter().collect::<String>();
        chunks.push(Chunk {
            text,
            char_start: start,
            char_end: end,
        });

        if end == len {
            break;
        }

        let next_start = end.saturating_sub(overlap);
        if next_start <= start {
            start = end;
        } else {
            start = next_start;
        }
    }

    chunks
}
