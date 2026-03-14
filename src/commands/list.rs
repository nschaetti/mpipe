use std::env;

use chromadb::collection::GetOptions;
use clap::Args;
use serde::Serialize;

use crate::commands::chroma::ChromaConnectArgs;

const DEFAULT_COLLECTION: &str = "mpipe";

#[derive(Debug, Args, Clone)]
pub struct ListArgs {
    #[arg(long)]
    collection: Option<String>,

    #[arg(long, default_value_t = 20)]
    limit: usize,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    #[arg(long)]
    json: bool,

    #[command(flatten)]
    chroma: ChromaConnectArgs,
}

#[derive(Debug, Serialize)]
struct ListedEntry {
    id: String,
    source: Option<String>,
    chunk_index: Option<usize>,
    chunk_count: Option<usize>,
    document: Option<String>,
}

pub async fn run(args: ListArgs) -> Result<(), String> {
    if args.limit == 0 {
        return Err("--limit must be > 0".to_string());
    }

    let collection_name = resolve_collection_name(args.collection.as_deref());
    let (client, _local_chroma) = crate::commands::chroma::connect(&args.chroma).await?;
    let collection = client
        .get_collection(&collection_name)
        .await
        .map_err(|err| format!("Failed to open collection '{collection_name}': {err}"))?;

    let result = collection
        .get(GetOptions {
            ids: Vec::new(),
            where_metadata: None,
            limit: Some(args.limit),
            offset: Some(args.offset),
            where_document: None,
            include: Some(vec!["metadatas".to_string(), "documents".to_string()]),
        })
        .await
        .map_err(|err| format!("Failed to list collection '{collection_name}': {err}"))?;

    let mut entries = Vec::with_capacity(result.ids.len());
    for idx in 0..result.ids.len() {
        let metadata = result
            .metadatas
            .as_ref()
            .and_then(|all| all.get(idx))
            .and_then(|item| item.as_ref());
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
        let chunk_count = metadata.and_then(|m| {
            m.get("chunk_count")
                .and_then(|value| value.as_u64())
                .map(|v| v as usize)
        });
        let document = result
            .documents
            .as_ref()
            .and_then(|all| all.get(idx))
            .and_then(|item| item.clone());

        entries.push(ListedEntry {
            id: result.ids[idx].clone(),
            source,
            chunk_index,
            chunk_count,
            document,
        });
    }

    if args.json {
        let payload = serde_json::to_string(&entries)
            .map_err(|err| format!("Failed to serialize list output: {err}"))?;
        println!("{payload}");
        return Ok(());
    }

    if entries.is_empty() {
        println!("no entries in collection '{collection_name}'");
        return Ok(());
    }

    for entry in entries {
        let source = entry.source.unwrap_or_else(|| "unknown".to_string());
        let chunk = match (entry.chunk_index, entry.chunk_count) {
            (Some(index), Some(count)) => format!("{}/{}", index + 1, count),
            _ => "-".to_string(),
        };
        let preview = entry
            .document
            .as_deref()
            .map(compact_preview)
            .unwrap_or_else(|| "".to_string());
        println!(
            "{}\tsource={}\tchunk={}\t{}",
            entry.id, source, chunk, preview
        );
    }

    Ok(())
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

fn compact_preview(input: &str) -> String {
    let compact = input.split_whitespace().collect::<Vec<_>>().join(" ");
    let max_chars = 120;
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let truncated = compact.chars().take(max_chars).collect::<String>();
    format!("{truncated}...")
}
