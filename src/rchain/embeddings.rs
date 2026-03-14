use std::env;
use std::error::Error;

use reqwest::blocking::Client;
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingProvider {
    Openai,
    Fireworks,
}

impl EmbeddingProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Fireworks => "fireworks",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    Paragraph,
    Sentence,
    Token,
}

impl ChunkStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Paragraph => "paragraph",
            Self::Sentence => "sentence",
            Self::Token => "token",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "paragraph" => Some(Self::Paragraph),
            "sentence" => Some(Self::Sentence),
            "token" => Some(Self::Token),
            _ => None,
        }
    }
}

pub struct EmbeddingsConfig {
    pub provider: EmbeddingProvider,
    pub model: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub chunk_strategy: ChunkStrategy,
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::Fireworks,
            model: "accounts/fireworks/models/qwen3-embedding-8b".to_string(),
            chunk_size: 8000,
            chunk_overlap: 10,
            chunk_strategy: ChunkStrategy::Paragraph,
        }
    }
}

pub struct EmbeddingResult {
    pub chunks: Vec<String>,
    pub embeddings: Vec<Vec<f64>>,
    pub model: String,
    pub provider: String,
}

pub fn chunk_text(
    text: &str,
    strategy: ChunkStrategy,
    chunk_size: usize,
    overlap_percent: usize,
) -> Vec<String> {
    match strategy {
        ChunkStrategy::Paragraph => chunk_by_paragraph(text, chunk_size, overlap_percent),
        ChunkStrategy::Sentence => chunk_by_sentence(text, chunk_size, overlap_percent),
        ChunkStrategy::Token => chunk_by_token(text, chunk_size, overlap_percent),
    }
}

fn chunk_by_paragraph(text: &str, chunk_size: usize, overlap_percent: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_size = 0;

    for para in paragraphs {
        let para_size = para.len();
        if current_size + para_size > chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            let _overlap_chars = (chunk_size * overlap_percent) / 100;
            current_chunk = String::new();
            current_size = 0;
        }

        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
            current_size += 2;
        }
        current_chunk.push_str(para);
        current_size += para_size;
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() && !text.trim().is_empty() {
        chunks.push(text.trim().to_string());
    }

    chunks
}

fn chunk_by_sentence(text: &str, chunk_size: usize, overlap_percent: usize) -> Vec<String> {
    let sentence_enders = ['.', '!', '?', '¿', '¡'];
    let mut sentences: Vec<&str> = Vec::new();
    let mut current_start = 0;
    let bytes = text.as_bytes();

    for (i, &byte) in bytes.iter().enumerate() {
        let c = byte as char;
        if sentence_enders.contains(&c) && i + 1 < bytes.len() && bytes[i + 1].is_ascii_whitespace()
        {
            sentences.push(&text[current_start..=i]);
            current_start = i + 1;
            while current_start < bytes.len() && bytes[current_start].is_ascii_whitespace() {
                current_start += 1;
            }
        }
    }

    if current_start < text.len() {
        sentences.push(&text[current_start..]);
    }

    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_size = 0;

    for sentence in sentences {
        let sentence_size = sentence.len();
        if current_size + sentence_size > chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            let _overlap_chars = (chunk_size * overlap_percent) / 100;
            current_chunk = String::new();
            current_size = 0;
        }

        if !current_chunk.is_empty() {
            current_chunk.push(' ');
            current_size += 1;
        }
        current_chunk.push_str(sentence);
        current_size += sentence_size;
    }

    if !current_chunk.trim().is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() && !text.trim().is_empty() {
        chunks.push(text.trim().to_string());
    }

    chunks
}

fn chunk_by_token(text: &str, chunk_size: usize, overlap_percent: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_size = 0;

    for word in words {
        let word_size = word
            .split(|c: char| c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .map(|s| s.chars().count())
            .sum::<usize>()
            .max(1);

        if current_size + word_size > chunk_size && !current_chunk.is_empty() {
            chunks.push(current_chunk.join(" "));
            let overlap_words = (chunk_size * overlap_percent) / 100 / 5;
            current_chunk = current_chunk[current_chunk
                .len()
                .saturating_sub(overlap_words.min(current_chunk.len()))..]
                .to_vec();
            current_size = current_chunk.iter().map(|w: &&str| w.len()).sum::<usize>();
        }

        current_chunk.push(word);
        current_size += word_size;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join(" "));
    }

    if chunks.is_empty() && !text.trim().is_empty() {
        chunks.push(text.trim().to_string());
    }

    chunks
}

pub fn embed_texts(
    config: &EmbeddingsConfig,
    texts: &[String],
) -> Result<EmbeddingResult, Box<dyn Error + Send + Sync>> {
    let all_chunks: Vec<String> = texts
        .iter()
        .flat_map(|text| {
            chunk_text(
                text,
                config.chunk_strategy,
                config.chunk_size,
                config.chunk_overlap,
            )
        })
        .collect();

    if all_chunks.is_empty() {
        return Ok(EmbeddingResult {
            chunks: vec![],
            embeddings: vec![],
            model: config.model.clone(),
            provider: config.provider.as_str().to_string(),
        });
    }

    let embeddings = embed_chunks(config, &all_chunks)?;

    Ok(EmbeddingResult {
        chunks: all_chunks,
        embeddings,
        model: config.model.clone(),
        provider: config.provider.as_str().to_string(),
    })
}

fn embed_chunks(
    config: &EmbeddingsConfig,
    chunks: &[String],
) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    match config.provider {
        EmbeddingProvider::Openai => embed_chunks_openai(&config.model, &config.api_key()?, chunks),
        EmbeddingProvider::Fireworks => {
            embed_chunks_fireworks(&config.model, &config.api_key()?, chunks)
        }
    }
}

pub fn embed_chunks_with_provider(
    provider: EmbeddingProvider,
    model: &str,
    chunks: &[String],
) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let config = EmbeddingsConfig {
        provider,
        model: model.to_string(),
        ..EmbeddingsConfig::default()
    };
    embed_chunks(&config, chunks)
}

impl EmbeddingsConfig {
    fn api_key(&self) -> Result<String, Box<dyn Error + Send + Sync>> {
        let env_key = match self.provider {
            EmbeddingProvider::Openai => "OPENAI_API_KEY",
            EmbeddingProvider::Fireworks => "FIREWORKS_API_KEY",
        };
        env::var(env_key).map_err(|_| format!("{env_key} is not set in the environment").into())
    }

    #[allow(dead_code)]
    fn endpoint(&self) -> &'static str {
        match self.provider {
            EmbeddingProvider::Openai => "https://api.openai.com/v1/embeddings",
            EmbeddingProvider::Fireworks => "https://api.fireworks.ai/inference/v1/embeddings",
        }
    }
}

fn embed_chunks_fireworks(
    model: &str,
    api_key: &str,
    chunks: &[String],
) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let base_url = "https://api.fireworks.ai/inference/v1/embeddings";

    let mut embeddings = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        let payload = json!({
            "model": model,
            "input": chunk,
        });

        let response = client
            .post(base_url)
            .bearer_auth(api_key)
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(format!("Fireworks API error {status}: {body}").into());
        }

        let body: serde_json::Value = response.json()?;
        let embedding = body["data"][0]["embedding"]
            .as_array()
            .ok_or("Missing embedding data from Fireworks API")?;

        let vector: Vec<f64> = embedding.iter().filter_map(|v| v.as_f64()).collect();

        embeddings.push(vector);
    }

    Ok(embeddings)
}

fn embed_chunks_openai(
    model: &str,
    api_key: &str,
    chunks: &[String],
) -> Result<Vec<Vec<f64>>, Box<dyn Error + Send + Sync>> {
    let client = Client::new();
    let base_url = "https://api.openai.com/v1/embeddings";

    let payload = json!({
        "model": model,
        "input": chunks,
    });

    let response = client
        .post(base_url)
        .bearer_auth(api_key)
        .json(&payload)
        .send()?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("OpenAI API error {status}: {body}").into());
    }

    let body: serde_json::Value = response.json()?;
    let data = body["data"]
        .as_array()
        .ok_or("Missing embedding data from OpenAI API")?;

    let mut embeddings: Vec<Vec<f64>> = Vec::with_capacity(data.len());

    for item in data {
        let embedding = item["embedding"]
            .as_array()
            .ok_or("Missing embedding array")?;

        let vector: Vec<f64> = embedding.iter().filter_map(|v| v.as_f64()).collect();

        embeddings.push(vector);
    }

    embeddings.sort_by_key(|e| e.len());

    Ok(embeddings)
}
