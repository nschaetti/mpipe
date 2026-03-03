use std::env;
use std::fmt;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::rchain::{fireworks, openai};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Openai,
    Fireworks,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Fireworks => "fireworks",
        }
    }
}

pub fn endpoint(provider: Provider) -> &'static str {
    match provider {
        Provider::Openai => "https://api.openai.com/v1/chat/completions",
        Provider::Fireworks => "https://api.fireworks.ai/inference/v1/chat/completions",
    }
}

pub fn api_key_env(provider: Provider) -> &'static str {
    match provider {
        Provider::Openai => "OPENAI_API_KEY",
        Provider::Fireworks => "FIREWORKS_API_KEY",
    }
}

pub fn is_api_key_present(provider: Provider) -> bool {
    env::var(api_key_env(provider))
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Simple(String),
    Multi(Vec<ContentPart>),
}

impl MessageContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Simple(text.into())
    }

    pub fn with_image(text: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self::Multi(vec![
            ContentPart::Text {
                text: text.into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: image_url.into(),
                },
            },
        ])
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Simple(s) => s.is_empty(),
            Self::Multi(parts) => parts.is_empty(),
        }
    }

    pub fn text_len(&self) -> usize {
        match self {
            Self::Simple(s) => s.chars().count(),
            Self::Multi(parts) => {
                parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } => text.chars().count(),
                        ContentPart::ImageUrl { .. } => 0,
                    })
                    .sum()
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        #[serde(rename = "image_url")]
        image_url: ImageUrl,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

impl ChatMessage {
    pub fn system(content: impl Into<MessageContent>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<MessageContent>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn user_with_text(text: impl Into<String>) -> Self {
        Self::user(MessageContent::text(text))
    }

    pub fn user_with_text_and_image(text: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self::user(MessageContent::with_image(text, image_url))
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Simple(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Simple(s.to_string())
    }
}

pub fn resolve_image_url(input: &str) -> Result<String, String> {
    let trimmed = input.trim();
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Ok(trimmed.to_string());
    }

    let path = std::path::Path::new(trimmed);
    if !path.exists() {
        return Err(format!("Image file not found: {}", trimmed));
    }

    let mime = if let Some(ext) = path.extension() {
        match ext.to_str().map(|e| e.to_lowercase()).as_deref() {
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("png") => "image/png",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("bmp") => "image/bmp",
            _ => "application/octet-stream",
        }
    } else {
        "application/octet-stream"
    };

    let data = std::fs::read(path)
        .map_err(|e| format!("Failed to read image file: {}", e))?;

    let base64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &data);
    Ok(format!("data:{};base64,{}", mime, base64))
}

#[derive(Debug, Clone, Copy)]
pub struct AskOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub timeout_secs: Option<u64>,
    pub retries: u32,
    pub retry_delay_ms: u64,
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: None,
            timeout_secs: None,
            retries: 0,
            retry_delay_ms: 500,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct AskResponse {
    pub content: String,
    pub usage: Option<Usage>,
}

#[derive(Debug)]
pub enum ProviderError {
    MissingApiKey {
        provider: Provider,
        key_env: &'static str,
    },
    Request {
        provider: Provider,
        source: reqwest::Error,
    },
    Api {
        provider: Provider,
        status: StatusCode,
        body: String,
    },
    EmptyResponse {
        provider: Provider,
    },
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingApiKey { key_env, .. } => {
                write!(f, "{key_env} is not set in the environment")
            }
            Self::Request { provider, source } => {
                write!(f, "{} request failed: {source}", provider.as_str())
            }
            Self::Api {
                provider,
                status,
                body,
            } => write!(f, "{} API error {status}: {body}", provider.as_str()),
            Self::EmptyResponse { provider } => {
                write!(
                    f,
                    "{} response did not contain message content",
                    provider.as_str()
                )
            }
        }
    }
}

impl std::error::Error for ProviderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub async fn ask(
    provider: Provider,
    model: &str,
    messages: &[ChatMessage],
    options: AskOptions,
) -> Result<AskResponse, ProviderError> {
    match provider {
        Provider::Openai => openai::ask_messages(messages, model, options).await,
        Provider::Fireworks => fireworks::ask_messages(messages, model, options).await,
    }
}
