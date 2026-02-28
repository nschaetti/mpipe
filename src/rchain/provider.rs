use std::env;
use std::fmt;

use reqwest::StatusCode;
use serde::Serialize;

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

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
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
    MissingApiKey { provider: Provider, key_env: &'static str },
    Request {
        provider: Provider,
        source: reqwest::Error,
    },
    Api {
        provider: Provider,
        status: StatusCode,
        body: String,
    },
    EmptyResponse { provider: Provider },
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
                write!(f, "{} response did not contain message content", provider.as_str())
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
