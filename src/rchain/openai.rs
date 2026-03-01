use std::env;

use serde::{Deserialize, Serialize};

use crate::rchain::chat_runtime::{RequestFailure, RetryConfig, send_chat_request_with_retry};
use crate::rchain::provider::{
    AskOptions, AskResponse, ChatMessage, Provider, ProviderError, Usage, api_key_env,
};

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    usage: Option<UsagePayload>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsagePayload {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

pub async fn ask(prompt: &str, model: &str) -> Result<String, ProviderError> {
    let response = ask_messages(&[ChatMessage::user(prompt)], model, AskOptions::default()).await?;
    Ok(response.content)
}

pub async fn ask_messages(
    messages: &[ChatMessage],
    model: &str,
    options: AskOptions,
) -> Result<AskResponse, ProviderError> {
    let provider = Provider::Openai;
    let key_env = api_key_env(provider);
    let api_key =
        env::var(key_env).map_err(|_| ProviderError::MissingApiKey { key_env, provider })?;

    let payload = ChatCompletionRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        temperature: options.temperature,
        max_tokens: options.max_tokens,
    };

    let client = reqwest::Client::new();
    let response = send_chat_request_with_retry(
        &client,
        OPENAI_CHAT_COMPLETIONS_URL,
        &api_key,
        &payload,
        RetryConfig {
            timeout_secs: options.timeout_secs,
            retries: options.retries,
            retry_delay_ms: options.retry_delay_ms,
        },
    )
    .await
    .map_err(|failure| match failure {
        RequestFailure::Request(source) => ProviderError::Request { provider, source },
        RequestFailure::Api { status, body } => ProviderError::Api {
            provider,
            status,
            body,
        },
    })?;

    let body: ChatCompletionResponse = response
        .json()
        .await
        .map_err(|source| ProviderError::Request { provider, source })?;
    let content = body
        .choices
        .first()
        .and_then(|choice| choice.message.content.clone())
        .filter(|content| !content.is_empty())
        .ok_or(ProviderError::EmptyResponse { provider })?;
    let usage = body.usage.map(|usage| Usage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    });

    Ok(AskResponse { content, usage })
}
