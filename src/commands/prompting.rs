use std::io::{self, IsTerminal, Read};

use crate::rchain::provider::{ChatMessage, MessageContent};

#[derive(Debug)]
pub struct PromptInput {
    pub text: String,
    pub source: PromptSource,
}

#[derive(Debug, Clone, Copy)]
pub enum PromptSource {
    Argument,
    Stdin,
}

impl PromptSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Argument => "argument",
            Self::Stdin => "stdin",
        }
    }
}

pub fn non_empty(value: Option<&str>) -> Option<&str> {
    value.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

pub fn build_messages(system: Option<&str>, prompt: &str) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(ChatMessage::system(MessageContent::text(system)));
    }
    messages.push(ChatMessage::user(MessageContent::text(prompt)));
    messages
}

pub fn build_messages_with_image(
    system: Option<&str>,
    prompt: &str,
    image_url: &str,
) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(ChatMessage::system(MessageContent::text(system)));
    }
    messages.push(ChatMessage::user_with_text_and_image(prompt, image_url));
    messages
}

pub fn compose_prompt(
    preprompt: Option<&str>,
    main_prompt: &str,
    postprompt: Option<&str>,
) -> String {
    let mut parts = Vec::new();

    if let Some(pre) = preprompt
        && !pre.trim().is_empty()
    {
        parts.push(pre.to_string());
    }

    parts.push(main_prompt.to_string());

    if let Some(post) = postprompt
        && !post.trim().is_empty()
    {
        parts.push(post.to_string());
    }

    parts.join("\n\n")
}

pub fn resolve_prompt(cli_prompt: Option<String>) -> Result<PromptInput, String> {
    if let Some(prompt) = cli_prompt {
        return Ok(PromptInput {
            text: prompt,
            source: PromptSource::Argument,
        });
    }

    if io::stdin().is_terminal() {
        return Err("No prompt provided. Pass an argument or pipe stdin.".to_string());
    }

    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .map_err(|err| format!("Failed to read stdin: {err}"))?;

    let text = buffer.trim().to_string();
    if text.is_empty() {
        return Err("Prompt is empty.".to_string());
    }

    Ok(PromptInput {
        text,
        source: PromptSource::Stdin,
    })
}
