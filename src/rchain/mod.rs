//! Lightweight LLM integration helpers.
//!
//! The module contains typed wrappers for chat models, embeddings, tool calls,
//! and multimodal helpers used by CLI commands and experiments.

/// Generic AI response traits and structures.
pub mod ai;
/// Chat model client abstractions.
pub mod chat_models;
pub(crate) mod chat_runtime;
/// Embedding model client abstractions.
pub mod embeddings;
/// Fireworks chat-completions helper functions.
pub mod fireworks;
/// Human/user message helper types.
pub mod human;
/// OpenAI chat-completions helper functions.
pub mod openai;
/// Provider-agnostic chat interfaces and dispatch.
pub mod provider;
/// Tool schema and invocation payload helpers.
pub mod tools;
