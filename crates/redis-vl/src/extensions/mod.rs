//! RedisVL extension modules.
//!
//! This module contains Redis-backed AI extensions:
//!
//! - [`cache`] – [`EmbeddingsCache`](cache::EmbeddingsCache) for deterministic
//!   embedding storage and [`SemanticCache`](cache::SemanticCache) for LLM
//!   response caching with semantic similarity lookup.
//! - [`history`] – [`MessageHistory`](history::MessageHistory) for conversation
//!   storage and [`SemanticMessageHistory`](history::SemanticMessageHistory)
//!   for vector-based semantic recall of past messages.
//! - [`router`] – [`SemanticRouter`](router::SemanticRouter) for classifying
//!   input text against predefined routes using vector similarity.

/// Cache-related extensions: [`EmbeddingsCache`](cache::EmbeddingsCache) and
/// [`SemanticCache`](cache::SemanticCache).
pub mod cache;
/// Conversation history extensions: [`MessageHistory`](history::MessageHistory) and
/// [`SemanticMessageHistory`](history::SemanticMessageHistory).
pub mod history;
/// Semantic routing: [`SemanticRouter`](router::SemanticRouter).
pub mod router;
