//! RedisVL extension modules.
//!
//! This module contains Redis-backed AI extensions:
//!
//! - `cache` – [`EmbeddingsCache`](crate::EmbeddingsCache) for deterministic
//!   embedding storage and [`SemanticCache`](crate::SemanticCache) for LLM
//!   response caching with semantic similarity lookup.
//! - `history` – [`MessageHistory`](crate::MessageHistory) for conversation
//!   storage and [`SemanticMessageHistory`](crate::SemanticMessageHistory)
//!   for vector-based semantic recall of past messages.
//! - `router` – [`SemanticRouter`](crate::SemanticRouter) for classifying
//!   input text against predefined routes using vector similarity.

/// Cache-related extensions: [`EmbeddingsCache`](crate::EmbeddingsCache) and
/// [`SemanticCache`](crate::SemanticCache).
pub mod cache;
/// Conversation history extensions: [`MessageHistory`](crate::MessageHistory) and
/// [`SemanticMessageHistory`](crate::SemanticMessageHistory).
pub mod history;
/// Semantic routing: [`SemanticRouter`](crate::SemanticRouter).
pub mod router;
