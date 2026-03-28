# Extensions

## Embeddings Cache

`EmbeddingsCache` stores embedding vectors in Redis to avoid redundant
vectorizer calls:

```rust,no_run
use redis_vl::EmbeddingsCache;

let cache = EmbeddingsCache::new("emb-cache", "redis://127.0.0.1:6379");
```

## Semantic Cache

`SemanticCache` provides LLM response caching with semantic similarity lookup:

```rust,no_run
use redis_vl::SemanticCache;
use redis_vl::vectorizers::OpenAITextVectorizer;

// Requires an OpenAI-compatible vectorizer for semantic matching
```

## Message History

`MessageHistory` stores conversation messages in Redis with ordered retrieval
and role-based filtering:

```rust,no_run
use redis_vl::{MessageHistory, Message, MessageRole};

let history = MessageHistory::new("session-1", "redis://127.0.0.1:6379");
```

## Semantic Message History

`SemanticMessageHistory` extends `MessageHistory` with vector-based semantic
recall of past messages.

## Semantic Router

`SemanticRouter` classifies input text against predefined routes using vector
similarity:

```rust,no_run
use redis_vl::{SemanticRouter, Route, RoutingConfig};

let routes = vec![
    Route::new("greeting", &["hello", "hi", "hey"]),
    Route::new("farewell", &["goodbye", "bye", "see you"]),
];
```

> **Note:** All extensions require a running Redis instance. Semantic extensions
> additionally require a configured vectorizer for embedding generation.

### Default vectorizer (zero-config)

When the `hf-local` feature is enabled, `SemanticCache`, `SemanticMessageHistory`,
and `SemanticRouter` provide a `with_default_vectorizer()` convenience method
that automatically uses `HuggingFaceTextVectorizer` with the default model
(`AllMiniLML6V2`). This is useful for quick prototyping without configuring an
external embedding API.

