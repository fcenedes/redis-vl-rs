# Extensions

RedisVL provides several Redis-backed AI extensions for caching, conversation
management, and semantic routing. All extensions require a running Redis
instance.

## Embeddings Cache

`EmbeddingsCache` stores embedding vectors in Redis to avoid redundant
vectorizer calls. Lookups are keyed by content hash and model name.

```rust,no_run
use redis_vl::EmbeddingsCache;

let cache = EmbeddingsCache::new("emb-cache", "redis://127.0.0.1:6379");

// Store an embedding
cache.set("hello world", "text-embedding-3-small", &[0.1_f32; 128]).unwrap();

// Retrieve it later (avoids a vectorizer API call)
if let Some(embedding) = cache.get("hello world", "text-embedding-3-small").unwrap() {
    println!("Cached embedding: {} dims", embedding.len());
}

// Check existence
let exists = cache.exists("hello world", "text-embedding-3-small").unwrap();
```

## Semantic Cache

`SemanticCache` provides LLM response caching with vector similarity lookup.
When a new prompt is semantically close to a cached prompt (within a distance
threshold), the cached response is returned instead of calling the LLM.

```rust,no_run
use redis_vl::{SemanticCache, CacheConfig};

// Create with 128-dimensional vectors and 0.2 distance threshold
let cache = SemanticCache::new(
    CacheConfig::new("llm-cache", "redis://127.0.0.1:6379"),
    128,
    0.2,
);

// Initialize the underlying index
cache.create().unwrap();

// Store a prompt/response pair (requires a configured vectorizer)
// cache.store("What is Redis?", "Redis is an in-memory database.").unwrap();

// Check for semantically similar cached responses
// let hit = cache.check("Tell me about Redis").unwrap();
```

## Message History

`MessageHistory` stores conversation messages in Redis with ordered retrieval
and role-based filtering. Messages are stored as Redis hashes.

```rust,no_run
use redis_vl::{MessageHistory, Message, MessageRole};

let history = MessageHistory::new("session-1", "redis://127.0.0.1:6379");

// Add messages
history.add_message(Message::new(MessageRole::User, "Hello!")).unwrap();
history.add_message(Message::new(MessageRole::Llm, "Hi there!")).unwrap();

// Retrieve recent messages
let recent = history.get_recent(10, None).unwrap();
println!("Got {} messages", recent.len());

// Filter by role
let user_msgs = history.get_recent(10, Some(MessageRole::User)).unwrap();

// Clear the session
history.clear().unwrap();
```

## Semantic Message History

`SemanticMessageHistory` extends `MessageHistory` with vector-based semantic
recall. In addition to ordered retrieval, it finds the most relevant past
messages for a given prompt using vector similarity search.

```rust,no_run
use redis_vl::SemanticMessageHistory;

// Create with 128-dimensional vectors
let history = SemanticMessageHistory::new(
    "semantic-session",
    "redis://127.0.0.1:6379",
    128,
);

// Initialize the index
history.create().unwrap();

// Messages are embedded and stored for semantic retrieval
// history.add_message(Message::new(MessageRole::User, "Tell me about caching")).unwrap();

// Find relevant past messages by semantic similarity
// let relevant = history.get_relevant("How does caching work?", 5).unwrap();
```

## Semantic Router

`SemanticRouter` classifies input text against predefined routes using vector
similarity. Each route has a set of reference utterances; incoming text is
matched against the closest references.

```rust,no_run
use redis_vl::{SemanticRouter, Route, RoutingConfig};

let routes = vec![
    Route::new("greeting", vec!["hello".into(), "hi".into(), "hey there".into()]),
    Route::new("farewell", vec!["goodbye".into(), "bye".into(), "see you later".into()]),
];

let config = RoutingConfig::new("my-router", "redis://127.0.0.1:6379", 128);
let router = SemanticRouter::new(config, routes);

// Initialize the router index
router.create().unwrap();

// Classify input (requires a configured vectorizer)
// let route_match = router.route("hey, how are you?").unwrap();
```

### Serialization

`SemanticRouter` supports serialization for persistence and sharing:

```rust,no_run
# use redis_vl::{SemanticRouter, Route, RoutingConfig};
# let routes = vec![Route::new("greeting", vec!["hello".into()])];
# let config = RoutingConfig::new("r", "redis://127.0.0.1:6379", 128);
# let router = SemanticRouter::new(config, routes);
// Export to JSON
let json_value = router.to_json_value();

// Export to dict (serde_json::Map)
let dict = router.to_dict();

// Reconnect to an existing router index
// let router = SemanticRouter::from_existing("my-router", "redis://127.0.0.1:6379").unwrap();
```

> **Note:** Semantic extensions (`SemanticCache`, `SemanticMessageHistory`,
> `SemanticRouter`) require a configured vectorizer for embedding generation.

## Default vectorizer (zero-config)

When the `hf-local` feature is enabled, `SemanticCache`, `SemanticMessageHistory`,
and `SemanticRouter` provide a `with_default_vectorizer()` convenience method
that automatically uses `HuggingFaceTextVectorizer` with the default model
(`AllMiniLML6V2`). This is useful for quick prototyping without configuring an
external embedding API:

```rust,no_run
# #[cfg(feature = "hf-local")]
# {
use redis_vl::{SemanticCache, CacheConfig};

let cache = SemanticCache::new(
    CacheConfig::new("cache", "redis://127.0.0.1:6379"),
    384, // AllMiniLML6V2 dimension
    0.2,
).with_default_vectorizer().unwrap();
# }
```

