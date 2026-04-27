# Vectorizers

Vectorizers convert text into embedding vectors for use with vector search,
semantic caching, and routing. The library provides a `Vectorizer` trait (sync)
and `AsyncVectorizer` trait (async) with several provider implementations.

## Available providers

| Provider | Feature flag | Type | Description |
| --- | --- | --- | --- |
| OpenAI | `openai` (default) | `OpenAITextVectorizer` | OpenAI embeddings API |
| LiteLLM | `litellm` (default) | `LiteLLMTextVectorizer` | LiteLLM-compatible proxy (uses OpenAI transport) |
| Custom | *(always available)* | `CustomTextVectorizer` | User-provided embedding function |
| Azure OpenAI | `azure-openai` | `AzureOpenAITextVectorizer` | Azure-hosted OpenAI embeddings |
| Cohere | `cohere` | `CohereTextVectorizer` | Cohere embed API |
| VoyageAI | `voyageai` | `VoyageAITextVectorizer` | VoyageAI embeddings API |
| Mistral | `mistral` | `MistralAITextVectorizer` | Mistral embeddings API |
| Vertex AI | `vertex-ai` | `VertexAITextVectorizer` | Google Vertex AI predict API |
| AWS Bedrock | `bedrock` | `BedrockTextVectorizer` | Amazon Bedrock Runtime `InvokeModel` API |
| Anthropic | `anthropic` | `AnthropicTextVectorizer` | Voyage AI-backed adapter (Anthropic recommends Voyage AI for embeddings) |
| HuggingFace local | `hf-local` | `HuggingFaceTextVectorizer` | Local ONNX embedding via `fastembed` — no external API required |

## The Vectorizer trait

All vectorizers implement `Vectorizer` for sync usage and optionally
`AsyncVectorizer` for async usage:

```rust,no_run
use redis_vl::vectorizers::Vectorizer;

// Sync: embed a single text
// let embedding: Vec<f32> = vectorizer.embed("hello world")?;

// Sync: embed multiple texts
// let embeddings: Vec<Vec<f32>> = vectorizer.embed_many(&["hello", "world"])?;
```

## OpenAI vectorizer

```rust,no_run
use redis_vl::OpenAITextVectorizer;

// Uses OPENAI_API_KEY environment variable by default
let vectorizer = OpenAITextVectorizer::default();

// Or with explicit configuration
// let vectorizer = OpenAITextVectorizer::new(OpenAICompatibleConfig { ... });
```

## LiteLLM vectorizer

```rust,no_run
use redis_vl::LiteLLMTextVectorizer;

// Connects to a LiteLLM proxy using the OpenAI-compatible transport
// let vectorizer = LiteLLMTextVectorizer::new(config);
```

## Custom vectorizer

The `CustomTextVectorizer` wraps any user-provided function:

```rust,no_run
use redis_vl::CustomTextVectorizer;

let vectorizer = CustomTextVectorizer::new(|texts: &[&str]| {
    // Your embedding logic here
    Ok(texts.iter().map(|_| vec![0.0_f32; 128]).collect())
});
```

## Azure OpenAI vectorizer

```rust,no_run
# #[cfg(feature = "azure-openai")]
# {
use redis_vl::{AzureOpenAITextVectorizer, AzureOpenAIConfig};

// Requires AZURE_OPENAI_API_KEY or explicit config
// let vectorizer = AzureOpenAITextVectorizer::new(config);
# }
```

## Cohere vectorizer

```rust,no_run
# #[cfg(feature = "cohere")]
# {
use redis_vl::{CohereTextVectorizer, CohereConfig};

// Requires COHERE_API_KEY or explicit config
// let vectorizer = CohereTextVectorizer::new(config);
# }
```

## VoyageAI vectorizer

```rust,no_run
# #[cfg(feature = "voyageai")]
# {
use redis_vl::{VoyageAITextVectorizer, VoyageAIConfig};

// Requires VOYAGEAI_API_KEY or explicit config
// let vectorizer = VoyageAITextVectorizer::new(config);
# }
```

## Mistral vectorizer

```rust,no_run
# #[cfg(feature = "mistral")]
# {
use redis_vl::{MistralAITextVectorizer, MistralConfig};

// Requires MISTRAL_API_KEY or explicit config
// let vectorizer = MistralAITextVectorizer::new(config);
# }
```

## Anthropic vectorizer

Anthropic does not provide its own embedding model. Instead, it
[recommends Voyage AI](https://docs.anthropic.com/en/docs/build-with-claude/embeddings)
for embedding tasks. The `AnthropicTextVectorizer` wraps `VoyageAITextVectorizer`
with Anthropic-oriented defaults (default model: `voyage-3-large`).

```rust,no_run
# #[cfg(feature = "anthropic")]
# {
use redis_vl::{AnthropicTextVectorizer, AnthropicConfig};

// Uses VOYAGE_API_KEY env var with the Anthropic-recommended default model
let config = AnthropicConfig::from_env(Some("document".into())).unwrap();
let vectorizer = AnthropicTextVectorizer::new(config);
# }
```

## HuggingFace local vectorizer

The `HuggingFaceTextVectorizer` runs embedding models locally using ONNX
Runtime via the [`fastembed`](https://crates.io/crates/fastembed) crate. No
external API or API key is required — models are downloaded from HuggingFace
Hub on first use and cached on disk.

```rust,no_run
# #[cfg(feature = "hf-local")]
# {
use redis_vl::vectorizers::HuggingFaceTextVectorizer;

// Uses the default model (AllMiniLML6V2)
let vectorizer = HuggingFaceTextVectorizer::new(Default::default()).unwrap();
// let embedding = vectorizer.embed("Hello, world!").unwrap();
# }
```

> **Note:** `HuggingFaceTextVectorizer` implements `Vectorizer` (sync) only.
> For async use cases, wrap calls with `tokio::task::spawn_blocking`.

## Using vectorizers with extensions

Semantic extensions accept any type implementing `Vectorizer`:

```rust,no_run
use redis_vl::{SemanticCache, CacheConfig};

let cache = SemanticCache::new(
    CacheConfig::new("cache", "redis://127.0.0.1:6379"),
    128, 0.2,
);
// cache = cache.with_vectorizer(my_vectorizer);
```

When the `hf-local` feature is enabled, semantic extensions also offer a
`with_default_vectorizer()` convenience method that automatically uses
`HuggingFaceTextVectorizer` with the default model — useful for quick
prototyping without configuring an external API:

```rust,no_run
# #[cfg(feature = "hf-local")]
# {
use redis_vl::{SemanticCache, CacheConfig};

let cache = SemanticCache::new(
    CacheConfig::new("cache", "redis://127.0.0.1:6379"),
    384, 0.2,
).with_default_vectorizer().unwrap();
# }
```
