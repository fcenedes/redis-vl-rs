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

### Planned (not yet implemented)

| Provider | Feature flag | Status |
| --- | --- | --- |
| Vertex AI | `vertex-ai` | *Deferred* |
| AWS Bedrock | `bedrock` | *Deferred* |
| HuggingFace local | `hf-local` | *Deferred* |
| Anthropic | `anthropic` | *Deferred* |

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
