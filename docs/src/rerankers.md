# Rerankers

Rerankers take a query and a set of documents and reorder the documents by
relevance to the query. This is useful as a second-stage ranking step after an
initial vector or text search retrieval.

## Traits

The library defines two reranker traits:

- **`Reranker`** – synchronous reranking
- **`AsyncReranker`** – async reranking

Both accept a query string, a list of documents, and configuration options,
and return a `RerankResult` with scored, reordered documents.

## Cohere Reranker

Enable the `rerankers` feature to use `CohereReranker`:

```bash
cargo add redis-vl --features rerankers
```

```rust,no_run
# #[cfg(feature = "rerankers")]
# {
use redis_vl::{CohereReranker, CohereRerankerConfig, Reranker, RerankDoc};

// Uses COHERE_API_KEY environment variable by default
let config = CohereRerankerConfig::from_env().unwrap();
let reranker = CohereReranker::new(config);

let docs = vec![
    RerankDoc::Text("Redis is a fast in-memory database".into()),
    RerankDoc::Text("Python is a programming language".into()),
    RerankDoc::Text("Vector search enables semantic queries".into()),
];

let result = reranker.rerank("vector database", &docs, 2, &[]).unwrap();
println!("Top result: {:?}", result.docs[0]);
# }
```

### Configuration

| Option | Environment variable | Default |
| --- | --- | --- |
| API key | `COHERE_API_KEY` | required |
| Model | — | `rerank-english-v3.0` |

### Structured documents

`RerankDoc` supports both plain text and structured field maps:

```rust,no_run
use redis_vl::RerankDoc;
use std::collections::HashMap;

// Plain text
let doc = RerankDoc::Text("some text".into());

// Structured: rerank using specific fields
let mut fields = HashMap::new();
fields.insert("title".into(), "Redis Vector Library".into());
fields.insert("content".into(), "A Rust implementation of RedisVL".into());
let doc = RerankDoc::Fields(fields);
```

When using structured documents, pass `rank_by` field names to control which
fields are used for scoring.
