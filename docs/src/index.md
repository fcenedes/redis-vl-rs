# RedisVL Rust

`redis-vl` is a Rust implementation of the
[Redis Vector Library](https://github.com/redis/redis-vl-python). It provides
vector search, semantic caching, message history, and routing on top of Redis.

## Current status

The library is pre-release (`0.1.0`). Core schema, index, filter, query,
vectorizer, and extension modules are functional and covered by integration
tests. See the [Parity Matrix](https://github.com/redis/redis-vl-rs/blob/main/PARITY_MATRIX.md)
for detailed coverage against the Python surface.

## Implemented modules

| Module | Description |
| --- | --- |
| `schema` | YAML/JSON index schema parsing with typed field attributes, multi-prefix support |
| `index` | Sync and async search index lifecycle, hybrid search, aggregate queries, multi-vector queries |
| `filter` | Composable filter DSL (Tag, Text, Num, Geo, Timestamp) |
| `query` | Vector, range, text, filter, count, hybrid, aggregate hybrid, multi-vector, and SQL queries |
| `vectorizers` | OpenAI, LiteLLM, Custom, Azure OpenAI, Cohere, VoyageAI, and Mistral vectorizers |
| `rerankers` | Reranker traits and Cohere reranker |
| `extensions` | Embeddings cache, semantic cache, message history, semantic router |

## Prerequisites

- Rust 1.85+
- Redis 8+ or Redis Stack (for Search module)

