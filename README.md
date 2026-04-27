# redis-vl

Rust implementation of the [Redis Vector Library](https://github.com/redis/redis-vl-python),
providing vector search, semantic caching, message history, and routing on top
of [Redis](https://redis.io/).

> **Status**: pre-release (`0.1.1`). The library is functional for core
> workflows but has not yet reached full parity with the Python `redisvl`
> package. See the [Parity Matrix](PARITY_MATRIX.md) for current coverage.

## Features

- **Schema** – define index schemas in YAML or JSON with typed field attributes
  (Tag, Text, Numeric, Geo, Timestamp, Vector), stopwords, multi-prefix support,
  and Hash/JSON storage selection.
- **Search Index** – sync and async index lifecycle: create, delete, load,
  fetch, search, query, batch operations, pagination, hybrid search, aggregate
  queries, multi-vector queries, and `from_existing`.
- **Filters** – composable filter DSL: `Tag`, `Text`, `Num`, `Geo`,
  `GeoRadius`, `Timestamp` with boolean composition (`&`, `|`, `!`).
- **Queries** – `VectorQuery`, `VectorRangeQuery`, `TextQuery`, `FilterQuery`,
  `CountQuery`, `HybridQuery` (Redis 8.4+ `FT.HYBRID`),
  `AggregateHybridQuery` (`FT.AGGREGATE`), and `MultiVectorQuery`.
- **SQL Queries** – `SQLQuery` behind the `sql` feature flag: translates SQL
  `SELECT` statements to Redis Search queries with `WHERE`, `ORDER BY`,
  `LIMIT`/`OFFSET`, field projection, aggregate functions
  (`COUNT`, `SUM`, `AVG`, `GROUP BY`), vector search functions
  (`vector_distance()`, `cosine_distance()`), and geo functions
  (`geo_distance()` in WHERE and SELECT clauses).
- **Vectorizers** – `OpenAITextVectorizer`, `LiteLLMTextVectorizer`,
  `CustomTextVectorizer`, `AzureOpenAITextVectorizer`, `CohereTextVectorizer`,
  `VoyageAITextVectorizer`, `MistralAITextVectorizer`, `VertexAITextVectorizer`,
  `BedrockTextVectorizer`, `AnthropicTextVectorizer` (Voyage AI-backed), and
  `HuggingFaceTextVectorizer` (local ONNX via `fastembed`).
- **Rerankers** – `CohereReranker` behind the `rerankers` feature flag with
  sync and async support.
- **Extensions** – `EmbeddingsCache`, `SemanticCache`, `MessageHistory`,
  `SemanticMessageHistory`, and `SemanticRouter`, all Redis-backed.
- **CLI** (`rvl`) – `version`, `index create/delete/destroy/info/listall`, and
  `stats` commands.
- **Benchmarks** – Criterion micro-benchmarks for schema parsing, filter
  rendering, query building, and Redis-backed index/search/cache operations.

### Not yet implemented

- SQL date functions (`YEAR()`), `IS NULL`/`IS NOT NULL`, `HAVING`
- Richer CLI command/flag parity (`load`, query commands)

## Quick start

Add `redis-vl` to your project:

```bash
cargo add redis-vl
```

To use only the core library without vectorizer dependencies:

```bash
cargo add redis-vl --no-default-features
```

### Defining a schema (YAML)

```yaml
index:
  name: my-index
  prefix: doc
  storage_type: hash
fields:
  - name: title
    type: tag
  - name: content
    type: text
  - name: score
    type: numeric
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 128
      distance_metric: cosine
      datatype: float32
```

### Creating an index and loading data

```rust,no_run
use redis_vl::{IndexSchema, SearchIndex};
use serde_json::json;

let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
index.create().unwrap();

let docs = vec![
    json!({"id": "doc:1", "title": "first", "content": "hello world", "score": 42, "embedding": vec![0.1_f32; 128]}),
];
index.load(&docs, "id", None).unwrap();
```

### Running a vector query

```rust,no_run
use redis_vl::{SearchIndex, IndexSchema, Vector, VectorQuery};

let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");

let vector = Vector::new(&[0.1_f32; 128] as &[f32]);
let query = VectorQuery::new(vector, "embedding", 5)
    .with_return_fields(["title", "score"]);
let result = index.search(&query).unwrap();

println!("Found {} documents", result.total);
for doc in &result.docs {
    println!("  {} (score: {})", doc["title"], doc["vector_distance"]);
}
```

### Composing filters

```rust,no_run
use redis_vl::filter::{Tag, Num, Text};

// Combine with & (AND), | (OR), and ! (NOT)
let filter = Tag::new("color").eq("red")
    & Num::new("price").between(10.0, 100.0, redis_vl::BetweenInclusivity::Both);

let text_filter = Text::new("description").eq("premium")
    | Text::new("description").eq("luxury");
```

### Message history

```rust,no_run
use redis_vl::{MessageHistory, Message, MessageRole};

let history = MessageHistory::new("session-1", "redis://127.0.0.1:6379");
history.add_message(Message::new(MessageRole::User, "Hello!")).unwrap();
history.add_message(Message::new(MessageRole::Llm, "Hi there!")).unwrap();

let recent = history.get_recent(10, None).unwrap();
for msg in &recent {
    println!("[{:?}] {}", msg.role, msg.content);
}
```

### Semantic router

```rust,no_run
use redis_vl::{SemanticRouter, Route, RoutingConfig};

let routes = vec![
    Route::new("greeting", vec!["hello".into(), "hi".into(), "hey".into()]),
    Route::new("farewell", vec!["goodbye".into(), "bye".into(), "see you".into()]),
];

// Create with a vectorizer:
// let router = SemanticRouter::new(vectorizer, routes, "my-router", "redis://...", RoutingConfig::default());
// let result = router.route(Some("howdy!"), None).unwrap();
```

## Redis 8.4+ hybrid search

Redis 8.4 introduces `FT.HYBRID` for combined text + vector search:

```rust,no_run
use redis_vl::query::{HybridQuery, HybridCombinationMethod, Vector};

let query = HybridQuery::new(
    "medical professional", "description",
    Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding",
)
.with_num_results(10)
.with_combination_method(HybridCombinationMethod::Rrf)
.with_return_fields(["user", "age", "job"]);
```

> **Note:** Hybrid and multi-vector queries require Redis 8.4+.
> See the [user guide](docs/src/redis-84.md) for `AggregateHybridQuery`
> and `MultiVectorQuery` details.

## CLI

Install the `rvl` binary:

```bash
cargo install --path crates/rvl
```

```bash
rvl version
rvl index create --schema schema.yaml
rvl index info --schema schema.yaml
rvl index listall --schema schema.yaml
rvl index delete --schema schema.yaml
rvl index destroy --schema schema.yaml   # alias for delete
rvl stats --schema schema.yaml
```

Set `REDIS_URL` or pass `--redis-url` to override the default
`redis://127.0.0.1:6379`.

## Feature flags

| Flag | Default | Description |
| --- | --- | --- |
| `openai` | ✓ | OpenAI-compatible vectorizer |
| `litellm` | ✓ | LiteLLM vectorizer (requires `openai`) |
| `azure-openai` | | Azure OpenAI vectorizer |
| `cohere` | | Cohere vectorizer |
| `voyageai` | | VoyageAI vectorizer |
| `mistral` | | Mistral vectorizer |
| `vertex-ai` | | Google Vertex AI vectorizer |
| `bedrock` | | AWS Bedrock vectorizer |
| `anthropic` | | Anthropic adapter (Voyage AI-backed; requires `voyageai`) |
| `hf-local` | | HuggingFace local ONNX embedding via `fastembed` |
| `sql` | | SQL query support (`SQLQuery`) |
| `rerankers` | | Reranker support (`CohereReranker`) |

## Examples

See the [`examples/`](examples/) directory for runnable code samples:

| Example | Description |
| --- | --- |
| `schema_basics` | Parse and validate index schemas from YAML |
| `filter_basics` | Build and compose filter expressions |
| `vector_search` | Create an index, load data, and run vector queries |
| `semantic_cache_basics` | Set up a semantic LLM response cache |
| `message_history_basics` | Store and retrieve conversation messages |
| `semantic_router_basics` | Route text to predefined categories |
| `sql_query_basics` | Translate SQL to Redis Search queries *(requires `sql` feature)* |

```bash
cargo run -p redis-vl --example schema_basics
cargo run -p redis-vl --example vector_search        # requires Redis
cargo run -p redis-vl --features sql --example sql_query_basics
```

## Benchmarks

Criterion micro-benchmarks cover schema parsing, filter rendering, query
building, and Redis-backed operations (index lifecycle, search, cache, history).

```bash
cargo bench -p redis-vl                                # pure-Rust benchmarks
REDISVL_RUN_INTEGRATION=1 cargo bench -p redis-vl     # includes Redis-backed benchmarks
```

See [`benches/README.md`](benches/README.md) for the full benchmark inventory.

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
```

Integration tests require a running Redis instance with the Search module
(Redis 8+ or Redis Stack):

```bash
REDISVL_RUN_INTEGRATION=1 cargo test --workspace
```

Redis 8.4+ hybrid/aggregate/multi-vector tests additionally require a Redis 8.4
server.

## Documentation

- **[API Reference (docs.rs)](https://docs.rs/redis-vl)** – auto-generated Rustdoc
- **[User Guide (mdBook)](docs/)** – getting started, schema, queries, extensions, CLI
- **[Parity Matrix](PARITY_MATRIX.md)** – feature-level tracking against Python RedisVL
- **[Publishing Guide](PUBLISHING.md)** – crates.io, docs, and release workflow notes

## License

MIT - see the LICENSE file for details.
