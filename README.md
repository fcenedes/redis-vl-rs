# redis-vl

Rust implementation of the [Redis Vector Library](https://github.com/redis/redis-vl-python),
providing vector search, semantic caching, message history, and routing on top
of [Redis](https://redis.io/).

> **Status**: pre-release (`0.1.0`). The library is functional for core
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
  `GeoRadius`, `Timestamp` with boolean composition.
- **Queries** – `VectorQuery`, `VectorRangeQuery`, `TextQuery`, `FilterQuery`,
  `CountQuery`, `HybridQuery` (generates `FT.HYBRID` for Redis 8.4+),
  `AggregateHybridQuery` (generates `FT.AGGREGATE`), and `MultiVectorQuery`.
- **SQL Queries** – `SQLQuery` behind the `sql` feature flag: translates SQL
  `SELECT` statements to Redis Search queries with `WHERE`, `ORDER BY`,
  `LIMIT`/`OFFSET`, and field projection.
- **Vectorizers** – `OpenAITextVectorizer`, `LiteLLMTextVectorizer`,
  `CustomTextVectorizer`, `AzureOpenAITextVectorizer`, `CohereTextVectorizer`,
  `VoyageAITextVectorizer`, and `MistralAITextVectorizer`.
- **Rerankers** – `CohereReranker` behind the `rerankers` feature flag with
  sync and async support.
- **Extensions** – `EmbeddingsCache`, `SemanticCache`, `MessageHistory`,
  `SemanticMessageHistory`, and `SemanticRouter`, all Redis-backed.
- **CLI** (`rvl`) – `version`, `index create/delete/destroy/info/listall`, and
  `stats` commands.

### Not yet implemented

- Aggregate SQL queries (`COUNT`, `GROUP BY`, vector/geo functions)
- Additional vectorizer providers (Vertex AI, Bedrock, HuggingFace local)
- Redis 8.4 end-to-end integration testing for hybrid/aggregate/multi-vector
- Semantic extension parity (dtype, default vectorizers, from-existing)

## Quick start

Add `redis-vl` to your project:

```bash
cargo add redis-vl
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

let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
index.create().unwrap();
```

### Running a vector query

```rust,no_run
use redis_vl::{SearchIndex, Vector, VectorQuery};

// Assuming `index` is already created and data is loaded:
let vector = Vector::new(&[0.1_f32; 128]);
let query = VectorQuery::new(vector, "embedding", 5);
// let results = index.search(&query).unwrap();
```

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
| `sql` | | SQL query support (`SQLQuery`) |
| `rerankers` | | Reranker support (`CohereReranker`) |
| `vertex-ai` | | Vertex AI adapter (planned) |
| `bedrock` | | AWS Bedrock adapter (planned) |
| `hf-local` | | HuggingFace local adapter (planned) |
| `anthropic` | | Anthropic adapter (planned) |

Flags marked *(planned)* are declared but not yet implemented.

## Development

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

Integration tests require a running Redis instance with the Search module
(Redis 8+ or Redis Stack):

```bash
REDISVL_RUN_INTEGRATION=1 cargo test --workspace
```

## License

MIT – see the LICENSE file for details.
