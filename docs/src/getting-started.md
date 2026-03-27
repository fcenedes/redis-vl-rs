# Getting Started

## Installation

Add `redis-vl` to your `Cargo.toml`:

```bash
cargo add redis-vl
```

The default features enable the `openai` and `litellm` vectorizers. To use only
the core library without vectorizer dependencies:

```bash
cargo add redis-vl --no-default-features
```

## Connecting to Redis

All index operations require a Redis URL. The library accepts any
`redis://` connection string:

```rust,no_run
use redis_vl::{IndexSchema, SearchIndex};

let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
```

For async usage:

```rust,no_run
use redis_vl::{IndexSchema, AsyncSearchIndex};

let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
let index = AsyncSearchIndex::new(schema, "redis://127.0.0.1:6379");
```

## Creating an index

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
index.create().unwrap();
```

## Loading data

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
use serde_json::json;

let docs = vec![
    json!({
        "id": "doc:1",
        "title": "first document",
        "embedding": vec![0.1_f32; 128]
    }),
];
index.load(&docs, "id", None).unwrap();
```

## Next steps

- [Schema](schema.md) – learn about field types and schema configuration
- [Queries & Filters](queries-and-filters.md) – search and filter your data
- [Extensions](extensions.md) – caching, message history, and routing
- [CLI](cli.md) – manage indices from the command line

