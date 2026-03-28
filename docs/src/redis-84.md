# Redis 8.4+ Features

Redis 8.4 introduces new search capabilities that `redis-vl` supports through
dedicated query types. These features require a Redis 8.4+ server.

## Hybrid search (`FT.HYBRID`)

`HybridQuery` combines text search and vector similarity search in a single
query using `FT.HYBRID`. It supports configurable fusion methods to merge
text and vector scores.

```rust,no_run
use redis_vl::query::{HybridQuery, HybridCombinationMethod, Vector};

let query = HybridQuery::new(
    "medical professional",  // text query
    "description",           // text field
    Vector::new(vec![0.1, 0.1, 0.5]),  // vector
    "user_embedding",        // vector field
)
.with_num_results(10)
.with_combination_method(HybridCombinationMethod::Rrf)  // Reciprocal Rank Fusion
.with_return_fields(["user", "age", "job"]);
```

### Combination methods

| Method | Description |
| --- | --- |
| `Rrf` | Reciprocal Rank Fusion – merges ranked lists from text and vector results |
| `Sum` | Weighted sum of text and vector scores |

### Running hybrid queries

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
# use redis_vl::query::{HybridQuery, Vector};
# let query = HybridQuery::new("query", "text_field", Vector::new(vec![0.1; 128]), "vec_field");
// Returns SearchResult
let result = index.hybrid_search(&query).unwrap();

// Returns QueryOutput (processed documents)
let output = index.hybrid_query(&query).unwrap();
```

## Aggregate hybrid search (`FT.AGGREGATE`)

`AggregateHybridQuery` uses `FT.AGGREGATE` for text + vector fusion. This is
useful when you need aggregation pipelines or when `FT.HYBRID` is not
available.

```rust,no_run
use redis_vl::query::{AggregateHybridQuery, HybridPolicy, Vector};

let query = AggregateHybridQuery::new(
    "search terms",
    "text_field",
    Vector::new(vec![0.1; 128]),
    "vector_field",
)
.with_num_results(20)
.with_return_fields(["title", "score"]);
```

### Running aggregate queries

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
# use redis_vl::query::{AggregateHybridQuery, Vector};
# let query = AggregateHybridQuery::new("query", "text_field", Vector::new(vec![0.1; 128]), "vec_field");
let output = index.aggregate_query(&query).unwrap();
```

## Multi-vector queries

`MultiVectorQuery` searches across multiple vector fields in a single
aggregate command, useful for documents with separate embedding spaces.

```rust,no_run
use redis_vl::query::{MultiVectorQuery, Vector};

let query = MultiVectorQuery::new(
    vec![
        ("title_embedding", Vector::new(vec![0.1; 128])),
        ("content_embedding", Vector::new(vec![0.2; 128])),
    ],
    10,
);
```

### Running multi-vector queries

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
# use redis_vl::query::{MultiVectorQuery, Vector};
# let query = MultiVectorQuery::new(vec![("f1", Vector::new(vec![0.1; 3]))], 5);
let output = index.multi_vector_query(&query).unwrap();
```

## Integration testing

Hybrid, aggregate, and multi-vector tests are environment-gated:

```bash
# Requires Redis 8.4+ with FT.HYBRID and FT.AGGREGATE support
REDISVL_RUN_INTEGRATION=1 cargo test --workspace python_parity_hybrid_aggregate
```

These tests are in `crates/redis-vl/tests/python_parity_hybrid_aggregate.rs`.

## Async usage

All hybrid/aggregate/multi-vector methods have async equivalents on
`AsyncSearchIndex`:

```rust,no_run
# async fn example() -> redis_vl::error::Result<()> {
# use redis_vl::{IndexSchema, AsyncSearchIndex};
# use redis_vl::query::{HybridQuery, Vector};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = AsyncSearchIndex::new(schema, "redis://127.0.0.1:6379");
# let query = HybridQuery::new("q", "tf", Vector::new(vec![0.1; 3]), "vf");
let result = index.hybrid_search(&query).await?;
let output = index.hybrid_query(&query).await?;
# Ok(())
# }
```
