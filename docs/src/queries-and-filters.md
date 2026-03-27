# Queries & Filters

## Query types

| Query | Description |
| --- | --- |
| `VectorQuery` | K-nearest neighbor vector similarity search |
| `VectorRangeQuery` | Vector search within a distance threshold |
| `TextQuery` | Full-text search with Redis Search syntax |
| `FilterQuery` | Filter-only search (no scoring) |
| `CountQuery` | Return only the count of matching documents |
| `HybridQuery` | Combined text + vector search via `FT.HYBRID` (Redis 8.4+) |
| `AggregateHybridQuery` | Text + vector fusion via `FT.AGGREGATE` |
| `MultiVectorQuery` | Multi-vector aggregate search across multiple vector fields |
| `SQLQuery` | SQL `SELECT` → Redis Search translation (behind `sql` feature) |

> **Note:** `HybridQuery`, `AggregateHybridQuery`, and `MultiVectorQuery` have
> full command builders but require Redis 8.4+ for `FT.HYBRID` / `FT.AGGREGATE`
> support. End-to-end integration testing against Redis 8.4 is still in progress.

## Filter DSL

Filters compose using `&` (AND), `|` (OR), and `!` (NOT):

```rust,no_run
use redis_vl::filter::{Tag, Num};

let filter = Tag::new("color").eq("red") & Num::new("price").lt(100.0);
```

### Available filters

- **Tag** – exact match: `Tag::new("field").eq("value")`
- **Text** – full-text match: `Text::new("field").eq("word")`
- **Num** – numeric comparisons: `.eq()`, `.ne()`, `.gt()`, `.lt()`, `.gte()`, `.lte()`, `.between()`
- **Geo** – geographic radius: `Geo::new("field").eq(GeoRadius::new(lon, lat, radius, "km"))`
- **GeoRadius** – radius specification: `GeoRadius::new(lon, lat, radius, unit)`
- **Timestamp** – Unix timestamp comparisons (same API as Num)

## Building a vector query

```rust,no_run
use redis_vl::{Vector, VectorQuery};

let vector = Vector::new(&[0.1_f32; 128] as &[f32]);
let query = VectorQuery::new(vector, "embedding", 10)
    .with_return_fields(["title", "score"]);
```

## Running queries

```rust,no_run
use redis_vl::{SearchIndex, IndexSchema, Vector, VectorQuery};

let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");

let vector = Vector::new(&[0.1_f32; 128] as &[f32]);
let query = VectorQuery::new(vector, "embedding", 5);
let result = index.search(&query).unwrap();

println!("Found {} documents", result.total);
for doc in &result.docs {
    println!("{:?}", doc);
}
```

## Pagination

```rust,no_run
# use redis_vl::{SearchIndex, IndexSchema, FilterQuery};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
let query = FilterQuery::new(redis_vl::filter::Tag::new("color").eq("red"));
let pages = index.paginate(&query, 10).unwrap();
for page in pages {
    println!("page with {} documents", page.len());
}
```

## Batch queries

```rust,no_run
# use redis_vl::{SearchIndex, IndexSchema, Vector, VectorQuery};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
let queries = vec![
    VectorQuery::new(Vector::new(&[0.1_f32; 128] as &[f32]), "embedding", 5),
    VectorQuery::new(Vector::new(&[0.2_f32; 128] as &[f32]), "embedding", 5),
];
let results = index.batch_search(queries.iter()).unwrap();
```

## Hybrid queries (Redis 8.4+)

`HybridQuery` combines text and vector search with configurable fusion:

```rust,no_run
use redis_vl::query::{HybridQuery, HybridCombinationMethod, Vector};

let query = HybridQuery::new(
    "medical professional",
    "description",
    Vector::new(vec![0.1, 0.1, 0.5]),
    "user_embedding",
)
.with_num_results(10)
.with_combination_method(HybridCombinationMethod::Rrf)
.with_return_fields(["user", "age", "job"]);
```

## SQL queries

`SQLQuery` (behind the `sql` feature flag) translates SQL `SELECT` statements
into Redis Search queries:

```rust,no_run
use redis_vl::{SQLQuery, SqlParam};

let query = SQLQuery::new("SELECT * FROM products WHERE category = 'electronics' AND price > :min")
    .with_param("min", SqlParam::Float(99.99));
```

Supported SQL features: `WHERE` (comparisons, `IN`/`NOT IN`, `LIKE`, `BETWEEN`,
`AND`/`OR`), `ORDER BY`, `LIMIT`/`OFFSET`, and field projection. Aggregate
queries (`COUNT`, `GROUP BY`) are not yet supported.

