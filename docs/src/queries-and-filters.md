# Queries & Filters

## Query types

| Query | Description |
| --- | --- |
| `VectorQuery` | K-nearest neighbor vector similarity search |
| `VectorRangeQuery` | Vector search within a distance threshold |
| `TextQuery` | Full-text search with Redis Search syntax |
| `FilterQuery` | Filter-only search (no scoring) |
| `CountQuery` | Return only the count of matching documents |

> `HybridQuery`, `AggregateHybridQuery`, and `MultiVectorQuery` types exist
> but runtime parity with the Python library is still in progress.

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

for doc in result.as_documents().unwrap_or(&[]) {
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

