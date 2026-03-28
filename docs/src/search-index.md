# Search Index

The search index is the central type for interacting with Redis Search. It
manages the index lifecycle, document loading/fetching, and query execution.

Two implementations are provided:
- **`SearchIndex`** – blocking (sync) operations
- **`AsyncSearchIndex`** – async operations (Tokio-based)

Both share the same schema, connection configuration, and query semantics.

## Creating an index

```rust,no_run
use redis_vl::{IndexSchema, SearchIndex};

// From a schema object
let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
index.create().unwrap();

// From YAML directly
let index = SearchIndex::from_yaml_file("schema.yaml", "redis://127.0.0.1:6379").unwrap();

// From JSON
let json = serde_json::json!({"index": {"name": "test"}, "fields": []});
let index = SearchIndex::from_json_value(json, "redis://127.0.0.1:6379").unwrap();

// Overwrite an existing index
index.create_with_options(/* overwrite */ true, /* drop_documents */ false).unwrap();
```

## Reconnecting to an existing index

```rust,no_run
use redis_vl::SearchIndex;

// Reconstruct a SearchIndex from an already-created Redis index
let index = SearchIndex::from_existing("my-index", "redis://127.0.0.1:6379").unwrap();
println!("Reconnected to index: {}", index.name());
```

## Loading documents

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
use serde_json::json;

let docs = vec![
    json!({"id": "doc:1", "title": "first", "embedding": vec![0.1_f32; 128]}),
    json!({"id": "doc:2", "title": "second", "embedding": vec![0.2_f32; 128]}),
];

// Basic load (returns written keys)
let keys = index.load(&docs, "id", None).unwrap();

// Load with TTL
index.load(&docs, "id", Some(3600)).unwrap();

// Load with preprocessing
index.load_with_preprocess(&docs, "id", None, |mut doc| {
    doc["processed"] = json!(true);
    doc
}).unwrap();
```

## Fetching documents

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
if let Some(doc) = index.fetch("doc:1").unwrap() {
    println!("Found: {}", doc);
}
```

## Querying

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex, Vector, VectorQuery, FilterQuery, CountQuery};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
use redis_vl::filter::Tag;

// search() returns SearchResult with total count and documents
let result = index.search(&VectorQuery::new(
    Vector::new(&[0.1_f32; 128] as &[f32]), "embedding", 5
)).unwrap();
println!("{} total matches", result.total);

// query() returns QueryOutput (documents or count depending on query type)
let output = index.query(&CountQuery::new(Tag::new("color").eq("red"))).unwrap();
println!("Count: {:?}", output.as_count());
```

## Batch and pagination

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex, Vector, VectorQuery, FilterQuery};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
# use redis_vl::filter::Tag;
// Batch search
let queries = vec![
    VectorQuery::new(Vector::new(&[0.1_f32; 128] as &[f32]), "embedding", 5),
    VectorQuery::new(Vector::new(&[0.2_f32; 128] as &[f32]), "embedding", 5),
];
let results = index.batch_search(queries.iter()).unwrap();

// Pagination
let filter = FilterQuery::new(Tag::new("category").eq("books"));
let pages = index.paginate(&filter, 10).unwrap();
for page in pages {
    println!("Page with {} documents", page.len());
}
```

## Index management

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
// Check existence
let exists = index.exists().unwrap();

// List all indices
let all = index.listall().unwrap();

// Get index metadata
let info = index.info().unwrap();

// Clear data (keep index)
let cleared = index.clear().unwrap();

// Delete index
index.delete(/* drop_documents */ false).unwrap();

// Drop index and documents
index.drop(/* delete_documents */ true).unwrap();
```

## Key and TTL management

```rust,no_run
# use redis_vl::{IndexSchema, SearchIndex};
# let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
# let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
// Build a key from prefix + suffix
let key = index.key("my-doc-id");

// Drop specific keys or documents
index.drop_key("doc:prefix:1").unwrap();
index.drop_document("doc:1").unwrap();

// Set TTL on keys
index.expire_key("doc:prefix:1", 3600).unwrap();
index.expire_keys(&["k1".into(), "k2".into()], 7200).unwrap();
```

## Async usage

`AsyncSearchIndex` mirrors the sync API. All I/O methods are async:

```rust,no_run
# async fn example() -> redis_vl::error::Result<()> {
use redis_vl::{IndexSchema, AsyncSearchIndex, Vector, VectorQuery};

let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
let index = AsyncSearchIndex::new(schema, "redis://127.0.0.1:6379");
index.create().await?;

let result = index.search(&VectorQuery::new(
    Vector::new(&[0.1_f32; 128] as &[f32]), "embedding", 5
)).await?;
# Ok(())
# }
```
