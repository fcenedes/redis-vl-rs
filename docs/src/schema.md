# Schema

An `IndexSchema` describes the structure of a Redis Search index, including its
name, key prefix, storage type, and searchable fields.

## YAML format

```yaml
index:
  name: my-index
  prefix: doc
  key_separator: ":"
  storage_type: hash
fields:
  - name: title
    type: tag
  - name: content
    type: text
  - name: score
    type: numeric
  - name: location
    type: geo
  - name: created_at
    type: timestamp
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 128
      distance_metric: cosine
      datatype: float32
```

## Loading schemas

```rust,no_run
use redis_vl::IndexSchema;

// From a YAML file
let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();

// From a YAML string
let schema = IndexSchema::from_yaml_str("index:\n  name: test\nfields: []").unwrap();

// From a JSON value
let value = serde_json::json!({"index": {"name": "test"}, "fields": []});
let schema = IndexSchema::from_json_value(value).unwrap();
```

## Field types

| Type | Rust filter | Description |
| --- | --- | --- |
| `tag` | `Tag` | Exact-match tag fields |
| `text` | `Text` | Full-text searchable fields |
| `numeric` | `Num` | Numeric range fields |
| `geo` | `Geo`, `GeoRadius` | Geographic coordinate fields |
| `timestamp` | `Timestamp` | Unix timestamp fields |
| `vector` | — | Vector similarity search fields |

## Storage types

- `hash` – stores documents as Redis Hashes (default)
- `json` – stores documents as RedisJSON documents

## Vector field attributes

| Attribute | Values | Default |
| --- | --- | --- |
| `algorithm` | `flat`, `hnsw` | `flat` |
| `dims` | positive integer | required |
| `distance_metric` | `cosine`, `l2`, `ip` | `cosine` |
| `datatype` | `float32`, `float64` | `float32` |

## Multi-prefix indexes

An index can span multiple key prefixes:

```yaml
index:
  name: multi-idx
  prefix:
    - products
    - inventory
fields:
  - name: title
    type: tag
```

```rust,no_run
use redis_vl::IndexSchema;

let schema = IndexSchema::from_json_value(serde_json::json!({
    "index": { "name": "multi", "prefix": ["pfx_a", "pfx_b"] },
    "fields": []
})).unwrap();
assert_eq!(schema.index.prefix.len(), 2);
```

## Stopwords

Custom stopwords can be configured at the index level:

```yaml
index:
  name: my-index
  prefix: doc
  stopwords:
    - the
    - a
    - an
fields: []
```

