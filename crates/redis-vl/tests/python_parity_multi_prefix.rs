//! Integration tests derived from the upstream Python
//! `tests/integration/test_multi_prefix.py` parity contract.
//!
//! These tests verify that queries return results from **all** configured
//! prefixes when using multi-prefix indexes.

use std::sync::atomic::{AtomicU64, Ordering};

use redis_vl::{
    CountQuery, FilterExpression, FilterQuery, Num, SearchIndex, Tag, TextQuery, Vector,
    VectorQuery, VectorRangeQuery,
};
use serde_json::{Map, Value, json};

static COUNTER: AtomicU64 = AtomicU64::new(1);

fn integration_enabled() -> bool {
    std::env::var("REDISVL_RUN_INTEGRATION")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned())
}

/// Creates a multi-prefix index, loads data under both prefixes, and returns
/// the index together with the unique id used in the prefix names.
fn setup_multi_prefix_index() -> Option<(SearchIndex, String)> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let unique = format!("mp{id}");

    let schema = json!({
        "index": {
            "name": format!("multi_prefix_test_{unique}"),
            "prefix": [format!("prefix_a_{unique}"), format!("prefix_b_{unique}")],
            "storage_type": "hash",
        },
        "fields": [
            { "name": "user", "type": "tag" },
            { "name": "credit_score", "type": "tag" },
            { "name": "job", "type": "text" },
            { "name": "age", "type": "numeric" },
            {
                "name": "user_embedding",
                "type": "vector",
                "attrs": {
                    "dims": 3,
                    "distance_metric": "COSINE",
                    "algorithm": "FLAT",
                    "datatype": "FLOAT32"
                }
            }
        ]
    });

    let index = SearchIndex::from_json_value(schema, redis_url()).expect("schema should parse");
    index
        .create_with_options(true, true)
        .expect("index should be created");

    // Data for prefix_a (2 docs)
    let data_a: Vec<Value> = vec![
        json!({"user":"john","credit_score":"high","job":"engineer at tech company","age":30,"user_embedding":[0.1,0.2,0.3]}),
        json!({"user":"jane","credit_score":"medium","job":"doctor at hospital","age":35,"user_embedding":[0.2,0.3,0.4]}),
    ];
    let keys_a = vec![
        format!("prefix_a_{unique}:doc1"),
        format!("prefix_a_{unique}:doc2"),
    ];

    // Data for prefix_b (2 docs)
    let data_b: Vec<Value> = vec![
        json!({"user":"bob","credit_score":"low","job":"teacher at school","age":40,"user_embedding":[0.3,0.4,0.5]}),
        json!({"user":"alice","credit_score":"high","job":"lawyer at firm","age":45,"user_embedding":[0.4,0.5,0.6]}),
    ];
    let keys_b = vec![
        format!("prefix_b_{unique}:doc1"),
        format!("prefix_b_{unique}:doc2"),
    ];

    index
        .load_with_keys(&data_a, &keys_a, None)
        .expect("load prefix_a should succeed");
    index
        .load_with_keys(&data_b, &keys_b, None)
        .expect("load prefix_b should succeed");

    Some((index, unique))
}

fn count_prefixes(results: &[Map<String, Value>], unique: &str) -> (usize, usize) {
    let a = results
        .iter()
        .filter(|r| {
            r.get("id")
                .and_then(Value::as_str)
                .map(|id| id.starts_with(&format!("prefix_a_{unique}:")))
                .unwrap_or(false)
        })
        .count();
    let b = results
        .iter()
        .filter(|r| {
            r.get("id")
                .and_then(Value::as_str)
                .map(|id| id.starts_with(&format!("prefix_b_{unique}:")))
                .unwrap_or(false)
        })
        .count();
    (a, b)
}

// ──────────────────────────── VectorQuery ────────────────────────────

#[test]
fn multi_prefix_vector_query_returns_both_prefixes() {
    let Some((index, unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = VectorQuery::new(Vector::new(vec![0.25, 0.35, 0.45]), "user_embedding", 10)
        .with_return_fields(["user", "credit_score", "job", "age"]);

    let results = index.query(&query).expect("query should succeed");
    let docs = results.as_documents().expect("documents");
    assert_eq!(docs.len(), 4, "expected 4 results, got {}", docs.len());

    let (a, b) = count_prefixes(docs, &unique);
    assert_eq!(a, 2, "expected 2 from prefix_a, got {a}");
    assert_eq!(b, 2, "expected 2 from prefix_b, got {b}");

    index.delete(true).ok();
}

#[test]
fn multi_prefix_vector_query_with_filter_both_prefixes() {
    let Some((index, unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = VectorQuery::new(Vector::new(vec![0.25, 0.35, 0.45]), "user_embedding", 10)
        .with_return_fields(["user", "credit_score"])
        .with_filter(Tag::new("credit_score").eq("high"));

    let results = index.query(&query).expect("query should succeed");
    let docs = results.as_documents().expect("documents");
    assert_eq!(docs.len(), 2, "expected 2 results, got {}", docs.len());

    let (a, b) = count_prefixes(docs, &unique);
    assert_eq!(a, 1, "expected 1 from prefix_a, got {a}");
    assert_eq!(b, 1, "expected 1 from prefix_b, got {b}");

    index.delete(true).ok();
}

// ──────────────────────── VectorRangeQuery ───────────────────────────

#[test]
fn multi_prefix_range_query_returns_both_prefixes() {
    let Some((index, unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = VectorRangeQuery::new(Vector::new(vec![0.25, 0.35, 0.45]), "user_embedding", 0.5)
        .with_return_fields(["user", "credit_score"]);

    let results = index.query(&query).expect("query should succeed");
    let docs = results.as_documents().expect("documents");

    let (a, b) = count_prefixes(docs, &unique);
    assert!(a > 0, "expected results from prefix_a");
    assert!(b > 0, "expected results from prefix_b");

    index.delete(true).ok();
}

// ──────────────────────── FilterQuery ────────────────────────────────

#[test]
fn multi_prefix_filter_query_returns_both_prefixes() {
    let Some((index, unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = FilterQuery::new(Num::new("age").gte(30.0))
        .with_return_fields(["user", "credit_score", "age"])
        .paging(0, 10);

    let results = index.query(&query).expect("query should succeed");
    let docs = results.as_documents().expect("documents");
    assert_eq!(docs.len(), 4, "expected 4 results, got {}", docs.len());

    let (a, b) = count_prefixes(docs, &unique);
    assert_eq!(a, 2, "expected 2 from prefix_a, got {a}");
    assert_eq!(b, 2, "expected 2 from prefix_b, got {b}");

    index.delete(true).ok();
}

#[test]
fn multi_prefix_filter_query_tag_both_prefixes() {
    let Some((index, unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = FilterQuery::new(Tag::new("credit_score").eq("high"))
        .with_return_fields(["user", "credit_score"])
        .paging(0, 10);

    let results = index.query(&query).expect("query should succeed");
    let docs = results.as_documents().expect("documents");
    assert_eq!(docs.len(), 2, "expected 2 results, got {}", docs.len());

    let (a, b) = count_prefixes(docs, &unique);
    assert_eq!(a, 1, "expected 1 from prefix_a, got {a}");
    assert_eq!(b, 1, "expected 1 from prefix_b, got {b}");

    index.delete(true).ok();
}

// ──────────────────────── CountQuery ─────────────────────────────────

#[test]
fn multi_prefix_count_query_counts_all_prefixes() {
    let Some((index, _unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = CountQuery::new().with_filter(Tag::new("credit_score").eq("high"));
    let result = index.query(&query).expect("query should succeed");
    let count = result.as_count().expect("should be a count");
    assert_eq!(count, 2, "expected count of 2, got {count}");

    index.delete(true).ok();
}

#[test]
fn multi_prefix_count_query_all_docs() {
    let Some((index, _unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = CountQuery::new().with_filter(FilterExpression::MatchAll);
    let result = index.query(&query).expect("query should succeed");
    let count = result.as_count().expect("should be a count");
    assert_eq!(count, 4, "expected count of 4, got {count}");

    index.delete(true).ok();
}

// ──────────────────────── TextQuery ──────────────────────────────────

#[test]
fn multi_prefix_text_query_returns_both_prefixes() {
    let Some((index, unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = TextQuery::new("engineer|doctor|teacher|lawyer")
        .for_field("job")
        .with_return_fields(["user", "job"])
        .paging(0, 10);

    let results = index.query(&query).expect("query should succeed");
    let docs = results.as_documents().expect("documents");
    assert_eq!(docs.len(), 4, "expected 4 results, got {}", docs.len());

    let (a, b) = count_prefixes(docs, &unique);
    assert_eq!(a, 2, "expected 2 from prefix_a, got {a}");
    assert_eq!(b, 2, "expected 2 from prefix_b, got {b}");

    index.delete(true).ok();
}

#[test]
fn multi_prefix_text_query_specific_term() {
    let Some((index, _unique)) = setup_multi_prefix_index() else {
        return;
    };

    let query = TextQuery::new("engineer")
        .for_field("job")
        .with_return_fields(["user", "job"])
        .paging(0, 10);

    let results = index.query(&query).expect("query should succeed");
    let docs = results.as_documents().expect("documents");
    assert_eq!(docs.len(), 1, "expected 1 result, got {}", docs.len());
    assert_eq!(docs[0].get("user").and_then(Value::as_str), Some("john"));

    index.delete(true).ok();
}

// ──────────────────────── Index creation ─────────────────────────────

#[test]
fn multi_prefix_create_index_with_prefix_list() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let unique = format!("cmp{id}");

    let schema = json!({
        "index": {
            "name": format!("create_multi_prefix_{unique}"),
            "prefix": [format!("pfx_a_{unique}"), format!("pfx_b_{unique}")],
            "storage_type": "hash",
        },
        "fields": [
            { "name": "user", "type": "tag" },
            { "name": "age", "type": "numeric" },
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 3,
                    "distance_metric": "COSINE",
                    "algorithm": "FLAT",
                    "datatype": "FLOAT32"
                }
            }
        ]
    });

    let index = SearchIndex::from_json_value(schema, redis_url()).expect("schema should parse");
    index
        .create_with_options(true, true)
        .expect("index should be created");

    assert!(index.exists().expect("exists check"));

    let data = vec![json!({"user":"test_user","age":25,"embedding":[0.1,0.2,0.3]})];

    let keys_a = vec![format!("pfx_a_{unique}:doc1")];
    index
        .load_with_keys(&data, &keys_a, None)
        .expect("load a should succeed");

    let keys_b = vec![format!("pfx_b_{unique}:doc1")];
    index
        .load_with_keys(&data, &keys_b, None)
        .expect("load b should succeed");

    let count_q = CountQuery::new().with_filter(FilterExpression::MatchAll);
    let result = index.query(&count_q).expect("query should succeed");
    let count = result.as_count().expect("should be a count");
    assert_eq!(count, 2, "expected 2 docs, got {count}");

    assert_eq!(
        index.prefixes(),
        vec![format!("pfx_a_{unique}"), format!("pfx_b_{unique}")]
    );
    assert_eq!(index.prefix(), format!("pfx_a_{unique}"));

    index.delete(true).ok();
}

// ──────────────────────── from_existing ──────────────────────────────

#[test]
fn multi_prefix_from_existing_preserves_prefixes() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let unique = format!("fe{id}");
    let index_name = format!("from_existing_mp_{unique}");

    let schema = json!({
        "index": {
            "name": &index_name,
            "prefix": [format!("fe_a_{unique}"), format!("fe_b_{unique}")],
            "storage_type": "hash",
        },
        "fields": [
            { "name": "tag", "type": "tag" }
        ]
    });

    let index = SearchIndex::from_json_value(schema, redis_url()).expect("schema should parse");
    index
        .create_with_options(true, true)
        .expect("index should be created");

    // Reconstruct from existing
    let reconstructed =
        SearchIndex::from_existing(&index_name, redis_url()).expect("from_existing should work");

    assert_eq!(reconstructed.prefixes().len(), 2);
    assert_eq!(
        reconstructed.prefixes(),
        vec![format!("fe_a_{unique}"), format!("fe_b_{unique}")]
    );

    index.delete(true).ok();
}
