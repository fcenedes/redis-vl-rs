//! Integration tests derived from the upstream Python
//! `test_async_search_index.py` parity contract.

use std::sync::atomic::{AtomicU64, Ordering};

use redis_vl::{AsyncSearchIndex, FilterExpression, FilterQuery, Tag};
use serde_json::{Value, json};

static COUNTER: AtomicU64 = AtomicU64::new(1);

/// Per-process unique run identifier to prevent stale-data collisions across
/// parallel test runs sharing the same Redis instance.
fn run_id() -> u32 {
    std::process::id()
}

fn integration_enabled() -> bool {
    std::env::var("REDISVL_RUN_INTEGRATION")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned())
}

fn unique_schema(storage_type: &str) -> Value {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    json!({
        "index": {
            "name": format!("python_parity_async_index_{storage_type}_{pid}_{id}"),
            "prefix": format!("python_parity_async_prefix_{storage_type}_{pid}_{id}"),
            "storage_type": storage_type,
        },
        "fields": [
            { "name": "id", "type": "tag" },
            { "name": "test", "type": "tag" }
        ]
    })
}

async fn create_index(storage_type: &str) -> Option<AsyncSearchIndex> {
    if !integration_enabled() {
        return None;
    }

    let index = AsyncSearchIndex::from_json_value(unique_schema(storage_type), redis_url())
        .expect("schema should parse");
    if index.create_with_options(true, true).await.is_err() {
        return None;
    }
    Some(index)
}

async fn create_query_index() -> Option<AsyncSearchIndex> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    let schema = json!({
        "index": {
            "name": format!("python_parity_async_query_index_{pid}_{id}"),
            "prefix": format!("python_parity_async_query_prefix_{pid}_{id}"),
            "storage_type": "hash",
        },
        "fields": [
            { "name": "user", "type": "tag" },
            { "name": "description", "type": "text" },
            { "name": "credit_score", "type": "tag" },
            { "name": "job", "type": "text" },
            { "name": "age", "type": "numeric" },
            { "name": "last_updated", "type": "numeric" },
            { "name": "location", "type": "geo" },
            {
                "name": "user_embedding",
                "type": "vector",
                "attrs": {
                    "dims": 3,
                    "distance_metric": "COSINE",
                    "algorithm": "HNSW",
                    "datatype": "FLOAT32"
                }
            }
        ]
    });

    let index =
        AsyncSearchIndex::from_json_value(schema, redis_url()).expect("schema should parse");
    if index.create_with_options(true, true).await.is_err() {
        return None;
    }
    Some(index)
}

fn query_sample_data() -> Vec<Value> {
    vec![
        json!({
            "user": "john",
            "age": 18,
            "job": "engineer",
            "description": "engineers conduct trains that ride on train tracks",
            "last_updated": 1737032400.0,
            "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
        }),
        json!({
            "user": "mary",
            "age": 14,
            "job": "doctor",
            "description": "a medical professional who treats diseases and helps people stay healthy",
            "last_updated": 1737032400.0,
            "credit_score": "low",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
        }),
        json!({
            "user": "nancy",
            "age": 94,
            "job": "doctor",
            "description": "a research scientist specializing in cancers and diseases of the lungs",
            "last_updated": 1739710800.0,
            "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.7, 0.1, 0.5],
        }),
        json!({
            "user": "tyler",
            "age": 100,
            "job": "engineer",
            "description": "a software developer with expertise in mathematics and computer science",
            "last_updated": 1739710800.0,
            "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.1, 0.4, 0.5],
        }),
        json!({
            "user": "tim",
            "age": 12,
            "job": "dermatologist",
            "description": "a medical professional specializing in diseases of the skin",
            "last_updated": 1739710800.0,
            "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.4, 0.4, 0.5],
        }),
        json!({
            "user": "taimur",
            "age": 15,
            "job": "CEO",
            "description": "high stress, but financially rewarding position at the head of a company",
            "last_updated": 1742130000.0,
            "credit_score": "low",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.6, 0.1, 0.5],
        }),
        json!({
            "user": "joe",
            "age": 35,
            "job": "dentist",
            "description": "like the tooth fairy because they'll take your teeth, but you have to pay them!",
            "last_updated": 1742130000.0,
            "credit_score": "medium",
            "location": "-110.0839,37.3861",
            "user_embedding": [-0.1, -0.1, -0.5],
        }),
    ]
}

#[tokio::test]
async fn python_test_async_search_index_load_and_fetch_hash() {
    let Some(index) = create_index("hash").await else {
        return;
    };

    index
        .load(&[json!({"id": "1", "test": "foo"})], "id", None)
        .await
        .expect("load should succeed");

    let record = index.fetch("1").await.expect("fetch should succeed");
    assert_eq!(
        record,
        Some(json!({
            "id": "1",
            "test": "foo"
        }))
    );

    index.delete(true).await.expect("cleanup should succeed");
}

#[tokio::test]
async fn python_test_async_search_index_clear() {
    let Some(index) = create_index("hash").await else {
        return;
    };

    index
        .load(
            &[
                json!({"id": "1", "test": "foo"}),
                json!({"id": "2", "test": "bar"}),
            ],
            "id",
            None,
        )
        .await
        .expect("load should succeed");

    let deleted = index.clear().await.expect("clear should succeed");
    assert_eq!(deleted, 2);

    let info = index.info().await.expect("info should succeed");
    assert_eq!(info["num_docs"], json!(0));

    index.delete(true).await.expect("cleanup should succeed");
}

#[tokio::test]
async fn python_test_async_batch_search_and_query_with_multiple_batches() {
    let Some(index) = create_index("hash").await else {
        return;
    };

    index
        .load(
            &[
                json!({"id": "1", "test": "foo"}),
                json!({"id": "2", "test": "bar"}),
            ],
            "id",
            None,
        )
        .await
        .expect("load should succeed");

    let raw = index
        .batch_search_with_size(
            ["@test:{foo}", "@test:{bar}", "@test:{baz}", "@test:{foo}"].iter(),
            2,
        )
        .await
        .expect("batched search should succeed");
    assert_eq!(raw.len(), 4);
    assert_eq!(raw[0].docs[0]["id"], json!(index.key("1")));
    assert_eq!(raw[1].docs[0]["id"], json!(index.key("2")));
    assert_eq!(raw[2].total, 0);
    assert_eq!(raw[3].docs[0]["id"], json!(index.key("1")));

    let query_a = FilterQuery::new(Tag::new("test").eq("foo"));
    let query_b = FilterQuery::new(Tag::new("test").eq("bar"));
    let query_c = FilterQuery::new(Tag::new("test").eq("baz"));
    let processed = index
        .batch_query_with_size([&query_a, &query_b, &query_c], 1)
        .await
        .expect("batched query should succeed");
    assert_eq!(processed.len(), 3);
    assert_eq!(
        processed[0].as_documents().expect("docs")[0]["id"],
        json!(index.key("1"))
    );
    assert_eq!(
        processed[1].as_documents().expect("docs")[0]["id"],
        json!(index.key("2"))
    );
    assert!(processed[2].as_documents().expect("docs").is_empty());

    index.delete(true).await.expect("cleanup should succeed");
}

#[tokio::test]
async fn python_test_async_search_results_process_json_without_projection() {
    let Some(index) = create_index("json").await else {
        return;
    };

    index
        .load(
            &[
                json!({"id": "1", "test": "foo"}),
                json!({"id": "2", "test": "foo"}),
            ],
            "id",
            None,
        )
        .await
        .expect("load should succeed");

    let query = FilterQuery::new(FilterExpression::MatchAll);
    let processed = index.query(&query).await.expect("query should succeed");
    let documents = processed.as_documents().expect("documents");

    assert_eq!(documents.len(), 2);
    assert!(
        documents
            .iter()
            .all(|doc| doc.get("test") == Some(&json!("foo")))
    );

    index.delete(true).await.expect("cleanup should succeed");
}

#[tokio::test]
async fn python_test_async_count_query() {
    let Some(index) = create_query_index().await else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .await
        .expect("load should succeed");

    let total = index
        .query(&redis_vl::CountQuery::new())
        .await
        .expect("count query should succeed");
    assert_eq!(total.as_count(), Some(7));

    let high_credit = index
        .query(&redis_vl::CountQuery::new().with_filter(Tag::new("credit_score").eq("high")))
        .await
        .expect("filtered count query should succeed");
    assert_eq!(high_credit.as_count(), Some(4));

    index.delete(true).await.expect("cleanup should succeed");
}

#[tokio::test]
async fn python_test_async_paginate_filter_query() {
    let Some(index) = create_query_index().await else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .await
        .expect("load should succeed");

    let query = FilterQuery::new(Tag::new("credit_score").eq("high"));
    let batches = index
        .paginate(&query, 3)
        .await
        .expect("paginate should succeed");
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].len(), 3);
    assert_eq!(batches[1].len(), 1);
    assert!(
        batches
            .iter()
            .flatten()
            .all(|doc| doc.get("credit_score") == Some(&json!("high")))
    );

    index.delete(true).await.expect("cleanup should succeed");
}
