//! Integration tests derived from the upstream Python `test_search_index.py`
//! and `test_search_results.py` parity contract.

use std::sync::atomic::{AtomicU64, Ordering};

use redis_vl::{
    BetweenInclusivity, FilterExpression, FilterQuery, Geo, GeoRadius, Num, SearchIndex, Tag, Text,
    Timestamp, Vector, VectorQuery, VectorRangeQuery,
};
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
            "name": format!("python_parity_index_{storage_type}_{pid}_{id}"),
            "prefix": format!("python_parity_prefix_{storage_type}_{pid}_{id}"),
            "storage_type": storage_type,
        },
        "fields": [
            { "name": "id", "type": "tag" },
            { "name": "test", "type": "tag" }
        ]
    })
}

fn create_index(storage_type: &str) -> Option<SearchIndex> {
    if !integration_enabled() {
        return None;
    }

    let index = SearchIndex::from_json_value(unique_schema(storage_type), redis_url())
        .expect("schema should parse");
    if index.create_with_options(true, true).is_err() {
        return None;
    }
    Some(index)
}

fn create_query_index() -> Option<SearchIndex> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    let schema = json!({
        "index": {
            "name": format!("python_parity_query_index_{pid}_{id}"),
            "prefix": format!("python_parity_query_prefix_{pid}_{id}"),
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

    let index = SearchIndex::from_json_value(schema, redis_url()).expect("schema should parse");
    if index.create_with_options(true, true).is_err() {
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

#[test]
fn python_test_search_index_load_and_fetch_hash() {
    let Some(index) = create_index("hash") else {
        return;
    };

    index
        .load(&[json!({"id": "1", "test": "foo"})], "id", None)
        .expect("load should succeed");

    let record = index.fetch("1").expect("fetch should succeed");
    assert_eq!(
        record,
        Some(json!({
            "id": "1",
            "test": "foo"
        }))
    );

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_search_index_load_and_fetch_json() {
    let Some(index) = create_index("json") else {
        return;
    };

    index
        .load(&[json!({"id": "1", "test": "foo"})], "id", None)
        .expect("load should succeed");

    let record = index.fetch("1").expect("fetch should succeed");
    assert_eq!(
        record,
        Some(json!({
            "id": "1",
            "test": "foo"
        }))
    );

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_batch_search_and_batch_query() {
    let Some(index) = create_index("hash") else {
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
        .expect("load should succeed");

    let raw_results = index
        .batch_search(["@test:{foo}", "@test:{bar}"].iter())
        .expect("batch search should succeed");
    assert_eq!(raw_results.len(), 2);
    assert_eq!(raw_results[0].total, 1);
    assert_eq!(raw_results[0].docs[0]["id"], json!(index.key("1")));
    assert_eq!(raw_results[1].total, 1);

    let query_a = FilterQuery::new(Tag::new("test").eq("foo"));
    let query_b = FilterQuery::new(Tag::new("test").eq("bar"));
    let processed = index
        .batch_query([&query_a, &query_b])
        .expect("batch query should succeed");
    assert_eq!(processed.len(), 2);
    assert_eq!(
        processed[0].as_documents().expect("docs")[0]["test"],
        json!("foo")
    );
    assert_eq!(
        processed[1].as_documents().expect("docs")[0]["test"],
        json!("bar")
    );

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_batch_search_and_query_with_multiple_batches() {
    let Some(index) = create_index("hash") else {
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
        .expect("load should succeed");

    let raw = index
        .batch_search_with_size(
            ["@test:{foo}", "@test:{bar}", "@test:{baz}", "@test:{foo}"].iter(),
            2,
        )
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

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_search_index_clear() {
    let Some(index) = create_index("hash") else {
        return;
    };

    index
        .load(
            &[
                json!({"id": "1", "test": "foo"}),
                json!({"id": "2", "test": "bar"}),
                json!({"id": "3", "test": "baz"}),
            ],
            "id",
            None,
        )
        .expect("load should succeed");

    let deleted = index.clear().expect("clear should succeed");
    assert_eq!(deleted, 3);

    let info = index.info().expect("info should succeed");
    assert_eq!(info["num_docs"], json!(0));

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_search_index_drop_documents_and_keys() {
    let Some(index) = create_index("hash") else {
        return;
    };

    let keys = index
        .load(
            &[
                json!({"id": "1", "test": "foo"}),
                json!({"id": "2", "test": "bar"}),
                json!({"id": "3", "test": "baz"}),
            ],
            "id",
            None,
        )
        .expect("load should succeed");

    let deleted_keys = index
        .drop_keys(&keys[..2])
        .expect("drop keys should succeed");
    assert_eq!(deleted_keys, 2);
    assert_eq!(index.fetch("1").expect("fetch should succeed"), None);
    assert_eq!(index.fetch("2").expect("fetch should succeed"), None);
    assert!(index.fetch("3").expect("fetch should succeed").is_some());

    let deleted_documents = index
        .drop_documents(&["3".to_owned()])
        .expect("drop documents should succeed");
    assert_eq!(deleted_documents, 1);
    assert_eq!(index.fetch("3").expect("fetch should succeed"), None);

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_search_index_expire_keys_and_ttl() {
    let Some(index) = create_index("hash") else {
        return;
    };

    let keys = index
        .load(
            &[
                json!({"id": "1", "test": "foo"}),
                json!({"id": "2", "test": "bar"}),
            ],
            "id",
            None,
        )
        .expect("load should succeed");

    let single = index
        .expire_key(&keys[0], 60)
        .expect("expire key should succeed");
    assert!(single);

    let multiple = index
        .expire_keys(&keys, 30)
        .expect("expire keys should succeed");
    assert_eq!(multiple, vec![true, true]);

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_search_index_load_with_preprocess() {
    let Some(index) = create_index("hash") else {
        return;
    };

    index
        .load_with_preprocess(&[json!({"id": "1", "test": "foo"})], "id", None, |record| {
            let mut record = record.clone();
            let object = record
                .as_object_mut()
                .expect("preprocess keeps the record as an object");
            object.insert("test".to_owned(), json!("bar"));
            Ok(record)
        })
        .expect("load with preprocess should succeed");

    let record = index.fetch("1").expect("fetch should succeed");
    assert_eq!(record, Some(json!({"id": "1", "test": "bar"})));

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_search_results_process_json_without_projection() {
    let Some(index) = create_index("json") else {
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
        .expect("load should succeed");

    let query = FilterQuery::new(FilterExpression::MatchAll);
    let processed = index.query(&query).expect("query should succeed");
    let documents = processed.as_documents().expect("documents");

    assert_eq!(documents.len(), 2);
    assert!(
        documents
            .iter()
            .all(|doc| doc.get("test") == Some(&json!("foo")))
    );

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_count_query() {
    let Some(index) = create_query_index() else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .expect("load should succeed");

    let total = index
        .query(&redis_vl::CountQuery::new())
        .expect("count query should succeed");
    assert_eq!(total.as_count(), Some(7));

    let high_credit = index
        .query(&redis_vl::CountQuery::new().with_filter(Tag::new("credit_score").eq("high")))
        .expect("filtered count query should succeed");
    assert_eq!(high_credit.as_count(), Some(4));

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_vector_query_search_and_query() {
    let Some(index) = create_query_index() else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .expect("load should succeed");

    let query = redis_vl::VectorQuery::new(
        redis_vl::Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
        7,
    )
    .with_return_fields(["user", "credit_score", "age", "job", "location"]);

    let raw = index.search(&query).expect("search should succeed");
    assert_eq!(raw.docs.len(), 7);

    let processed = index.query(&query).expect("query should succeed");
    let documents = processed.as_documents().expect("documents");
    assert_eq!(documents.len(), 7);
    assert!(documents.iter().all(|doc| doc.get("user").is_some()));
    assert_eq!(documents[0], raw.docs[0].to_map());

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_range_query() {
    let Some(index) = create_query_index() else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .expect("load should succeed");

    let mut query = redis_vl::VectorRangeQuery::new(
        redis_vl::Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
        0.2,
    )
    .with_return_fields(["user", "credit_score", "age", "job"]);

    let processed = index.query(&query).expect("range query should succeed");
    let documents = processed.as_documents().expect("documents");
    assert_eq!(documents.len(), 4);
    assert!(documents.iter().all(|doc| {
        doc["vector_distance"]
            .as_str()
            .and_then(|value| value.parse::<f64>().ok())
            .map(|distance| distance <= 0.2)
            .unwrap_or(false)
    }));

    query.set_distance_threshold(0.1);
    let tightened = index
        .query(&query)
        .expect("tightened range query should succeed");
    let tightened_docs = tightened.as_documents().expect("documents");
    assert_eq!(tightened_docs.len(), 2);

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_paginate_filter_query() {
    let Some(index) = create_query_index() else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .expect("load should succeed");

    let query = FilterQuery::new(Tag::new("credit_score").eq("high"));
    let batches = index.paginate(&query, 3).expect("paginate should succeed");
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].len(), 3);
    assert_eq!(batches[1].len(), 1);
    assert!(
        batches
            .iter()
            .flatten()
            .all(|doc| doc.get("credit_score") == Some(&json!("high")))
    );

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_vector_query_filters_tag_numeric_text_geo_timestamp() {
    let Some(index) = create_query_index() else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .expect("load should succeed");

    let base = VectorQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 7)
        .with_return_fields([
            "user",
            "credit_score",
            "age",
            "job",
            "location",
            "last_updated",
        ]);

    let high_credit = base
        .clone()
        .with_filter(Tag::new("credit_score").eq("high"));
    assert_eq!(
        index
            .search(&high_credit)
            .expect("search should succeed")
            .docs
            .len(),
        4
    );

    let high_or_low = base
        .clone()
        .with_filter(Tag::new("credit_score").one_of(["high", "low"]));
    assert_eq!(
        index
            .search(&high_or_low)
            .expect("search should succeed")
            .docs
            .len(),
        6
    );

    let adults = base.clone().with_filter(Num::new("age").gte(18.0));
    assert_eq!(
        index
            .search(&adults)
            .expect("search should succeed")
            .docs
            .len(),
        4
    );

    let age_window = base
        .clone()
        .with_filter(Num::new("age").gte(18.0) & Num::new("age").lt(100.0));
    assert_eq!(
        index
            .search(&age_window)
            .expect("search should succeed")
            .docs
            .len(),
        3
    );

    let not_eighteen = base.clone().with_filter(Num::new("age").ne(18.0));
    assert_eq!(
        index
            .search(&not_eighteen)
            .expect("search should succeed")
            .docs
            .len(),
        6
    );

    let geo = base
        .clone()
        .with_filter(Geo::new("location").eq(GeoRadius::new(-122.4194, 37.7749, 1.0, "m")));
    let geo_results = index.search(&geo).expect("search should succeed");
    assert_eq!(geo_results.docs.len(), 3);
    assert!(
        geo_results
            .docs
            .iter()
            .all(|doc| doc["location"] == json!("-122.4194,37.7749"))
    );

    let engineers = base.clone().with_filter(Text::new("job").eq("engineer"));
    assert_eq!(
        index
            .search(&engineers)
            .expect("search should succeed")
            .docs
            .len(),
        2
    );

    let non_engineers = base.clone().with_filter(Text::new("job").ne("engineer"));
    assert_eq!(
        index
            .search(&non_engineers)
            .expect("search should succeed")
            .docs
            .len(),
        5
    );

    let wildcard_jobs = base
        .clone()
        .with_filter(Text::new("job").like("engine*|doctor"));
    assert_eq!(
        index
            .search(&wildcard_jobs)
            .expect("search should succeed")
            .docs
            .len(),
        4
    );

    let after_mid = base
        .clone()
        .with_filter(Timestamp::new("last_updated").after(1739710800.0));
    assert_eq!(
        index
            .search(&after_mid)
            .expect("search should succeed")
            .docs
            .len(),
        2
    );

    let at_or_after_mid = base
        .clone()
        .with_filter(Timestamp::new("last_updated").gte(1739710800.0));
    assert_eq!(
        index
            .search(&at_or_after_mid)
            .expect("search should succeed")
            .docs
            .len(),
        5
    );

    let between = base
        .clone()
        .with_filter(Timestamp::new("last_updated").between(
            1737032401.0,
            1742129999.0,
            BetweenInclusivity::Both,
        ));
    assert_eq!(
        index
            .search(&between)
            .expect("search should succeed")
            .docs
            .len(),
        3
    );

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_vector_range_query_filters_and_pagination() {
    let Some(index) = create_query_index() else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .expect("load should succeed");

    let query = VectorRangeQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 0.2)
        .with_return_fields(["user", "credit_score", "age", "job"])
        .with_filter(Tag::new("credit_score").eq("high"));

    let results = index.query(&query).expect("query should succeed");
    let documents = results.as_documents().expect("documents");
    // 3 docs have credit_score=high within the vector range
    assert_eq!(documents.len(), 3);
    assert!(
        documents
            .iter()
            .all(|doc| doc["credit_score"] == json!("high"))
    );

    let paged = index
        .paginate(
            &VectorQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 7)
                .with_return_fields(["user", "age"]),
            2,
        )
        .expect("paginate should succeed");
    assert_eq!(paged.len(), 4);
    assert!(paged.iter().all(|batch| batch.len() <= 2));

    index.delete(true).expect("cleanup should succeed");
}

#[test]
fn python_test_manual_string_filters_and_combinations() {
    let Some(index) = create_query_index() else {
        return;
    };

    index
        .load(&query_sample_data(), "user", None)
        .expect("load should succeed");

    let raw_tag = VectorQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 7)
        .with_return_fields(["user", "credit_score", "age", "job", "location"])
        .with_filter(FilterExpression::raw("@credit_score:{high}"));
    assert_eq!(
        index
            .search(&raw_tag)
            .expect("search should succeed")
            .docs
            .len(),
        4
    );

    let raw_numeric = VectorQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 7)
        .with_return_fields(["user", "credit_score", "age", "job", "location"])
        .with_filter(FilterExpression::raw("@age:[18 +inf]"));
    assert_eq!(
        index
            .search(&raw_numeric)
            .expect("search should succeed")
            .docs
            .len(),
        4
    );

    let typed_combo = VectorQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 7)
        .with_return_fields(["user", "credit_score", "age", "job", "location"])
        .with_filter(Tag::new("credit_score").eq("high") & Text::new("job").eq("engineer"));
    let combo_results = index.search(&typed_combo).expect("search should succeed");
    assert_eq!(combo_results.docs.len(), 2);
    assert!(
        combo_results
            .docs
            .iter()
            .all(|doc| doc["credit_score"] == json!("high"))
    );

    let geo_combo = VectorQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 7)
        .with_return_fields(["user", "credit_score", "age", "job", "location"])
        .with_filter(
            Geo::new("location").eq(GeoRadius::new(-122.4194, 37.7749, 1.0, "m"))
                & Text::new("job").eq("engineer"),
        );
    let geo_combo_results = index.search(&geo_combo).expect("search should succeed");
    assert_eq!(geo_combo_results.docs.len(), 1);
    assert_eq!(geo_combo_results.docs[0]["user"], json!("john"));

    index.delete(true).expect("cleanup should succeed");
}
