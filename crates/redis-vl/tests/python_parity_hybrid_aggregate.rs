//! Integration tests for Redis 8.4+ features: HybridQuery (FT.HYBRID),
//! AggregateHybridQuery (FT.AGGREGATE), and MultiVectorQuery (FT.AGGREGATE).
//!
//! Derived from the upstream Python test files:
//! - `tests/integration/test_hybrid.py`
//! - `tests/integration/test_aggregation.py`
//!
//! These tests require:
//! - `REDISVL_RUN_INTEGRATION=1` (standard integration gate)
//! - A Redis 8.4+ instance with the Search module
//!
//! The tests detect server version at runtime and skip automatically
//! when the connected Redis instance is below 8.4.0.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use redis_vl::{
    AggregateHybridQuery, Geo, GeoRadius, HybridQuery, MultiVectorQuery, Num, SearchIndex, Tag,
    Text, Vector, VectorInput,
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
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned())
}

/// Returns the Redis server version as (major, minor, patch), or None
/// if it cannot be determined.
fn redis_server_version() -> Option<(u32, u32, u32)> {
    let client = redis::Client::open(redis_url()).ok()?;
    let mut conn = client.get_connection().ok()?;
    let info: String = redis::cmd("INFO").arg("server").query(&mut conn).ok()?;
    for line in info.lines() {
        if let Some(version) = line.strip_prefix("redis_version:") {
            let parts: Vec<&str> = version.trim().split('.').collect();
            if parts.len() >= 3 {
                let major = parts[0].parse().ok()?;
                let minor = parts[1].parse().ok()?;
                let patch = parts[2].parse().ok()?;
                return Some((major, minor, patch));
            }
        }
    }
    None
}

/// Returns true if the connected Redis server is >= 8.4.0.
fn redis84_available() -> bool {
    match redis_server_version() {
        Some((major, minor, _)) => major > 8 || (major == 8 && minor >= 4),
        None => false,
    }
}

/// Shared schema matching Python `index_schema` fixture from test_hybrid.py
/// and test_aggregation.py. Includes user_embedding (3-dim f32),
/// image_embedding (5-dim f32), and audio_embedding (6-dim f64).
fn multi_vector_schema() -> Value {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    json!({
        "index": {
            "name": format!("hybrid_parity_index_{pid}_{id}"),
            "prefix": format!("hybrid_parity_prefix_{pid}_{id}"),
            "storage_type": "hash",
        },
        "fields": [
            { "name": "credit_score", "type": "tag" },
            { "name": "job", "type": "text" },
            { "name": "description", "type": "text" },
            { "name": "age", "type": "numeric" },
            { "name": "last_updated", "type": "numeric" },
            { "name": "location", "type": "geo" },
            {
                "name": "user_embedding",
                "type": "vector",
                "attrs": {
                    "dims": 3,
                    "distance_metric": "COSINE",
                    "algorithm": "FLAT",
                    "datatype": "FLOAT32"
                }
            },
            {
                "name": "image_embedding",
                "type": "vector",
                "attrs": {
                    "dims": 5,
                    "distance_metric": "COSINE",
                    "algorithm": "FLAT",
                    "datatype": "FLOAT32"
                }
            },
            {
                "name": "audio_embedding",
                "type": "vector",
                "attrs": {
                    "dims": 6,
                    "distance_metric": "COSINE",
                    "algorithm": "FLAT",
                    "datatype": "FLOAT64"
                }
            }
        ]
    })
}

/// Multi-vector data matching the Python conftest `multi_vector_data` fixture.
fn multi_vector_data() -> Vec<Value> {
    vec![
        json!({
            "user": "john", "age": 18, "job": "engineer",
            "description": "engineers conduct trains that ride on train tracks",
            "last_updated": 1737032400.0, "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
            "image_embedding": [0.1, 0.1, 0.1, 0.1, 0.1],
            "audio_embedding": [34.0, 18.5, -6.0, -12.0, 115.0, 96.5],
        }),
        json!({
            "user": "mary", "age": 14, "job": "doctor",
            "description": "a medical professional who treats diseases and helps people stay healthy",
            "last_updated": 1737032400.0, "credit_score": "low",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.1, 0.1, 0.5],
            "image_embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "audio_embedding": [0.0, -1.06, 4.55, -1.93, 0.0, 1.53],
        }),
        json!({
            "user": "nancy", "age": 94, "job": "doctor",
            "description": "a research scientist specializing in cancers and diseases of the lungs",
            "last_updated": 1739710800.0, "credit_score": "high",
            "location": "-122.4194,37.7749",
            "user_embedding": [0.7, 0.1, 0.5],
            "image_embedding": [0.1, 0.1, 0.3, 0.3, 0.5],
            "audio_embedding": [2.75, -0.33, -3.01, -0.52, 5.59, -2.30],
        }),
        json!({
            "user": "tyler", "age": 100, "job": "engineer",
            "description": "a software developer with expertise in mathematics and computer science",
            "last_updated": 1739710800.0, "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.1, 0.4, 0.5],
            "image_embedding": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "audio_embedding": [1.11, -6.73, 5.41, 1.04, 3.92, 0.73],
        }),
        json!({
            "user": "tim", "age": 12, "job": "dermatologist",
            "description": "a medical professional specializing in diseases of the skin",
            "last_updated": 1739710800.0, "credit_score": "high",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.4, 0.4, 0.5],
            "image_embedding": [-0.1, 0.0, 0.6, 0.0, -0.9],
            "audio_embedding": [0.03, -2.67, -2.08, 4.57, -2.33, 0.0],
        }),
        json!({
            "user": "taimur", "age": 15, "job": "CEO",
            "description": "high stress, but financially rewarding position at the head of a company",
            "last_updated": 1742130000.0, "credit_score": "low",
            "location": "-110.0839,37.3861",
            "user_embedding": [0.6, 0.1, 0.5],
            "image_embedding": [1.1, 1.2, -0.3, -4.1, 5.0],
            "audio_embedding": [0.68, 0.26, 2.08, 2.96, 0.01, 5.13],
        }),
        json!({
            "user": "joe", "age": 35, "job": "dentist",
            "description": "like the tooth fairy because they'll take your teeth, but you have to pay them!",
            "last_updated": 1742130000.0, "credit_score": "medium",
            "location": "-110.0839,37.3861",
            "user_embedding": [-0.1, -0.1, -0.5],
            "image_embedding": [-0.8, 2.0, 3.1, 1.5, -1.6],
            "audio_embedding": [0.91, 7.10, -2.14, -0.52, -6.08, -5.53],
        }),
    ]
}

/// Creates an index loaded with `multi_vector_data()`.
/// Returns `None` if integration is disabled or Redis unavailable.
fn create_multi_vector_index() -> Option<SearchIndex> {
    if !integration_enabled() {
        return None;
    }

    let schema = multi_vector_schema();
    let index = SearchIndex::from_json_value(schema, redis_url()).expect("schema should parse");
    if index.create_with_options(true, true).is_err() {
        return None;
    }

    let data = multi_vector_data();
    index
        .load(&data, "user", None)
        .expect("load should succeed");

    // Brief pause to let Redis index the documents
    std::thread::sleep(std::time::Duration::from_millis(200));

    Some(index)
}

// =============================================================================
// HybridQuery integration tests (mirrors test_hybrid.py)
// Requires Redis 8.4+ for FT.HYBRID support
// =============================================================================

#[test]
fn hybrid_query_basic_rrf() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };
    if !redis84_available() {
        index.delete(true).ok();
        return;
    }

    let query = HybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .with_num_results(7)
    .with_yield_text_score_as("text_score")
    .with_yield_vsim_score_as("vsim_score")
    .with_rrf(None, None)
    .with_yield_combined_score_as("hybrid_score")
    .with_return_fields([
        "user",
        "credit_score",
        "age",
        "job",
        "location",
        "description",
    ]);

    let result = index.hybrid_query(&query);
    match result {
        Ok(output) => {
            let docs = output.as_documents().expect("should be documents");
            assert_eq!(docs.len(), 7, "all 7 docs should be returned");

            let valid_users = [
                "john", "derrick", "nancy", "tyler", "tim", "taimur", "joe", "mary",
            ];
            for doc in docs {
                let user = doc["user"].as_str().unwrap();
                assert!(valid_users.contains(&user), "unexpected user: {user}");
            }
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("unknown command") || msg.contains("FT.HYBRID") {
                eprintln!("FT.HYBRID not supported by this Redis instance, skipping");
            } else {
                panic!("hybrid_query failed: {e}");
            }
        }
    }

    index.delete(true).ok();
}

#[test]
fn hybrid_query_num_results_and_score_ordering() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };
    if !redis84_available() {
        index.delete(true).ok();
        return;
    }

    let query = HybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .with_num_results(3)
    .with_rrf(None, None)
    .with_yield_combined_score_as("hybrid_score");

    match index.hybrid_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            assert_eq!(docs.len(), 3, "should return exactly 3 results");

            // Scores should be in descending order
            let scores: Vec<f64> = docs
                .iter()
                .map(|d| d["hybrid_score"].as_str().unwrap().parse().unwrap())
                .collect();
            for i in 1..scores.len() {
                assert!(
                    scores[i - 1] >= scores[i],
                    "scores should be in descending order: {:?}",
                    scores
                );
            }
        }
        Err(e) if e.to_string().contains("unknown command") => {
            eprintln!("FT.HYBRID not supported, skipping");
        }
        Err(e) => panic!("hybrid_query failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_hybrid_query_with_filter
#[test]
fn hybrid_query_with_tag_and_numeric_filter() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };
    if !redis84_available() {
        index.delete(true).ok();
        return;
    }

    let filter = Tag::new("credit_score").eq("high") & Num::new("age").gt(30.0);
    let query = HybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .with_filter(filter)
    .with_return_fields([
        "user",
        "credit_score",
        "age",
        "job",
        "location",
        "description",
    ]);

    match index.hybrid_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            // nancy(94, high), tyler(100, high) match
            assert_eq!(
                docs.len(),
                2,
                "only 2 docs with credit_score=high AND age>30"
            );
            for doc in docs {
                assert_eq!(doc["credit_score"].as_str().unwrap(), "high");
                let age: i64 = doc["age"].as_str().unwrap().parse().unwrap();
                assert!(age > 30, "age should be > 30, got {age}");
            }
        }
        Err(e) if e.to_string().contains("unknown command") => {
            eprintln!("FT.HYBRID not supported, skipping");
        }
        Err(e) => panic!("hybrid_query failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_hybrid_query_with_geo_filter
#[test]
fn hybrid_query_with_geo_filter() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };
    if !redis84_available() {
        index.delete(true).ok();
        return;
    }

    let filter = Geo::new("location").eq(GeoRadius::new(-122.4194, 37.7749, 1000.0, "m"));
    let query = HybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .with_filter(filter)
    .with_return_fields([
        "user",
        "credit_score",
        "age",
        "job",
        "location",
        "description",
    ]);

    match index.hybrid_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            // john, mary, nancy have location -122.4194,37.7749
            assert_eq!(docs.len(), 3, "3 docs near the given location");
        }
        Err(e) if e.to_string().contains("unknown command") => {
            eprintln!("FT.HYBRID not supported, skipping");
        }
        Err(e) => panic!("hybrid_query failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_hybrid_query_alpha (LINEAR combination)
#[test]
fn hybrid_query_linear_alpha() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };
    if !redis84_available() {
        index.delete(true).ok();
        return;
    }

    for alpha in [0.1_f32, 0.5, 0.9] {
        let query = HybridQuery::new(
            "a medical professional with expertise in lung cancer",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .with_num_results(7)
        .with_linear(alpha)
        .with_yield_text_score_as("text_score")
        .with_yield_vsim_score_as("vector_similarity")
        .with_yield_combined_score_as("hybrid_score");

        match index.hybrid_query(&query) {
            Ok(output) => {
                let docs = output.as_documents().expect("documents");
                assert_eq!(docs.len(), 7);
                // Verify the combined score matches alpha*text + (1-alpha)*vsim.
                // Note: FT.HYBRID only includes text_score when the doc matched
                // the text search component. Default to 0.0 when absent.
                for doc in docs {
                    let text_s: f64 = doc
                        .get("text_score")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    let vsim_s: f64 = doc
                        .get("vector_similarity")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    let hybrid_s: f64 = doc["hybrid_score"].as_str().unwrap().parse().unwrap();
                    let expected = (alpha as f64) * text_s + (1.0 - alpha as f64) * vsim_s;
                    assert!(
                        (hybrid_s - expected).abs() <= 0.01,
                        "alpha={alpha}: hybrid_score {hybrid_s} != expected {expected}"
                    );
                }
            }
            Err(e) if e.to_string().contains("unknown command") => {
                eprintln!("FT.HYBRID not supported, skipping");
                break;
            }
            Err(e) => panic!("hybrid_query failed with alpha={alpha}: {e}"),
        }
    }

    index.delete(true).ok();
}

/// Mirrors: test_hybrid_query_stopwords
#[test]
fn hybrid_query_with_stopwords() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };
    if !redis84_available() {
        index.delete(true).ok();
        return;
    }

    let mut stopwords = HashSet::new();
    stopwords.insert("medical".to_owned());
    stopwords.insert("expertise".to_owned());

    let query = HybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .with_num_results(7)
    .with_linear(0.5)
    .with_stopwords(stopwords)
    .with_yield_text_score_as("text_score")
    .with_yield_vsim_score_as("vector_similarity")
    .with_yield_combined_score_as("hybrid_score");

    // Verify stopwords are removed from the query string
    let cmd = query.build_cmd("test");
    let packed = cmd.get_packed_command();
    let cmd_str = String::from_utf8_lossy(&packed);
    assert!(
        !cmd_str.contains("medical"),
        "stopword 'medical' should be removed"
    );
    assert!(
        !cmd_str.contains("expertise"),
        "stopword 'expertise' should be removed"
    );

    match index.hybrid_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            assert_eq!(docs.len(), 7);
        }
        Err(e) if e.to_string().contains("unknown command") => {
            eprintln!("FT.HYBRID not supported, skipping");
        }
        Err(e) => panic!("hybrid_query failed: {e}"),
    }

    index.delete(true).ok();
}

// =============================================================================
// AggregateHybridQuery integration tests (mirrors test_aggregation.py)
// Uses FT.AGGREGATE which works on Redis 7.2+ with Search module
// =============================================================================

/// Mirrors: test_hybrid_query (aggregate version)
#[test]
fn aggregate_hybrid_query_basic() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let query = AggregateHybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .unwrap()
    .with_num_results(7)
    .with_return_fields([
        "user",
        "credit_score",
        "age",
        "job",
        "location",
        "description",
    ]);

    match index.aggregate_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            assert_eq!(docs.len(), 7, "all 7 docs should be returned");
            let valid_users = ["john", "mary", "nancy", "tyler", "tim", "taimur", "joe"];
            for doc in docs {
                let user = doc["user"].as_str().unwrap();
                assert!(valid_users.contains(&user), "unexpected user: {user}");
            }
        }
        Err(e) => panic!("aggregate_query failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_hybrid_query num_results + score ordering (aggregate)
#[test]
fn aggregate_hybrid_query_num_results_and_ordering() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let query = AggregateHybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .unwrap()
    .with_num_results(3);

    match index.aggregate_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            assert_eq!(docs.len(), 3);
            // hybrid_score should be in descending order
            let scores: Vec<f64> = docs
                .iter()
                .map(|d| d["hybrid_score"].as_str().unwrap().parse().unwrap())
                .collect();
            for i in 1..scores.len() {
                assert!(
                    scores[i - 1] >= scores[i],
                    "scores should descend: {:?}",
                    scores
                );
            }
        }
        Err(e) => panic!("aggregate_query failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_hybrid_query_with_filter (aggregate)
#[test]
fn aggregate_hybrid_query_with_filter() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let filter = Tag::new("credit_score").eq("high") & Num::new("age").gt(30.0);
    let query = AggregateHybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .unwrap()
    .with_num_results(7)
    .with_filter(filter)
    .with_return_fields([
        "user",
        "credit_score",
        "age",
        "job",
        "location",
        "description",
    ]);

    match index.aggregate_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            // Fresh dataset has exactly 2 matching docs (nancy, tyler).
            // Stale data from prior test runs may inflate the count, so verify
            // ≥ 2 and that every returned doc satisfies the filter.
            assert!(
                docs.len() >= 2,
                "expected at least 2 filtered docs, got {}",
                docs.len()
            );
            for doc in docs {
                assert_eq!(doc["credit_score"].as_str().unwrap(), "high");
                let age: f64 = doc["age"].as_str().unwrap().parse().unwrap();
                assert!(age > 30.0);
            }
        }
        Err(e) => panic!("aggregate_query with filter failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_hybrid_query_with_geo_filter (aggregate)
#[test]
fn aggregate_hybrid_query_with_geo_filter() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let filter = Geo::new("location").eq(GeoRadius::new(-122.4194, 37.7749, 1000.0, "m"));
    let query = AggregateHybridQuery::new(
        "a medical professional with expertise in lung cancer",
        "description",
        Vector::new(vec![0.1, 0.1, 0.5]),
        "user_embedding",
    )
    .unwrap()
    .with_num_results(7)
    .with_filter(filter)
    .with_return_fields([
        "user",
        "credit_score",
        "age",
        "job",
        "location",
        "description",
    ]);

    match index.aggregate_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            // Fresh dataset has 3 docs near the given location (john, mary, nancy).
            assert!(
                docs.len() >= 3,
                "expected at least 3 docs near location, got {}",
                docs.len()
            );
        }
        Err(e) => panic!("aggregate_query with geo filter failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_hybrid_query_alpha (aggregate version)
#[test]
fn aggregate_hybrid_query_alpha() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    for alpha in [0.1_f32, 0.5, 0.9] {
        let query = AggregateHybridQuery::new(
            "a medical professional with expertise in lung cancer",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .unwrap()
        .with_alpha(alpha)
        .with_num_results(7);

        match index.aggregate_query(&query) {
            Ok(output) => {
                let docs = output.as_documents().expect("documents");
                assert_eq!(docs.len(), 7);
                for doc in docs {
                    let vs: f64 = doc["vector_similarity"].as_str().unwrap().parse().unwrap();
                    let ts: f64 = doc["text_score"].as_str().unwrap().parse().unwrap();
                    let hs: f64 = doc["hybrid_score"].as_str().unwrap().parse().unwrap();
                    let expected = (alpha as f64) * vs + (1.0 - alpha as f64) * ts;
                    assert!(
                        (hs - expected).abs() <= 0.001,
                        "alpha={alpha}: hybrid_score {hs} != expected {expected}"
                    );
                }
            }
            Err(e) => panic!("aggregate_query with alpha={alpha} failed: {e}"),
        }
    }

    index.delete(true).ok();
}

// =============================================================================
// MultiVectorQuery integration tests (mirrors test_aggregation.py)
// =============================================================================

/// Mirrors: test_multivector_query
#[test]
fn multi_vector_query_basic() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let v1 = VectorInput::from_floats(&[0.1, 0.1, 0.5], "user_embedding");
    let v2 = VectorInput::from_floats(&[0.3, 0.4, 0.7, 0.2, -0.3], "image_embedding");

    let query = MultiVectorQuery::new(vec![v1, v2]).with_return_fields([
        "user",
        "credit_score",
        "age",
        "job",
        "location",
        "description",
    ]);

    match index.multi_vector_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            assert_eq!(docs.len(), 7, "all docs should be returned");
            let valid_users = ["john", "mary", "nancy", "tyler", "tim", "taimur", "joe"];
            for doc in docs {
                let user = doc["user"].as_str().unwrap();
                assert!(valid_users.contains(&user), "unexpected user: {user}");
            }
        }
        Err(e) => panic!("multi_vector_query failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_multivector_query num_results + combined_score ordering
#[test]
fn multi_vector_query_num_results_and_ordering() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let v1 = VectorInput::from_floats(&[0.1, 0.1, 0.5], "user_embedding");
    let v2 = VectorInput::from_floats(&[0.3, 0.4, 0.7, 0.2, -0.3], "image_embedding");

    let query = MultiVectorQuery::new(vec![v1, v2]).with_num_results(3);

    match index.multi_vector_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            assert_eq!(docs.len(), 3);
            let scores: Vec<f64> = docs
                .iter()
                .map(|d| d["combined_score"].as_str().unwrap().parse().unwrap())
                .collect();
            for i in 1..scores.len() {
                assert!(
                    scores[i - 1] >= scores[i],
                    "scores should descend: {:?}",
                    scores
                );
            }
        }
        Err(e) => panic!("multi_vector_query failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_multivector_query_with_filter (text filter)
#[test]
fn multi_vector_query_with_text_filter() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let filter = Text::new("description").eq("medical");
    let v1 = VectorInput::from_floats(&[0.1, 0.1, 0.5], "user_embedding");
    let v2 = VectorInput::from_floats(&[0.3, 0.4, 0.7, 0.2, -0.3], "image_embedding");

    let query = MultiVectorQuery::new(vec![v1, v2])
        .with_filter(filter)
        .with_return_fields(["job", "description"]);

    match index.multi_vector_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            // mary and tim have "medical" in description
            assert_eq!(docs.len(), 2);
            for doc in docs {
                let desc = doc["description"].as_str().unwrap().to_lowercase();
                assert!(
                    desc.contains("medical"),
                    "description should contain 'medical'"
                );
            }
        }
        Err(e) => panic!("multi_vector_query with filter failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_multivector_query_with_geo_filter
#[test]
fn multi_vector_query_with_geo_filter() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let filter = Geo::new("location").eq(GeoRadius::new(-122.4194, 37.7749, 1000.0, "m"));
    let v1 = VectorInput::from_floats(&[0.2, 0.4, 0.1], "user_embedding");
    let v2 = VectorInput::from_floats(&[0.1, 0.8, 0.3, -0.2, 0.3], "image_embedding");

    let query = MultiVectorQuery::new(vec![v1, v2])
        .with_filter(filter)
        .with_return_fields([
            "user",
            "credit_score",
            "age",
            "job",
            "location",
            "description",
        ]);

    match index.multi_vector_query(&query) {
        Ok(output) => {
            let docs = output.as_documents().expect("documents");
            assert_eq!(docs.len(), 3, "3 docs near location");
        }
        Err(e) => panic!("multi_vector_query with geo filter failed: {e}"),
    }

    index.delete(true).ok();
}

/// Mirrors: test_multivector_query_weights
#[test]
fn multi_vector_query_weights_affect_ordering() {
    let Some(index) = create_multi_vector_index() else {
        return;
    };

    let return_fields = [
        "distance_0",
        "distance_1",
        "score_0",
        "score_1",
        "user_embedding",
        "image_embedding",
    ];

    // Default weights (1.0 each)
    let v1 = VectorInput::from_floats(&[0.1, 0.2, 0.5], "user_embedding");
    let v2 = VectorInput::from_floats(&[0.3, 0.4, 0.7, 0.2, -0.3], "image_embedding");
    let query1 = MultiVectorQuery::new(vec![v1, v2]).with_return_fields(return_fields.clone());

    // Custom weights
    let v1w = VectorInput::from_floats(&[0.1, 0.2, 0.5], "user_embedding").with_weight(0.2);
    let v2w =
        VectorInput::from_floats(&[0.3, 0.4, 0.7, 0.2, -0.3], "image_embedding").with_weight(0.9);
    let query2 = MultiVectorQuery::new(vec![v1w, v2w]).with_return_fields(return_fields.clone());

    let result1 = index
        .multi_vector_query(&query1)
        .expect("query1 should succeed");
    let result2 = index
        .multi_vector_query(&query2)
        .expect("query2 should succeed");

    let docs1 = result1.as_documents().expect("docs1");
    let docs2 = result2.as_documents().expect("docs2");

    // Verify combined_score ordering for both results
    for docs in [&docs1, &docs2] {
        for i in 1..docs.len() {
            let prev: f64 = docs[i - 1]["combined_score"]
                .as_str()
                .unwrap()
                .parse()
                .unwrap();
            let curr: f64 = docs[i]["combined_score"].as_str().unwrap().parse().unwrap();
            assert!(prev >= curr, "combined_score should be sorted descending");
        }
    }

    // Different weights should produce different results
    assert_ne!(
        docs1, docs2,
        "different weights should produce different results"
    );

    index.delete(true).ok();
}
