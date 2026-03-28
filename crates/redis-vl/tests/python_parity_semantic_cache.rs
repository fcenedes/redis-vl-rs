//! Integration tests derived from the upstream Python `test_llmcache.py`
//! semantic-cache contract.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::Duration,
};

use redis_vl::{CacheConfig, CustomTextVectorizer, SemanticCache};
use serde_json::{Map, json};

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

fn embed_text(text: &str) -> Vec<f32> {
    match text {
        "This is a test prompt." => vec![1.0, 0.0, 0.0],
        "This is also test prompt." => vec![0.9, 0.1, 0.0],
        "This is another test prompt." => vec![0.8, 0.2, 0.0],
        "This is another metadata prompt." => vec![0.0, 1.0, 0.0],
        "some random sentence" => vec![0.0, 0.0, 1.0],
        other => {
            // Use a simple hash-based approach to produce unique vectors for
            // different texts, even when they have the same length.
            let hash = other
                .bytes()
                .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
            let a = ((hash % 1000) as f32) / 1000.0;
            let b = (((hash / 1000) % 1000) as f32) / 1000.0;
            let c = 1.0 - a.max(b);
            vec![a, b, c]
        }
    }
}

fn build_cache(ttl_seconds: Option<u64>) -> Option<SemanticCache> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    let mut config = CacheConfig::new(format!("python_parity_semcache_{pid}_{id}"), redis_url());
    if let Some(ttl_seconds) = ttl_seconds {
        config = config.with_ttl(ttl_seconds);
    }

    Some(
        SemanticCache::new(config, 0.2, 3)
            .expect("cache should initialize")
            .with_vectorizer(CustomTextVectorizer::new(|text| Ok(embed_text(text)))),
    )
}

#[test]
fn python_test_semantic_cache_store_and_check() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    cache
        .store(
            "This is a test prompt.",
            "This is a test response.",
            None,
            None,
            None,
            None,
        )
        .expect("store should succeed");

    let hits = cache
        .check(
            Some("This is a test prompt."),
            None,
            1,
            None,
            None,
            Some(0.4),
        )
        .expect("check should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["response"], json!("This is a test response."));
    assert_eq!(hits[0]["prompt"], json!("This is a test prompt."));
    assert!(hits[0].get("vector_distance").is_some());
    assert!(hits[0].get("key").is_some());

    cache.delete().expect("delete should succeed");
}

#[test]
fn python_test_semantic_cache_metadata_update_and_drop() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    cache
        .store(
            "This is another metadata prompt.",
            "This is another metadata response.",
            Some(&embed_text("This is another metadata prompt.")),
            Some(json!({"source": "test"})),
            None,
            None,
        )
        .expect("store should succeed");

    let initial = cache
        .check(
            None,
            Some(&embed_text("This is another metadata prompt.")),
            1,
            Some(&["updated_at", "metadata", "entry_id"]),
            None,
            None,
        )
        .expect("check should succeed");
    let key = initial[0]["key"]
        .as_str()
        .expect("key should be a string")
        .to_owned();
    let entry_id = initial[0]["entry_id"]
        .as_str()
        .expect("entry id should be a string")
        .to_owned();
    let original_updated_at = initial[0]["updated_at"]
        .as_f64()
        .expect("updated_at should be numeric");

    thread::sleep(Duration::from_secs(1));

    let mut update = Map::new();
    update.insert("metadata".to_owned(), json!({"foo": "bar"}));
    cache.update(&key, update).expect("update should succeed");

    let updated = cache
        .check(
            Some("This is another metadata prompt."),
            None,
            1,
            Some(&["updated_at", "metadata"]),
            None,
            None,
        )
        .expect("check should succeed");
    assert_eq!(updated[0]["metadata"], json!({"foo": "bar"}));
    assert!(
        updated[0]["updated_at"]
            .as_f64()
            .expect("updated_at should be numeric")
            > original_updated_at
    );

    cache
        .drop_ids(&[entry_id])
        .expect("drop ids should succeed");
    let after_drop = cache
        .check(
            Some("This is another metadata prompt."),
            None,
            1,
            None,
            None,
            None,
        )
        .expect("check should succeed");
    assert!(after_drop.is_empty());

    cache.delete().expect("delete should succeed");
}

#[test]
fn python_test_semantic_cache_clear_and_ttl_behaviour() {
    let Some(cache) = build_cache(Some(2)) else {
        return;
    };

    cache
        .store(
            "This is a test prompt.",
            "This is a test response.",
            Some(&embed_text("This is a test prompt.")),
            None,
            None,
            None,
        )
        .expect("store should succeed");
    thread::sleep(Duration::from_secs(3));
    let expired = cache
        .check(
            None,
            Some(&embed_text("This is a test prompt.")),
            1,
            None,
            None,
            None,
        )
        .expect("check should succeed");
    assert!(expired.is_empty());

    cache
        .store(
            "This is a test prompt.",
            "This is a test response.",
            Some(&embed_text("This is a test prompt.")),
            None,
            None,
            Some(5),
        )
        .expect("store with custom ttl should succeed");
    thread::sleep(Duration::from_secs(3));
    let refreshed = cache
        .check(
            None,
            Some(&embed_text("This is a test prompt.")),
            1,
            None,
            None,
            None,
        )
        .expect("check should succeed");
    assert_eq!(refreshed.len(), 1);

    for _ in 0..3 {
        thread::sleep(Duration::from_secs(1));
        let hits = cache
            .check(
                None,
                Some(&embed_text("This is a test prompt.")),
                1,
                None,
                None,
                None,
            )
            .expect("ttl refresh check should succeed");
        assert_eq!(hits.len(), 1);
    }

    let deleted = cache.clear().expect("clear should succeed");
    assert!(deleted >= 1);
    let empty = cache
        .check(
            None,
            Some(&embed_text("This is a test prompt.")),
            1,
            None,
            None,
            None,
        )
        .expect("check should succeed");
    assert!(empty.is_empty());

    cache.delete().expect("delete should succeed");
}

#[tokio::test]
async fn python_test_async_semantic_cache_store_and_check() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    cache
        .astore(
            "This is a test prompt.",
            "This is a test response.",
            None,
            None,
            None,
            None,
        )
        .await
        .expect("astore should succeed");

    // Brief pause to let Redis index the newly stored document before querying.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let hits = cache
        .acheck(
            Some("This is a test prompt."),
            None,
            1,
            None,
            None,
            Some(0.4),
        )
        .await
        .expect("acheck should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["response"], json!("This is a test response."));

    cache.adelete().await.expect("adelete should succeed");
}

#[tokio::test]
async fn python_test_async_semantic_cache_update_and_clear() {
    let Some(cache) = build_cache(Some(2)) else {
        return;
    };

    cache
        .astore(
            "This is another metadata prompt.",
            "This is another metadata response.",
            Some(&embed_text("This is another metadata prompt.")),
            Some(json!({"source": "test"})),
            None,
            Some(5),
        )
        .await
        .expect("astore should succeed");

    let initial = cache
        .acheck(
            None,
            Some(&embed_text("This is another metadata prompt.")),
            1,
            Some(&["updated_at"]),
            None,
            None,
        )
        .await
        .expect("acheck should succeed");
    let key = initial[0]["key"]
        .as_str()
        .expect("key should be a string")
        .to_owned();

    tokio::time::sleep(Duration::from_secs(1)).await;
    let mut update = Map::new();
    update.insert("metadata".to_owned(), json!({"foo": "bar"}));
    cache
        .aupdate(&key, update)
        .await
        .expect("aupdate should succeed");

    let updated = cache
        .acheck(
            Some("This is another metadata prompt."),
            None,
            1,
            Some(&["metadata"]),
            None,
            None,
        )
        .await
        .expect("acheck should succeed");
    assert_eq!(updated[0]["metadata"], json!({"foo": "bar"}));

    let deleted = cache.aclear().await.expect("aclear should succeed");
    assert!(deleted >= 1);
    let empty = cache
        .acheck(
            Some("This is another metadata prompt."),
            None,
            1,
            None,
            None,
            None,
        )
        .await
        .expect("acheck should succeed");
    assert!(empty.is_empty());

    cache.adelete().await.expect("adelete should succeed");
}

/// Mirrors Python `test_check_no_match` — empty results for unmatched queries.
#[test]
fn python_test_check_no_match() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    let hits = cache
        .check(Some("some random sentence"), None, 1, None, None, None)
        .expect("check should succeed");
    assert!(hits.is_empty());

    cache.delete().expect("delete");
}

/// Mirrors Python `test_check_invalid_input` — no prompt or vector raises error.
#[test]
fn python_test_check_invalid_input() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    let err = cache.check(None, None, 1, None, None, None);
    assert!(err.is_err());

    cache.delete().expect("delete");
}

/// Mirrors Python `test_store_with_invalid_metadata` — non-object metadata
/// is rejected.
#[test]
fn python_test_store_with_invalid_metadata() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    let err = cache.store(
        "prompt",
        "response",
        Some(&embed_text("prompt")),
        Some(json!("string_metadata")),
        None,
        None,
    );
    assert!(err.is_err());

    cache.delete().expect("delete");
}

/// Mirrors Python `test_store_with_empty_metadata` — empty object is allowed.
#[test]
fn python_test_store_with_empty_metadata() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    cache
        .store(
            "empty metadata prompt",
            "empty metadata response",
            Some(&embed_text("empty metadata prompt")),
            Some(json!({})),
            None,
            None,
        )
        .expect("store with empty metadata should succeed");

    let hits = cache
        .check(
            Some("empty metadata prompt"),
            None,
            1,
            Some(&["metadata"]),
            None,
            Some(0.4),
        )
        .expect("check should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["metadata"], json!({}));

    cache.delete().expect("delete");
}

/// Mirrors Python `test_distance_threshold` / `test_distance_threshold_out_of_range`.
#[test]
fn python_test_distance_threshold_validation() {
    let Some(mut cache) = build_cache(None) else {
        return;
    };

    cache.set_threshold(0.1).expect("valid threshold");
    assert!((cache.distance_threshold - 0.1).abs() < f32::EPSILON);

    let err = cache.set_threshold(-1.0);
    assert!(err.is_err());

    cache.delete().expect("delete");
}

/// Mirrors Python `test_vector_size` — dimension mismatch is rejected.
#[test]
fn python_test_vector_size_mismatch() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    // Store with correct size
    cache
        .store(
            "test prompt",
            "test response",
            Some(&[1.0, 0.0, 0.0]),
            None,
            None,
            None,
        )
        .expect("store with correct dims");

    // Store with wrong size
    let err = cache.store(
        "test prompt",
        "test response",
        Some(&[1.0, 0.0]),
        None,
        None,
        None,
    );
    assert!(err.is_err());

    // Check with wrong size
    let err = cache.check(None, Some(&[1.0, 0.0]), 1, None, None, None);
    assert!(err.is_err());

    cache.delete().expect("delete");
}

/// Mirrors Python `test_drop_documents` — dropping multiple entries by id.
#[test]
fn python_test_drop_multiple_documents() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    let prompts = [
        "This is a test prompt.",
        "This is also test prompt.",
        "This is another test prompt.",
    ];
    let responses = [
        "This is a test response.",
        "This is also test response.",
        "This is another test response.",
    ];
    for (prompt, response) in prompts.iter().zip(responses.iter()) {
        cache
            .store(
                prompt,
                response,
                Some(&embed_text(prompt)),
                None,
                None,
                None,
            )
            .expect("store should succeed");
    }

    let hits = cache
        .check(
            Some("This is another test prompt."),
            None,
            3,
            None,
            None,
            Some(0.5),
        )
        .expect("check should succeed");
    assert!(hits.len() >= 2);

    let ids: Vec<String> = hits[..2]
        .iter()
        .map(|h| h["entry_id"].as_str().unwrap().to_owned())
        .collect();
    cache.drop_ids(&ids).expect("drop ids");

    let after = cache
        .check(
            Some("This is another test prompt."),
            None,
            3,
            None,
            None,
            Some(0.5),
        )
        .expect("recheck");
    assert_eq!(after.len(), hits.len() - 2);

    cache.delete().expect("delete");
}

/// Mirrors Python `test_cache_bad_filters` — reserved names, duplicates,
/// and invalid types are rejected.
#[test]
fn python_test_cache_bad_filters() {
    if !integration_enabled() {
        return;
    }

    // Invalid field type
    let config = CacheConfig::new("bad_filter_type", redis_url());
    let err = SemanticCache::with_filterable_fields(
        config,
        0.2,
        3,
        &[
            json!({"name": "label", "type": "tag"}),
            json!({"name": "test", "type": "nothing"}),
        ],
    );
    assert!(err.is_err());

    // Duplicate field name
    let config = CacheConfig::new("bad_filter_dup", redis_url());
    let err = SemanticCache::with_filterable_fields(
        config,
        0.2,
        3,
        &[
            json!({"name": "label", "type": "tag"}),
            json!({"name": "label", "type": "tag"}),
        ],
    );
    assert!(err.is_err());

    // Reserved field name
    let config = CacheConfig::new("bad_filter_reserved", redis_url());
    let err = SemanticCache::with_filterable_fields(
        config,
        0.2,
        3,
        &[
            json!({"name": "label", "type": "tag"}),
            json!({"name": "metadata", "type": "tag"}),
        ],
    );
    assert!(err.is_err());
}

/// Mirrors Python `test_cache_with_filters` — filterable fields appear in schema.
#[test]
fn python_test_cache_with_filters() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    let config = CacheConfig::new(format!("parity_filter_cache_{pid}_{id}"), redis_url());
    let cache = SemanticCache::with_filterable_fields(
        config,
        0.2,
        3,
        &[json!({"name": "label", "type": "tag"})],
    )
    .expect("should create cache with filters")
    .with_vectorizer(CustomTextVectorizer::new(|text| Ok(embed_text(text))));

    assert!(
        cache
            .index
            .schema()
            .fields
            .iter()
            .any(|f| f.name == "label")
    );
    cache.delete().expect("delete");
}

/// Mirrors Python `test_return_fields` — verify default and custom return fields.
#[test]
fn python_test_return_fields() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    cache
        .store(
            "This is a test prompt.",
            "This is a test response.",
            Some(&embed_text("This is a test prompt.")),
            None,
            None,
            None,
        )
        .expect("store");

    // Check default return fields
    let hits = cache
        .check(
            None,
            Some(&embed_text("This is a test prompt.")),
            1,
            None,
            None,
            None,
        )
        .expect("check");
    assert_eq!(hits.len(), 1);
    let keys: std::collections::HashSet<&str> = hits[0].keys().map(String::as_str).collect();
    assert!(keys.contains("key"));
    assert!(keys.contains("entry_id"));
    assert!(keys.contains("prompt"));
    assert!(keys.contains("response"));
    assert!(keys.contains("vector_distance"));
    assert!(keys.contains("inserted_at"));
    assert!(keys.contains("updated_at"));

    // Check specific return fields
    let fields = &["entry_id", "prompt", "response", "vector_distance"];
    let hits = cache
        .check(
            None,
            Some(&embed_text("This is a test prompt.")),
            1,
            Some(fields),
            None,
            None,
        )
        .expect("check with fields");
    let keys: std::collections::HashSet<&str> = hits[0].keys().map(String::as_str).collect();
    // key is always included
    assert!(keys.contains("key"));
    assert!(keys.contains("entry_id"));
    assert!(keys.contains("prompt"));
    assert!(!keys.contains("inserted_at"));
    assert!(!keys.contains("updated_at"));

    cache.delete().expect("delete");
}

/// Mirrors Python `test_multiple_items` — storing and checking multiple items.
#[test]
fn python_test_multiple_items() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    let items = vec![
        ("prompt1", "response1"),
        ("prompt2", "response2"),
        ("prompt3", "response3"),
    ];

    for (prompt, response) in &items {
        cache
            .store(prompt, response, None, None, None, None)
            .expect("store should succeed");
    }

    for (prompt, expected_response) in &items {
        let hits = cache
            .check(Some(prompt), None, 1, None, None, None)
            .expect("check");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["response"], json!(*expected_response));
    }

    cache.delete().expect("delete");
}

/// Mirrors Python `test_set_ttl` / `test_reset_ttl` — setting and clearing TTL.
#[test]
fn python_test_set_and_reset_ttl() {
    let Some(mut cache) = build_cache(None) else {
        return;
    };

    assert!(cache.ttl().is_none());
    cache.set_ttl(Some(5));
    assert_eq!(cache.ttl(), Some(5));
    cache.set_ttl(None);
    assert!(cache.ttl().is_none());

    cache.delete().expect("delete");
}

/// Mirrors Python async TTL expiration test.
#[tokio::test]
async fn python_test_async_ttl_expiration() {
    let Some(cache) = build_cache(Some(2)) else {
        return;
    };

    cache
        .astore(
            "This is a test prompt.",
            "This is a test response.",
            Some(&embed_text("This is a test prompt.")),
            None,
            None,
            None,
        )
        .await
        .expect("astore");

    tokio::time::sleep(Duration::from_secs(3)).await;

    let expired = cache
        .acheck(
            None,
            Some(&embed_text("This is a test prompt.")),
            1,
            None,
            None,
            None,
        )
        .await
        .expect("acheck");
    assert!(expired.is_empty());

    cache.adelete().await.expect("adelete");
}

/// Mirrors Python async drop document test.
#[tokio::test]
async fn python_test_async_drop_document() {
    let Some(cache) = build_cache(None) else {
        return;
    };

    cache
        .astore(
            "This is a test prompt.",
            "This is a test response.",
            Some(&embed_text("This is a test prompt.")),
            None,
            None,
            None,
        )
        .await
        .expect("astore");

    let hits = cache
        .acheck(
            None,
            Some(&embed_text("This is a test prompt.")),
            1,
            None,
            None,
            None,
        )
        .await
        .expect("acheck");
    assert_eq!(hits.len(), 1);

    let entry_id = hits[0]["entry_id"].as_str().expect("entry_id").to_owned();
    cache.adrop_ids(&[entry_id]).await.expect("adrop_ids");

    let after = cache
        .acheck(
            None,
            Some(&embed_text("This is a test prompt.")),
            1,
            None,
            None,
            None,
        )
        .await
        .expect("acheck");
    assert!(after.is_empty());

    cache.adelete().await.expect("adelete");
}
