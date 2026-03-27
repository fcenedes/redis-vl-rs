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
            let score = (other.len() % 10) as f32 / 10.0;
            vec![score, 1.0 - score, 0.0]
        }
    }
}

fn build_cache(ttl_seconds: Option<u64>) -> Option<SemanticCache> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut config = CacheConfig::new(format!("python_parity_semcache_{id}"), redis_url());
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
