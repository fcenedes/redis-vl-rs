//! Integration tests derived from the upstream Python `test_embedcache.py`
//! parity contract.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::Duration,
};

use redis_vl::{CacheConfig, EmbeddingCacheItem, EmbeddingsCache};
use serde_json::json;

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

fn create_cache(ttl_seconds: Option<u64>) -> Option<EmbeddingsCache> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    let mut config = CacheConfig::new(format!("python_parity_embedcache_{pid}_{id}"), redis_url());
    if let Some(ttl_seconds) = ttl_seconds {
        config = config.with_ttl(ttl_seconds);
    }
    Some(EmbeddingsCache::new(config))
}

fn sample_items() -> Vec<EmbeddingCacheItem> {
    vec![
        EmbeddingCacheItem {
            content: "What is machine learning?".to_owned(),
            model_name: "text-embedding-ada-002".to_owned(),
            embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            metadata: Some(json!({"source": "user_query", "category": "ai"})),
        },
        EmbeddingCacheItem {
            content: "How do neural networks work?".to_owned(),
            model_name: "text-embedding-ada-002".to_owned(),
            embedding: vec![0.2, 0.3, 0.4, 0.5, 0.6],
            metadata: Some(json!({"source": "documentation", "category": "ai"})),
        },
        EmbeddingCacheItem {
            content: "What's the weather like today?".to_owned(),
            model_name: "text-embedding-ada-002".to_owned(),
            embedding: vec![0.5, 0.6, 0.7, 0.8, 0.9],
            metadata: Some(json!({"source": "user_query", "category": "weather"})),
        },
    ]
}

#[test]
fn python_test_make_entry_id_and_cache_key() {
    let cache = EmbeddingsCache::default();
    let entry_id = cache.make_entry_id("Hello world", "text-embedding-ada-002");
    assert_eq!(
        entry_id,
        "368dacc611e96e4189a9809faaca1a70b3c3306352bbcfc9ab6291359a5dfca0"
    );
    assert_eq!(
        cache.make_cache_key("Hello world", "text-embedding-ada-002"),
        format!("embedcache:{entry_id}")
    );
}

#[test]
fn python_test_set_get_exists_and_drop() {
    let Some(cache) = create_cache(None) else {
        return;
    };
    let sample = &sample_items()[0];

    assert!(
        !cache
            .exists(&sample.content, &sample.model_name)
            .expect("exists should succeed")
    );

    let key = cache
        .set(
            &sample.content,
            &sample.model_name,
            &sample.embedding,
            sample.metadata.clone(),
            None,
        )
        .expect("set should succeed");

    assert!(cache.exists_by_key(&key).expect("exists by key"));
    let entry = cache
        .get(&sample.content, &sample.model_name)
        .expect("get should succeed")
        .expect("entry should exist");
    assert_eq!(entry.content, sample.content);
    assert_eq!(entry.model_name, sample.model_name);
    assert_eq!(entry.embedding, sample.embedding);
    assert_eq!(entry.metadata, sample.metadata);
    assert_eq!(
        entry.entry_id,
        cache.make_entry_id(&sample.content, &sample.model_name)
    );

    let by_key = cache
        .get_by_key(&key)
        .expect("get by key should succeed")
        .expect("entry should exist");
    assert_eq!(by_key.content, sample.content);

    cache
        .drop(&sample.content, &sample.model_name)
        .expect("drop should succeed");
    assert!(!cache.exists_by_key(&key).expect("exists by key"));

    cache.clear().expect("clear");
}

#[test]
fn python_test_mset_mget_mexists_and_mdrop() {
    let Some(cache) = create_cache(None) else {
        return;
    };
    let items = sample_items();
    let keys = cache.mset(&items, None).expect("mset should succeed");
    assert_eq!(keys.len(), items.len());

    let contents = items
        .iter()
        .map(|item| item.content.as_str())
        .collect::<Vec<_>>();
    let results = cache
        .mget(contents.iter().copied(), "text-embedding-ada-002")
        .expect("mget should succeed");
    assert_eq!(results.len(), items.len());
    assert_eq!(
        results[0].as_ref().map(|entry| entry.content.as_str()),
        Some(items[0].content.as_str())
    );

    let by_keys = cache
        .mget_by_keys(keys.iter().map(String::as_str))
        .expect("mget by keys should succeed");
    assert_eq!(by_keys.len(), items.len());
    assert!(by_keys.iter().all(Option::is_some));

    let exists = cache
        .mexists(contents.iter().copied(), "text-embedding-ada-002")
        .expect("mexists should succeed");
    assert!(exists.into_iter().all(|value| value));

    let subset = contents[..2].to_vec();
    cache
        .mdrop(subset.iter().copied(), "text-embedding-ada-002")
        .expect("mdrop should succeed");
    assert!(
        !cache
            .exists(&items[0].content, &items[0].model_name)
            .expect("exists should succeed")
    );
    assert!(
        cache
            .exists(&items[2].content, &items[2].model_name)
            .expect("exists should succeed")
    );

    cache.clear().expect("clear");
}

#[test]
fn python_test_ttl_and_custom_ttl_override() {
    let Some(cache) = create_cache(Some(2)) else {
        return;
    };
    let sample = &sample_items()[0];

    let key = cache
        .set(
            &sample.content,
            &sample.model_name,
            &sample.embedding,
            sample.metadata.clone(),
            None,
        )
        .expect("set should succeed");
    assert!(cache.exists_by_key(&key).expect("exists by key"));
    thread::sleep(Duration::from_secs(3));
    assert!(!cache.exists_by_key(&key).expect("exists by key"));

    let custom_key = cache
        .set(
            &sample.content,
            &sample.model_name,
            &sample.embedding,
            sample.metadata.clone(),
            Some(5),
        )
        .expect("set with custom ttl should succeed");
    thread::sleep(Duration::from_secs(3));
    assert!(cache.exists_by_key(&custom_key).expect("exists by key"));

    cache.clear().expect("clear");
}

#[test]
fn python_test_batch_operations_with_missing_data() {
    let Some(cache) = create_cache(None) else {
        return;
    };

    assert!(
        cache
            .mget_by_keys(std::iter::empty::<&str>())
            .expect("empty mget")
            .is_empty()
    );
    assert!(
        cache
            .mexists_by_keys(std::iter::empty::<&str>())
            .expect("empty mexists")
            .is_empty()
    );
    cache
        .mdrop_by_keys(std::iter::empty::<&str>())
        .expect("empty mdrop");

    let missing = ["missing:key:1", "missing:key:2"];
    let results = cache
        .mget_by_keys(missing.iter().copied())
        .expect("mget by missing keys");
    assert_eq!(results, vec![None, None]);

    let exists = cache
        .mexists_by_keys(missing.iter().copied())
        .expect("mexists by missing keys");
    assert_eq!(exists, vec![false, false]);
}

#[tokio::test]
async fn python_test_async_set_get_and_exists() {
    let Some(cache) = create_cache(None) else {
        return;
    };
    let sample = &sample_items()[0];

    let key = cache
        .aset(
            &sample.content,
            &sample.model_name,
            &sample.embedding,
            sample.metadata.clone(),
            None,
        )
        .await
        .expect("aset should succeed");
    assert!(cache.aexists_by_key(&key).await.expect("aexists by key"));

    let result = cache
        .aget(&sample.content, &sample.model_name)
        .await
        .expect("aget should succeed")
        .expect("entry should exist");
    assert_eq!(result.content, sample.content);
    assert_eq!(result.metadata, sample.metadata);

    cache.adrop_by_key(&key).await.expect("adrop by key");
    assert!(!cache.aexists_by_key(&key).await.expect("aexists by key"));

    cache.aclear().await.expect("aclear");
}

#[tokio::test]
async fn python_test_async_batch_operations() {
    let Some(cache) = create_cache(None) else {
        return;
    };
    let items = sample_items();
    let keys = cache
        .amset(&items, None)
        .await
        .expect("amset should succeed");
    assert_eq!(keys.len(), items.len());

    let contents = items
        .iter()
        .map(|item| item.content.as_str())
        .collect::<Vec<_>>();
    let results = cache
        .amget(contents.iter().copied(), "text-embedding-ada-002")
        .await
        .expect("amget should succeed");
    assert_eq!(results.len(), items.len());
    assert!(results.iter().all(Option::is_some));

    let exists = cache
        .amexists_by_keys(keys.iter().map(String::as_str))
        .await
        .expect("amexists by keys");
    assert!(exists.into_iter().all(|value| value));

    cache
        .amdrop_by_keys(keys[..2].iter().map(String::as_str))
        .await
        .expect("amdrop by keys");
    assert!(
        !cache
            .aexists(&items[0].content, &items[0].model_name)
            .await
            .expect("aexists")
    );
    assert!(
        cache
            .aexists(&items[2].content, &items[2].model_name)
            .await
            .expect("aexists")
    );

    cache.aclear().await.expect("aclear");
}

#[tokio::test]
async fn python_test_async_ttl_expiration() {
    let Some(cache) = create_cache(Some(2)) else {
        return;
    };
    let sample = &sample_items()[0];

    let key = cache
        .aset(
            &sample.content,
            &sample.model_name,
            &sample.embedding,
            sample.metadata.clone(),
            None,
        )
        .await
        .expect("aset should succeed");
    assert!(cache.aexists_by_key(&key).await.expect("aexists by key"));
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(!cache.aexists_by_key(&key).await.expect("aexists by key"));
}

/// Mirrors Python `test_batch_with_ttl` — mset with default TTL and custom
/// TTL override.
#[test]
fn python_test_batch_with_ttl() {
    let Some(cache) = create_cache(Some(2)) else {
        return;
    };
    let items = sample_items();

    // Store with default TTL of 2 seconds
    let keys = cache.mset(&items, None).expect("mset should succeed");
    let exists = cache
        .mexists_by_keys(keys.iter().map(String::as_str))
        .expect("mexists by keys");
    assert!(exists.iter().all(|&e| e));

    thread::sleep(Duration::from_secs(3));
    let after_expire = cache
        .mexists_by_keys(keys.iter().map(String::as_str))
        .expect("mexists by keys after TTL");
    assert!(after_expire.iter().all(|&e| !e));

    // Store with custom TTL override of 5 seconds
    let keys = cache.mset(&items, Some(5)).expect("mset with custom ttl");
    thread::sleep(Duration::from_secs(3));
    let still_present = cache
        .mexists_by_keys(keys.iter().map(String::as_str))
        .expect("mexists by keys after 3s with 5s ttl");
    assert!(still_present.iter().all(|&e| e));

    cache.clear().expect("clear");
}

/// Mirrors Python `test_large_batch_operations` — 100 items batch test.
#[test]
fn python_test_large_batch_operations() {
    let Some(cache) = create_cache(None) else {
        return;
    };

    let large_batch: Vec<EmbeddingCacheItem> = (0..100)
        .map(|i| EmbeddingCacheItem {
            content: format!("Sample text {i}"),
            model_name: "test-model".to_owned(),
            embedding: vec![i as f32 / 100.0; 5],
            metadata: Some(json!({"index": i})),
        })
        .collect();

    let keys = cache.mset(&large_batch, None).expect("mset 100 items");
    assert_eq!(keys.len(), 100);

    let results = cache
        .mget_by_keys(keys.iter().map(String::as_str))
        .expect("mget by keys");
    assert_eq!(results.len(), 100);
    assert!(results.iter().all(Option::is_some));

    let contents: Vec<&str> = large_batch.iter().map(|i| i.content.as_str()).collect();
    let results_by_content = cache
        .mget(contents.iter().copied(), "test-model")
        .expect("mget by content");
    assert_eq!(results_by_content.len(), 100);
    assert!(results_by_content.iter().all(Option::is_some));

    let exists = cache
        .mexists_by_keys(keys.iter().map(String::as_str))
        .expect("mexists");
    assert_eq!(exists.len(), 100);
    assert!(exists.iter().all(|&e| e));

    // Delete first half
    cache
        .mdrop_by_keys(keys[..50].iter().map(String::as_str))
        .expect("mdrop first half");
    for (i, key) in keys.iter().enumerate() {
        let exists = cache.exists_by_key(key).expect("exists check");
        if i < 50 {
            assert!(!exists, "key {i} should be deleted");
        } else {
            assert!(exists, "key {i} should still exist");
        }
    }

    cache.clear().expect("clear");
}

/// Mirrors Python `test_mget_by_keys` with mixed existing/non-existing keys.
#[test]
fn python_test_mget_by_keys_mixed() {
    let Some(cache) = create_cache(None) else {
        return;
    };
    let items = sample_items();
    let keys = cache.mset(&items, None).expect("mset");

    let non_existent_key = format!("{}_nonexistent", cache.config.name);
    let mixed_keys: Vec<String> = vec![keys[0].clone(), non_existent_key, keys[1].clone()];
    let results = cache
        .mget_by_keys(mixed_keys.iter().map(String::as_str))
        .expect("mget by mixed keys");
    assert_eq!(results.len(), 3);
    assert!(results[0].is_some());
    assert!(results[1].is_none());
    assert!(results[2].is_some());

    cache.clear().expect("clear");
}

/// Mirrors Python `test_entry_id_consistency` — entry IDs are consistent
/// between make_entry_id and the key returned by set.
#[test]
fn python_test_entry_id_consistency() {
    let Some(cache) = create_cache(None) else {
        return;
    };
    let sample = &sample_items()[0];

    let expected_id = cache.make_entry_id(&sample.content, &sample.model_name);
    let key = cache
        .set(
            &sample.content,
            &sample.model_name,
            &sample.embedding,
            sample.metadata.clone(),
            None,
        )
        .expect("set");

    // Key should be cache_name:entry_id
    let parts: Vec<&str> = key.splitn(2, ':').collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[1], expected_id);

    let result = cache
        .get_by_key(&key)
        .expect("get by key")
        .expect("entry exists");
    assert_eq!(result.entry_id, expected_id);

    cache.clear().expect("clear");
}
