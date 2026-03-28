//! Criterion benchmarks for Redis-backed `redis-vl` operations.
//!
//! These benchmarks require a running Redis instance and are **opt-in** via
//! environment variable. Run with:
//!
//! ```bash
//! REDISVL_RUN_BENCHMARKS=1 cargo bench --bench redis_benchmarks
//! ```
//!
//! The benchmarks use a test Redis database and clean up after themselves.
#![allow(missing_docs)]

use std::{
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use redis_vl::{
    CacheConfig, CountQuery, EmbeddingsCache, FilterQuery, IndexSchema, Message, MessageHistory,
    MessageRole, SearchIndex, SemanticCache, SemanticMessageHistory, Vector, VectorQuery,
    filter::{Num, Tag},
    vectorizers::CustomTextVectorizer,
};
use serde_json::{Value, json};

static BENCH_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Maximum iterations per benchmark sample to avoid exhausting macOS
/// ephemeral ports.  Each iteration opens 1-3 TCP connections that linger
/// in TIME_WAIT (~30 s on macOS with default MSL), so we cap total
/// connections per benchmark function to stay well under the ~16 k port
/// limit.
const MAX_ITERS: u64 = 10;

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

fn benchmarks_enabled() -> bool {
    std::env::var("REDISVL_RUN_BENCHMARKS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Run a benchmark operation with a capped iteration count, extrapolating
/// timing so Criterion computes valid per-iteration estimates.
fn capped_iter_custom<F>(b: &mut criterion::Bencher, mut op: F)
where
    F: FnMut(),
{
    b.iter_custom(|iters| {
        let actual = iters.min(MAX_ITERS);
        let start = std::time::Instant::now();
        for _ in 0..actual {
            op();
        }
        let elapsed = start.elapsed();
        if actual > 0 && actual < iters {
            elapsed.mul_f64(iters as f64 / actual as f64)
        } else {
            elapsed
        }
    });
}

fn unique_name(prefix: &str) -> String {
    let id = BENCH_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_bench_{id}")
}

// ---------------------------------------------------------------------------
// Search Index: Creation and Lifecycle
// ---------------------------------------------------------------------------

const BASIC_SCHEMA_YAML: &str = r#"
index:
  name: bench-index
  prefix: doc
  storage_type: hash
fields:
  - name: title
    type: tag
  - name: content
    type: text
  - name: score
    type: numeric
  - name: embedding
    type: vector
    attrs:
      algorithm: FLAT
      dims: 128
      distance_metric: COSINE
      datatype: FLOAT32
"#;

fn bench_index_create(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    // Use iter_custom to cap iterations — each create+delete cycle opens
    // multiple TCP connections that linger in TIME_WAIT on macOS.  We cap
    // at MAX_ITERS and extrapolate timing so Criterion gets valid estimates.
    c.bench_function("index_create", |b| {
        b.iter_custom(|iters| {
            let actual = iters.min(MAX_ITERS);
            let mut total = std::time::Duration::ZERO;
            for _ in 0..actual {
                let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
                schema.index.name = unique_name("idx_create");
                let index = SearchIndex::new(schema, redis_url());
                let start = std::time::Instant::now();
                index.create().unwrap();
                total += start.elapsed();
                let _ = index.delete(true);
            }
            // Extrapolate to the requested iteration count so Criterion
            // calculates per-iteration time correctly.
            if actual > 0 && actual < iters {
                total.mul_f64(iters as f64 / actual as f64)
            } else {
                total
            }
        });
    });
}

fn bench_index_exists(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_exists");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    c.bench_function("index_exists", |b| {
        capped_iter_custom(b, || {
            black_box(index.exists().unwrap());
        });
    });

    let _ = index.delete(true);
}

fn bench_index_info(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_info");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    c.bench_function("index_info", |b| {
        capped_iter_custom(b, || {
            black_box(index.info().unwrap());
        });
    });

    let _ = index.delete(true);
}

// ---------------------------------------------------------------------------
// Search Index: Load and Fetch
// ---------------------------------------------------------------------------

fn bench_index_load_single(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_load_single");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    let doc = json!({
        "id": "doc:1",
        "title": "test",
        "content": "benchmark document",
        "score": 100,
        "embedding": vec![0.1_f32; 128]
    });

    c.bench_function("index_load_single", |b| {
        capped_iter_custom(b, || {
            index.load(&[black_box(doc.clone())], "id", None).unwrap();
        });
    });

    let _ = index.delete(true);
}

fn bench_index_load_batch(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_load_batch");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    let mut group = c.benchmark_group("index_load_batch");
    for size in [10, 50, 100, 500].iter() {
        let docs: Vec<Value> = (0..*size)
            .map(|i| {
                json!({
                    "id": format!("doc:{i}"),
                    "title": format!("title-{i}"),
                    "content": "benchmark document with more content here",
                    "score": i,
                    "embedding": vec![0.1_f32; 128]
                })
            })
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &docs, |b, docs| {
            capped_iter_custom(b, || {
                index.clear().unwrap();
                index.load(black_box(docs), "id", None).unwrap();
            });
        });
    }
    group.finish();

    let _ = index.delete(true);
}

fn bench_index_fetch(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_fetch");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    // Load test data — use string-only fields (no raw binary vectors) because
    // hash-storage fetch decodes values as UTF-8 strings.
    let docs: Vec<Value> = (0..100)
        .map(|i| {
            json!({
                "id": format!("doc:{i}"),
                "title": format!("title-{i}"),
                "content": "benchmark document",
                "score": i,
            })
        })
        .collect();
    index.load(&docs, "id", None).unwrap();

    c.bench_function("index_fetch_single", |b| {
        capped_iter_custom(b, || {
            black_box(index.fetch("doc:50").unwrap());
        });
    });

    c.bench_function("index_fetch_batch", |b| {
        let ids: Vec<&str> = (0..20).map(|i| docs[i]["id"].as_str().unwrap()).collect();
        capped_iter_custom(b, || {
            for id in black_box(&ids) {
                black_box(index.fetch(id).unwrap());
            }
        });
    });

    let _ = index.delete(true);
}

// ---------------------------------------------------------------------------
// Search: Vector Query Performance
// ---------------------------------------------------------------------------

fn bench_search_vector_small(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_search_vec_small");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    // Load 100 documents
    let docs: Vec<Value> = (0..100)
        .map(|i| {
            json!({
                "id": format!("doc:{i}"),
                "title": format!("title-{i}"),
                "content": "benchmark document with searchable content",
                "score": i,
                "embedding": (0..128).map(|_| rand::random::<f32>()).collect::<Vec<_>>()
            })
        })
        .collect();
    index.load(&docs, "id", None).unwrap();

    let query_vec = (0..128).map(|_| rand::random::<f32>()).collect::<Vec<_>>();
    let query = VectorQuery::new(Vector::new(&query_vec), "embedding", 10);

    c.bench_function("search_vector_k10_n100", |b| {
        capped_iter_custom(b, || {
            black_box(index.search(&query).unwrap());
        });
    });

    let _ = index.delete(true);
}

fn bench_search_vector_with_filter(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_search_vec_filter");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    // Load 100 documents
    let docs: Vec<Value> = (0..100)
        .map(|i| {
            json!({
                "id": format!("doc:{i}"),
                "title": if i % 2 == 0 { "even" } else { "odd" },
                "content": "benchmark document",
                "score": i,
                "embedding": (0..128).map(|_| rand::random::<f32>()).collect::<Vec<_>>()
            })
        })
        .collect();
    index.load(&docs, "id", None).unwrap();

    let query_vec = (0..128).map(|_| rand::random::<f32>()).collect::<Vec<_>>();
    let filter = Tag::new("title").eq("even") & Num::new("score").gte(50.0);
    let query = VectorQuery::new(Vector::new(&query_vec), "embedding", 10).with_filter(filter);

    c.bench_function("search_vector_with_filter", |b| {
        capped_iter_custom(b, || {
            black_box(index.search(&query).unwrap());
        });
    });

    let _ = index.delete(true);
}

// ---------------------------------------------------------------------------
// Search: Filter and Count Queries
// ---------------------------------------------------------------------------

fn bench_search_filter(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_search_filter");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    // Load test data
    let docs: Vec<Value> = (0..500)
        .map(|i| {
            json!({
                "id": format!("doc:{i}"),
                "title": format!("category-{}", i % 10),
                "content": format!("content for document {i}"),
                "score": i,
                "embedding": vec![0.1_f32; 128]
            })
        })
        .collect();
    index.load(&docs, "id", None).unwrap();

    c.bench_function("search_filter_simple", |b| {
        let filter = Tag::new("title").eq("category-5");
        let query = FilterQuery::new(filter);
        capped_iter_custom(b, || {
            black_box(index.search(&query).unwrap());
        });
    });

    c.bench_function("search_filter_compound", |b| {
        let filter = Tag::new("title").eq("category-5")
            & Num::new("score").between(100.0, 300.0, redis_vl::BetweenInclusivity::Both);
        let query = FilterQuery::new(filter);
        capped_iter_custom(b, || {
            black_box(index.search(&query).unwrap());
        });
    });

    let _ = index.delete(true);
}

fn bench_search_count(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_search_count");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    // Load test data
    let docs: Vec<Value> = (0..1000)
        .map(|i| {
            json!({
                "id": format!("doc:{i}"),
                "title": format!("category-{}", i % 10),
                "content": "benchmark content",
                "score": i,
                "embedding": vec![0.1_f32; 128]
            })
        })
        .collect();
    index.load(&docs, "id", None).unwrap();

    c.bench_function("search_count", |b| {
        let filter = Tag::new("title").eq("category-5");
        let query = CountQuery::new().with_filter(filter);
        capped_iter_custom(b, || {
            black_box(index.query(&query).unwrap());
        });
    });

    let _ = index.delete(true);
}

// ---------------------------------------------------------------------------
// Search: Batch and Pagination
// ---------------------------------------------------------------------------

fn bench_search_batch(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_search_batch");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    // Load test data
    let docs: Vec<Value> = (0..200)
        .map(|i| {
            json!({
                "id": format!("doc:{i}"),
                "title": format!("title-{i}"),
                "content": "benchmark content",
                "score": i,
                "embedding": (0..128).map(|_| rand::random::<f32>()).collect::<Vec<_>>()
            })
        })
        .collect();
    index.load(&docs, "id", None).unwrap();

    let mut group = c.benchmark_group("search_batch");
    for count in [5, 10, 20].iter() {
        let queries: Vec<VectorQuery> = (0..*count)
            .map(|_| {
                let vec = (0..128).map(|_| rand::random::<f32>()).collect::<Vec<_>>();
                VectorQuery::new(Vector::new(vec), "embedding", 5)
            })
            .collect();

        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &queries,
            |b, queries| {
                capped_iter_custom(b, || {
                    black_box(index.batch_search(queries.iter()).unwrap());
                });
            },
        );
    }
    group.finish();

    let _ = index.delete(true);
}

fn bench_search_paginate(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let mut schema = IndexSchema::from_yaml_str(BASIC_SCHEMA_YAML).unwrap();
    schema.index.name = unique_name("idx_search_paginate");
    let index = SearchIndex::new(schema, redis_url());
    index.create().unwrap();

    // Load test data
    let docs: Vec<Value> = (0..100)
        .map(|i| {
            json!({
                "id": format!("doc:{i}"),
                "title": "match",
                "content": "benchmark content",
                "score": i,
                "embedding": vec![0.1_f32; 128]
            })
        })
        .collect();
    index.load(&docs, "id", None).unwrap();

    c.bench_function("search_paginate", |b| {
        let filter = Tag::new("title").eq("match");
        let query = FilterQuery::new(filter);
        capped_iter_custom(b, || {
            black_box(index.paginate(&query, 10).unwrap());
        });
    });

    let _ = index.delete(true);
}

// ---------------------------------------------------------------------------
// Cache: Embeddings Cache
// ---------------------------------------------------------------------------

fn bench_embeddings_cache_set(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let config = CacheConfig {
        name: unique_name("embedcache"),
        connection: redis_vl::index::RedisConnectionInfo::new(redis_url()),
        ttl_seconds: None,
    };
    let cache = EmbeddingsCache::new(config);

    c.bench_function("embedcache_set", |b| {
        let embedding = vec![0.1_f32; 128];
        capped_iter_custom(b, || {
            cache
                .set(
                    "test content",
                    "text-embedding-ada-002",
                    black_box(&embedding),
                    None,
                    None,
                )
                .unwrap();
        });
    });

    // Cleanup
    cache.clear().unwrap();
}

fn bench_embeddings_cache_get_hit(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let config = CacheConfig {
        name: unique_name("embedcache"),
        connection: redis_vl::index::RedisConnectionInfo::new(redis_url()),
        ttl_seconds: None,
    };
    let cache = EmbeddingsCache::new(config);

    // Pre-populate
    let embedding = vec![0.1_f32; 128];
    cache
        .set(
            "cached content",
            "text-embedding-ada-002",
            &embedding,
            None,
            None,
        )
        .unwrap();

    c.bench_function("embedcache_get_hit", |b| {
        capped_iter_custom(b, || {
            black_box(
                cache
                    .get("cached content", "text-embedding-ada-002")
                    .unwrap(),
            );
        });
    });

    cache.clear().unwrap();
}

fn bench_embeddings_cache_get_miss(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let config = CacheConfig {
        name: unique_name("embedcache"),
        connection: redis_vl::index::RedisConnectionInfo::new(redis_url()),
        ttl_seconds: None,
    };
    let cache = EmbeddingsCache::new(config);

    c.bench_function("embedcache_get_miss", |b| {
        capped_iter_custom(b, || {
            black_box(
                cache
                    .get("uncached content", "text-embedding-ada-002")
                    .unwrap(),
            );
        });
    });

    cache.clear().unwrap();
}

// ---------------------------------------------------------------------------
// Cache: Semantic Cache
// ---------------------------------------------------------------------------

fn simple_vectorizer(text: &str) -> Vec<f32> {
    let mut vec = vec![0.0_f32; 128];
    for (i, byte) in text.bytes().enumerate() {
        vec[i % 128] += byte as f32 / 255.0;
    }
    vec
}

fn bench_semantic_cache_store(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let config = CacheConfig {
        name: unique_name("semcache"),
        connection: redis_vl::index::RedisConnectionInfo::new(redis_url()),
        ttl_seconds: None,
    };
    let cache = SemanticCache::new(config, 0.3, 128)
        .unwrap()
        .with_vectorizer(CustomTextVectorizer::new(|text| {
            Ok(simple_vectorizer(text))
        }));

    c.bench_function("semcache_store", |b| {
        capped_iter_custom(b, || {
            cache
                .store(
                    "what is the capital of France?",
                    "Paris",
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
        });
    });

    cache.delete().unwrap();
}

fn bench_semantic_cache_check_hit(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let config = CacheConfig {
        name: unique_name("semcache"),
        connection: redis_vl::index::RedisConnectionInfo::new(redis_url()),
        ttl_seconds: None,
    };
    let cache = SemanticCache::new(config, 0.3, 128)
        .unwrap()
        .with_vectorizer(CustomTextVectorizer::new(|text| {
            Ok(simple_vectorizer(text))
        }));

    // Pre-populate
    cache
        .store(
            "what is the capital of France?",
            "Paris",
            None,
            None,
            None,
            None,
        )
        .unwrap();

    c.bench_function("semcache_check_hit", |b| {
        capped_iter_custom(b, || {
            black_box(
                cache
                    .check(
                        Some("what is the capital of France?"),
                        None,
                        1,
                        None,
                        None,
                        None,
                    )
                    .unwrap(),
            );
        });
    });

    cache.delete().unwrap();
}

fn bench_semantic_cache_check_miss(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let config = CacheConfig {
        name: unique_name("semcache"),
        connection: redis_vl::index::RedisConnectionInfo::new(redis_url()),
        ttl_seconds: None,
    };
    let cache = SemanticCache::new(config, 0.3, 128)
        .unwrap()
        .with_vectorizer(CustomTextVectorizer::new(|text| {
            Ok(simple_vectorizer(text))
        }));

    c.bench_function("semcache_check_miss", |b| {
        capped_iter_custom(b, || {
            black_box(
                cache
                    .check(
                        Some("completely different query"),
                        None,
                        1,
                        None,
                        None,
                        None,
                    )
                    .unwrap(),
            );
        });
    });

    cache.delete().unwrap();
}

// ---------------------------------------------------------------------------
// Message History: Basic Operations
// ---------------------------------------------------------------------------

fn bench_message_history_add(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let history = MessageHistory::new(unique_name("msghist"), redis_url());

    c.bench_function("msghist_add_single", |b| {
        capped_iter_custom(b, || {
            let msg = Message::new(MessageRole::User, "benchmark message");
            history.add_message(black_box(msg)).unwrap();
        });
    });

    history.delete().unwrap();
}

fn bench_message_history_get_recent(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let history = MessageHistory::new(unique_name("msghist"), redis_url());

    // Pre-populate
    for i in 0..100 {
        history
            .add_message(Message::new(MessageRole::User, format!("message {i}")))
            .unwrap();
    }

    c.bench_function("msghist_get_recent_10", |b| {
        capped_iter_custom(b, || {
            black_box(history.get_recent(10, None).unwrap());
        });
    });

    history.delete().unwrap();
}

// ---------------------------------------------------------------------------
// Message History: Semantic Operations
// ---------------------------------------------------------------------------

fn bench_semantic_history_add(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let history = SemanticMessageHistory::new(
        unique_name("semhist"),
        redis_url(),
        0.3,
        128,
        CustomTextVectorizer::new(|text| Ok(simple_vectorizer(text))),
    )
    .unwrap();

    c.bench_function("semhist_add_single", |b| {
        capped_iter_custom(b, || {
            let msg = Message::new(MessageRole::User, "benchmark message");
            history.add_message(black_box(msg)).unwrap();
        });
    });

    history.delete().unwrap();
}

fn bench_semantic_history_get_recent(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let history = SemanticMessageHistory::new(
        unique_name("semhist"),
        redis_url(),
        0.3,
        128,
        CustomTextVectorizer::new(|text| Ok(simple_vectorizer(text))),
    )
    .unwrap();

    // Pre-populate
    for i in 0..50 {
        history
            .add_message(Message::new(MessageRole::User, format!("message {i}")))
            .unwrap();
    }

    c.bench_function("semhist_get_recent_10", |b| {
        capped_iter_custom(b, || {
            black_box(history.get_recent(10, None).unwrap());
        });
    });

    history.delete().unwrap();
}

fn bench_semantic_history_get_relevant(c: &mut Criterion) {
    if !benchmarks_enabled() {
        return;
    }

    let history = SemanticMessageHistory::new(
        unique_name("semhist"),
        redis_url(),
        0.3,
        128,
        CustomTextVectorizer::new(|text| Ok(simple_vectorizer(text))),
    )
    .unwrap();

    // Pre-populate with diverse content
    for topic in &["weather", "sports", "politics", "technology", "food"] {
        for i in 0..10 {
            history
                .add_message(Message::new(
                    MessageRole::User,
                    format!("{topic} related message {i}"),
                ))
                .unwrap();
        }
    }

    c.bench_function("semhist_get_relevant", |b| {
        capped_iter_custom(b, || {
            black_box(
                history
                    .get_relevant_with_options("technology query", 5, None, None, None, false)
                    .unwrap(),
            );
        });
    });

    history.delete().unwrap();
}

// Redis benchmarks use short measurement windows and small sample counts to
// avoid exhausting macOS ephemeral ports (~16 k) — each iteration opens TCP
// connections that linger in TIME_WAIT for ~60 s.

fn redis_criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(100))
        .measurement_time(std::time::Duration::from_millis(500))
}

criterion_group! {
    name = index_lifecycle;
    config = redis_criterion();
    targets = bench_index_create, bench_index_exists, bench_index_info
}

criterion_group! {
    name = index_data;
    config = redis_criterion();
    targets = bench_index_load_single, bench_index_load_batch, bench_index_fetch
}

criterion_group! {
    name = search_vector;
    config = redis_criterion();
    targets = bench_search_vector_small, bench_search_vector_with_filter
}

criterion_group! {
    name = search_filter_count;
    config = redis_criterion();
    targets = bench_search_filter, bench_search_count
}

criterion_group! {
    name = search_batch_pagination;
    config = redis_criterion();
    targets = bench_search_batch, bench_search_paginate
}

criterion_group! {
    name = embeddings_cache;
    config = redis_criterion();
    targets = bench_embeddings_cache_set, bench_embeddings_cache_get_hit, bench_embeddings_cache_get_miss
}

criterion_group! {
    name = semantic_cache;
    config = redis_criterion();
    targets = bench_semantic_cache_store, bench_semantic_cache_check_hit, bench_semantic_cache_check_miss
}

criterion_group! {
    name = message_history;
    config = redis_criterion();
    targets = bench_message_history_add, bench_message_history_get_recent
}

criterion_group! {
    name = semantic_history;
    config = redis_criterion();
    targets = bench_semantic_history_add, bench_semantic_history_get_recent, bench_semantic_history_get_relevant
}

criterion_main!(
    index_lifecycle,
    index_data,
    search_vector,
    search_filter_count,
    search_batch_pagination,
    embeddings_cache,
    semantic_cache,
    message_history,
    semantic_history,
);
