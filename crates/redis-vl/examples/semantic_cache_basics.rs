//! Demonstrates setting up a `SemanticCache` for LLM response caching.
//!
//! This example shows how to construct a semantic cache. Full store/check
//! operations require a configured vectorizer (e.g., OpenAI) and a running
//! Redis instance.
//!
//! Run with:
//! ```bash
//! cargo run -p redis-vl --example semantic_cache_basics
//! ```

use redis_vl::CacheConfig;

fn main() {
    // Configure the cache
    let config = CacheConfig::new("llm-cache-example", "redis://127.0.0.1:6379").with_ttl(3600); // 1 hour TTL

    println!("Cache name: {}", config.name);
    println!("TTL: {:?} seconds", config.ttl_seconds);

    // Create a semantic cache (requires Redis for index creation)
    // Uncomment the following to use with a live Redis instance:
    //
    // let cache = SemanticCache::new(config, 0.2, 128).unwrap();
    // println!("Cache created with {} dimensions", cache.vector_dimensions);
    // println!("Distance threshold: {}", cache.distance_threshold);
    //
    // // Attach a vectorizer for automatic embedding
    // // let cache = cache.with_vectorizer(my_openai_vectorizer);
    //
    // // Store a cached response
    // cache.store("What is Redis?", "Redis is an in-memory data store.", None, None).unwrap();
    //
    // // Check for semantically similar prompts
    // let hits = cache.check("Tell me about Redis", None, None, None).unwrap();
    // if let Some(docs) = hits.as_documents() {
    //     for doc in docs {
    //         println!("Cache hit: {:?}", doc);
    //     }
    // }
    //
    // // Cleanup
    // cache.delete().unwrap();

    println!("\nTo run the full example, uncomment the Redis sections above and ensure:");
    println!("  1. Redis 8+ or Redis Stack is running");
    println!("  2. Set REDIS_URL if not at redis://127.0.0.1:6379");
    println!("  3. Configure an OpenAI API key for vectorizer usage");
}
