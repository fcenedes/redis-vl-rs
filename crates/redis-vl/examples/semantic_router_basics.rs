//! Demonstrates configuring a `SemanticRouter` with predefined routes.
//!
//! Full routing requires a configured vectorizer and a running Redis instance.
//! This example shows how to define routes and routing configuration.
//!
//! Run with:
//! ```bash
//! cargo run -p redis-vl --example semantic_router_basics
//! ```

use redis_vl::{DistanceAggregationMethod, Route, RoutingConfig};

fn main() {
    // Define routes with reference utterances
    let routes = vec![
        Route::new(
            "greeting",
            vec![
                "hello".into(),
                "hi there".into(),
                "hey".into(),
                "good morning".into(),
            ],
        ),
        Route::new(
            "farewell",
            vec![
                "goodbye".into(),
                "bye".into(),
                "see you later".into(),
                "take care".into(),
            ],
        ),
        Route::new(
            "technical_support",
            vec![
                "how do I fix this error".into(),
                "something is broken".into(),
                "help with configuration".into(),
                "debug this issue".into(),
            ],
        ),
        Route::new(
            "billing",
            vec![
                "what is my invoice".into(),
                "payment failed".into(),
                "upgrade my plan".into(),
                "cancel subscription".into(),
            ],
        ),
    ];

    println!("Defined {} routes:", routes.len());
    for route in &routes {
        println!("  {} ({} references)", route.name, route.references.len());
    }

    // Configure routing behavior
    let config = RoutingConfig {
        max_k: 5,
        aggregation_method: DistanceAggregationMethod::Avg,
    };
    println!("\nRouting config:");
    println!("  max_k: {}", config.max_k);
    println!("  aggregation: {:?}", config.aggregation_method);

    // To create and use the router with a live Redis instance:
    //
    // let router = SemanticRouter::new(
    //     vectorizer, routes, "my-router", "redis://127.0.0.1:6379", config
    // );
    //
    // let result = router.route(Some("howdy!"), None).unwrap();
    // println!("Best route: {}", result.name);
    //
    // let matches = router.route_many(Some("hey, my payment failed"), None, 3).unwrap();
    // for m in &matches {
    //     println!("  {} (distance: {})", m.name, m.distance);
    // }

    println!("\nTo run the full example, ensure:");
    println!("  1. Redis 8+ or Redis Stack is running");
    println!("  2. Configure a vectorizer (e.g., OpenAI)");
}
