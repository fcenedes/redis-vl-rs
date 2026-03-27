//! Integration tests derived from the upstream Python `test_semantic_router.py`
//! parity contract.

use std::sync::atomic::{AtomicU64, Ordering};

use redis_vl::{
    CustomTextVectorizer, DistanceAggregationMethod, Route, RoutingConfig, SemanticRouter,
};
use serde_json::json;

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
        "hello" | "hi" => vec![1.0, 0.0, 0.0],
        "bye" | "goodbye" => vec![0.0, 1.0, 0.0],
        "political speech" | "who will you vote for?" | "are you liberal or conservative?" => {
            vec![0.0, 0.0, 1.0]
        }
        "unknown_phrase" => vec![0.2, 0.2, 0.2],
        other => {
            let score = (other.len() % 10) as f32 / 10.0;
            vec![score, 1.0 - score, 0.0]
        }
    }
}

fn routes() -> Vec<Route> {
    vec![
        Route {
            name: "greeting".to_owned(),
            references: vec!["hello".to_owned(), "hi".to_owned()],
            metadata: serde_json::Map::from_iter([("type".to_owned(), json!("greeting"))]),
            distance_threshold: Some(0.3),
        },
        Route {
            name: "farewell".to_owned(),
            references: vec!["bye".to_owned(), "goodbye".to_owned()],
            metadata: serde_json::Map::from_iter([("type".to_owned(), json!("farewell"))]),
            distance_threshold: Some(0.2),
        },
    ]
}

fn create_router() -> Option<SemanticRouter> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    Some(
        SemanticRouter::new(
            format!("python_parity_router_{id}"),
            redis_url(),
            routes(),
            RoutingConfig {
                max_k: 2,
                aggregation_method: DistanceAggregationMethod::Avg,
            },
            CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        )
        .expect("router should initialize"),
    )
}

#[test]
fn python_test_router_properties_and_getters() {
    let Some(router) = create_router() else {
        return;
    };

    assert_eq!(router.routes.len(), 2);
    assert_eq!(router.routing_config.max_k, 2);
    assert!(router.route_names().contains(&"greeting".to_owned()));
    assert!(router.route_names().contains(&"farewell".to_owned()));
    assert_eq!(router.route_thresholds()["greeting"], 0.3);
    assert_eq!(router.route_thresholds()["farewell"], 0.2);
    assert_eq!(
        router.get("greeting").map(|route| route.name.as_str()),
        Some("greeting")
    );
    assert!(router.get("non_existent_route").is_none());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_router_single_and_multi_query() {
    let Some(router) = create_router() else {
        return;
    };

    let single = router
        .route(Some("hello"), None)
        .expect("route should succeed");
    assert_eq!(single.name.as_deref(), Some("greeting"));
    assert!(single.distance.expect("distance should exist") <= 0.3);

    let vector = embed_text("goodbye");
    let vector_match = router
        .route(None, Some(&vector))
        .expect("vector route should succeed");
    assert_eq!(vector_match.name.as_deref(), Some("farewell"));

    let no_match = router
        .route(Some("unknown_phrase"), None)
        .expect("route should succeed");
    assert!(no_match.name.is_none());

    let many = router
        .route_many(Some("hello"), None, Some(2), None)
        .expect("route many should succeed");
    assert!(!many.is_empty());
    assert_eq!(many[0].name.as_deref(), Some("greeting"));

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_router_update_config_add_and_remove_routes() {
    let Some(mut router) = create_router() else {
        return;
    };

    router.update_routing_config(RoutingConfig {
        max_k: 27,
        aggregation_method: DistanceAggregationMethod::Min,
    });
    assert_eq!(router.routing_config.max_k, 27);
    assert_eq!(
        router.routing_config.aggregation_method,
        DistanceAggregationMethod::Min
    );

    let politics = Route {
        name: "politics".to_owned(),
        references: vec![
            "are you liberal or conservative?".to_owned(),
            "who will you vote for?".to_owned(),
            "political speech".to_owned(),
        ],
        metadata: serde_json::Map::from_iter([("type".to_owned(), json!("politics"))]),
        distance_threshold: Some(0.25),
    };
    router
        .add_routes(std::slice::from_ref(&politics))
        .expect("add routes should succeed");

    assert_eq!(
        router.get("politics").map(|route| route.name.as_str()),
        Some("politics")
    );
    let match_result = router
        .route(Some("political speech"), None)
        .expect("route should succeed");
    assert_eq!(match_result.name.as_deref(), Some("politics"));

    router
        .remove_route("greeting")
        .expect("remove route should succeed");
    assert!(router.get("greeting").is_none());
    router
        .remove_route("unknown_route")
        .expect("removing an unknown route should still succeed");

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_router_clear_and_to_json_value() {
    let Some(mut router) = create_router() else {
        return;
    };

    let json_value = router.to_json_value().expect("to json should succeed");
    assert_eq!(json_value["name"], json!(router.name));
    assert_eq!(json_value["routes"].as_array().map(Vec::len), Some(2));
    assert_eq!(json_value["vectorizer"]["type"], json!("custom"));

    let deleted = router.clear().expect("clear should succeed");
    assert!(deleted >= 1);
    assert!(router.routes.is_empty());

    router.delete().expect("delete should succeed");
}
