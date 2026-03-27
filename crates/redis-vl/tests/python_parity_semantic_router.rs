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


#[test]
fn python_test_router_add_route_references() {
    let Some(mut router) = create_router() else {
        return;
    };

    // Add new references to the existing "greeting" route
    let new_refs = vec!["howdy".to_owned(), "hey there".to_owned()];
    let keys = router
        .add_route_references("greeting", &new_refs)
        .expect("add_route_references should succeed");

    // Should return 2 keys (one per reference)
    assert_eq!(keys.len(), 2);

    // In-memory route should now have 4 references (original 2 + new 2)
    let greeting = router.get("greeting").expect("greeting route should exist");
    assert_eq!(greeting.references.len(), 4);
    assert!(greeting.references.contains(&"howdy".to_owned()));
    assert!(greeting.references.contains(&"hey there".to_owned()));

    // Adding references to a non-existent route should fail
    let result = router.add_route_references("no_such_route", &["test".to_owned()]);
    assert!(result.is_err());

    // Adding empty references should return empty keys
    let empty_keys = router
        .add_route_references("greeting", &[])
        .expect("empty refs should succeed");
    assert!(empty_keys.is_empty());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_router_get_route_references() {
    let Some(router) = create_router() else {
        return;
    };

    // Wait a moment for Redis to index
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Get references by route name
    let refs = router
        .get_route_references(Some("greeting"), None)
        .expect("get_route_references should succeed");

    // Should find 2 references for "greeting"
    assert_eq!(refs.len(), 2, "expected 2 greeting references, got {:?}", refs);

    // Each reference should have the expected fields
    for r in &refs {
        assert!(r.contains_key("reference_id"), "missing reference_id: {:?}", r);
        assert!(r.contains_key("route_name"), "missing route_name: {:?}", r);
        assert!(r.contains_key("reference"), "missing reference: {:?}", r);
        assert_eq!(
            r.get("route_name").and_then(|v| v.as_str()),
            Some("greeting"),
            "unexpected route_name: {:?}", r
        );
    }

    // Must provide at least one parameter
    let result = router.get_route_references(None, None);
    assert!(result.is_err());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_router_delete_route_references_by_route_name() {
    let Some(mut router) = create_router() else {
        return;
    };

    // Wait for indexing
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Delete references by route name
    let deleted = router
        .delete_route_references(Some("greeting"), None, None)
        .expect("delete_route_references should succeed");
    assert_eq!(deleted, 2, "expected 2 deleted, got {deleted}");

    // In-memory route should have 0 references
    let greeting = router.get("greeting").expect("greeting route should still exist");
    assert!(
        greeting.references.is_empty(),
        "expected empty references, got {:?}", greeting.references
    );

    // Must provide at least one parameter
    let result = router.delete_route_references(None, None, None);
    assert!(result.is_err());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_router_delete_route_references_by_keys() {
    let Some(mut router) = create_router() else {
        return;
    };

    // Wait for indexing
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Get the references to learn their keys
    let refs = router
        .get_route_references(Some("farewell"), None)
        .expect("get should succeed");
    assert_eq!(refs.len(), 2);

    // Build full keys from the reference_ids
    let keys: Vec<String> = refs
        .iter()
        .filter_map(|r| {
            r.get("reference_id")
                .and_then(|v| v.as_str())
                .map(|id| router.index.key(id))
        })
        .collect();
    assert_eq!(keys.len(), 2);

    // Delete by explicit keys
    let deleted = router
        .delete_route_references(None, None, Some(&keys))
        .expect("delete by keys should succeed");
    assert_eq!(deleted, 2);

    // In-memory route should have 0 references
    let farewell = router.get("farewell").expect("farewell route should exist");
    assert!(farewell.references.is_empty());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_add_route_references() {
    let Some(mut router) = create_router() else {
        return;
    };

    // Add new references to an existing route
    let new_refs = vec!["hey there".to_owned(), "howdy".to_owned()];
    let keys = router
        .add_route_references("greeting", &new_refs)
        .expect("add_route_references should succeed");

    // Should return one key per reference
    assert_eq!(keys.len(), 2);

    // In-memory route should be updated
    let greeting = router.get("greeting").expect("greeting route should exist");
    assert!(greeting.references.contains(&"hey there".to_owned()));
    assert!(greeting.references.contains(&"howdy".to_owned()));
    assert_eq!(greeting.references.len(), 4); // 2 original + 2 new

    // Adding to a non-existent route should error
    let result = router.add_route_references("nonexistent", &["test".to_owned()]);
    assert!(result.is_err());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_get_route_references() {
    let Some(router) = create_router() else {
        return;
    };

    // Give Redis a moment to index
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Get references by route name
    let refs = router
        .get_route_references(Some("greeting"), None)
        .expect("get_route_references should succeed");

    // Should find the 2 greeting references
    assert_eq!(refs.len(), 2);
    for r in &refs {
        assert_eq!(
            r.get("route_name").and_then(|v| v.as_str()),
            Some("greeting")
        );
        assert!(r.get("reference").is_some());
        assert!(r.get("reference_id").is_some());
    }

    // Calling with neither name nor IDs should error
    let result = router.get_route_references(None, None);
    assert!(result.is_err());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_delete_route_references_by_route_name() {
    let Some(mut router) = create_router() else {
        return;
    };

    // Give Redis a moment to index
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Delete all references for a route by name
    let deleted = router
        .delete_route_references(Some("greeting"), None, None)
        .expect("delete_route_references should succeed");

    assert!(deleted >= 1);

    // In-memory route should have no references left
    let greeting = router.get("greeting").expect("greeting route should exist");
    assert!(greeting.references.is_empty());

    // Calling with no arguments should error
    let result = router.delete_route_references(None, None, None);
    assert!(result.is_err());

    router.delete().expect("delete should succeed");
}


#[test]
fn python_test_router_add_route_references() {
    let Some(mut router) = create_router() else {
        return;
    };

    // Add new references to the existing "greeting" route
    let new_refs = vec!["hey there".to_owned(), "howdy".to_owned()];
    let keys = router
        .add_route_references("greeting", &new_refs)
        .expect("add_route_references should succeed");

    // Should return 2 keys, one per added reference
    assert_eq!(keys.len(), 2);

    // In-memory route should reflect the added references
    let route = router.get("greeting").expect("greeting route should exist");
    assert!(route.references.contains(&"hey there".to_owned()));
    assert!(route.references.contains(&"howdy".to_owned()));
    assert_eq!(route.references.len(), 4); // original 2 + new 2

    // Adding to a non-existent route should fail
    let result = router.add_route_references("non_existent", &["test".to_owned()]);
    assert!(result.is_err());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_router_get_route_references() {
    let Some(router) = create_router() else {
        return;
    };

    // Allow the index to settle
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Get references by route name
    let refs = router
        .get_route_references(Some("greeting"), None)
        .expect("get_route_references should succeed");

    // The greeting route has 2 references ("hello", "hi")
    assert_eq!(refs.len(), 2);
    for r in &refs {
        assert_eq!(
            r.get("route_name").and_then(|v| v.as_str()),
            Some("greeting")
        );
        let ref_text = r.get("reference").and_then(|v| v.as_str()).unwrap();
        assert!(ref_text == "hello" || ref_text == "hi");
    }

    // Must provide at least one of route_name or reference_ids
    let err = router.get_route_references(None, None);
    assert!(err.is_err());

    router.delete().expect("delete should succeed");
}

#[test]
fn python_test_router_delete_route_references_by_route_name() {
    let Some(mut router) = create_router() else {
        return;
    };

    // Delete all references for "farewell"
    let deleted = router
        .delete_route_references(Some("farewell"), None, None)
        .expect("delete_route_references should succeed");

    assert!(deleted >= 1);

    // In-memory route should have no references
    let route = router.get("farewell").expect("farewell route should exist");
    assert!(route.references.is_empty());

    // Must provide at least one filter
    let err = router.delete_route_references(None, None, None);
    assert!(err.is_err());

    router.delete().expect("delete should succeed");
}
