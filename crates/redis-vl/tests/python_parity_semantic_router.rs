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
    assert_eq!(
        refs.len(),
        2,
        "expected 2 greeting references, got {:?}",
        refs
    );

    // Each reference should have the expected fields
    for r in &refs {
        assert!(
            r.contains_key("reference_id"),
            "missing reference_id: {:?}",
            r
        );
        assert!(r.contains_key("route_name"), "missing route_name: {:?}", r);
        assert!(r.contains_key("reference"), "missing reference: {:?}", r);
        assert_eq!(
            r.get("route_name").and_then(|v| v.as_str()),
            Some("greeting"),
            "unexpected route_name: {:?}",
            r
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
    let greeting = router
        .get("greeting")
        .expect("greeting route should still exist");
    assert!(
        greeting.references.is_empty(),
        "expected empty references, got {:?}",
        greeting.references
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

/// Mirrors `test_to_dict` from Python.
#[test]
fn python_test_router_to_dict() {
    let Some(router) = create_router() else {
        return;
    };

    let dict = router.to_dict().expect("to_dict should succeed");
    assert_eq!(dict["name"], json!(router.name));
    assert_eq!(dict["routes"].as_array().map(Vec::len), Some(2));
    assert_eq!(dict["vectorizer"]["type"], json!("custom"));
    assert!(dict.get("routing_config").is_some());

    router.delete().expect("delete should succeed");
}

/// Mirrors `test_from_dict` from Python: round-trip via to_dict → from_dict.
#[test]
fn python_test_router_from_dict() {
    let Some(router) = create_router() else {
        return;
    };

    let dict = router.to_dict().expect("to_dict should succeed");
    let new_router = SemanticRouter::from_dict(
        dict.clone(),
        redis_url(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        true, // overwrite to reuse the same index
    )
    .expect("from_dict should succeed");

    let new_dict = new_router.to_dict().expect("to_dict should succeed");
    assert_eq!(dict, new_dict);

    new_router.delete().expect("delete should succeed");
}

/// Mirrors `test_to_dict_missing_fields` from Python.
#[test]
fn python_test_router_from_dict_missing_fields() {
    let data = json!({
        "name": "incomplete-router",
        "routes": [],
        "vectorizer": {"type": "custom"},
    });

    let result = SemanticRouter::from_dict(
        data,
        redis_url(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        false,
    );
    // Missing routing_config → should fail
    assert!(result.is_err(), "should fail with missing routing_config");
}

/// Mirrors `test_from_existing` from Python.
#[test]
fn python_test_router_from_existing() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = format!("python_parity_from_existing_{id}");
    let router = SemanticRouter::new(
        name.clone(),
        redis_url(),
        routes(),
        RoutingConfig {
            max_k: 2,
            aggregation_method: DistanceAggregationMethod::Avg,
        },
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
    )
    .expect("router should initialize");

    // Reconnect from persisted state
    let router2 = SemanticRouter::from_existing(
        name.clone(),
        redis_url(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
    )
    .expect("from_existing should succeed");

    let dict1 = router.to_dict().expect("to_dict should succeed");
    let dict2 = router2.to_dict().expect("to_dict should succeed");
    assert_eq!(dict1, dict2);

    // Verify routing still works on the reconnected instance
    let match_result = router2
        .route(Some("hello"), None)
        .expect("route should succeed");
    assert_eq!(match_result.name.as_deref(), Some("greeting"));

    router.delete().expect("delete should succeed");
}

/// Mirrors `test_to_yaml` / `test_from_yaml` from Python.
#[test]
fn python_test_router_yaml_round_trip() {
    let Some(router) = create_router() else {
        return;
    };

    let yaml_path = std::env::temp_dir().join(format!(
        "test_router_yaml_{}.yaml",
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));

    router
        .to_yaml(&yaml_path, true)
        .expect("to_yaml should succeed");
    assert!(yaml_path.exists());

    let new_router = SemanticRouter::from_yaml(
        &yaml_path,
        redis_url(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        true, // overwrite
    )
    .expect("from_yaml should succeed");

    let mut d1 = router.to_dict().expect("to_dict");
    let mut d2 = new_router.to_dict().expect("to_dict");
    // Remove name since it may differ in the reconstructed router
    d1.as_object_mut().unwrap().remove("name");
    d2.as_object_mut().unwrap().remove("name");
    // The routes/config should match
    assert_eq!(d1["routes"], d2["routes"]);
    assert_eq!(d1["routing_config"], d2["routing_config"]);

    let _ = std::fs::remove_file(&yaml_path);
    new_router.delete().expect("delete should succeed");
    router.delete().expect("delete should succeed");
}

/// Mirrors `test_yaml_invalid_file_path` from Python.
#[test]
fn python_test_router_yaml_invalid_file_path() {
    let result = SemanticRouter::from_yaml(
        "nonexistent_path_xyz.yaml",
        redis_url(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        false,
    );
    assert!(result.is_err());
}

/// Mirrors `test_idempotent_to_dict` from Python.
#[test]
fn python_test_router_idempotent_to_dict() {
    let Some(router) = create_router() else {
        return;
    };

    let dict = router.to_dict().expect("to_dict should succeed");
    let new_router = SemanticRouter::from_dict(
        dict.clone(),
        redis_url(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        true,
    )
    .expect("from_dict should succeed");
    let new_dict = new_router.to_dict().expect("to_dict should succeed");
    assert_eq!(dict, new_dict);

    new_router.delete().expect("delete should succeed");
}

/// Mirrors `test_routes_different_distance_thresholds_get_two` from Python.
/// Routes with generous per-route thresholds should both match.
#[test]
fn python_test_routes_different_distance_thresholds_get_two() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut test_routes = routes();
    test_routes[0].distance_threshold = Some(0.5);
    test_routes[1].distance_threshold = Some(0.7);

    let router = SemanticRouter::new_with_options(
        format!("python_parity_thresh_two_{id}"),
        redis_url(),
        test_routes,
        RoutingConfig {
            max_k: 2,
            aggregation_method: DistanceAggregationMethod::Avg,
        },
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        true,
    )
    .expect("router should initialize");

    let matches = router
        .route_many(Some("hello"), None, Some(2), None)
        .expect("route_many should succeed");
    assert_eq!(
        matches.len(),
        2,
        "both routes should match with generous thresholds"
    );
    assert_eq!(matches[0].name.as_deref(), Some("greeting"));
    assert_eq!(matches[1].name.as_deref(), Some("farewell"));

    router.delete().expect("delete should succeed");
}

/// Mirrors `test_routes_different_distance_thresholds_get_one` from Python.
/// Only the route with a generous threshold should match; the tight one shouldn't.
#[test]
fn python_test_routes_different_distance_thresholds_get_one() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut test_routes = routes();
    test_routes[0].distance_threshold = Some(0.5); // greeting: generous
    test_routes[1].distance_threshold = Some(0.3); // farewell: tight, won't match "hello"

    let router = SemanticRouter::new_with_options(
        format!("python_parity_thresh_one_{id}"),
        redis_url(),
        test_routes,
        RoutingConfig {
            max_k: 2,
            aggregation_method: DistanceAggregationMethod::Avg,
        },
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        true,
    )
    .expect("router should initialize");

    let matches = router
        .route_many(Some("hello"), None, Some(2), None)
        .expect("route_many should succeed");
    assert_eq!(
        matches.len(),
        1,
        "only greeting should match with tight farewell threshold"
    );
    assert_eq!(matches[0].name.as_deref(), Some("greeting"));

    router.delete().expect("delete should succeed");
}

/// Tests that `add_route_references` persists state so `from_existing` sees changes.
#[test]
fn python_test_router_persist_after_add_references() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = format!("python_parity_persist_refs_{id}");
    let mut router = SemanticRouter::new(
        name.clone(),
        redis_url(),
        routes(),
        RoutingConfig::default(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
    )
    .expect("router should initialize");

    router
        .add_route_references("greeting", &["howdy".to_owned()])
        .expect("add refs should succeed");

    // Reconnect and verify the persisted config contains the new reference
    let router2 = SemanticRouter::from_existing(
        name,
        redis_url(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
    )
    .expect("from_existing should succeed");

    let greeting = router2.get("greeting").expect("greeting should exist");
    assert!(
        greeting.references.contains(&"howdy".to_owned()),
        "persisted state should include the added reference"
    );

    router.delete().expect("delete should succeed");
}

/// Mirrors Python `test_bad_dtype_connecting_to_existing_router`.
///
/// Creating a router with one dtype and then connecting to it with a
/// different dtype (overwrite=false) should error.
#[test]
fn python_test_bad_dtype_connecting_to_existing_router() {
    if !integration_enabled() {
        return;
    }

    let name = format!("test_bad_dtype_{}", COUNTER.fetch_add(1, Ordering::Relaxed));

    // Create a router with float32 dtype.
    let _router = SemanticRouter::new_with_options(
        name.clone(),
        redis_url(),
        routes(),
        RoutingConfig::default(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        false,
    )
    .expect("initial creation should succeed");

    // Attempt to connect with float64 dtype — should fail.
    let result = SemanticRouter::new_with_options(
        name.clone(),
        redis_url(),
        routes(),
        RoutingConfig::default(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float64,
        false,
    );

    assert!(
        result.is_err(),
        "connecting with mismatched dtype should fail"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("schema does not match"),
        "error should mention schema mismatch, got: {err_msg}"
    );

    // Clean up — delete via a properly-typed router.
    _router.delete().expect("cleanup should succeed");
}

/// Creating a router twice with the same dtype and overwrite=false should succeed
/// (idempotent reconnect).
#[test]
fn python_test_same_dtype_reconnect_succeeds() {
    if !integration_enabled() {
        return;
    }

    let name = format!(
        "test_same_dtype_{}",
        COUNTER.fetch_add(1, Ordering::Relaxed)
    );

    let router1 = SemanticRouter::new_with_options(
        name.clone(),
        redis_url(),
        routes(),
        RoutingConfig::default(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        false,
    )
    .expect("initial creation should succeed");

    // Reconnect with same dtype — should succeed.
    let router2 = SemanticRouter::new_with_options(
        name.clone(),
        redis_url(),
        routes(),
        RoutingConfig::default(),
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        false,
    );

    assert!(
        router2.is_ok(),
        "reconnect with same dtype should succeed: {:?}",
        router2.err()
    );

    router1.delete().expect("cleanup should succeed");
}
