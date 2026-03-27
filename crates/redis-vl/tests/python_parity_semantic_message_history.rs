//! Integration tests derived from the upstream Python semantic
//! `test_message_history.py` contract.

use std::sync::atomic::{AtomicU64, Ordering};

use redis_vl::{CustomTextVectorizer, Message, MessageRole, SemanticMessageHistory};
use serde_json::json;

static COUNTER: AtomicU64 = AtomicU64::new(1);

/// The dimensionality used by the deterministic test vectorizer.
const VECTOR_DIMENSIONS: usize = 3;

fn integration_enabled() -> bool {
    std::env::var("REDISVL_RUN_INTEGRATION")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned())
}

fn embed_text(text: &str) -> Vec<f32> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("winter sports in the olympics") {
        vec![0.0, 1.0, 0.0]
    } else if lower.contains("skiing, skating, luge") {
        vec![0.0, 0.995, 0.1]
    } else if lower.contains("downhill skiing") || lower.contains("ice skating") {
        vec![0.0, 0.94, 0.34]
    } else if lower.contains("winter sports") || lower.contains("skiing") || lower.contains("luge")
    {
        vec![0.0, 1.0, 0.0]
    } else if lower.contains("fruits and vegetables") {
        vec![0.65, 0.76, 0.0]
    } else if lower.contains("vegetable")
        || lower.contains("carrots")
        || lower.contains("broccoli")
        || lower.contains("onions")
        || lower.contains("spinach")
    {
        vec![0.6, 0.8, 0.0]
    } else if lower.contains("fruit")
        || lower.contains("apple")
        || lower.contains("orange")
        || lower.contains("banana")
        || lower.contains("strawberr")
    {
        vec![1.0, 0.0, 0.0]
    } else if lower.contains("cars") || lower.contains("vehicles") {
        vec![0.0, 0.0, 1.0]
    } else if lower.contains("configuration") {
        vec![0.2, 0.2, 0.96]
    } else {
        vec![0.3, 0.3, 0.4]
    }
}

fn create_history() -> Option<SemanticMessageHistory> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    Some(
        SemanticMessageHistory::new(
            format!("python_parity_semantic_history_{id}"),
            redis_url(),
            0.3,
            3,
            CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        )
        .expect("semantic history should initialize"),
    )
}

#[test]
fn python_test_semantic_store_and_get_recent() {
    let Some(history) = create_history() else {
        return;
    };

    assert_eq!(history.get_recent(5, None).expect("recent"), Vec::new());

    history
        .store("first prompt", "first response")
        .expect("store should succeed");
    history
        .store("second prompt", "second response")
        .expect("store should succeed");
    history
        .store("third prompt", "third response")
        .expect("store should succeed");
    history
        .store("fourth prompt", "fourth response")
        .expect("store should succeed");
    history
        .add_message(Message {
            tool_call_id: Some("tool id".to_owned()),
            ..Message::new(MessageRole::Tool, "tool result")
        })
        .expect("tool message should store");
    history
        .add_message(Message {
            tool_call_id: Some("tool id".to_owned()),
            metadata: Some(json!("return value from tool")),
            ..Message::new(MessageRole::Tool, "tool result")
        })
        .expect("tool message should store");

    let default_context = history.get_recent(5, None).expect("recent");
    assert_eq!(default_context.len(), 5);

    let partial = history.get_recent(4, None).expect("recent");
    assert_eq!(partial.len(), 4);

    let full_context = history.get_recent(10, None).expect("recent");
    assert_eq!(full_context.len(), 10);
    assert_eq!(full_context[0].content, "first prompt");
    assert_eq!(full_context[1].content, "first response");
    assert_eq!(full_context[8].role, MessageRole::Tool);
    assert_eq!(full_context[8].tool_call_id.as_deref(), Some("tool id"));
    assert_eq!(
        full_context[9].metadata,
        Some(json!("return value from tool"))
    );

    history.delete().expect("delete should succeed");
}

#[test]
fn python_test_semantic_messages_property_and_scope() {
    let Some(history) = create_history() else {
        return;
    };

    history
        .add_messages(&[
            Message::new(MessageRole::User, "first prompt"),
            Message::new(MessageRole::Llm, "first response"),
            Message {
                tool_call_id: Some("tool call one".to_owned()),
                metadata: Some(json!(42)),
                ..Message::new(MessageRole::Tool, "tool result 1")
            },
            Message {
                tool_call_id: Some("tool call two".to_owned()),
                metadata: Some(json!([1, 2, 3])),
                ..Message::new(MessageRole::Tool, "tool result 2")
            },
            Message::new(MessageRole::User, "second prompt"),
        ])
        .expect("add messages should succeed");
    history
        .store_in_session("scoped prompt", "scoped response", Some("session-b"))
        .expect("store in scoped session should succeed");

    let messages = history.messages().expect("messages");
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0].content, "first prompt");
    assert_eq!(messages[2].metadata, Some(json!(42)));
    assert_eq!(messages[3].metadata, Some(json!([1, 2, 3])));

    let scoped = history
        .get_recent(10, Some("session-b"))
        .expect("scoped recent should succeed");
    assert_eq!(scoped.len(), 2);
    assert_eq!(scoped[0].content, "scoped prompt");

    let missing = history
        .get_recent(10, Some("missing-session"))
        .expect("missing recent should succeed");
    assert!(missing.is_empty());

    history.delete().expect("delete should succeed");
}

#[test]
fn python_test_semantic_add_and_get_relevant() {
    let Some(mut history) = create_history() else {
        return;
    };

    history
        .add_message(Message::new(
            MessageRole::System,
            "discussing common fruits and vegetables",
        ))
        .expect("system message should store");
    history
        .store(
            "list of common fruits",
            "apples, oranges, bananas, strawberries",
        )
        .expect("fruit store should succeed");
    history
        .store(
            "list of common vegetables",
            "carrots, broccoli, onions, spinach",
        )
        .expect("vegetable store should succeed");
    history
        .store(
            "winter sports in the olympics",
            "downhill skiing, ice skating, luge",
        )
        .expect("sports store should succeed");
    history
        .add_message(Message {
            tool_call_id: Some("winter_sports()".to_owned()),
            ..Message::new(MessageRole::Tool, "skiing, skating, luge")
        })
        .expect("tool message should store");

    let fruit_context = history
        .get_relevant("set of common fruits like apples and bananas")
        .expect("relevant should succeed");
    assert_eq!(fruit_context.len(), 2);
    assert_eq!(fruit_context[0].role, MessageRole::User);
    assert_eq!(fruit_context[0].content, "list of common fruits");
    assert_eq!(fruit_context[1].role, MessageRole::Llm);
    assert_eq!(
        fruit_context[1].content,
        "apples, oranges, bananas, strawberries"
    );

    history
        .set_distance_threshold(0.5)
        .expect("threshold update should succeed");
    let broader = history
        .get_relevant("list of fruits and vegetables")
        .expect("relevant should succeed");
    assert_eq!(broader.len(), 5);
    assert_eq!(
        broader,
        history
            .get_relevant_with_options(
                "list of fruits and vegetables",
                5,
                None,
                None,
                Some(0.5),
                false,
            )
            .expect("explicit threshold should succeed")
    );

    let winter = history
        .get_relevant("winter sports like skiing")
        .expect("relevant should succeed");
    assert_eq!(winter.len(), 3);
    assert_eq!(winter[0].content, "winter sports in the olympics");
    assert_eq!(winter[1].content, "skiing, skating, luge");
    assert_eq!(winter[1].tool_call_id.as_deref(), Some("winter_sports()"));
    assert_eq!(winter[2].content, "downhill skiing, ice skating, luge");

    history.delete().expect("delete should succeed");
}

#[test]
fn python_test_semantic_drop_and_count() {
    let Some(history) = create_history() else {
        return;
    };

    history
        .store("first prompt", "first response")
        .expect("store should succeed");
    history
        .store("second prompt", "second response")
        .expect("store should succeed");
    history
        .store("third prompt", "third response")
        .expect("store should succeed");
    history
        .store("fourth prompt", "fourth response")
        .expect("store should succeed");

    assert_eq!(history.count(None).expect("count should succeed"), 8);

    history.drop(None).expect("drop should succeed");
    let after_drop = history.get_recent(3, None).expect("recent");
    assert_eq!(after_drop.len(), 3);
    assert_eq!(after_drop[0].content, "third prompt");
    assert_eq!(after_drop[2].content, "fourth prompt");

    let raw = history.get_recent(5, None).expect("recent");
    let middle_id = raw[2].entry_id.clone().expect("entry id should exist");
    history
        .drop(Some(&middle_id))
        .expect("drop by id should succeed");

    let after_id_drop = history.get_recent(10, None).expect("recent");
    assert_eq!(after_id_drop.len(), 6);
    assert!(
        after_id_drop
            .iter()
            .all(|message| message.entry_id.as_deref() != Some(middle_id.as_str()))
    );

    history.clear().expect("clear should succeed");
    assert_eq!(history.count(None).expect("count should succeed"), 0);

    history.delete().expect("delete should succeed");
}

/// Tests that `new_with_options(overwrite=true)` drops and recreates the index.
#[test]
fn python_test_semantic_history_overwrite() {
    if !integration_enabled() {
        return;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let name = format!("python_parity_semhist_overwrite_{id}");
    let url = redis_url();

    let history = SemanticMessageHistory::new_with_options(
        name.clone(),
        url.clone(),
        0.5,
        VECTOR_DIMENSIONS,
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        false,
    )
    .expect("first create should succeed");

    history
        .store("hello world", "hello response")
        .expect("store should succeed");
    assert_eq!(history.count(None).expect("count"), 2);

    // Recreate with overwrite = true → should start fresh
    let history2 = SemanticMessageHistory::new_with_options(
        name.clone(),
        url.clone(),
        0.5,
        VECTOR_DIMENSIONS,
        CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        redis_vl::schema::VectorDataType::Float32,
        true,
    )
    .expect("overwrite create should succeed");

    assert_eq!(history2.count(None).expect("count"), 0);

    history2.delete().expect("delete should succeed");
}
