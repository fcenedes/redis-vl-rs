//! Integration tests derived from the upstream Python role-filter contracts for
//! standard and semantic message history.

use std::sync::atomic::{AtomicU64, Ordering};

use redis_vl::{
    CustomTextVectorizer, Message, MessageHistory, MessageRole, SemanticMessageHistory,
};

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
    let lower = text.to_ascii_lowercase();
    if lower.contains("fruits") || lower.contains("apples") {
        vec![1.0, 0.0, 0.0]
    } else if lower.contains("configuration") || lower.contains("system") {
        vec![0.0, 1.0, 0.0]
    } else {
        vec![0.0, 0.0, 1.0]
    }
}

fn create_history() -> Option<MessageHistory> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    Some(MessageHistory::new(
        format!("python_parity_role_history_{pid}_{id}"),
        redis_url(),
    ))
}

fn create_semantic_history() -> Option<SemanticMessageHistory> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = run_id();
    Some(
        SemanticMessageHistory::new(
            format!("python_parity_role_semantic_history_{pid}_{id}"),
            redis_url(),
            0.4,
            3,
            CustomTextVectorizer::new(|text| Ok(embed_text(text))),
        )
        .expect("semantic history should initialize"),
    )
}

#[test]
fn python_test_standard_get_recent_with_role_filters() {
    let Some(history) = create_history() else {
        return;
    };

    history
        .add_messages(&[
            Message::new(MessageRole::System, "System initialization"),
            Message::new(MessageRole::User, "Hello"),
            Message::new(MessageRole::Llm, "Hi there"),
            Message::new(MessageRole::System, "System configuration updated"),
            Message {
                tool_call_id: Some("call1".to_owned()),
                ..Message::new(MessageRole::Tool, "Function executed")
            },
        ])
        .expect("messages should store");

    let systems = history
        .get_recent_with_roles(10, None, Some(&[MessageRole::System]))
        .expect("role-filtered recent should succeed");
    assert_eq!(systems.len(), 2);
    assert!(
        systems
            .iter()
            .all(|message| message.role == MessageRole::System)
    );
    assert_eq!(systems[0].content, "System initialization");
    assert_eq!(systems[1].content, "System configuration updated");

    let mixed = history
        .get_recent_with_roles(10, None, Some(&[MessageRole::System, MessageRole::User]))
        .expect("role-filtered recent should succeed");
    assert_eq!(mixed.len(), 3);
    assert_eq!(mixed[0].role, MessageRole::System);
    assert_eq!(mixed[1].role, MessageRole::User);

    let raw = history
        .get_recent_with_roles(10, None, Some(&[MessageRole::Tool]))
        .expect("role-filtered recent should succeed");
    assert_eq!(raw.len(), 1);
    assert!(raw[0].entry_id.is_some());
    assert!(raw[0].timestamp.is_some());
    assert!(raw[0].session_tag.is_some());

    assert!(history.get_recent_with_roles(10, None, Some(&[])).is_err());

    history.delete().expect("delete should succeed");
}

#[test]
fn python_test_standard_role_filters_respect_session_scope() {
    let Some(history) = create_history() else {
        return;
    };

    history
        .add_messages_in_session(
            &[
                Message::new(MessageRole::System, "System for session1"),
                Message::new(MessageRole::User, "User for session1"),
            ],
            Some("session1"),
        )
        .expect("session1 messages should store");
    history
        .add_messages_in_session(
            &[
                Message::new(MessageRole::System, "System for session2"),
                Message::new(MessageRole::Llm, "LLM for session2"),
            ],
            Some("session2"),
        )
        .expect("session2 messages should store");

    let filtered = history
        .get_recent_with_roles(10, Some("session2"), Some(&[MessageRole::System]))
        .expect("role-filtered recent should succeed");
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].content, "System for session2");

    history.delete().expect("delete should succeed");
}

#[test]
fn python_test_semantic_get_recent_and_relevant_with_role_filters() {
    let Some(history) = create_semantic_history() else {
        return;
    };

    history
        .add_messages(&[
            Message::new(MessageRole::System, "System instructions about fruits"),
            Message::new(MessageRole::User, "Tell me about apples"),
            Message::new(MessageRole::Llm, "Apples are a type of fruit"),
            Message::new(MessageRole::User, "What about cars?"),
            Message::new(MessageRole::Llm, "Cars are vehicles for transportation"),
        ])
        .expect("messages should store");

    let systems = history
        .get_recent_with_roles(10, None, Some(&[MessageRole::System]))
        .expect("role-filtered recent should succeed");
    assert_eq!(systems.len(), 1);
    assert_eq!(systems[0].role, MessageRole::System);

    let relevant_system = history
        .get_relevant_with_options(
            "fruits",
            10,
            None,
            Some(&[MessageRole::System]),
            None,
            false,
        )
        .expect("role-filtered relevant should succeed");
    if !relevant_system.is_empty() {
        assert!(
            relevant_system
                .iter()
                .all(|message| message.role == MessageRole::System)
        );
    }

    let relevant_user = history
        .get_relevant_with_options("apples", 10, None, Some(&[MessageRole::User]), None, false)
        .expect("role-filtered relevant should succeed");
    if !relevant_user.is_empty() {
        assert!(
            relevant_user
                .iter()
                .all(|message| message.role == MessageRole::User)
        );
    }

    history.delete().expect("delete should succeed");
}
