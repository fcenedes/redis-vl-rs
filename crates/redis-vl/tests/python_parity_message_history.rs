//! Integration tests derived from the upstream Python
//! `test_message_history.py` standard-history contract.

use std::sync::atomic::{AtomicU64, Ordering};

use redis_vl::{Message, MessageHistory, MessageRole};
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

fn create_history() -> Option<MessageHistory> {
    if !integration_enabled() {
        return None;
    }

    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    Some(MessageHistory::new(
        format!("python_parity_history_{id}"),
        redis_url(),
    ))
}

#[test]
fn python_test_standard_store_and_get_recent() {
    let Some(history) = create_history() else {
        return;
    };

    assert_eq!(history.get_recent(5, None).expect("get recent"), Vec::new());

    history
        .store("first prompt", "first response")
        .expect("store");
    history
        .store("second prompt", "second response")
        .expect("store");
    history
        .store("third prompt", "third response")
        .expect("store");

    let recent = history.get_recent(10, None).expect("get recent");
    assert_eq!(recent.len(), 6);
    assert_eq!(recent[0].content, "first prompt");
    assert_eq!(recent[1].content, "first response");
    assert_eq!(recent[4].content, "third prompt");
    assert_eq!(recent[5].content, "third response");

    history.clear().expect("clear");
}

#[test]
fn python_test_standard_add_and_scope() {
    let Some(history) = create_history() else {
        return;
    };

    history
        .add_message(Message::new(MessageRole::User, "some prompt"))
        .expect("add message");
    history
        .add_message(Message::new(MessageRole::Llm, "some response"))
        .expect("add message");

    history
        .store_in_session("new user prompt", "new user response", Some("abc"))
        .expect("store in session");

    let default_session = history.get_recent(10, None).expect("default session");
    assert_eq!(default_session.len(), 2);
    assert_eq!(default_session[0].content, "some prompt");

    let scoped = history.get_recent(10, Some("abc")).expect("scoped session");
    assert_eq!(scoped.len(), 2);
    assert_eq!(scoped[0].content, "new user prompt");

    history.clear().expect("clear");
    history.clear_session(Some("abc")).expect("clear scoped");
}

#[test]
fn python_test_standard_add_messages_and_messages_property() {
    let Some(history) = create_history() else {
        return;
    };

    history
        .add_messages(&[
            Message::new(MessageRole::User, "first prompt"),
            Message::new(MessageRole::Llm, "first response"),
            Message {
                metadata: Some(json!({"params": "abc"})),
                ..Message::new(MessageRole::Tool, "tool result")
            },
        ])
        .expect("add messages");

    let messages = history.messages().expect("messages");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].content, "first prompt");
    assert_eq!(messages[1].content, "first response");
    assert_eq!(messages[2].metadata, Some(json!({"params": "abc"})));

    let text = history.get_recent_as_text(3, None).expect("as text");
    assert_eq!(text, vec!["first prompt", "first response", "tool result"]);

    history.clear().expect("clear");
}

#[test]
fn python_test_standard_drop_clear_and_count() {
    let Some(history) = create_history() else {
        return;
    };

    history
        .store("first prompt", "first response")
        .expect("store");
    history
        .store("second prompt", "second response")
        .expect("store");
    history
        .store("third prompt", "third response")
        .expect("store");
    assert_eq!(
        history
            .count(history.default_session_tag())
            .expect("count should succeed"),
        6
    );

    history.drop(None).expect("drop last");
    let after_drop = history.get_recent(10, None).expect("get recent");
    assert_eq!(after_drop.len(), 5);
    assert_eq!(after_drop.last().expect("last").content, "third prompt");

    let entry_id = after_drop[1].entry_id.clone().expect("entry id");
    history.drop(Some(&entry_id)).expect("drop by id");
    let after_id_drop = history.get_recent(10, None).expect("get recent");
    assert_eq!(after_id_drop.len(), 4);
    assert!(
        after_id_drop
            .iter()
            .all(|message| message.entry_id.as_deref() != Some(&entry_id))
    );

    let cleared = history.clear().expect("clear");
    assert_eq!(cleared, 4);
    assert_eq!(history.get_recent(10, None).expect("empty"), Vec::new());
}
