//! Demonstrates using `MessageHistory` for conversation storage.
//!
//! **Requires a running Redis instance** (Redis 8+ or Redis Stack).
//! Set `REDIS_URL` to override the default `redis://127.0.0.1:6379`.
//!
//! Run with:
//! ```bash
//! cargo run -p redis-vl --example message_history_basics
//! ```

use redis_vl::{Message, MessageHistory, MessageRole};

fn redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned())
}

fn main() -> Result<(), redis_vl::Error> {
    let history = MessageHistory::new("history-example", redis_url());
    println!("Session tag: {}", history.default_session_tag());

    // Add individual messages
    history.add_message(Message::new(
        MessageRole::System,
        "You are a helpful assistant.",
    ))?;
    history.add_message(Message::new(MessageRole::User, "What is Redis?"))?;
    history.add_message(Message::new(
        MessageRole::Llm,
        "Redis is an in-memory data store used as a database, cache, and message broker.",
    ))?;

    // Store a prompt/response pair (adds User + Llm messages)
    history.store(
        "Can Redis do vector search?",
        "Yes! Redis supports vector similarity search through the Search module.",
    )?;

    // Retrieve recent messages
    let recent = history.get_recent(10, None)?;
    println!("\nAll messages ({}):", recent.len());
    for msg in &recent {
        println!("  [{:?}] {}", msg.role, msg.content);
    }

    // Filter by role
    let user_msgs = history.get_recent_with_roles(10, None, Some(&[MessageRole::User]))?;
    println!("\nUser messages only ({}):", user_msgs.len());
    for msg in &user_msgs {
        println!("  {}", msg.content);
    }

    // Get as plain text
    let texts = history.get_recent_as_text(5, None)?;
    println!("\nAs text:");
    for line in &texts {
        println!("  {line}");
    }

    // Count and cleanup
    let session = history.default_session_tag().to_owned();
    let count = history.count(&session)?;
    println!("\nMessage count: {count}");

    history.clear()?;
    println!("Session cleared.");

    history.delete()?;
    println!("History deleted.");

    Ok(())
}
