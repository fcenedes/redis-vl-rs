//! Error types for `redis-vl`.
//!
//! All fallible public APIs return `Result<T, Error>`. The [`Error`] enum
//! wraps underlying Redis, JSON, YAML, HTTP, I/O errors as well as
//! domain-specific schema validation and input errors.

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by RedisVL operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Wrapper for Redis transport errors.
    #[error("redis error: {0}")]
    Redis(#[from] redis::RedisError),
    /// Wrapper for JSON serialization and parsing errors.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Wrapper for YAML serialization and parsing errors.
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// Wrapper for HTTP client errors.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    /// Wrapper for URL parsing errors.
    #[error("url parse error: {0}")]
    Url(#[from] url::ParseError),
    /// Wrapper for I/O errors.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Returned when the schema is invalid.
    #[error("schema validation error: {0}")]
    SchemaValidation(String),
    /// Returned when a caller provides invalid input.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Returned when an operation depends on an unimplemented parity feature.
    #[error("unsupported feature: {0}")]
    Unsupported(&'static str),
}
