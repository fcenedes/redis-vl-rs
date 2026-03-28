//! Anthropic embedding adapter.
//!
//! Enabled by the `anthropic` feature flag. Anthropic does not provide its own
//! embedding model; instead it recommends [Voyage AI](https://www.voyageai.com/).
//! This adapter wraps [`VoyageAITextVectorizer`] with Anthropic-oriented
//! defaults and environment variable conventions.
//!
//! The default model is `voyage-3-large`, which Anthropic recommends for
//! general-purpose and multilingual retrieval.

use async_trait::async_trait;

use super::voyageai::{VoyageAIConfig, VoyageAITextVectorizer};
use super::{AsyncVectorizer, Vectorizer};
use crate::error::Result;

/// Default Voyage AI model recommended by Anthropic.
const DEFAULT_MODEL: &str = "voyage-3-large";

/// Configuration for the Anthropic (Voyage AI) embedding provider.
///
/// Since Anthropic does not offer its own embeddings, this delegates to Voyage AI.
/// The API key can be supplied directly or read from the `VOYAGE_API_KEY`
/// environment variable.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Underlying Voyage AI configuration.
    pub(crate) inner: VoyageAIConfig,
}

impl AnthropicConfig {
    /// Creates a new Anthropic config with the given API key and model.
    ///
    /// The `api_key` is the Voyage AI API key (Anthropic recommends Voyage AI
    /// for embeddings). The `model` selects the Voyage AI embedding model.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        input_type: Option<String>,
    ) -> Self {
        Self {
            inner: VoyageAIConfig::new(api_key, model, input_type),
        }
    }

    /// Creates an Anthropic config using the Anthropic-recommended default
    /// model (`voyage-3-large`).
    pub fn with_defaults(api_key: impl Into<String>, input_type: Option<String>) -> Self {
        Self::new(api_key, DEFAULT_MODEL, input_type)
    }

    /// Constructs from the `VOYAGE_API_KEY` environment variable with the
    /// Anthropic-recommended default model.
    pub fn from_env(input_type: Option<String>) -> Result<Self> {
        let inner = VoyageAIConfig::from_env(DEFAULT_MODEL, input_type)?;
        Ok(Self { inner })
    }

    /// Constructs from the `VOYAGE_API_KEY` environment variable with a
    /// specific model.
    pub fn from_env_with_model(
        model: impl Into<String>,
        input_type: Option<String>,
    ) -> Result<Self> {
        let inner = VoyageAIConfig::from_env(model, input_type)?;
        Ok(Self { inner })
    }
}

/// Anthropic embedding adapter backed by Voyage AI.
///
/// Anthropic [recommends Voyage AI](https://docs.anthropic.com/en/docs/build-with-claude/embeddings)
/// for embedding tasks. This adapter provides a convenience wrapper around
/// [`VoyageAITextVectorizer`] with Anthropic-oriented defaults.
///
/// # Example
///
/// ```no_run
/// use redis_vl::vectorizers::{AnthropicConfig, AnthropicTextVectorizer, Vectorizer};
///
/// let config = AnthropicConfig::with_defaults("my-voyage-key", Some("document".into()));
/// let vectorizer = AnthropicTextVectorizer::new(config);
/// let embedding = vectorizer.embed("Hello, world!").unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AnthropicTextVectorizer {
    inner: VoyageAITextVectorizer,
}

impl AnthropicTextVectorizer {
    /// Creates a new Anthropic embedding adapter.
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            inner: VoyageAITextVectorizer::new(config.inner),
        }
    }
}

impl Vectorizer for AnthropicTextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Vectorizer::embed(&self.inner, text)
    }

    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Vectorizer::embed_many(&self.inner, texts)
    }
}

#[async_trait]
impl AsyncVectorizer for AnthropicTextVectorizer {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        AsyncVectorizer::embed(&self.inner, text).await
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        AsyncVectorizer::embed_many(&self.inner, texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_config_stores_fields() {
        let cfg = AnthropicConfig::new("key", "voyage-3-large", Some("document".into()));
        assert_eq!(cfg.inner.api_key, "key");
        assert_eq!(cfg.inner.model, "voyage-3-large");
        assert_eq!(cfg.inner.input_type.as_deref(), Some("document"));
    }

    #[test]
    fn anthropic_config_with_defaults_uses_default_model() {
        let cfg = AnthropicConfig::with_defaults("key", None);
        assert_eq!(cfg.inner.model, DEFAULT_MODEL);
        assert!(cfg.inner.input_type.is_none());
    }

    #[test]
    fn anthropic_vectorizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AnthropicTextVectorizer>();
    }
}
