//! Mistral AI embedding adapter.
//!
//! Enabled by the `mistral` feature flag. Mistral's embedding API is similar to
//! OpenAI but uses `inputs` (plural) instead of `input` in the request body.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{AsyncVectorizer, Vectorizer};
use crate::error::Result;

/// Configuration for the Mistral AI embedding provider.
#[derive(Debug, Clone)]
pub struct MistralConfig {
    /// API key for Mistral.
    pub api_key: String,
    /// Embedding model name (default: `mistral-embed`).
    pub model: String,
}

impl MistralConfig {
    /// Creates a new Mistral config.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// Constructs from `MISTRAL_API_KEY` environment variable.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let api_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| crate::error::Error::InvalidInput("MISTRAL_API_KEY not set".into()))?;
        Ok(Self::new(api_key, model))
    }
}

const MISTRAL_EMBED_URL: &str = "https://api.mistral.ai/v1/embeddings";

/// Mistral uses `inputs` instead of `input`.
#[derive(Debug, Serialize)]
struct MistralEmbedRequest<'a> {
    model: &'a str,
    #[serde(rename = "input")]
    inputs: Vec<&'a str>,
}

#[derive(Debug, Deserialize)]
struct MistralEmbedResponse {
    data: Vec<MistralEmbedDatum>,
}

#[derive(Debug, Deserialize)]
struct MistralEmbedDatum {
    embedding: Vec<f32>,
}

/// Mistral AI embedding adapter.
///
/// Uses the Mistral embeddings API. The request format is similar to OpenAI but
/// the Python client sends the field as `inputs`; the actual Mistral REST API
/// accepts `input` (same as OpenAI), so we use the standard field name.
#[derive(Debug, Clone)]
pub struct MistralAITextVectorizer {
    config: MistralConfig,
    client: reqwest::Client,
    blocking_client: reqwest::blocking::Client,
}

impl MistralAITextVectorizer {
    /// Creates a new Mistral AI adapter.
    pub fn new(config: MistralConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            blocking_client: reqwest::blocking::Client::new(),
        }
    }

    async fn embed_many_inner(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let resp: MistralEmbedResponse = self
            .client
            .post(MISTRAL_EMBED_URL)
            .bearer_auth(&self.config.api_key)
            .json(&MistralEmbedRequest {
                model: &self.config.model,
                inputs: texts.to_vec(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

impl Vectorizer for MistralAITextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let resp: MistralEmbedResponse = self
            .blocking_client
            .post(MISTRAL_EMBED_URL)
            .bearer_auth(&self.config.api_key)
            .json(&MistralEmbedRequest {
                model: &self.config.model,
                inputs: vec![text],
            })
            .send()?
            .error_for_status()?
            .json()?;
        Ok(resp
            .data
            .into_iter()
            .next()
            .map_or_else(Vec::new, |d| d.embedding))
    }
}

#[async_trait]
impl AsyncVectorizer for MistralAITextVectorizer {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed_many_inner(&[text]).await?;
        Ok(v.pop().unwrap_or_default())
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_many_inner(texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mistral_config_stores_fields() {
        let cfg = MistralConfig::new("key", "mistral-embed");
        assert_eq!(cfg.api_key, "key");
        assert_eq!(cfg.model, "mistral-embed");
    }

    #[test]
    fn mistral_request_serializes_input_field() {
        let body = MistralEmbedRequest {
            model: "mistral-embed",
            inputs: vec!["hello"],
        };
        let json = serde_json::to_value(&body).unwrap();
        // Mistral REST API uses "input" field name
        assert_eq!(json["model"], "mistral-embed");
        assert_eq!(json["input"], serde_json::json!(["hello"]));
    }

    #[test]
    fn mistral_vectorizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MistralAITextVectorizer>();
    }
}
