//! VoyageAI embedding adapter.
//!
//! Enabled by the `voyageai` feature flag. VoyageAI has its own REST API shape
//! at `https://api.voyageai.com/v1/embeddings`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{AsyncVectorizer, Vectorizer};
use crate::error::Result;

/// Configuration for the VoyageAI embedding provider.
#[derive(Debug, Clone)]
pub struct VoyageAIConfig {
    /// API key for VoyageAI.
    pub api_key: String,
    /// Embedding model name (default: `voyage-3-large`).
    pub model: String,
    /// The VoyageAI `input_type` to use (e.g. `"document"`, `"query"`).
    pub input_type: Option<String>,
}

impl VoyageAIConfig {
    /// Creates a new VoyageAI config.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        input_type: Option<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            input_type,
        }
    }

    /// Constructs from `VOYAGE_API_KEY` environment variable.
    pub fn from_env(model: impl Into<String>, input_type: Option<String>) -> Result<Self> {
        let api_key = std::env::var("VOYAGE_API_KEY")
            .map_err(|_| crate::error::Error::InvalidInput("VOYAGE_API_KEY not set".into()))?;
        Ok(Self::new(api_key, model, input_type))
    }
}

const VOYAGEAI_EMBED_URL: &str = "https://api.voyageai.com/v1/embeddings";

#[derive(Serialize)]
struct VoyageAIEmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
}

#[derive(Deserialize)]
struct VoyageAIEmbedResponse {
    data: Vec<VoyageAIEmbedDatum>,
}

#[derive(Deserialize)]
struct VoyageAIEmbedDatum {
    embedding: Vec<f32>,
}

/// VoyageAI embedding adapter.
///
/// Uses the VoyageAI `/v1/embeddings` REST API.
#[derive(Debug, Clone)]
pub struct VoyageAITextVectorizer {
    config: VoyageAIConfig,
    client: reqwest::Client,
    blocking_client: reqwest::blocking::Client,
}

impl VoyageAITextVectorizer {
    /// Creates a new VoyageAI adapter.
    pub fn new(config: VoyageAIConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            blocking_client: reqwest::blocking::Client::new(),
        }
    }

    fn build_request<'a>(&'a self, texts: &[&'a str]) -> VoyageAIEmbedRequest<'a> {
        VoyageAIEmbedRequest {
            model: &self.config.model,
            input: texts.to_vec(),
            input_type: self.config.input_type.as_deref(),
        }
    }

    async fn embed_many_inner(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let resp: VoyageAIEmbedResponse = self
            .client
            .post(VOYAGEAI_EMBED_URL)
            .bearer_auth(&self.config.api_key)
            .json(&self.build_request(texts))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

impl Vectorizer for VoyageAITextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let resp: VoyageAIEmbedResponse = self
            .blocking_client
            .post(VOYAGEAI_EMBED_URL)
            .bearer_auth(&self.config.api_key)
            .json(&self.build_request(&[text]))
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
impl AsyncVectorizer for VoyageAITextVectorizer {
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
    fn voyageai_config_stores_fields() {
        let cfg = VoyageAIConfig::new("key", "voyage-3-large", Some("document".into()));
        assert_eq!(cfg.api_key, "key");
        assert_eq!(cfg.model, "voyage-3-large");
        assert_eq!(cfg.input_type.as_deref(), Some("document"));
    }

    #[test]
    fn voyageai_request_serializes_with_input_type() {
        let cfg = VoyageAIConfig::new("k", "voyage-3-large", Some("query".into()));
        let v = VoyageAITextVectorizer::new(cfg);
        let body = v.build_request(&["hello"]);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "voyage-3-large");
        assert_eq!(json["input"], serde_json::json!(["hello"]));
        assert_eq!(json["input_type"], "query");
    }

    #[test]
    fn voyageai_request_omits_none_input_type() {
        let cfg = VoyageAIConfig::new("k", "voyage-3-large", None);
        let v = VoyageAITextVectorizer::new(cfg);
        let body = v.build_request(&["hello"]);
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("input_type").is_none());
    }

    #[test]
    fn voyageai_vectorizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VoyageAITextVectorizer>();
    }
}
