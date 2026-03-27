//! Cohere embedding adapter.
//!
//! Enabled by the `cohere` feature flag. Cohere uses a distinct `/embed` API
//! that differs from the OpenAI-compatible format: it takes `texts`,
//! `input_type`, and `embedding_types` fields.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{AsyncVectorizer, Vectorizer};
use crate::error::Result;

/// Configuration for the Cohere embedding provider.
#[derive(Debug, Clone)]
pub struct CohereConfig {
    /// API key for Cohere.
    pub api_key: String,
    /// Embedding model name (default: `embed-english-v3.0`).
    pub model: String,
    /// The Cohere `input_type` to use (e.g. `"search_document"`, `"search_query"`).
    pub input_type: String,
}

impl CohereConfig {
    /// Creates a new Cohere config.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        input_type: impl Into<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            input_type: input_type.into(),
        }
    }

    /// Constructs from `COHERE_API_KEY` environment variable.
    pub fn from_env(model: impl Into<String>, input_type: impl Into<String>) -> Result<Self> {
        let api_key = std::env::var("COHERE_API_KEY")
            .map_err(|_| crate::error::Error::InvalidInput("COHERE_API_KEY not set".into()))?;
        Ok(Self::new(api_key, model, input_type))
    }
}

const COHERE_EMBED_URL: &str = "https://api.cohere.com/v1/embed";

#[derive(Serialize)]
struct CohereEmbedRequest<'a> {
    model: &'a str,
    texts: Vec<&'a str>,
    input_type: &'a str,
    embedding_types: Vec<&'a str>,
}

#[derive(Deserialize)]
struct CohereEmbedResponse {
    embeddings: CohereEmbeddings,
}

#[derive(Deserialize)]
struct CohereEmbeddings {
    float: Option<Vec<Vec<f32>>>,
}

/// Cohere embedding adapter.
///
/// Uses the Cohere `/embed` API which differs from the OpenAI-compatible format.
#[derive(Debug, Clone)]
pub struct CohereTextVectorizer {
    config: CohereConfig,
    client: reqwest::Client,
    blocking_client: reqwest::blocking::Client,
}

impl CohereTextVectorizer {
    /// Creates a new Cohere adapter.
    pub fn new(config: CohereConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            blocking_client: reqwest::blocking::Client::new(),
        }
    }

    fn build_request_body<'a>(&'a self, texts: &[&'a str]) -> CohereEmbedRequest<'a> {
        CohereEmbedRequest {
            model: &self.config.model,
            texts: texts.to_vec(),
            input_type: &self.config.input_type,
            embedding_types: vec!["float"],
        }
    }

    fn parse_response(response: CohereEmbedResponse) -> Result<Vec<Vec<f32>>> {
        response.embeddings.float.ok_or_else(|| {
            crate::error::Error::InvalidInput("no float embeddings in response".into())
        })
    }

    async fn embed_many_inner(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let resp: CohereEmbedResponse = self
            .client
            .post(COHERE_EMBED_URL)
            .bearer_auth(&self.config.api_key)
            .json(&self.build_request_body(texts))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Self::parse_response(resp)
    }
}

impl Vectorizer for CohereTextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let resp: CohereEmbedResponse = self
            .blocking_client
            .post(COHERE_EMBED_URL)
            .bearer_auth(&self.config.api_key)
            .json(&self.build_request_body(&[text]))
            .send()?
            .error_for_status()?
            .json()?;
        let mut embeddings = Self::parse_response(resp)?;
        Ok(embeddings.pop().unwrap_or_default())
    }
}

#[async_trait]
impl AsyncVectorizer for CohereTextVectorizer {
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
    fn cohere_config_stores_fields() {
        let cfg = CohereConfig::new("key", "embed-english-v3.0", "search_document");
        assert_eq!(cfg.api_key, "key");
        assert_eq!(cfg.model, "embed-english-v3.0");
        assert_eq!(cfg.input_type, "search_document");
    }

    #[test]
    fn cohere_request_serializes_correctly() {
        let cfg = CohereConfig::new("k", "model", "search_query");
        let v = CohereTextVectorizer::new(cfg);
        let body = v.build_request_body(&["hello", "world"]);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "model");
        assert_eq!(json["input_type"], "search_query");
        assert_eq!(json["embedding_types"], serde_json::json!(["float"]));
        assert_eq!(json["texts"], serde_json::json!(["hello", "world"]));
    }

    #[test]
    fn cohere_parse_response_extracts_floats() {
        let resp = CohereEmbedResponse {
            embeddings: CohereEmbeddings {
                float: Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
            },
        };
        let result = CohereTextVectorizer::parse_response(resp).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0]);
    }

    #[test]
    fn cohere_parse_response_errors_on_missing_float() {
        let resp = CohereEmbedResponse {
            embeddings: CohereEmbeddings { float: None },
        };
        assert!(CohereTextVectorizer::parse_response(resp).is_err());
    }

    #[test]
    fn cohere_vectorizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CohereTextVectorizer>();
    }
}
