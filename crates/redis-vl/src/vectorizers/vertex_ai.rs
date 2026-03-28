//! Google Vertex AI embedding adapter.
//!
//! Enabled by the `vertex-ai` feature flag. Uses the Vertex AI `predict`
//! REST API endpoint for text embedding models such as
//! `textembedding-gecko@003`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{AsyncVectorizer, Vectorizer};
use crate::error::Result;

/// Configuration for the Vertex AI embedding provider.
#[derive(Debug, Clone)]
pub struct VertexAIConfig {
    /// GCP project ID.
    pub project_id: String,
    /// GCP location / region (e.g. `us-central1`).
    pub location: String,
    /// Embedding model name (default: `textembedding-gecko@003`).
    pub model: String,
    /// API key or OAuth2 access token used for authentication.
    pub api_key: String,
}

impl VertexAIConfig {
    /// Creates a new Vertex AI config.
    pub fn new(
        project_id: impl Into<String>,
        location: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            project_id: project_id.into(),
            location: location.into(),
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Constructs from environment variables:
    /// `GCP_PROJECT_ID`, `GCP_LOCATION`, `GCP_API_KEY`.
    /// Model defaults to `textembedding-gecko@003`.
    pub fn from_env(model: Option<String>) -> Result<Self> {
        let project_id = std::env::var("GCP_PROJECT_ID")
            .map_err(|_| crate::error::Error::InvalidInput("GCP_PROJECT_ID not set".into()))?;
        let location = std::env::var("GCP_LOCATION")
            .map_err(|_| crate::error::Error::InvalidInput("GCP_LOCATION not set".into()))?;
        let api_key = std::env::var("GCP_API_KEY")
            .map_err(|_| crate::error::Error::InvalidInput("GCP_API_KEY not set".into()))?;
        Ok(Self::new(
            project_id,
            location,
            model.unwrap_or_else(|| "textembedding-gecko@003".to_string()),
            api_key,
        ))
    }

    fn predict_url(&self) -> String {
        format!(
            "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict",
            location = self.location,
            project = self.project_id,
            model = self.model,
        )
    }
}

#[derive(Serialize)]
struct VertexAIInstance<'a> {
    content: &'a str,
}

#[derive(Serialize)]
struct VertexAIPredictRequest<'a> {
    instances: Vec<VertexAIInstance<'a>>,
}

#[derive(Deserialize)]
struct VertexAIPredictResponse {
    predictions: Vec<VertexAIPrediction>,
}

#[derive(Deserialize)]
struct VertexAIPrediction {
    embeddings: VertexAIEmbeddings,
}

#[derive(Deserialize)]
struct VertexAIEmbeddings {
    values: Vec<f32>,
}

/// Vertex AI embedding adapter.
///
/// Uses the Vertex AI `predict` REST endpoint which takes `instances`
/// with `content` fields and returns `predictions` with `embeddings.values`.
#[derive(Debug, Clone)]
pub struct VertexAITextVectorizer {
    config: VertexAIConfig,
    client: reqwest::Client,
    blocking_client: reqwest::blocking::Client,
}

impl VertexAITextVectorizer {
    /// Creates a new Vertex AI adapter.
    pub fn new(config: VertexAIConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            blocking_client: reqwest::blocking::Client::new(),
        }
    }

    fn build_request<'a>(&self, texts: &[&'a str]) -> VertexAIPredictRequest<'a> {
        VertexAIPredictRequest {
            instances: texts
                .iter()
                .map(|t| VertexAIInstance { content: t })
                .collect(),
        }
    }

    fn parse_response(resp: VertexAIPredictResponse) -> Result<Vec<Vec<f32>>> {
        if resp.predictions.is_empty() {
            return Err(crate::error::Error::InvalidInput(
                "no predictions in Vertex AI response".into(),
            ));
        }
        Ok(resp
            .predictions
            .into_iter()
            .map(|p| p.embeddings.values)
            .collect())
    }

    async fn embed_many_inner(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let resp: VertexAIPredictResponse = self
            .client
            .post(self.config.predict_url())
            .bearer_auth(&self.config.api_key)
            .json(&self.build_request(texts))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Self::parse_response(resp)
    }
}

impl Vectorizer for VertexAITextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let resp: VertexAIPredictResponse = self
            .blocking_client
            .post(self.config.predict_url())
            .bearer_auth(&self.config.api_key)
            .json(&self.build_request(&[text]))
            .send()?
            .error_for_status()?
            .json()?;
        let mut embeddings = Self::parse_response(resp)?;
        Ok(embeddings.pop().unwrap_or_default())
    }
}

#[async_trait]
impl AsyncVectorizer for VertexAITextVectorizer {
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
    fn vertex_ai_config_stores_fields() {
        let cfg = VertexAIConfig::new(
            "my-project",
            "us-central1",
            "textembedding-gecko@003",
            "key",
        );
        assert_eq!(cfg.project_id, "my-project");
        assert_eq!(cfg.location, "us-central1");
        assert_eq!(cfg.model, "textembedding-gecko@003");
        assert_eq!(cfg.api_key, "key");
    }

    #[test]
    fn vertex_ai_config_builds_predict_url() {
        let cfg = VertexAIConfig::new("proj", "us-central1", "textembedding-gecko@003", "k");
        let url = cfg.predict_url();
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/google/models/textembedding-gecko@003:predict"
        );
    }

    #[test]
    fn vertex_ai_request_serializes_correctly() {
        let cfg = VertexAIConfig::new("p", "us-central1", "model", "k");
        let v = VertexAITextVectorizer::new(cfg);
        let body = v.build_request(&["hello", "world"]);
        let json = serde_json::to_value(&body).unwrap();
        let instances = json["instances"].as_array().unwrap();
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0]["content"], "hello");
        assert_eq!(instances[1]["content"], "world");
    }

    #[test]
    fn vertex_ai_parse_response_extracts_values() {
        let resp = VertexAIPredictResponse {
            predictions: vec![
                VertexAIPrediction {
                    embeddings: VertexAIEmbeddings {
                        values: vec![1.0, 2.0, 3.0],
                    },
                },
                VertexAIPrediction {
                    embeddings: VertexAIEmbeddings {
                        values: vec![4.0, 5.0, 6.0],
                    },
                },
            ],
        };
        let result = VertexAITextVectorizer::parse_response(resp).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(result[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn vertex_ai_parse_response_errors_on_empty() {
        let resp = VertexAIPredictResponse {
            predictions: vec![],
        };
        assert!(VertexAITextVectorizer::parse_response(resp).is_err());
    }

    #[test]
    fn vertex_ai_vectorizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VertexAITextVectorizer>();
    }
}
