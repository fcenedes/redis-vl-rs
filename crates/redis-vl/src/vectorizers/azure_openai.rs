//! Azure OpenAI embedding adapter.
//!
//! Enabled by the `azure-openai` feature flag.

use async_trait::async_trait;

use super::{AsyncVectorizer, EmbeddingRequest, EmbeddingResponse, Vectorizer};
use crate::error::Result;

/// Configuration for connecting to an Azure OpenAI deployment.
#[derive(Debug, Clone)]
pub struct AzureOpenAIConfig {
    /// Azure OpenAI resource endpoint (e.g. `https://myresource.openai.azure.com/`).
    pub azure_endpoint: url::Url,
    /// API key for authentication.
    pub api_key: String,
    /// Deployment name (not the model name).
    pub deployment: String,
    /// API version, e.g. `"2024-02-01"`.
    pub api_version: String,
}

impl AzureOpenAIConfig {
    /// Creates a new Azure OpenAI configuration.
    pub fn new(
        azure_endpoint: impl AsRef<str>,
        api_key: impl Into<String>,
        deployment: impl Into<String>,
        api_version: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            azure_endpoint: url::Url::parse(azure_endpoint.as_ref())?,
            api_key: api_key.into(),
            deployment: deployment.into(),
            api_version: api_version.into(),
        })
    }

    /// Constructs from environment variables:
    /// `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`.
    pub fn from_env(deployment: impl Into<String>) -> Result<Self> {
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").map_err(|_| {
            crate::error::Error::InvalidInput("AZURE_OPENAI_ENDPOINT not set".into())
        })?;
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").map_err(|_| {
            crate::error::Error::InvalidInput("AZURE_OPENAI_API_KEY not set".into())
        })?;
        let api_version =
            std::env::var("OPENAI_API_VERSION").unwrap_or_else(|_| "2024-02-01".to_string());
        Self::new(endpoint, api_key, deployment, api_version)
    }

    fn embeddings_url(&self) -> Result<url::Url> {
        let path = format!(
            "openai/deployments/{}/embeddings?api-version={}",
            self.deployment, self.api_version
        );
        Ok(self.azure_endpoint.join(&path)?)
    }
}

/// Azure OpenAI embedding adapter.
///
/// Uses the Azure-specific endpoint format with `api-key` header authentication.
#[derive(Debug, Clone)]
pub struct AzureOpenAITextVectorizer {
    config: AzureOpenAIConfig,
    client: reqwest::Client,
    blocking_client: reqwest::blocking::Client,
}

impl AzureOpenAITextVectorizer {
    /// Creates a new Azure OpenAI adapter.
    pub fn new(config: AzureOpenAIConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            blocking_client: reqwest::blocking::Client::new(),
        }
    }

    async fn embed_many_inner(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let response: EmbeddingResponse = self
            .client
            .post(self.config.embeddings_url()?)
            .header("api-key", &self.config.api_key)
            .json(&EmbeddingRequest {
                model: &self.config.deployment,
                input: texts.to_vec(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(response.data.into_iter().map(|d| d.embedding).collect())
    }
}

impl Vectorizer for AzureOpenAITextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response: EmbeddingResponse = self
            .blocking_client
            .post(self.config.embeddings_url()?)
            .header("api-key", &self.config.api_key)
            .json(&EmbeddingRequest {
                model: &self.config.deployment,
                input: vec![text],
            })
            .send()?
            .error_for_status()?
            .json()?;
        Ok(response
            .data
            .into_iter()
            .next()
            .map_or_else(Vec::new, |d| d.embedding))
    }
}

#[async_trait]
impl AsyncVectorizer for AzureOpenAITextVectorizer {
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
    fn azure_config_builds_embeddings_url() {
        let cfg = AzureOpenAIConfig::new(
            "https://myresource.openai.azure.com/",
            "test-key",
            "my-deployment",
            "2024-02-01",
        )
        .unwrap();
        let url = cfg.embeddings_url().unwrap();
        assert!(
            url.as_str()
                .contains("openai/deployments/my-deployment/embeddings"),
            "URL was: {url}"
        );
        assert!(
            url.as_str().contains("api-version=2024-02-01"),
            "URL was: {url}"
        );
    }

    #[test]
    fn azure_config_rejects_bad_url() {
        let result = AzureOpenAIConfig::new("not a url", "key", "dep", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn azure_vectorizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AzureOpenAITextVectorizer>();
    }
}
