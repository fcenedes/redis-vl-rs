//! Embedding provider abstractions and OpenAI-compatible adapters.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

#[cfg(feature = "azure-openai")]
mod azure_openai;
#[cfg(feature = "azure-openai")]
pub use azure_openai::{AzureOpenAIConfig, AzureOpenAITextVectorizer};

#[cfg(feature = "cohere")]
mod cohere;
#[cfg(feature = "cohere")]
pub use self::cohere::{CohereConfig, CohereTextVectorizer};

#[cfg(feature = "mistral")]
mod mistral;
#[cfg(feature = "mistral")]
pub use self::mistral::{MistralAITextVectorizer, MistralConfig};

#[cfg(feature = "voyageai")]
mod voyageai;
#[cfg(feature = "voyageai")]
pub use self::voyageai::{VoyageAIConfig, VoyageAITextVectorizer};

/// Shared embedding request payload.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingRequest<'a> {
    /// Model name.
    pub model: &'a str,
    /// Input texts.
    pub input: Vec<&'a str>,
}

/// Synchronous vectorizer abstraction.
pub trait Vectorizer: Send + Sync {
    /// Embeds a single string.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embeds many strings.
    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

/// Asynchronous vectorizer abstraction.
#[async_trait]
pub trait AsyncVectorizer: Send + Sync {
    /// Embeds a single string asynchronously.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embeds many strings asynchronously.
    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }
}

/// Shared configuration for OpenAI-compatible embedding providers.
#[derive(Debug, Clone)]
pub struct OpenAICompatibleConfig {
    /// Base URL for the provider.
    pub base_url: url::Url,
    /// API key used for authentication.
    pub api_key: String,
    /// Embedding model name.
    pub model: String,
}

impl OpenAICompatibleConfig {
    /// Creates a new OpenAI-compatible config.
    pub fn new(
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            base_url: url::Url::parse(base_url.as_ref())?,
            api_key: api_key.into(),
            model: model.into(),
        })
    }

    fn embeddings_url(&self) -> Result<url::Url> {
        Ok(self.base_url.join("embeddings")?)
    }
}

/// OpenAI embedding adapter.
#[derive(Debug, Clone)]
pub struct OpenAITextVectorizer {
    config: OpenAICompatibleConfig,
    client: reqwest::Client,
    blocking_client: reqwest::blocking::Client,
}

impl OpenAITextVectorizer {
    /// Creates a new OpenAI adapter.
    pub fn new(config: OpenAICompatibleConfig) -> Self {
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
            .bearer_auth(&self.config.api_key)
            .json(&EmbeddingRequest {
                model: &self.config.model,
                input: texts.to_vec(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect())
    }
}

impl Vectorizer for OpenAITextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response: EmbeddingResponse = self
            .blocking_client
            .post(self.config.embeddings_url()?)
            .bearer_auth(&self.config.api_key)
            .json(&EmbeddingRequest {
                model: &self.config.model,
                input: vec![text],
            })
            .send()?
            .error_for_status()?
            .json()?;
        Ok(response
            .data
            .into_iter()
            .next()
            .map_or_else(Vec::new, |item| item.embedding))
    }

    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let response: EmbeddingResponse = self
            .blocking_client
            .post(self.config.embeddings_url()?)
            .bearer_auth(&self.config.api_key)
            .json(&EmbeddingRequest {
                model: &self.config.model,
                input: texts.to_vec(),
            })
            .send()?
            .error_for_status()?
            .json()?;
        Ok(response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect())
    }
}

#[async_trait]
impl AsyncVectorizer for OpenAITextVectorizer {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut embeddings = self.embed_many_inner(&[text]).await?;
        Ok(embeddings.pop().unwrap_or_default())
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_many_inner(texts).await
    }
}

/// LiteLLM embedding adapter built on the same OpenAI-compatible transport.
#[derive(Debug, Clone)]
pub struct LiteLLMTextVectorizer {
    inner: OpenAITextVectorizer,
}

impl LiteLLMTextVectorizer {
    /// Creates a new LiteLLM adapter.
    pub fn new(config: OpenAICompatibleConfig) -> Self {
        Self {
            inner: OpenAITextVectorizer::new(config),
        }
    }
}

impl Vectorizer for LiteLLMTextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Vectorizer::embed(&self.inner, text)
    }

    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Vectorizer::embed_many(&self.inner, texts)
    }
}

#[async_trait]
impl AsyncVectorizer for LiteLLMTextVectorizer {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        AsyncVectorizer::embed(&self.inner, text).await
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        AsyncVectorizer::embed_many(&self.inner, texts).await
    }
}

/// Custom synchronous vectorizer backed by a user callback.
pub struct CustomTextVectorizer<F>
where
    F: Fn(&str) -> Result<Vec<f32>> + Send + Sync + 'static,
{
    embedder: Arc<F>,
}

impl<F> CustomTextVectorizer<F>
where
    F: Fn(&str) -> Result<Vec<f32>> + Send + Sync + 'static,
{
    /// Creates a custom synchronous vectorizer.
    pub fn new(embedder: F) -> Self {
        Self {
            embedder: Arc::new(embedder),
        }
    }
}

impl<F> Vectorizer for CustomTextVectorizer<F>
where
    F: Fn(&str) -> Result<Vec<f32>> + Send + Sync + 'static,
{
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        (self.embedder)(text)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingResponse {
    pub(crate) data: Vec<EmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingDatum {
    pub(crate) embedding: Vec<f32>,
}
