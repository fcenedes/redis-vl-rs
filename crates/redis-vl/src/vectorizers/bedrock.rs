//! AWS Bedrock embedding adapter.
//!
//! Enabled by the `bedrock` feature flag. Uses the AWS SDK for Rust to call
//! Amazon Bedrock Runtime's `InvokeModel` API with SigV4 authentication.
//! The default model is `amazon.titan-embed-text-v2:0`.
//!
//! Bedrock does not support batch embedding — each text is embedded
//! individually via `invoke_model`.

use async_trait::async_trait;
use aws_sdk_bedrockruntime::primitives::Blob;

use super::{AsyncVectorizer, Vectorizer};
use crate::error::{Error, Result};

/// Configuration for the AWS Bedrock embedding provider.
///
/// Credentials are resolved through the standard AWS credential chain
/// (environment variables, shared config/credentials files, IAM roles, etc.)
/// unless explicit values are provided.
#[derive(Debug, Clone)]
pub struct BedrockConfig {
    /// Bedrock model ID (default: `amazon.titan-embed-text-v2:0`).
    pub model: String,
    /// AWS region (default: `us-east-1`).
    pub region: String,
    /// Optional explicit AWS access key ID.
    pub access_key_id: Option<String>,
    /// Optional explicit AWS secret access key.
    pub secret_access_key: Option<String>,
}

impl Default for BedrockConfig {
    fn default() -> Self {
        Self {
            model: "amazon.titan-embed-text-v2:0".into(),
            region: "us-east-1".into(),
            access_key_id: None,
            secret_access_key: None,
        }
    }
}

impl BedrockConfig {
    /// Creates a new Bedrock config with the given model ID.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Sets the AWS region.
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Sets explicit AWS credentials.
    #[must_use]
    pub fn with_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id.into());
        self.secret_access_key = Some(secret_access_key.into());
        self
    }

    /// Constructs a config from environment variables.
    ///
    /// Reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
    /// `AWS_REGION` (defaults to `us-east-1`). The `BEDROCK_MODEL_ID`
    /// env var overrides the default model if set.
    pub fn from_env() -> Result<Self> {
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into());
        let model = std::env::var("BEDROCK_MODEL_ID")
            .unwrap_or_else(|_| "amazon.titan-embed-text-v2:0".into());
        // Credentials are optional here; the AWS SDK will resolve them
        // through its default chain if not explicitly provided.
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID").ok();
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok();
        Ok(Self {
            model,
            region,
            access_key_id,
            secret_access_key,
        })
    }
}

/// Bedrock Titan embedding request body.
#[derive(Debug, serde::Serialize)]
struct TitanEmbedRequest<'a> {
    /// Text to embed.
    #[serde(rename = "inputText")]
    input_text: &'a str,
}

/// Bedrock Titan embedding response body.
#[derive(Debug, serde::Deserialize)]
struct TitanEmbedResponse {
    /// The embedding vector.
    embedding: Vec<f32>,
}

/// AWS Bedrock embedding adapter.
///
/// Uses the Bedrock Runtime `InvokeModel` API with SigV4 authentication.
/// Each text is embedded individually since Bedrock does not support batch
/// embedding.
///
/// # Example
///
/// ```no_run
/// use redis_vl::vectorizers::{BedrockConfig, BedrockTextVectorizer, Vectorizer};
///
/// # fn main() -> redis_vl::error::Result<()> {
/// let config = BedrockConfig::from_env()?;
/// let rt = tokio::runtime::Runtime::new().unwrap();
/// let vectorizer = rt.block_on(BedrockTextVectorizer::new(config))?;
/// # Ok(())
/// # }
/// ```
pub struct BedrockTextVectorizer {
    config: BedrockConfig,
    client: aws_sdk_bedrockruntime::Client,
}

impl std::fmt::Debug for BedrockTextVectorizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockTextVectorizer")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl BedrockTextVectorizer {
    /// Creates a new Bedrock adapter by building an AWS SDK client from the
    /// provided configuration.
    ///
    /// This is an `async` constructor because the AWS SDK credential
    /// resolution is asynchronous.
    pub async fn new(config: BedrockConfig) -> Result<Self> {
        let mut aws_config_loader =
            aws_config::from_env().region(aws_config::Region::new(config.region.clone()));

        if let (Some(ref key_id), Some(ref secret)) =
            (&config.access_key_id, &config.secret_access_key)
        {
            aws_config_loader = aws_config_loader.credentials_provider(
                aws_sdk_bedrockruntime::config::Credentials::new(
                    key_id.clone(),
                    secret.clone(),
                    None, // session token
                    None, // expiry
                    "redis-vl-bedrock",
                ),
            );
        }

        let sdk_config = aws_config_loader.load().await;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);

        Ok(Self { config, client })
    }

    /// Invokes the Bedrock model for a single text and returns the embedding.
    async fn invoke_embed(&self, text: &str) -> Result<Vec<f32>> {
        let body = serde_json::to_vec(&TitanEmbedRequest { input_text: text })?;

        let response = self
            .client
            .invoke_model()
            .model_id(&self.config.model)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body))
            .send()
            .await
            .map_err(|e| Error::InvalidInput(format!("Bedrock invoke_model failed: {e}")))?;

        let response_bytes = response.body().as_ref();
        let parsed: TitanEmbedResponse = serde_json::from_slice(response_bytes)?;
        Ok(parsed.embedding)
    }
}

impl Vectorizer for BedrockTextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Build a current-thread runtime for the blocking path.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::InvalidInput(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(self.invoke_embed(text))
    }

    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::InvalidInput(format!("failed to build tokio runtime: {e}")))?;
        rt.block_on(async {
            let mut embeddings = Vec::with_capacity(texts.len());
            for text in texts {
                embeddings.push(self.invoke_embed(text).await?);
            }
            Ok(embeddings)
        })
    }
}

#[async_trait]
impl AsyncVectorizer for BedrockTextVectorizer {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.invoke_embed(text).await
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.invoke_embed(text).await?);
        }
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bedrock_config_defaults() {
        let cfg = BedrockConfig::default();
        assert_eq!(cfg.model, "amazon.titan-embed-text-v2:0");
        assert_eq!(cfg.region, "us-east-1");
        assert!(cfg.access_key_id.is_none());
        assert!(cfg.secret_access_key.is_none());
    }

    #[test]
    fn bedrock_config_builder() {
        let cfg = BedrockConfig::new("amazon.titan-embed-text-v1")
            .with_region("eu-west-1")
            .with_credentials("AKID", "SECRET");
        assert_eq!(cfg.model, "amazon.titan-embed-text-v1");
        assert_eq!(cfg.region, "eu-west-1");
        assert_eq!(cfg.access_key_id.as_deref(), Some("AKID"));
        assert_eq!(cfg.secret_access_key.as_deref(), Some("SECRET"));
    }

    #[test]
    fn titan_request_serializes_correctly() {
        let req = TitanEmbedRequest {
            input_text: "hello world",
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["inputText"], "hello world");
        // Must not contain any other top-level keys
        assert_eq!(json.as_object().unwrap().len(), 1);
    }

    #[test]
    fn titan_response_deserializes_correctly() {
        let json = r#"{"embedding": [0.1, 0.2, 0.3]}"#;
        let resp: TitanEmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn bedrock_vectorizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BedrockTextVectorizer>();
    }
}
