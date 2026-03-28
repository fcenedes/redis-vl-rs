//! HuggingFace local embedding adapter using ONNX Runtime via `fastembed`.
//!
//! Enabled by the `hf-local` feature flag. This vectorizer runs embedding
//! models locally without requiring an external API. Models are automatically
//! downloaded from the HuggingFace Hub on first use.
//!
//! # Example
//!
//! ```rust,no_run
//! use redis_vl::vectorizers::HuggingFaceTextVectorizer;
//!
//! // Uses the default model (AllMiniLML6V2)
//! let vectorizer = HuggingFaceTextVectorizer::new(Default::default()).unwrap();
//! let embedding = vectorizer.embed("Hello, world!").unwrap();
//! ```

use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding};

use super::Vectorizer;
use crate::error::{Error, Result};

/// Configuration for the HuggingFace local embedding provider.
#[derive(Debug, Clone)]
pub struct HuggingFaceConfig {
    /// The embedding model to use.
    ///
    /// Defaults to [`EmbeddingModel::AllMiniLML6V2`].
    pub model: EmbeddingModel,
    /// Whether to show download progress when fetching the model.
    ///
    /// Defaults to `false`.
    pub show_download_progress: bool,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::AllMiniLML6V2,
            show_download_progress: false,
        }
    }
}

impl HuggingFaceConfig {
    /// Creates a new config with the given model.
    #[must_use]
    pub fn new(model: EmbeddingModel) -> Self {
        Self {
            model,
            show_download_progress: false,
        }
    }

    /// Enables download progress output.
    #[must_use]
    pub fn with_show_download_progress(mut self, show: bool) -> Self {
        self.show_download_progress = show;
        self
    }
}

/// HuggingFace local embedding adapter backed by ONNX Runtime.
///
/// Uses the [`fastembed`] crate to run embedding models locally. Models are
/// automatically downloaded from the HuggingFace Hub on first use and cached
/// on disk.
///
/// This vectorizer implements [`Vectorizer`] for synchronous embedding
/// generation. For async use cases, wrap it with
/// [`tokio::task::spawn_blocking`] or use it with the synchronous semantic
/// extension APIs.
pub struct HuggingFaceTextVectorizer {
    model: Mutex<TextEmbedding>,
}

impl HuggingFaceTextVectorizer {
    /// Creates a new HuggingFace local vectorizer.
    ///
    /// This may download the model from HuggingFace Hub on first invocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn new(config: HuggingFaceConfig) -> Result<Self> {
        let init_options = fastembed::InitOptions::new(config.model)
            .with_show_download_progress(config.show_download_progress);

        let model = TextEmbedding::try_new(init_options)
            .map_err(|e| Error::InvalidInput(format!("failed to load HF model: {e}")))?;

        Ok(Self {
            model: Mutex::new(model),
        })
    }
}

impl Vectorizer for HuggingFaceTextVectorizer {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::InvalidInput(format!("lock poisoned: {e}")))?;
        let mut embeddings = model
            .embed(vec![text], None)
            .map_err(|e| Error::InvalidInput(format!("embedding failed: {e}")))?;
        Ok(embeddings.pop().unwrap_or_default())
    }

    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::InvalidInput(format!("lock poisoned: {e}")))?;
        model
            .embed(texts.to_vec(), None)
            .map_err(|e| Error::InvalidInput(format!("embedding failed: {e}")))
    }
}

// Safety: Mutex<TextEmbedding> provides thread-safe access.
unsafe impl Send for HuggingFaceTextVectorizer {}
unsafe impl Sync for HuggingFaceTextVectorizer {}

impl std::fmt::Debug for HuggingFaceTextVectorizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HuggingFaceTextVectorizer")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_all_mini_lm() {
        let cfg = HuggingFaceConfig::default();
        assert!(!cfg.show_download_progress);
        // EmbeddingModel doesn't implement PartialEq, so verify Debug output.
        assert!(format!("{:?}", cfg.model).contains("AllMiniLML6V2"));
    }

    #[test]
    fn config_builder_chain() {
        let cfg =
            HuggingFaceConfig::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true);
        assert!(cfg.show_download_progress);
    }

    #[test]
    fn vectorizer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HuggingFaceTextVectorizer>();
    }

    #[test]
    fn debug_impl_does_not_panic() {
        // We can't easily construct a vectorizer in unit tests without downloading
        // a model, but we can verify the Debug impl compiles and the config Debug works.
        let cfg = HuggingFaceConfig::default();
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("HuggingFaceConfig"));
    }
}
