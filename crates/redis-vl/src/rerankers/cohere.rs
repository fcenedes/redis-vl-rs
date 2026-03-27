//! Cohere reranker adapter.
//!
//! Uses the Cohere `/v1/rerank` REST API to rerank documents.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{AsyncReranker, RerankDoc, RerankResult, Reranker};
use crate::error::Result;

/// Configuration for the Cohere reranker.
#[derive(Debug, Clone)]
pub struct CohereRerankerConfig {
    /// API key for Cohere.
    pub api_key: String,
    /// Rerank model name (default: `rerank-english-v3.0`).
    pub model: String,
    /// Fields to rank by when documents are structured.
    pub rank_by: Vec<String>,
    /// Whether to include relevance scores in the output.
    pub return_score: bool,
}

impl CohereRerankerConfig {
    /// Creates a new Cohere reranker config.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "rerank-english-v3.0".to_string(),
            rank_by: Vec::new(),
            return_score: true,
        }
    }

    /// Constructs from `COHERE_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("COHERE_API_KEY")
            .map_err(|_| crate::error::Error::InvalidInput("COHERE_API_KEY not set".into()))?;
        Ok(Self::new(api_key))
    }

    /// Sets the model name.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the rank-by fields.
    pub fn rank_by(mut self, fields: Vec<String>) -> Self {
        self.rank_by = fields;
        self
    }

    /// Sets whether to return scores.
    pub fn return_score(mut self, val: bool) -> Self {
        self.return_score = val;
        self
    }
}

const COHERE_RERANK_URL: &str = "https://api.cohere.com/v1/rerank";

#[derive(Serialize)]
struct CohereRerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    top_n: usize,
    documents: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rank_fields: Option<Vec<&'a str>>,
}

#[derive(Deserialize)]
struct CohereRerankResponse {
    results: Vec<CohereRerankResult>,
}

#[derive(Deserialize)]
struct CohereRerankResult {
    index: usize,
    relevance_score: f64,
}

/// Cohere reranker adapter.
///
/// Mirrors the Python `CohereReranker` using the Cohere `/v1/rerank` REST API.
#[derive(Debug, Clone)]
pub struct CohereReranker {
    config: CohereRerankerConfig,
    client: reqwest::Client,
    blocking_client: reqwest::blocking::Client,
}

impl CohereReranker {
    /// Creates a new Cohere reranker.
    pub fn new(config: CohereRerankerConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            blocking_client: reqwest::blocking::Client::new(),
        }
    }

    fn prepare_texts<'a>(&'a self, docs: &'a [RerankDoc]) -> Vec<&'a str> {
        docs.iter()
            .filter_map(|d| match d {
                RerankDoc::Text(s) => Some(s.as_str()),
                RerankDoc::Fields(map) => {
                    if self.config.rank_by.is_empty() {
                        map.get("content").map(|s| s.as_str())
                    } else {
                        // For Cohere with rank_fields we send the raw text of the first field
                        self.config
                            .rank_by
                            .first()
                            .and_then(|k| map.get(k).map(|s| s.as_str()))
                    }
                }
            })
            .collect()
    }

    fn build_result(&self, docs: &[RerankDoc], response: CohereRerankResponse) -> RerankResult {
        let mut reranked = Vec::with_capacity(response.results.len());
        let mut scores = Vec::with_capacity(response.results.len());
        for item in &response.results {
            if item.index < docs.len() {
                reranked.push(docs[item.index].clone());
                scores.push(item.relevance_score);
            }
        }
        RerankResult {
            docs: reranked,
            scores: if self.config.return_score {
                Some(scores)
            } else {
                None
            },
        }
    }
}

impl Reranker for CohereReranker {
    fn rank(&self, query: &str, docs: &[RerankDoc], limit: Option<usize>) -> Result<RerankResult> {
        let texts = self.prepare_texts(docs);
        let top_n = limit.unwrap_or(texts.len());
        let resp: CohereRerankResponse = self
            .blocking_client
            .post(COHERE_RERANK_URL)
            .bearer_auth(&self.config.api_key)
            .json(&CohereRerankRequest {
                model: &self.config.model,
                query,
                top_n,
                documents: texts,
                rank_fields: None,
            })
            .send()?
            .error_for_status()?
            .json()?;
        Ok(self.build_result(docs, resp))
    }
}

#[async_trait]
impl AsyncReranker for CohereReranker {
    async fn rank(
        &self,
        query: &str,
        docs: &[RerankDoc],
        limit: Option<usize>,
    ) -> Result<RerankResult> {
        let texts = self.prepare_texts(docs);
        let top_n = limit.unwrap_or(texts.len());
        let resp: CohereRerankResponse = self
            .client
            .post(COHERE_RERANK_URL)
            .bearer_auth(&self.config.api_key)
            .json(&CohereRerankRequest {
                model: &self.config.model,
                query,
                top_n,
                documents: texts,
                rank_fields: None,
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(self.build_result(docs, resp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohere_reranker_config_defaults() {
        let cfg = CohereRerankerConfig::new("test-key");
        assert_eq!(cfg.api_key, "test-key");
        assert_eq!(cfg.model, "rerank-english-v3.0");
        assert!(cfg.rank_by.is_empty());
        assert!(cfg.return_score);
    }

    #[test]
    fn cohere_reranker_config_builder() {
        let cfg = CohereRerankerConfig::new("key")
            .model("rerank-multilingual-v3.0")
            .rank_by(vec!["content".into()])
            .return_score(false);
        assert_eq!(cfg.model, "rerank-multilingual-v3.0");
        assert_eq!(cfg.rank_by, vec!["content"]);
        assert!(!cfg.return_score);
    }

    #[test]
    fn cohere_reranker_build_result_with_scores() {
        let cfg = CohereRerankerConfig::new("key");
        let reranker = CohereReranker::new(cfg);
        let docs = vec![
            RerankDoc::Text("a".into()),
            RerankDoc::Text("b".into()),
            RerankDoc::Text("c".into()),
        ];
        let response = CohereRerankResponse {
            results: vec![
                CohereRerankResult {
                    index: 2,
                    relevance_score: 0.9,
                },
                CohereRerankResult {
                    index: 0,
                    relevance_score: 0.5,
                },
            ],
        };
        let result = reranker.build_result(&docs, response);
        assert_eq!(result.docs.len(), 2);
        let scores = result.scores.unwrap();
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 0.9).abs() < f64::EPSILON);
        assert!((scores[1] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn cohere_reranker_build_result_without_scores() {
        let cfg = CohereRerankerConfig::new("key").return_score(false);
        let reranker = CohereReranker::new(cfg);
        let docs = vec![RerankDoc::Text("a".into())];
        let response = CohereRerankResponse {
            results: vec![CohereRerankResult {
                index: 0,
                relevance_score: 0.9,
            }],
        };
        let result = reranker.build_result(&docs, response);
        assert!(result.scores.is_none());
    }

    #[test]
    fn cohere_reranker_prepare_texts_plain() {
        let cfg = CohereRerankerConfig::new("key");
        let reranker = CohereReranker::new(cfg);
        let docs = vec![
            RerankDoc::Text("doc1".into()),
            RerankDoc::Text("doc2".into()),
        ];
        let texts = reranker.prepare_texts(&docs);
        assert_eq!(texts, vec!["doc1", "doc2"]);
    }

    #[test]
    fn cohere_reranker_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CohereReranker>();
    }
}
