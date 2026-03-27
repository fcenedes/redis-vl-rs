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

/// Documents sent to the Cohere rerank API can be either plain strings or
/// structured objects (maps of field→value). When `rank_fields` is provided,
/// Cohere expects structured documents so it can rank by the specified fields.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum CohereDocument<'a> {
    /// A plain text document.
    Text(&'a str),
    /// A structured document with named fields — sent as a JSON object.
    Fields(std::collections::HashMap<&'a str, &'a str>),
}

#[derive(Debug, Serialize)]
struct CohereRerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    top_n: usize,
    documents: Vec<CohereDocument<'a>>,
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

    /// Builds the request body matching the upstream Python `CohereReranker._preprocess` contract:
    ///
    /// - **String docs**: sent as plain strings, no `rank_fields`.
    /// - **Dict docs + `rank_by`**: sent as structured JSON objects with `rank_fields` set.
    /// - **Dict docs without `rank_by`**: returns `InvalidInput` error (matches Python `ValueError`).
    fn prepare_request<'a>(
        &'a self,
        query: &'a str,
        docs: &'a [RerankDoc],
        limit: Option<usize>,
    ) -> Result<CohereRerankRequest<'a>> {
        let all_fields = docs.iter().all(|d| matches!(d, RerankDoc::Fields(_)));

        let (documents, rank_fields) = if all_fields {
            // Dict-style docs: must have rank_by, send structured documents + rank_fields
            if self.config.rank_by.is_empty() {
                return Err(crate::error::Error::InvalidInput(
                    "If reranking dictionary-like docs, you must provide a list of rank_by fields"
                        .into(),
                ));
            }
            let structured: Vec<CohereDocument<'a>> = docs
                .iter()
                .map(|d| match d {
                    RerankDoc::Fields(map) => {
                        let obj: std::collections::HashMap<&str, &str> =
                            map.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                        CohereDocument::Fields(obj)
                    }
                    // unreachable given `all_fields` guard, but handle gracefully
                    RerankDoc::Text(s) => CohereDocument::Text(s.as_str()),
                })
                .collect();
            let rf: Vec<&str> = self.config.rank_by.iter().map(|s| s.as_str()).collect();
            (structured, Some(rf))
        } else {
            // String docs (or mixed): flatten to plain text, no rank_fields
            let plain: Vec<CohereDocument<'a>> = docs
                .iter()
                .filter_map(|d| match d {
                    RerankDoc::Text(s) => Some(CohereDocument::Text(s.as_str())),
                    RerankDoc::Fields(map) => {
                        map.get("content").map(|s| CohereDocument::Text(s.as_str()))
                    }
                })
                .collect();
            (plain, None)
        };

        let top_n = limit.unwrap_or(documents.len());
        Ok(CohereRerankRequest {
            model: &self.config.model,
            query,
            top_n,
            documents,
            rank_fields,
        })
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
        let request = self.prepare_request(query, docs, limit)?;
        let resp: CohereRerankResponse = self
            .blocking_client
            .post(COHERE_RERANK_URL)
            .bearer_auth(&self.config.api_key)
            .json(&request)
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
        let request = self.prepare_request(query, docs, limit)?;
        let resp: CohereRerankResponse = self
            .client
            .post(COHERE_RERANK_URL)
            .bearer_auth(&self.config.api_key)
            .json(&request)
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
    fn prepare_request_string_docs_sends_plain_text_no_rank_fields() {
        let cfg = CohereRerankerConfig::new("key");
        let reranker = CohereReranker::new(cfg);
        let docs = vec![
            RerankDoc::Text("doc1".into()),
            RerankDoc::Text("doc2".into()),
        ];
        let req = reranker.prepare_request("query", &docs, Some(2)).unwrap();
        assert!(req.rank_fields.is_none());
        assert_eq!(req.documents.len(), 2);
        assert_eq!(req.top_n, 2);
        // Verify they serialize as plain strings
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["documents"], serde_json::json!(["doc1", "doc2"]));
        // rank_fields is omitted from JSON when None (skip_serializing_if)
        assert!(json.get("rank_fields").is_none());
    }

    #[test]
    fn prepare_request_dict_docs_with_rank_by_sends_structured_plus_rank_fields() {
        let cfg = CohereRerankerConfig::new("key").rank_by(vec!["content".into()]);
        let reranker = CohereReranker::new(cfg);
        let mut map1 = std::collections::HashMap::new();
        map1.insert("content".to_string(), "document 1".to_string());
        let mut map2 = std::collections::HashMap::new();
        map2.insert("content".to_string(), "document 2".to_string());
        let docs = vec![RerankDoc::Fields(map1), RerankDoc::Fields(map2)];

        let req = reranker.prepare_request("query", &docs, None).unwrap();
        assert_eq!(req.rank_fields, Some(vec!["content"]));
        assert_eq!(req.documents.len(), 2);
        // Verify they serialize as JSON objects
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["documents"][0]["content"],
            serde_json::json!("document 1")
        );
        assert_eq!(json["rank_fields"], serde_json::json!(["content"]));
    }

    #[test]
    fn prepare_request_dict_docs_without_rank_by_errors_like_python() {
        // Python raises ValueError: "If reranking dictionary-like docs, you must provide
        // a list of rank_by fields"
        let cfg = CohereRerankerConfig::new("key"); // rank_by is empty
        let reranker = CohereReranker::new(cfg);
        let mut map = std::collections::HashMap::new();
        map.insert("content".to_string(), "doc".to_string());
        let docs = vec![RerankDoc::Fields(map)];

        let err = reranker.prepare_request("query", &docs, None);
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("rank_by"),
            "Error should mention rank_by: {msg}"
        );
    }

    #[test]
    fn prepare_request_limit_defaults_to_doc_count() {
        let cfg = CohereRerankerConfig::new("key");
        let reranker = CohereReranker::new(cfg);
        let docs = vec![
            RerankDoc::Text("a".into()),
            RerankDoc::Text("b".into()),
            RerankDoc::Text("c".into()),
        ];
        let req = reranker.prepare_request("q", &docs, None).unwrap();
        assert_eq!(req.top_n, 3);
    }

    #[test]
    fn cohere_reranker_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CohereReranker>();
    }
}
