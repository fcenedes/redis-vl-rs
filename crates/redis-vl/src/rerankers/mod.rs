//! Reranker abstractions and provider adapters.
//!
//! Rerankers take a query and a set of documents and reorder the documents
//! by relevance to the query. This module mirrors the Python `redisvl.utils.rerank`
//! surface.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::error::Result;

#[cfg(feature = "rerankers")]
mod cohere;
#[cfg(feature = "rerankers")]
pub use self::cohere::{CohereReranker, CohereRerankerConfig};

/// A single document that can be reranked.
///
/// Documents can be plain strings or dictionaries of field values.
#[derive(Debug, Clone)]
pub enum RerankDoc {
    /// A plain text document.
    Text(String),
    /// A structured document with named fields.
    Fields(HashMap<String, String>),
}

impl RerankDoc {
    /// Returns the text content of this document, using `rank_by` fields if
    /// this is a structured document.
    pub fn text(&self, rank_by: &[String]) -> Option<String> {
        match self {
            Self::Text(s) => Some(s.clone()),
            Self::Fields(map) => {
                if rank_by.is_empty() {
                    // Fall back to "content" key
                    map.get("content").cloned()
                } else {
                    let parts: Vec<&str> = rank_by
                        .iter()
                        .filter_map(|k| map.get(k).map(|v| v.as_str()))
                        .collect();
                    if parts.is_empty() {
                        None
                    } else {
                        Some(parts.join(" "))
                    }
                }
            }
        }
    }
}

/// Output from a reranking operation.
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Reranked documents in order of relevance.
    pub docs: Vec<RerankDoc>,
    /// Relevance scores aligned with `docs`, if requested.
    pub scores: Option<Vec<f64>>,
}

/// Synchronous reranker abstraction.
pub trait Reranker: Send + Sync {
    /// Reranks `docs` by relevance to `query`, returning at most `limit` results.
    fn rank(&self, query: &str, docs: &[RerankDoc], limit: Option<usize>) -> Result<RerankResult>;
}

/// Asynchronous reranker abstraction.
#[async_trait]
pub trait AsyncReranker: Send + Sync {
    /// Asynchronously reranks `docs` by relevance to `query`.
    async fn rank(
        &self,
        query: &str,
        docs: &[RerankDoc],
        limit: Option<usize>,
    ) -> Result<RerankResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerank_doc_text_returns_plain_text() {
        let doc = RerankDoc::Text("hello world".to_string());
        assert_eq!(doc.text(&[]), Some("hello world".to_string()));
    }

    #[test]
    fn rerank_doc_fields_uses_content_key_by_default() {
        let mut map = HashMap::new();
        map.insert("content".to_string(), "doc text".to_string());
        map.insert("title".to_string(), "ignored".to_string());
        let doc = RerankDoc::Fields(map);
        assert_eq!(doc.text(&[]), Some("doc text".to_string()));
    }

    #[test]
    fn rerank_doc_fields_uses_rank_by() {
        let mut map = HashMap::new();
        map.insert("content".to_string(), "doc text".to_string());
        map.insert("title".to_string(), "the title".to_string());
        let doc = RerankDoc::Fields(map);
        assert_eq!(
            doc.text(&["title".to_string()]),
            Some("the title".to_string())
        );
    }

    #[test]
    fn rerank_doc_fields_joins_multiple_rank_by() {
        let mut map = HashMap::new();
        map.insert("a".to_string(), "first".to_string());
        map.insert("b".to_string(), "second".to_string());
        let doc = RerankDoc::Fields(map);
        let result = doc.text(&["a".to_string(), "b".to_string()]);
        assert_eq!(result, Some("first second".to_string()));
    }

    #[test]
    fn rerank_doc_fields_returns_none_for_missing_keys() {
        let map = HashMap::new();
        let doc = RerankDoc::Fields(map);
        assert_eq!(doc.text(&["nonexistent".to_string()]), None);
    }
}
