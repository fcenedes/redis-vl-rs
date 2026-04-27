#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
//! # redis-vl
//!
//! Async-first Rust implementation of the [Redis Vector Library](https://github.com/redis/redis-vl-python).
//!
//! This crate provides vector search, semantic caching, message history, and
//! routing on top of Redis. It targets feature parity with the Python `redisvl`
//! package while using idiomatic Rust APIs.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use redis_vl::{IndexSchema, SearchIndex, Vector, VectorQuery};
//!
//! let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
//! let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
//! index.create().unwrap();
//!
//! let query = VectorQuery::new(
//!     Vector::new(&[0.1_f32; 128] as &[f32]),
//!     "embedding",
//!     5,
//! );
//! let result = index.search(&query).unwrap();
//! ```
//!
//! ## Feature flags
//!
//! - `openai` *(default)* – OpenAI-compatible vectorizer
//! - `litellm` *(default)* – LiteLLM vectorizer
//! - `azure-openai` – Azure OpenAI vectorizer
//! - `bedrock` – AWS Bedrock vectorizer
//! - `cohere` – Cohere vectorizer and reranker support
//! - `voyageai` – VoyageAI vectorizer
//! - `mistral` – Mistral vectorizer
//! - `anthropic` – Anthropic adapter (Voyage AI-backed)
//! - `vertex-ai` – Google Vertex AI vectorizer
//! - `hf-local` – HuggingFace local ONNX embedding via `fastembed`
//! - `sql` – SQL query support (`SQLQuery`)
//! - `rerankers` – Reranker support (`CohereReranker`)

/// Error types returned by the library.
///
/// The [`Error`](error::Error) enum wraps Redis, JSON, YAML, HTTP, I/O,
/// schema validation, and input errors. All fallible public APIs return
/// `Result<T, Error>`.
pub mod error;

/// Cache and storage extensions: [`EmbeddingsCache`](extensions::cache::EmbeddingsCache),
/// [`SemanticCache`](extensions::cache::SemanticCache),
/// [`MessageHistory`](extensions::history::MessageHistory),
/// [`SemanticMessageHistory`](extensions::history::SemanticMessageHistory),
/// and [`SemanticRouter`](extensions::router::SemanticRouter).
pub mod extensions;

/// Filter DSL for Redis Search queries.
///
/// Build composable filters with [`Tag`](filter::Tag), [`Text`](filter::Text),
/// [`Num`](filter::Num), [`Geo`](filter::Geo), [`GeoRadius`](filter::GeoRadius),
/// and [`Timestamp`](filter::Timestamp). Combine with `&` (AND), `|` (OR),
/// and `!` (NOT).
pub mod filter;

/// Search index lifecycle and Redis transport.
///
/// [`SearchIndex`](index::SearchIndex) provides blocking operations while
/// [`AsyncSearchIndex`](index::AsyncSearchIndex) provides Tokio-based async
/// operations. Both support create, delete, load, fetch, search, query,
/// batch, pagination, hybrid search, aggregate queries, and multi-vector queries.
pub mod index;

/// Query builders for Redis Search.
///
/// Includes [`VectorQuery`](query::VectorQuery), [`VectorRangeQuery`](query::VectorRangeQuery),
/// [`TextQuery`](query::TextQuery), [`FilterQuery`](query::FilterQuery),
/// [`CountQuery`](query::CountQuery), [`HybridQuery`](query::HybridQuery),
/// [`AggregateHybridQuery`](query::AggregateHybridQuery),
/// [`MultiVectorQuery`](query::MultiVectorQuery), and (with `sql` feature)
/// `SQLQuery`.
pub mod query;

/// Reranker abstractions and provider adapters.
///
/// The [`Reranker`](rerankers::Reranker) and [`AsyncReranker`](rerankers::AsyncReranker)
/// traits define the reranking interface. Enable the `rerankers` feature for
/// `CohereReranker`.
pub mod rerankers;

/// Schema types and Redis Search schema serialization.
///
/// [`IndexSchema`](schema::IndexSchema) describes the structure of a Redis
/// Search index including name, key prefix, storage type, and field definitions.
/// Supports YAML/JSON parsing, multi-prefix indexes, and stopword configuration.
pub mod schema;

/// Embedding provider abstractions and adapters.
///
/// The [`Vectorizer`](vectorizers::Vectorizer) trait (sync) and
/// [`AsyncVectorizer`](vectorizers::AsyncVectorizer) trait (async) define the
/// embedding interface. Provider implementations are gated behind feature flags.
pub mod vectorizers;

pub use error::Error;
pub use extensions::cache::{
    CacheConfig, EmbeddingCacheEntry, EmbeddingCacheItem, EmbeddingsCache, SemanticCache,
};
pub use extensions::history::{Message, MessageHistory, MessageRole, SemanticMessageHistory};
pub use extensions::router::{
    DistanceAggregationMethod, Route, RouteMatch, RoutingConfig, SemanticRouter,
};
pub use filter::{BetweenInclusivity, FilterExpression, Geo, GeoRadius, Num, Tag, Text, Timestamp};
pub use index::{
    AsyncSearchIndex, QueryOutput, RedisConnectionInfo, SearchDocument, SearchIndex, SearchResult,
};
pub use query::{
    AggregateHybridQuery, CountQuery, FilterQuery, GeoFilter, HybridCombinationMethod,
    HybridPolicy, HybridQuery, MultiVectorQuery, PageableQuery, QueryKind, QueryLimit, QueryParam,
    QueryParamValue, QueryRender, QueryString, SearchHistoryMode, SortBy, SortDirection, TextQuery,
    Vector, VectorDtype, VectorInput, VectorQuery, VectorRangeQuery, VectorSearchMethod,
};
#[cfg(feature = "sql")]
pub use query::{SQLQuery, SqlParam};
pub use rerankers::{AsyncReranker, RerankDoc, RerankResult, Reranker};
#[cfg(feature = "rerankers")]
pub use rerankers::{CohereReranker, CohereRerankerConfig};
pub use schema::{
    Field, FieldKind, GeoFieldAttributes, IndexDefinition, IndexSchema, NumericFieldAttributes,
    Prefix, StorageType, SvsCompressionType, TagFieldAttributes, TextFieldAttributes,
    TimestampFieldAttributes, VectorAlgorithm, VectorDataType, VectorDistanceMetric,
    VectorFieldAttributes,
};
#[cfg(feature = "anthropic")]
pub use vectorizers::{AnthropicConfig, AnthropicTextVectorizer};
pub use vectorizers::{
    AsyncVectorizer, CustomTextVectorizer, EmbeddingRequest, LiteLLMTextVectorizer,
    OpenAICompatibleConfig, OpenAITextVectorizer, Vectorizer,
};
#[cfg(feature = "azure-openai")]
pub use vectorizers::{AzureOpenAIConfig, AzureOpenAITextVectorizer};
#[cfg(feature = "bedrock")]
pub use vectorizers::{BedrockConfig, BedrockTextVectorizer};
#[cfg(feature = "cohere")]
pub use vectorizers::{CohereConfig, CohereTextVectorizer};
#[cfg(feature = "mistral")]
pub use vectorizers::{MistralAITextVectorizer, MistralConfig};
#[cfg(feature = "vertex-ai")]
pub use vectorizers::{VertexAIConfig, VertexAITextVectorizer};
#[cfg(feature = "voyageai")]
pub use vectorizers::{VoyageAIConfig, VoyageAITextVectorizer};
