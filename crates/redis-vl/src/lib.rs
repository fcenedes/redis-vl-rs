#![deny(missing_docs)]
#![doc = include_str!("../../../README.md")]

//! Async-first Rust implementation of Redis Vector Library concepts.

/// Error types returned by the library.
pub mod error;
/// Cache and storage extensions.
pub mod extensions;
/// Filter DSL for Redis Search queries.
pub mod filter;
/// Index lifecycle and Redis transport helpers.
pub mod index;
/// Query builders and vector payload helpers.
pub mod query;
/// Reranker abstractions and provider adapters.
pub mod rerankers;
/// Schema types and Redis Search schema serialization.
pub mod schema;
/// Embedding provider abstractions and adapters.
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
    AggregateHybridQuery, CountQuery, FilterQuery, HybridCombinationMethod, HybridPolicy,
    HybridQuery, MultiVectorQuery, PageableQuery, QueryKind, QueryLimit, QueryParam,
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
pub use vectorizers::{
    AsyncVectorizer, CustomTextVectorizer, EmbeddingRequest, LiteLLMTextVectorizer,
    OpenAICompatibleConfig, OpenAITextVectorizer, Vectorizer,
};
#[cfg(feature = "azure-openai")]
pub use vectorizers::{AzureOpenAIConfig, AzureOpenAITextVectorizer};
#[cfg(feature = "cohere")]
pub use vectorizers::{CohereConfig, CohereTextVectorizer};
#[cfg(feature = "mistral")]
pub use vectorizers::{MistralAITextVectorizer, MistralConfig};
#[cfg(feature = "voyageai")]
pub use vectorizers::{VoyageAIConfig, VoyageAITextVectorizer};
