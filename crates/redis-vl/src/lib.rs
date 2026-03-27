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
    AggregateHybridQuery, CountQuery, FilterQuery, HybridCombinationMethod, HybridQuery,
    MultiVectorQuery, PageableQuery, QueryKind, QueryLimit, QueryParam, QueryParamValue,
    QueryRender, QueryString, SortBy, SortDirection, TextQuery, Vector, VectorQuery,
    VectorRangeQuery,
};
pub use schema::{
    Field, FieldKind, GeoFieldAttributes, IndexDefinition, IndexSchema, NumericFieldAttributes,
    StorageType, TagFieldAttributes, TextFieldAttributes, TimestampFieldAttributes,
    VectorAlgorithm, VectorDataType, VectorDistanceMetric, VectorFieldAttributes,
};
pub use vectorizers::{
    AsyncVectorizer, CustomTextVectorizer, EmbeddingRequest, LiteLLMTextVectorizer,
    OpenAICompatibleConfig, OpenAITextVectorizer, Vectorizer,
};
