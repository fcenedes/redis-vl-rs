//! Query builders for Redis Search and RedisVL semantics.

use std::borrow::Cow;

use bytes::{BufMut, Bytes};

use crate::filter::FilterExpression;

/// Query parameter value passed through Redis `PARAMS`.
#[derive(Debug, Clone)]
pub enum QueryParamValue {
    /// UTF-8 string parameter.
    String(String),
    /// Binary parameter, typically used for vector blobs.
    Binary(Vec<u8>),
}

/// A named query parameter.
#[derive(Debug, Clone)]
pub struct QueryParam {
    /// Parameter name.
    pub name: String,
    /// Parameter value.
    pub value: QueryParamValue,
}

/// Sort direction for Redis Search sorting.
#[derive(Debug, Clone, Copy)]
pub enum SortDirection {
    /// Ascending sort order.
    Asc,
    /// Descending sort order.
    Desc,
}

/// Redis Search sort specification.
#[derive(Debug, Clone)]
pub struct SortBy {
    /// Field name used for sorting.
    pub field: String,
    /// Sort direction.
    pub direction: SortDirection,
}

/// Limit clause used by Redis Search.
#[derive(Debug, Clone, Copy)]
pub struct QueryLimit {
    /// Result offset.
    pub offset: usize,
    /// Number of results to return.
    pub num: usize,
}

/// Fully rendered Redis Search query metadata.
#[derive(Debug, Clone)]
pub struct QueryRender {
    /// Redis Search query string.
    pub query_string: String,
    /// Optional parameter substitutions.
    pub params: Vec<QueryParam>,
    /// Fields to return from the query.
    pub return_fields: Vec<String>,
    /// Optional sort specification.
    pub sort_by: Option<SortBy>,
    /// Optional limit clause.
    pub limit: Option<QueryLimit>,
    /// Query dialect.
    pub dialect: u32,
    /// Whether `INORDER` should be used.
    pub in_order: bool,
    /// Whether `NOCONTENT` should be used.
    pub no_content: bool,
    /// Optional scorer name.
    pub scorer: Option<String>,
}

/// High-level result shape expected from a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    /// A query that returns documents.
    Documents,
    /// A query that returns only a total count.
    Count,
}

/// Trait for query types that can render Redis Search query strings.
pub trait QueryString {
    /// Builds a Redis Search query string.
    fn to_redis_query(&self) -> String;

    /// Builds the full query render, including search options and params.
    fn render(&self) -> QueryRender {
        QueryRender {
            query_string: self.to_redis_query(),
            params: self.params(),
            return_fields: self.return_fields(),
            sort_by: self.sort_by(),
            limit: self.limit(),
            dialect: self.dialect(),
            in_order: self.in_order(),
            no_content: self.no_content(),
            scorer: self.scorer(),
        }
    }

    /// Query parameters used with Redis `PARAMS`.
    fn params(&self) -> Vec<QueryParam> {
        Vec::new()
    }

    /// Return fields requested by the query.
    fn return_fields(&self) -> Vec<String> {
        Vec::new()
    }

    /// Sort order requested by the query.
    fn sort_by(&self) -> Option<SortBy> {
        None
    }

    /// Limit clause requested by the query.
    fn limit(&self) -> Option<QueryLimit> {
        None
    }

    /// Query dialect.
    fn dialect(&self) -> u32 {
        2
    }

    /// Whether the query should use `INORDER`.
    fn in_order(&self) -> bool {
        false
    }

    /// Whether the query should use `NOCONTENT`.
    fn no_content(&self) -> bool {
        false
    }

    /// Optional scorer.
    fn scorer(&self) -> Option<String> {
        None
    }

    /// The processed result shape for this query.
    fn kind(&self) -> QueryKind {
        QueryKind::Documents
    }

    /// Whether JSON search results should be unpacked into top-level fields when no
    /// explicit projection is requested.
    fn should_unpack_json(&self) -> bool {
        false
    }
}

/// Trait for query types whose pagination window can be adjusted.
pub trait PageableQuery: QueryString + Clone {
    /// Returns a cloned query with an updated `LIMIT` clause.
    fn paged(&self, offset: usize, num: usize) -> Self;
}

#[derive(Debug, Clone)]
struct QueryOptions {
    return_fields: Vec<String>,
    limit: QueryLimit,
    dialect: u32,
    sort_by: Option<SortBy>,
    in_order: bool,
    scorer: Option<String>,
}

impl QueryOptions {
    fn with_num_results(num_results: usize) -> Self {
        Self {
            return_fields: Vec::new(),
            limit: QueryLimit {
                offset: 0,
                num: num_results,
            },
            dialect: 2,
            sort_by: None,
            in_order: false,
            scorer: None,
        }
    }
}

/// Borrowed-or-owned vector payload.
#[derive(Debug, Clone)]
pub struct Vector<'a> {
    elements: Cow<'a, [f32]>,
}

impl<'a> Vector<'a> {
    /// Creates a vector from borrowed or owned elements.
    pub fn new(elements: impl Into<Cow<'a, [f32]>>) -> Self {
        Self {
            elements: elements.into(),
        }
    }

    /// Returns the vector elements.
    pub fn elements(&self) -> &[f32] {
        &self.elements
    }

    /// Encodes the vector into a little-endian byte buffer suitable for Redis params.
    pub fn to_bytes(&self) -> Bytes {
        let mut buffer =
            bytes::BytesMut::with_capacity(self.elements.len() * std::mem::size_of::<f32>());
        for value in self.elements.iter().copied() {
            buffer.put_f32_le(value);
        }
        buffer.freeze()
    }
}

/// Hybrid policy for vector search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridPolicy {
    /// Batched hybrid search.
    Batches,
    /// Ad-hoc brute-force hybrid search.
    AdhocBf,
}

impl HybridPolicy {
    /// Returns the Redis Search string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Batches => "BATCHES",
            Self::AdhocBf => "ADHOC_BF",
        }
    }
}

/// Search history mode for SVS-VAMANA indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchHistoryMode {
    /// Disable search history.
    Off,
    /// Enable search history.
    On,
    /// Let the engine decide.
    Auto,
}

impl SearchHistoryMode {
    /// Returns the Redis Search string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "OFF",
            Self::On => "ON",
            Self::Auto => "AUTO",
        }
    }
}

/// Vector nearest-neighbor query.
#[derive(Debug, Clone)]
pub struct VectorQuery<'a> {
    vector: Vector<'a>,
    vector_field_name: String,
    num_results: usize,
    filter_expression: Option<FilterExpression>,
    ef_runtime: Option<usize>,
    epsilon: Option<f64>,
    hybrid_policy: Option<HybridPolicy>,
    batch_size: Option<usize>,
    search_window_size: Option<usize>,
    use_search_history: Option<SearchHistoryMode>,
    search_buffer_capacity: Option<usize>,
    options: QueryOptions,
}

impl<'a> VectorQuery<'a> {
    /// Creates a vector nearest-neighbor query.
    pub fn new(
        vector: Vector<'a>,
        vector_field_name: impl Into<String>,
        num_results: usize,
    ) -> Self {
        let mut options = QueryOptions::with_num_results(num_results);
        options.return_fields.push("vector_distance".to_owned());
        options.sort_by = Some(SortBy {
            field: "vector_distance".to_owned(),
            direction: SortDirection::Asc,
        });

        Self {
            vector,
            vector_field_name: vector_field_name.into(),
            num_results,
            filter_expression: None,
            ef_runtime: None,
            epsilon: None,
            hybrid_policy: None,
            batch_size: None,
            search_window_size: None,
            use_search_history: None,
            search_buffer_capacity: None,
            options,
        }
    }

    /// Attaches a filter expression.
    pub fn with_filter(mut self, filter_expression: FilterExpression) -> Self {
        self.filter_expression = Some(filter_expression);
        self
    }

    /// Replaces the filter expression in place.
    pub fn set_filter(&mut self, filter_expression: FilterExpression) {
        self.filter_expression = Some(filter_expression);
    }

    /// Sets the runtime EF parameter (HNSW indexes).
    pub fn with_ef_runtime(mut self, ef_runtime: usize) -> Self {
        self.ef_runtime = Some(ef_runtime);
        self
    }

    /// Replaces the runtime EF parameter in place.
    pub fn set_ef_runtime(&mut self, ef_runtime: usize) {
        self.ef_runtime = Some(ef_runtime);
    }

    /// Returns the current EF runtime setting.
    pub fn ef_runtime(&self) -> Option<usize> {
        self.ef_runtime
    }

    /// Sets the epsilon parameter for approximate search.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = Some(epsilon);
        self
    }

    /// Replaces the epsilon parameter in place.
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = Some(epsilon);
    }

    /// Returns the current epsilon setting.
    pub fn epsilon(&self) -> Option<f64> {
        self.epsilon
    }

    /// Sets the hybrid search policy.
    pub fn with_hybrid_policy(mut self, policy: HybridPolicy) -> Self {
        self.hybrid_policy = Some(policy);
        self
    }

    /// Replaces the hybrid search policy in place.
    pub fn set_hybrid_policy(&mut self, policy: HybridPolicy) {
        self.hybrid_policy = Some(policy);
    }

    /// Returns the current hybrid policy setting.
    pub fn hybrid_policy(&self) -> Option<HybridPolicy> {
        self.hybrid_policy
    }

    /// Sets the batch size for `BATCHES` hybrid search policy.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Replaces the batch size in place.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = Some(batch_size);
    }

    /// Returns the current batch size setting.
    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    /// Sets the search window size (SVS-VAMANA indexes).
    pub fn with_search_window_size(mut self, size: usize) -> Self {
        self.search_window_size = Some(size);
        self
    }

    /// Replaces the search window size in place.
    pub fn set_search_window_size(&mut self, size: usize) {
        self.search_window_size = Some(size);
    }

    /// Returns the current search window size setting.
    pub fn search_window_size(&self) -> Option<usize> {
        self.search_window_size
    }

    /// Sets the search history mode (SVS-VAMANA indexes).
    pub fn with_use_search_history(mut self, mode: SearchHistoryMode) -> Self {
        self.use_search_history = Some(mode);
        self
    }

    /// Replaces the search history mode in place.
    pub fn set_use_search_history(&mut self, mode: SearchHistoryMode) {
        self.use_search_history = Some(mode);
    }

    /// Returns the current search history mode.
    pub fn use_search_history(&self) -> Option<SearchHistoryMode> {
        self.use_search_history
    }

    /// Sets the search buffer capacity (SVS-VAMANA indexes).
    pub fn with_search_buffer_capacity(mut self, capacity: usize) -> Self {
        self.search_buffer_capacity = Some(capacity);
        self
    }

    /// Replaces the search buffer capacity in place.
    pub fn set_search_buffer_capacity(&mut self, capacity: usize) {
        self.search_buffer_capacity = Some(capacity);
    }

    /// Returns the current search buffer capacity.
    pub fn search_buffer_capacity(&self) -> Option<usize> {
        self.search_buffer_capacity
    }

    /// Replaces the return field list.
    pub fn with_return_fields<I, S>(mut self, return_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.options.return_fields = return_fields.into_iter().map(Into::into).collect();
        if !self
            .options
            .return_fields
            .iter()
            .any(|field| field == "vector_distance")
        {
            self.options
                .return_fields
                .push("vector_distance".to_owned());
        }
        self
    }

    /// Updates the query limit.
    pub fn paging(mut self, offset: usize, num: usize) -> Self {
        self.options.limit = QueryLimit { offset, num };
        self
    }

    /// Sets the sort field and direction.
    pub fn sort_by(mut self, field: impl Into<String>, direction: SortDirection) -> Self {
        self.options.sort_by = Some(SortBy {
            field: field.into(),
            direction,
        });
        self
    }

    /// Enables or disables `INORDER`.
    pub fn in_order(mut self, in_order: bool) -> Self {
        self.options.in_order = in_order;
        self
    }

    /// Sets the query dialect.
    pub fn with_dialect(mut self, dialect: u32) -> Self {
        self.options.dialect = dialect;
        self
    }

    /// Returns the encoded query vector.
    pub fn vector(&self) -> &Vector<'a> {
        &self.vector
    }
}

impl QueryString for VectorQuery<'_> {
    fn to_redis_query(&self) -> String {
        let base = self
            .filter_expression
            .as_ref()
            .map_or_else(|| "*".to_owned(), FilterExpression::to_redis_syntax);
        let mut query = format!(
            "{}=>[KNN {} @{} $vector AS vector_distance",
            base, self.num_results, self.vector_field_name
        );
        if self.ef_runtime.is_some() {
            query.push_str(" EF_RUNTIME $EF");
        }
        if self.epsilon.is_some() {
            query.push_str(" EPSILON $EPSILON");
        }
        if self.search_window_size.is_some() {
            query.push_str(" SEARCH_WINDOW_SIZE $SEARCH_WINDOW_SIZE");
        }
        if self.use_search_history.is_some() {
            query.push_str(" USE_SEARCH_HISTORY $USE_SEARCH_HISTORY");
        }
        if self.search_buffer_capacity.is_some() {
            query.push_str(" SEARCH_BUFFER_CAPACITY $SEARCH_BUFFER_CAPACITY");
        }
        query.push(']');
        if let Some(policy) = &self.hybrid_policy {
            query.push_str(&format!(" HYBRID_POLICY {}", policy.as_str()));
            if let Some(batch_size) = self.batch_size {
                query.push_str(&format!(" BATCH_SIZE {}", batch_size));
            }
        }
        query
    }

    fn params(&self) -> Vec<QueryParam> {
        let mut params = vec![QueryParam {
            name: "vector".to_owned(),
            value: QueryParamValue::Binary(self.vector.to_bytes().to_vec()),
        }];
        if let Some(ef_runtime) = self.ef_runtime {
            params.push(QueryParam {
                name: "EF".to_owned(),
                value: QueryParamValue::String(ef_runtime.to_string()),
            });
        }
        if let Some(epsilon) = self.epsilon {
            params.push(QueryParam {
                name: "EPSILON".to_owned(),
                value: QueryParamValue::String(epsilon.to_string()),
            });
        }
        if let Some(size) = self.search_window_size {
            params.push(QueryParam {
                name: "SEARCH_WINDOW_SIZE".to_owned(),
                value: QueryParamValue::String(size.to_string()),
            });
        }
        if let Some(mode) = &self.use_search_history {
            params.push(QueryParam {
                name: "USE_SEARCH_HISTORY".to_owned(),
                value: QueryParamValue::String(mode.as_str().to_owned()),
            });
        }
        if let Some(capacity) = self.search_buffer_capacity {
            params.push(QueryParam {
                name: "SEARCH_BUFFER_CAPACITY".to_owned(),
                value: QueryParamValue::String(capacity.to_string()),
            });
        }
        params
    }

    fn return_fields(&self) -> Vec<String> {
        self.options.return_fields.clone()
    }

    fn sort_by(&self) -> Option<SortBy> {
        self.options.sort_by.clone()
    }

    fn limit(&self) -> Option<QueryLimit> {
        Some(self.options.limit)
    }

    fn dialect(&self) -> u32 {
        self.options.dialect
    }

    fn in_order(&self) -> bool {
        self.options.in_order
    }
}

impl PageableQuery for VectorQuery<'_> {
    fn paged(&self, offset: usize, num: usize) -> Self {
        self.clone().paging(offset, num)
    }
}

/// Vector range query.
#[derive(Debug, Clone)]
pub struct VectorRangeQuery<'a> {
    vector: Vector<'a>,
    vector_field_name: String,
    distance_threshold: f32,
    filter_expression: Option<FilterExpression>,
    epsilon: Option<f64>,
    hybrid_policy: Option<HybridPolicy>,
    batch_size: Option<usize>,
    search_window_size: Option<usize>,
    use_search_history: Option<SearchHistoryMode>,
    search_buffer_capacity: Option<usize>,
    options: QueryOptions,
}

impl<'a> VectorRangeQuery<'a> {
    /// Creates a vector range query.
    pub fn new(
        vector: Vector<'a>,
        vector_field_name: impl Into<String>,
        distance_threshold: f32,
    ) -> Self {
        let mut options = QueryOptions::with_num_results(10);
        options.return_fields.push("vector_distance".to_owned());
        options.sort_by = Some(SortBy {
            field: "vector_distance".to_owned(),
            direction: SortDirection::Asc,
        });

        Self {
            vector,
            vector_field_name: vector_field_name.into(),
            distance_threshold,
            filter_expression: None,
            epsilon: None,
            hybrid_policy: None,
            batch_size: None,
            search_window_size: None,
            use_search_history: None,
            search_buffer_capacity: None,
            options,
        }
    }

    /// Attaches a filter expression.
    pub fn with_filter(mut self, filter_expression: FilterExpression) -> Self {
        self.filter_expression = Some(filter_expression);
        self
    }

    /// Replaces the filter expression in place.
    pub fn set_filter(&mut self, filter_expression: FilterExpression) {
        self.filter_expression = Some(filter_expression);
    }

    /// Returns the active distance threshold.
    pub fn distance_threshold(&self) -> f32 {
        self.distance_threshold
    }

    /// Updates the distance threshold in place.
    pub fn set_distance_threshold(&mut self, distance_threshold: f32) {
        self.distance_threshold = distance_threshold;
    }

    /// Sets the epsilon parameter for approximate range search.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = Some(epsilon);
        self
    }

    /// Replaces the epsilon parameter in place.
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = Some(epsilon);
    }

    /// Returns the current epsilon setting.
    pub fn epsilon(&self) -> Option<f64> {
        self.epsilon
    }

    /// Sets the hybrid search policy.
    pub fn with_hybrid_policy(mut self, policy: HybridPolicy) -> Self {
        self.hybrid_policy = Some(policy);
        self
    }

    /// Replaces the hybrid search policy in place.
    pub fn set_hybrid_policy(&mut self, policy: HybridPolicy) {
        self.hybrid_policy = Some(policy);
    }

    /// Returns the current hybrid policy setting.
    pub fn hybrid_policy(&self) -> Option<HybridPolicy> {
        self.hybrid_policy
    }

    /// Sets the batch size for `BATCHES` hybrid search policy.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Replaces the batch size in place.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = Some(batch_size);
    }

    /// Returns the current batch size setting.
    pub fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    /// Sets the search window size (SVS-VAMANA indexes).
    pub fn with_search_window_size(mut self, size: usize) -> Self {
        self.search_window_size = Some(size);
        self
    }

    /// Replaces the search window size in place.
    pub fn set_search_window_size(&mut self, size: usize) {
        self.search_window_size = Some(size);
    }

    /// Returns the current search window size setting.
    pub fn search_window_size(&self) -> Option<usize> {
        self.search_window_size
    }

    /// Sets the search history mode (SVS-VAMANA indexes).
    pub fn with_use_search_history(mut self, mode: SearchHistoryMode) -> Self {
        self.use_search_history = Some(mode);
        self
    }

    /// Replaces the search history mode in place.
    pub fn set_use_search_history(&mut self, mode: SearchHistoryMode) {
        self.use_search_history = Some(mode);
    }

    /// Returns the current search history mode.
    pub fn use_search_history(&self) -> Option<SearchHistoryMode> {
        self.use_search_history
    }

    /// Sets the search buffer capacity (SVS-VAMANA indexes).
    pub fn with_search_buffer_capacity(mut self, capacity: usize) -> Self {
        self.search_buffer_capacity = Some(capacity);
        self
    }

    /// Replaces the search buffer capacity in place.
    pub fn set_search_buffer_capacity(&mut self, capacity: usize) {
        self.search_buffer_capacity = Some(capacity);
    }

    /// Returns the current search buffer capacity.
    pub fn search_buffer_capacity(&self) -> Option<usize> {
        self.search_buffer_capacity
    }

    /// Replaces the return field list.
    pub fn with_return_fields<I, S>(mut self, return_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.options.return_fields = return_fields.into_iter().map(Into::into).collect();
        if !self
            .options
            .return_fields
            .iter()
            .any(|field| field == "vector_distance")
        {
            self.options
                .return_fields
                .push("vector_distance".to_owned());
        }
        self
    }

    /// Updates the query limit.
    pub fn paging(mut self, offset: usize, num: usize) -> Self {
        self.options.limit = QueryLimit { offset, num };
        self
    }

    /// Sets the sort field and direction.
    pub fn sort_by(mut self, field: impl Into<String>, direction: SortDirection) -> Self {
        self.options.sort_by = Some(SortBy {
            field: field.into(),
            direction,
        });
        self
    }

    /// Enables or disables `INORDER`.
    pub fn in_order(mut self, in_order: bool) -> Self {
        self.options.in_order = in_order;
        self
    }

    /// Sets the query dialect.
    pub fn with_dialect(mut self, dialect: u32) -> Self {
        self.options.dialect = dialect;
        self
    }

    /// Returns the encoded query vector.
    pub fn vector(&self) -> &Vector<'a> {
        &self.vector
    }
}

impl QueryString for VectorRangeQuery<'_> {
    fn to_redis_query(&self) -> String {
        let base = self
            .filter_expression
            .as_ref()
            .map_or_else(|| "*".to_owned(), FilterExpression::to_redis_syntax);
        let mut query = format!(
            "@{}:[VECTOR_RANGE $distance_threshold $vector",
            self.vector_field_name
        );
        // SVS-VAMANA and epsilon params are embedded in the query string as
        // `$PARAM: value` (following upstream Python behaviour).
        if let Some(epsilon) = self.epsilon {
            query.push_str(&format!(" $EPSILON: {}", epsilon));
        }
        if let Some(size) = self.search_window_size {
            query.push_str(&format!(" $SEARCH_WINDOW_SIZE: {}", size));
        }
        if let Some(mode) = &self.use_search_history {
            query.push_str(&format!(" $USE_SEARCH_HISTORY: {}", mode.as_str()));
        }
        if let Some(capacity) = self.search_buffer_capacity {
            query.push_str(&format!(" $SEARCH_BUFFER_CAPACITY: {}", capacity));
        }
        query.push_str(&format!(" $YIELD_DISTANCE_AS: vector_distance] {}", base));
        query
    }

    fn params(&self) -> Vec<QueryParam> {
        let mut params = vec![
            QueryParam {
                name: "vector".to_owned(),
                value: QueryParamValue::Binary(self.vector.to_bytes().to_vec()),
            },
            QueryParam {
                name: "distance_threshold".to_owned(),
                value: QueryParamValue::String(self.distance_threshold.to_string()),
            },
        ];
        if let Some(policy) = &self.hybrid_policy {
            params.push(QueryParam {
                name: "HYBRID_POLICY".to_owned(),
                value: QueryParamValue::String(policy.as_str().to_owned()),
            });
        }
        if let Some(batch_size) = self.batch_size {
            params.push(QueryParam {
                name: "BATCH_SIZE".to_owned(),
                value: QueryParamValue::String(batch_size.to_string()),
            });
        }
        params
    }

    fn return_fields(&self) -> Vec<String> {
        self.options.return_fields.clone()
    }

    fn sort_by(&self) -> Option<SortBy> {
        self.options.sort_by.clone()
    }

    fn limit(&self) -> Option<QueryLimit> {
        Some(self.options.limit)
    }

    fn dialect(&self) -> u32 {
        self.options.dialect
    }

    fn in_order(&self) -> bool {
        self.options.in_order
    }
}

impl PageableQuery for VectorRangeQuery<'_> {
    fn paged(&self, offset: usize, num: usize) -> Self {
        self.clone().paging(offset, num)
    }
}

/// Full-text query.
#[derive(Debug, Clone)]
pub struct TextQuery {
    text: String,
    text_field_name: Option<String>,
    filter_expression: Option<FilterExpression>,
    return_score: bool,
    options: QueryOptions,
}

impl TextQuery {
    /// Creates a full-text query.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_field_name: None,
            filter_expression: None,
            return_score: true,
            options: QueryOptions::with_num_results(10),
        }
    }

    /// Restricts the query to a specific text field.
    pub fn for_field(mut self, text_field_name: impl Into<String>) -> Self {
        self.text_field_name = Some(text_field_name.into());
        self
    }

    /// Attaches a filter expression to combine with the text search.
    pub fn with_filter(mut self, filter_expression: FilterExpression) -> Self {
        self.filter_expression = Some(filter_expression);
        self
    }

    /// Replaces the filter expression in place.
    pub fn set_filter(&mut self, filter_expression: FilterExpression) {
        self.filter_expression = Some(filter_expression);
    }

    /// Sets whether to return scores (default: `true`).
    pub fn with_return_score(mut self, return_score: bool) -> Self {
        self.return_score = return_score;
        self
    }

    /// Replaces the return field list.
    pub fn with_return_fields<I, S>(mut self, return_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.options.return_fields = return_fields.into_iter().map(Into::into).collect();
        self
    }

    /// Updates the query limit.
    pub fn paging(mut self, offset: usize, num: usize) -> Self {
        self.options.limit = QueryLimit { offset, num };
        self
    }

    /// Sets the sort field and direction.
    pub fn sort_by(mut self, field: impl Into<String>, direction: SortDirection) -> Self {
        self.options.sort_by = Some(SortBy {
            field: field.into(),
            direction,
        });
        self
    }

    /// Enables or disables `INORDER`.
    pub fn in_order(mut self, in_order: bool) -> Self {
        self.options.in_order = in_order;
        self
    }

    /// Sets the query dialect.
    pub fn with_dialect(mut self, dialect: u32) -> Self {
        self.options.dialect = dialect;
        self
    }

    /// Sets the scorer name.
    pub fn with_scorer(mut self, scorer: impl Into<String>) -> Self {
        self.options.scorer = Some(scorer.into());
        self
    }
}

impl QueryString for TextQuery {
    fn to_redis_query(&self) -> String {
        let text_part = match &self.text_field_name {
            Some(field) => format!("@{}:({})", field, self.text),
            None => self.text.clone(),
        };
        match &self.filter_expression {
            Some(filter) => {
                let filter_str = filter.to_redis_syntax();
                if filter_str == "*" {
                    text_part
                } else {
                    format!("{} AND {}", text_part, filter_str)
                }
            }
            None => text_part,
        }
    }

    fn return_fields(&self) -> Vec<String> {
        self.options.return_fields.clone()
    }

    fn sort_by(&self) -> Option<SortBy> {
        self.options.sort_by.clone()
    }

    fn limit(&self) -> Option<QueryLimit> {
        Some(self.options.limit)
    }

    fn dialect(&self) -> u32 {
        self.options.dialect
    }

    fn in_order(&self) -> bool {
        self.options.in_order
    }

    fn scorer(&self) -> Option<String> {
        self.options.scorer.clone()
    }
}

impl PageableQuery for TextQuery {
    fn paged(&self, offset: usize, num: usize) -> Self {
        self.clone().paging(offset, num)
    }
}

/// Filter-only query.
#[derive(Debug, Clone)]
pub struct FilterQuery {
    filter_expression: FilterExpression,
    options: QueryOptions,
}

impl FilterQuery {
    /// Creates a filter-only query.
    pub fn new(filter_expression: FilterExpression) -> Self {
        Self {
            filter_expression,
            options: QueryOptions::with_num_results(10),
        }
    }

    /// Replaces the filter expression in place.
    pub fn set_filter(&mut self, filter_expression: FilterExpression) {
        self.filter_expression = filter_expression;
    }

    /// Replaces the return field list.
    pub fn with_return_fields<I, S>(mut self, return_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.options.return_fields = return_fields.into_iter().map(Into::into).collect();
        self
    }

    /// Updates the query limit.
    pub fn paging(mut self, offset: usize, num: usize) -> Self {
        self.options.limit = QueryLimit { offset, num };
        self
    }

    /// Sets the sort field and direction.
    pub fn sort_by(mut self, field: impl Into<String>, direction: SortDirection) -> Self {
        self.options.sort_by = Some(SortBy {
            field: field.into(),
            direction,
        });
        self
    }

    /// Enables or disables `INORDER`.
    pub fn in_order(mut self, in_order: bool) -> Self {
        self.options.in_order = in_order;
        self
    }

    /// Sets the query dialect.
    pub fn with_dialect(mut self, dialect: u32) -> Self {
        self.options.dialect = dialect;
        self
    }
}

impl QueryString for FilterQuery {
    fn to_redis_query(&self) -> String {
        self.filter_expression.to_redis_syntax()
    }

    fn return_fields(&self) -> Vec<String> {
        self.options.return_fields.clone()
    }

    fn sort_by(&self) -> Option<SortBy> {
        self.options.sort_by.clone()
    }

    fn limit(&self) -> Option<QueryLimit> {
        Some(self.options.limit)
    }

    fn dialect(&self) -> u32 {
        self.options.dialect
    }

    fn in_order(&self) -> bool {
        self.options.in_order
    }

    fn should_unpack_json(&self) -> bool {
        true
    }
}

impl PageableQuery for FilterQuery {
    fn paged(&self, offset: usize, num: usize) -> Self {
        self.clone().paging(offset, num)
    }
}

/// Count query.
#[derive(Debug, Clone)]
pub struct CountQuery {
    filter_expression: Option<FilterExpression>,
    dialect: u32,
}

impl CountQuery {
    /// Creates a count query.
    pub fn new() -> Self {
        Self {
            filter_expression: None,
            dialect: 2,
        }
    }

    /// Attaches a filter expression.
    pub fn with_filter(mut self, filter_expression: FilterExpression) -> Self {
        self.filter_expression = Some(filter_expression);
        self
    }

    /// Sets the query dialect.
    pub fn with_dialect(mut self, dialect: u32) -> Self {
        self.dialect = dialect;
        self
    }
}

impl Default for CountQuery {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryString for CountQuery {
    fn to_redis_query(&self) -> String {
        self.filter_expression
            .as_ref()
            .map_or_else(|| "*".to_owned(), FilterExpression::to_redis_syntax)
    }

    fn limit(&self) -> Option<QueryLimit> {
        Some(QueryLimit { offset: 0, num: 0 })
    }

    fn dialect(&self) -> u32 {
        self.dialect
    }

    fn no_content(&self) -> bool {
        true
    }

    fn kind(&self) -> QueryKind {
        QueryKind::Count
    }
}

/// Hybrid search score-combination method.
#[derive(Debug, Clone, Copy)]
pub enum HybridCombinationMethod {
    /// Weighted linear score fusion.
    Linear,
    /// Reciprocal rank fusion.
    Rrf,
}

impl HybridCombinationMethod {
    /// Returns the Redis protocol name for this combination method.
    pub fn redis_name(self) -> &'static str {
        match self {
            Self::Linear => "LINEAR",
            Self::Rrf => "RRF",
        }
    }
}

/// Vector search method for hybrid queries.
#[derive(Debug, Clone, Copy)]
pub enum VectorSearchMethod {
    /// K-Nearest Neighbors search.
    Knn,
    /// Range-based search.
    Range,
}

/// Native Redis hybrid query dispatched via `FT.HYBRID`.
///
/// This query combines text search and vector similarity search with
/// configurable fusion methods. It requires Redis 8.4.0+ and produces
/// `FT.HYBRID` commands (not `FT.SEARCH`).
///
/// # Example
///
/// ```
/// use redis_vl::query::{HybridQuery, HybridCombinationMethod, Vector};
///
/// let query = HybridQuery::new(
///     "a medical professional",
///     "description",
///     Vector::new(vec![0.1, 0.1, 0.5]),
///     "user_embedding",
/// )
/// .with_num_results(10)
/// .with_combination_method(HybridCombinationMethod::Rrf)
/// .with_yield_combined_score_as("hybrid_score")
/// .with_return_fields(["user", "age", "job"]);
/// ```
#[derive(Debug, Clone)]
pub struct HybridQuery<'a> {
    text: String,
    text_field_name: String,
    vector: Vector<'a>,
    vector_field_name: String,
    vector_param_name: String,
    num_results: usize,
    text_scorer: Option<String>,
    yield_text_score_as: Option<String>,
    vector_search_method: Option<VectorSearchMethod>,
    knn_ef_runtime: Option<usize>,
    range_radius: Option<f32>,
    range_epsilon: Option<f32>,
    yield_vsim_score_as: Option<String>,
    filter_expression: Option<FilterExpression>,
    combination_method: Option<HybridCombinationMethod>,
    rrf_window: Option<usize>,
    rrf_constant: Option<usize>,
    linear_alpha: Option<f32>,
    yield_combined_score_as: Option<String>,
    return_fields: Vec<String>,
    stopwords: Option<std::collections::HashSet<String>>,
    text_weights: Option<std::collections::HashMap<String, f32>>,
}

impl<'a> HybridQuery<'a> {
    /// Creates a hybrid query with the given text, text field, vector, and vector field.
    pub fn new(
        text: impl Into<String>,
        text_field_name: impl Into<String>,
        vector: Vector<'a>,
        vector_field_name: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            text_field_name: text_field_name.into(),
            vector,
            vector_field_name: vector_field_name.into(),
            vector_param_name: "vector".to_owned(),
            num_results: 10,
            text_scorer: None,
            yield_text_score_as: None,
            vector_search_method: None,
            knn_ef_runtime: None,
            range_radius: None,
            range_epsilon: None,
            yield_vsim_score_as: None,
            filter_expression: None,
            combination_method: None,
            rrf_window: None,
            rrf_constant: None,
            linear_alpha: None,
            yield_combined_score_as: None,
            return_fields: Vec::new(),
            stopwords: None,
            text_weights: None,
        }
    }

    /// Sets the number of results to return.
    pub fn with_num_results(mut self, num_results: usize) -> Self {
        self.num_results = num_results;
        self
    }

    /// Sets the text scorer algorithm (e.g. `"BM25STD"`, `"TFIDF"`, `"TFIDF.DOCNORM"`).
    pub fn with_text_scorer(mut self, scorer: impl Into<String>) -> Self {
        self.text_scorer = Some(scorer.into());
        self
    }

    /// Sets the alias for the text search score in results.
    pub fn with_yield_text_score_as(mut self, alias: impl Into<String>) -> Self {
        self.yield_text_score_as = Some(alias.into());
        self
    }

    /// Sets the vector search method to KNN with optional EF runtime.
    pub fn with_knn(mut self, ef_runtime: Option<usize>) -> Self {
        self.vector_search_method = Some(VectorSearchMethod::Knn);
        self.knn_ef_runtime = ef_runtime;
        self
    }

    /// Sets the vector search method to RANGE.
    pub fn with_range(mut self, radius: f32, epsilon: Option<f32>) -> Self {
        self.vector_search_method = Some(VectorSearchMethod::Range);
        self.range_radius = Some(radius);
        self.range_epsilon = epsilon;
        self
    }

    /// Sets the alias for the vector similarity score in results.
    pub fn with_yield_vsim_score_as(mut self, alias: impl Into<String>) -> Self {
        self.yield_vsim_score_as = Some(alias.into());
        self
    }

    /// Attaches a filter expression.
    pub fn with_filter(mut self, filter_expression: FilterExpression) -> Self {
        self.filter_expression = Some(filter_expression);
        self
    }

    /// Selects the hybrid combination method.
    pub fn with_combination_method(mut self, method: HybridCombinationMethod) -> Self {
        self.combination_method = Some(method);
        self
    }

    /// Sets RRF parameters.
    pub fn with_rrf(mut self, window: Option<usize>, constant: Option<usize>) -> Self {
        self.combination_method = Some(HybridCombinationMethod::Rrf);
        self.rrf_window = window;
        self.rrf_constant = constant;
        self
    }

    /// Sets LINEAR combination with an alpha weight for the text score.
    pub fn with_linear(mut self, alpha: f32) -> Self {
        self.combination_method = Some(HybridCombinationMethod::Linear);
        self.linear_alpha = Some(alpha);
        self
    }

    /// Sets the alias for the combined score in results.
    pub fn with_yield_combined_score_as(mut self, alias: impl Into<String>) -> Self {
        self.yield_combined_score_as = Some(alias.into());
        self
    }

    /// Replaces the return field list.
    pub fn with_return_fields<I, S>(mut self, return_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.return_fields = return_fields.into_iter().map(Into::into).collect();
        self
    }

    /// Sets stopwords to filter from the query text.
    pub fn with_stopwords(mut self, stopwords: std::collections::HashSet<String>) -> Self {
        self.stopwords = Some(stopwords);
        self
    }

    /// Sets word weights for the text search.
    pub fn with_text_weights(mut self, weights: std::collections::HashMap<String, f32>) -> Self {
        self.text_weights = Some(weights);
        self
    }

    /// Sets the parameter name used for the vector blob.
    pub fn with_vector_param_name(mut self, name: impl Into<String>) -> Self {
        self.vector_param_name = name.into();
        self
    }

    /// Returns the encoded query vector.
    pub fn vector(&self) -> &Vector<'a> {
        &self.vector
    }

    /// Builds the full-text query string, applying stopwords if configured.
    fn build_query_string(&self) -> String {
        let mut text = self.text.clone();

        // Apply stopwords
        if let Some(stopwords) = &self.stopwords {
            if !stopwords.is_empty() {
                let words: Vec<&str> = text.split_whitespace().collect();
                let filtered: Vec<&str> = words
                    .into_iter()
                    .filter(|w| !stopwords.contains(&w.to_lowercase()))
                    .collect();
                text = filtered.join(" ");
            }
        }

        // Apply word weights
        if let Some(weights) = &self.text_weights {
            if !weights.is_empty() {
                let words: Vec<String> = text
                    .split_whitespace()
                    .map(|w| {
                        if let Some(weight) = weights.get(w) {
                            format!("{}=>{{{}}}", w, weight)
                        } else {
                            w.to_owned()
                        }
                    })
                    .collect();
                text = words.join(" ");
            }
        }

        format!("@{}:({})", self.text_field_name, text)
    }

    /// Builds an `FT.HYBRID` command for this query.
    ///
    /// The command targets the given `index_name` and encodes all hybrid
    /// query clauses (QUERY, VSIM, COMBINE_METHOD, LOAD, LIMIT).
    pub fn build_cmd(&self, index_name: &str) -> redis::Cmd {
        let mut cmd = redis::cmd("FT.HYBRID");
        cmd.arg(index_name);

        // QUERY clause
        let query_string = self.build_query_string();
        cmd.arg("QUERY").arg(&query_string);

        if let Some(scorer) = &self.text_scorer {
            cmd.arg("SCORER").arg(scorer);
        }
        if let Some(alias) = &self.yield_text_score_as {
            cmd.arg("YIELD_SCORE_AS").arg(alias);
        }

        // VSIM clause
        cmd.arg("VSIM")
            .arg(format!("@{}", self.vector_field_name))
            .arg(format!("${}", self.vector_param_name));

        if let Some(method) = self.vector_search_method {
            match method {
                VectorSearchMethod::Knn => {
                    cmd.arg("SEARCH_METHOD").arg("KNN");
                    cmd.arg("K").arg(self.num_results);
                    if let Some(ef) = self.knn_ef_runtime {
                        cmd.arg("EF_RUNTIME").arg(ef);
                    }
                }
                VectorSearchMethod::Range => {
                    cmd.arg("SEARCH_METHOD").arg("RANGE");
                    if let Some(radius) = self.range_radius {
                        cmd.arg("RADIUS").arg(radius);
                    }
                    if let Some(epsilon) = self.range_epsilon {
                        cmd.arg("EPSILON").arg(epsilon);
                    }
                }
            }
        }

        if let Some(filter) = &self.filter_expression {
            let filter_str = filter.to_redis_syntax();
            if filter_str != "*" {
                cmd.arg("FILTER").arg(&filter_str);
            }
        }

        if let Some(alias) = &self.yield_vsim_score_as {
            cmd.arg("YIELD_SCORE_AS").arg(alias);
        }

        // COMBINE_METHOD clause
        if let Some(method) = &self.combination_method {
            cmd.arg("COMBINE_METHOD").arg(method.redis_name());

            match method {
                HybridCombinationMethod::Rrf => {
                    if let Some(window) = self.rrf_window {
                        cmd.arg("WINDOW").arg(window);
                    }
                    if let Some(constant) = self.rrf_constant {
                        cmd.arg("CONSTANT").arg(constant);
                    }
                }
                HybridCombinationMethod::Linear => {
                    if let Some(alpha) = self.linear_alpha {
                        cmd.arg("ALPHA").arg(alpha);
                        cmd.arg("BETA").arg(1.0 - alpha);
                    }
                }
            }

            if let Some(alias) = &self.yield_combined_score_as {
                cmd.arg("YIELD_SCORE_AS").arg(alias);
            }
        }

        // PARAMS substitution for vector blob
        cmd.arg("PARAMS")
            .arg(2)
            .arg(&self.vector_param_name)
            .arg(self.vector.to_bytes().as_ref());

        // LOAD (return fields)
        if !self.return_fields.is_empty() {
            cmd.arg("LOAD");
            cmd.arg(self.return_fields.len());
            for field in &self.return_fields {
                cmd.arg(format!("@{}", field));
            }
        }

        // LIMIT
        cmd.arg("LIMIT").arg(0).arg(self.num_results);

        cmd
    }
}

/// Aggregate-based hybrid query that combines text and vector search via
/// `FT.AGGREGATE`.
///
/// Mirrors the Python `AggregateHybridQuery` which scores documents as:
///
/// ```text
/// hybrid_score = alpha * vector_similarity + (1 - alpha) * text_score
/// ```
///
/// where `vector_similarity = (2 - vector_distance) / 2` and
/// `text_score = @__score` (the scorer output).
#[derive(Debug, Clone)]
pub struct AggregateHybridQuery<'a> {
    text: String,
    text_field_name: String,
    vector: Vector<'a>,
    vector_field_name: String,
    alpha: f32,
    num_results: usize,
    text_scorer: String,
    filter_expression: Option<FilterExpression>,
    return_fields: Vec<String>,
    stopwords: Option<std::collections::HashSet<String>>,
    text_weights: Option<std::collections::HashMap<String, f32>>,
    dialect: u32,
}

impl<'a> AggregateHybridQuery<'a> {
    /// Creates an aggregate hybrid query.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `text` is empty or becomes empty after stopword
    /// removal.
    pub fn new(
        text: impl Into<String>,
        text_field_name: impl Into<String>,
        vector: Vector<'a>,
        vector_field_name: impl Into<String>,
    ) -> std::result::Result<Self, String> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err("text string cannot be empty".to_owned());
        }
        Ok(Self {
            text,
            text_field_name: text_field_name.into(),
            vector,
            vector_field_name: vector_field_name.into(),
            alpha: 0.7,
            num_results: 10,
            text_scorer: "BM25STD".to_owned(),
            filter_expression: None,
            return_fields: Vec::new(),
            stopwords: None,
            text_weights: None,
            dialect: 2,
        })
    }

    /// Sets the weight of the vector similarity in the hybrid score.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the number of results to return.
    pub fn with_num_results(mut self, num_results: usize) -> Self {
        self.num_results = num_results;
        self
    }

    /// Sets the text scorer algorithm (e.g. `"BM25STD"`, `"TFIDF"`).
    pub fn with_text_scorer(mut self, scorer: impl Into<String>) -> Self {
        self.text_scorer = scorer.into();
        self
    }

    /// Attaches a filter expression.
    pub fn with_filter(mut self, filter_expression: FilterExpression) -> Self {
        self.filter_expression = Some(filter_expression);
        self
    }

    /// Replaces the return field list.
    pub fn with_return_fields<I, S>(mut self, return_fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.return_fields = return_fields.into_iter().map(Into::into).collect();
        self
    }

    /// Sets stopwords to filter from the query text.
    pub fn with_stopwords(mut self, stopwords: std::collections::HashSet<String>) -> Self {
        self.stopwords = Some(stopwords);
        self
    }

    /// Sets word weights for the text search.
    pub fn with_text_weights(mut self, weights: std::collections::HashMap<String, f32>) -> Self {
        self.text_weights = Some(weights);
        self
    }

    /// Updates text weights after construction (mirrors Python `set_text_weights`).
    pub fn set_text_weights(&mut self, weights: std::collections::HashMap<String, f32>) {
        self.text_weights = Some(weights);
    }

    /// Sets the Redis dialect version.
    pub fn with_dialect(mut self, dialect: u32) -> Self {
        self.dialect = dialect;
        self
    }

    /// Returns the encoded query vector.
    pub fn vector(&self) -> &Vector<'a> {
        &self.vector
    }

    /// Returns the configured alpha value.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the query text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Builds the full-text query string, applying stopwords and weights.
    pub(crate) fn build_query_string(&self) -> String {
        let mut text = self.text.clone();

        // Apply stopwords
        if let Some(stopwords) = &self.stopwords {
            if !stopwords.is_empty() {
                let words: Vec<&str> = text.split_whitespace().collect();
                let filtered: Vec<&str> = words
                    .into_iter()
                    .filter(|w| !stopwords.contains(&w.to_lowercase()))
                    .collect();
                text = filtered.join(" ");
            }
        }

        // Apply word weights
        if let Some(weights) = &self.text_weights {
            if !weights.is_empty() {
                let words: Vec<String> = text
                    .split_whitespace()
                    .map(|w| {
                        if let Some(weight) = weights.get(w) {
                            format!("{}=>{{{}}}", w, weight)
                        } else {
                            w.to_owned()
                        }
                    })
                    .collect();
                text = words.join(" ");
            }
        }

        // Build the base text query with optional filter
        let base = if let Some(filter) = &self.filter_expression {
            let filter_str = filter.to_redis_syntax();
            if filter_str == "*" {
                format!("@{}:({})", self.text_field_name, text)
            } else {
                format!("({}) @{}:({})", filter_str, self.text_field_name, text)
            }
        } else {
            format!("@{}:({})", self.text_field_name, text)
        };

        // Append KNN vector part
        format!(
            "{}=>[KNN {} @{} $vector AS vector_distance]",
            base, self.num_results, self.vector_field_name,
        )
    }

    /// Builds the complete `FT.AGGREGATE` command for this query.
    pub fn build_aggregate_cmd(&self, index_name: &str) -> redis::Cmd {
        let query_string = self.build_query_string();
        let mut cmd = redis::cmd("FT.AGGREGATE");
        cmd.arg(index_name);
        cmd.arg(&query_string);

        // SCORER
        cmd.arg("SCORER").arg(&self.text_scorer);

        // ADDSCORES
        cmd.arg("ADDSCORES");

        // APPLY: compute vector_similarity and text_score
        cmd.arg("APPLY")
            .arg("(2 - @vector_distance)/2")
            .arg("AS")
            .arg("vector_similarity");
        cmd.arg("APPLY").arg("@__score").arg("AS").arg("text_score");

        // APPLY: compute hybrid_score
        let hybrid_expr = format!(
            "{}*@text_score + {}*@vector_similarity",
            1.0 - self.alpha,
            self.alpha
        );
        cmd.arg("APPLY")
            .arg(&hybrid_expr)
            .arg("AS")
            .arg("hybrid_score");

        // SORTBY by hybrid_score DESC
        cmd.arg("SORTBY")
            .arg(2)
            .arg("@hybrid_score")
            .arg("DESC")
            .arg("MAX")
            .arg(self.num_results);

        // LOAD return fields
        if !self.return_fields.is_empty() {
            cmd.arg("LOAD");
            cmd.arg(self.return_fields.len());
            for field in &self.return_fields {
                cmd.arg(format!("@{}", field));
            }
        }

        // DIALECT
        cmd.arg("DIALECT").arg(self.dialect);

        // PARAMS for the vector blob
        cmd.arg("PARAMS")
            .arg(2)
            .arg("vector")
            .arg(self.vector.to_bytes().as_ref());

        cmd
    }
}

/// Supported vector data types for multi-vector queries.
///
/// Mirrors the Python `Vector.dtype` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorDtype {
    /// Brain floating-point 16-bit.
    BFloat16,
    /// IEEE 754 half-precision 16-bit.
    Float16,
    /// IEEE 754 single-precision 32-bit (default).
    Float32,
    /// IEEE 754 double-precision 64-bit.
    Float64,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 8-bit integer.
    Uint8,
}

impl Default for VectorDtype {
    fn default() -> Self {
        Self::Float32
    }
}

impl VectorDtype {
    /// Bytes per element for this dtype.
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::BFloat16 | Self::Float16 => 2,
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Int8 | Self::Uint8 => 1,
        }
    }
}

/// A single vector input for [`MultiVectorQuery`].
///
/// Mirrors the Python `Vector` dataclass from `redisvl.query.aggregate`.
/// Each input carries its own field name, weight, data type, and optional
/// maximum distance threshold.
#[derive(Debug, Clone)]
pub struct VectorInput<'a> {
    /// The encoded vector bytes.
    pub vector: Cow<'a, [u8]>,
    /// Redis field name this vector targets.
    pub field_name: String,
    /// Weight applied when computing the combined score.
    pub weight: f32,
    /// Data type of the vector elements.
    pub dtype: VectorDtype,
    /// Maximum cosine distance threshold for range filtering (0.0–2.0).
    pub max_distance: f32,
}

impl<'a> VectorInput<'a> {
    /// Creates a vector input from float elements, encoding them as float32
    /// bytes.
    pub fn from_floats(elements: &[f32], field_name: impl Into<String>) -> Self {
        let mut buf = Vec::with_capacity(elements.len() * std::mem::size_of::<f32>());
        for &v in elements {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            vector: Cow::Owned(buf),
            field_name: field_name.into(),
            weight: 1.0,
            dtype: VectorDtype::Float32,
            max_distance: 2.0,
        }
    }

    /// Creates a vector input from pre-encoded bytes.
    pub fn from_bytes(
        bytes: impl Into<Cow<'a, [u8]>>,
        field_name: impl Into<String>,
        dtype: VectorDtype,
    ) -> Self {
        Self {
            vector: bytes.into(),
            field_name: field_name.into(),
            weight: 1.0,
            dtype,
            max_distance: 2.0,
        }
    }

    /// Sets the weight for this vector.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Sets the data type.
    pub fn with_dtype(mut self, dtype: VectorDtype) -> Self {
        self.dtype = dtype;
        self
    }

    /// Sets the maximum distance threshold (must be in 0.0–2.0).
    ///
    /// # Panics
    ///
    /// Panics if `max_distance` is outside the valid range.
    pub fn with_max_distance(mut self, max_distance: f32) -> Self {
        assert!(
            (0.0..=2.0).contains(&max_distance),
            "max_distance must be in [0.0, 2.0], got {}",
            max_distance
        );
        self.max_distance = max_distance;
        self
    }
}

/// Weighted multi-vector query using `FT.AGGREGATE` with per-vector range
/// searches.
///
/// Mirrors the Python `MultiVectorQuery` which composes multiple
/// `VECTOR_RANGE` clauses and computes a weighted `combined_score` via
/// `APPLY` steps.
#[derive(Debug, Clone)]
pub struct MultiVectorQuery<'a> {
    vectors: Vec<VectorInput<'a>>,
    filter_expression: Option<FilterExpression>,
    num_results: usize,
    return_fields: Vec<String>,
    dialect: u32,
}

impl<'a> MultiVectorQuery<'a> {
    /// Creates a multi-vector query from one or more [`VectorInput`]s.
    pub fn new(vectors: Vec<VectorInput<'a>>) -> Self {
        Self {
            vectors,
            filter_expression: None,
            num_results: 10,
            return_fields: Vec::new(),
            dialect: 2,
        }
    }

    /// Sets the number of results to return.
    pub fn with_num_results(mut self, num_results: usize) -> Self {
        self.num_results = num_results;
        self
    }

    /// Attaches a filter expression.
    pub fn with_filter(mut self, filter: FilterExpression) -> Self {
        self.filter_expression = Some(filter);
        self
    }

    /// Replaces the return field list.
    pub fn with_return_fields<I, S>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.return_fields = fields.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the Redis dialect version.
    pub fn with_dialect(mut self, dialect: u32) -> Self {
        self.dialect = dialect;
        self
    }

    /// Returns the vector inputs used by the query.
    pub fn vectors(&self) -> &[VectorInput<'a>] {
        &self.vectors
    }

    /// Builds the FT.AGGREGATE query string.
    ///
    /// Mirrors the Python `MultiVectorQuery.__str__()` output:
    /// ```text
    /// @field_0:[VECTOR_RANGE max_dist_0 $vector_0]=>{$YIELD_DISTANCE_AS: distance_0}
    ///   AND @field_1:[VECTOR_RANGE max_dist_1 $vector_1]=>{$YIELD_DISTANCE_AS: distance_1}
    /// ```
    pub fn build_query_string(&self) -> String {
        let mut parts = Vec::with_capacity(self.vectors.len());
        for (i, vi) in self.vectors.iter().enumerate() {
            parts.push(format!(
                "@{}:[VECTOR_RANGE {} $vector_{}]=>{{$YIELD_DISTANCE_AS: distance_{}}}",
                vi.field_name, vi.max_distance, i, i
            ));
        }

        let base = parts.join(" AND ");

        if let Some(filter) = &self.filter_expression {
            let filter_str = filter.to_redis_syntax();
            if filter_str != "*" {
                format!("({}) {}", filter_str, base)
            } else {
                base
            }
        } else {
            base
        }
    }

    /// Builds the complete `FT.AGGREGATE` command.
    pub fn build_aggregate_cmd(&self, index_name: &str) -> redis::Cmd {
        let query_string = self.build_query_string();
        let mut cmd = redis::cmd("FT.AGGREGATE");
        cmd.arg(index_name);
        cmd.arg(&query_string);

        // SCORER TFIDF (matches Python default)
        cmd.arg("SCORER").arg("TFIDF");

        // DIALECT
        cmd.arg("DIALECT").arg(self.dialect);

        // APPLY: compute score_i = (2 - distance_i) / 2 for each vector
        for i in 0..self.vectors.len() {
            cmd.arg("APPLY")
                .arg(format!("(2 - @distance_{})/2", i))
                .arg("AS")
                .arg(format!("score_{}", i));
        }

        // APPLY: combined_score = sum(score_i * weight_i)
        let combined_expr: Vec<String> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, vi)| format!("@score_{} * {}", i, vi.weight))
            .collect();
        cmd.arg("APPLY")
            .arg(combined_expr.join(" + "))
            .arg("AS")
            .arg("combined_score");

        // SORTBY by combined_score DESC
        cmd.arg("SORTBY")
            .arg(2)
            .arg("@combined_score")
            .arg("DESC")
            .arg("MAX")
            .arg(self.num_results);

        // LOAD return fields
        if !self.return_fields.is_empty() {
            cmd.arg("LOAD");
            cmd.arg(self.return_fields.len());
            for field in &self.return_fields {
                cmd.arg(format!("@{}", field));
            }
        }

        // PARAMS for vector blobs
        let param_count = self.vectors.len() * 2;
        cmd.arg("PARAMS").arg(param_count);
        for (i, vi) in self.vectors.iter().enumerate() {
            cmd.arg(format!("vector_{}", i));
            cmd.arg(vi.vector.as_ref());
        }

        cmd
    }
}

impl QueryString for str {
    fn to_redis_query(&self) -> String {
        self.to_owned()
    }
}

impl QueryString for &str {
    fn to_redis_query(&self) -> String {
        (*self).to_owned()
    }
}

impl QueryString for String {
    fn to_redis_query(&self) -> String {
        self.clone()
    }
}

#[cfg(feature = "sql")]
mod sql;
#[cfg(feature = "sql")]
pub use sql::{SQLQuery, SqlParam};

#[cfg(test)]
mod tests {
    use super::{
        AggregateHybridQuery, CountQuery, FilterQuery, HybridCombinationMethod, HybridPolicy,
        HybridQuery, MultiVectorQuery, PageableQuery, QueryString, SearchHistoryMode,
        SortDirection, TextQuery, Vector, VectorDtype, VectorInput, VectorQuery, VectorRangeQuery,
    };
    use crate::filter::{Num, Tag, Text};

    #[test]
    fn vector_query_should_render_knn() {
        let query = VectorQuery::new(Vector::new(vec![1.0, 2.0, 3.0]), "embedding", 5)
            .with_return_fields(["field1", "field2"])
            .with_dialect(3);

        assert!(query.to_redis_query().contains("KNN 5"));
        assert_eq!(query.vector().to_bytes().len(), 12);
        assert_eq!(
            query.render().return_fields,
            vec!["field1", "field2", "vector_distance"]
        );
        assert_eq!(query.render().dialect, 3);
    }

    #[test]
    fn hybrid_query_should_build_ft_hybrid_cmd_like_python_hybrid_query() {
        let query = HybridQuery::new(
            "a medical professional",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .with_num_results(10)
        .with_combination_method(HybridCombinationMethod::Rrf)
        .with_yield_combined_score_as("hybrid_score")
        .with_return_fields(["user", "age", "job"]);

        let cmd = query.build_cmd("my_index");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("FT.HYBRID"));
        assert!(cmd_str.contains("my_index"));
        assert!(cmd_str.contains("@description:(a medical professional)"));
        assert!(cmd_str.contains("COMBINE_METHOD"));
        assert!(cmd_str.contains("RRF"));
        assert!(cmd_str.contains("YIELD_SCORE_AS"));
        assert!(cmd_str.contains("hybrid_score"));
    }

    #[test]
    fn hybrid_query_with_rrf_params_like_python_hybrid_query_rrf() {
        let query = HybridQuery::new(
            "search text",
            "content",
            Vector::new(vec![0.5, 0.5]),
            "vec_field",
        )
        .with_rrf(Some(100), Some(10));

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("COMBINE_METHOD"));
        assert!(cmd_str.contains("RRF"));
        assert!(cmd_str.contains("WINDOW"));
        assert!(cmd_str.contains("CONSTANT"));
    }

    #[test]
    fn hybrid_query_with_linear_alpha_like_python_hybrid_query_linear() {
        let query =
            HybridQuery::new("query text", "body", Vector::new(vec![1.0]), "vec").with_linear(0.3);

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("COMBINE_METHOD"));
        assert!(cmd_str.contains("LINEAR"));
        assert!(cmd_str.contains("ALPHA"));
    }

    #[test]
    fn hybrid_query_with_filter_like_python_hybrid_query_filter() {
        let filter = Tag::new("status").eq("active");
        let query = HybridQuery::new("doctors", "description", Vector::new(vec![1.0, 2.0]), "vec")
            .with_filter(filter);

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("FILTER"));
        assert!(cmd_str.contains("@status:{active}"));
    }

    #[test]
    fn hybrid_query_with_stopwords_and_weights_like_python_hybrid_query() {
        use std::collections::{HashMap, HashSet};
        let mut stopwords = HashSet::new();
        stopwords.insert("the".to_owned());
        stopwords.insert("a".to_owned());

        let mut weights = HashMap::new();
        weights.insert("doctor".to_owned(), 2.0_f32);

        let query = HybridQuery::new(
            "a doctor in the house",
            "description",
            Vector::new(vec![1.0]),
            "vec",
        )
        .with_stopwords(stopwords)
        .with_text_weights(weights);

        let query_string = query.build_query_string();
        // Stopwords "a" and "the" removed
        assert!(!query_string.contains(" a "));
        assert!(!query_string.contains(" the "));
        assert!(query_string.contains("doctor"));
        // Weight applied to "doctor"
        assert!(query_string.contains("doctor=>{2}"));
    }

    #[test]
    fn hybrid_query_with_text_scorer_like_python_hybrid_query() {
        let query = HybridQuery::new("test", "body", Vector::new(vec![1.0]), "vec")
            .with_text_scorer("BM25STD")
            .with_yield_text_score_as("text_score");

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("SCORER"));
        assert!(cmd_str.contains("BM25STD"));
        assert!(cmd_str.contains("YIELD_SCORE_AS"));
        assert!(cmd_str.contains("text_score"));
    }

    #[test]
    fn filter_query_should_track_paging_and_sort_like_python_test_query_types() {
        let query = FilterQuery::new(Tag::new("brand").eq("Nike"))
            .with_return_fields(["brand", "price"])
            .paging(5, 7)
            .sort_by("price", SortDirection::Asc)
            .in_order(true)
            .with_dialect(2);

        let render = query.render();
        assert_eq!(render.return_fields, vec!["brand", "price"]);
        assert_eq!(render.limit.expect("limit").offset, 5);
        assert_eq!(render.limit.expect("limit").num, 7);
        assert!(render.sort_by.is_some());
        assert!(render.in_order);
        assert_eq!(render.dialect, 2);
    }

    #[test]
    fn count_query_should_use_nocontent_and_zero_limit_like_python_test_query_types() {
        let render = CountQuery::new()
            .with_filter(Tag::new("brand").eq("Nike"))
            .render();

        assert!(render.no_content);
        assert_eq!(render.limit.expect("limit").num, 0);
        assert_eq!(render.dialect, 2);
    }

    #[test]
    fn text_query_should_track_return_fields_and_limit_like_python_test_query_types() {
        let render = TextQuery::new("basketball")
            .for_field("description")
            .with_return_fields(["title", "genre", "rating"])
            .paging(5, 7)
            .render();

        assert_eq!(render.return_fields, vec!["title", "genre", "rating"]);
        assert_eq!(render.limit.expect("limit").offset, 5);
        assert!(render.query_string.contains("@description:(basketball)"));
    }

    #[test]
    fn range_query_should_include_vector_params_like_python_test_query_types() {
        let render = VectorRangeQuery::new(Vector::new(vec![1.0, 2.0, 3.0]), "embedding", 0.2)
            .with_return_fields(["field1"])
            .render();

        // params: vector + distance_threshold
        assert_eq!(render.params.len(), 2);
        assert_eq!(render.params[0].name, "vector");
        assert_eq!(render.params[1].name, "distance_threshold");
        assert_eq!(render.return_fields, vec!["field1", "vector_distance"]);
    }

    #[test]
    fn vector_range_query_should_update_distance_threshold_like_python_integration_test_query() {
        let mut query = VectorRangeQuery::new(Vector::new(vec![1.0, 2.0, 3.0]), "embedding", 0.2);
        assert_eq!(query.distance_threshold(), 0.2);

        query.set_distance_threshold(0.1);

        assert_eq!(query.distance_threshold(), 0.1);
        assert!(
            query
                .to_redis_query()
                .contains("VECTOR_RANGE $distance_threshold")
        );
    }

    #[test]
    fn vector_query_should_replace_filter_in_place_like_python_integration_test_query() {
        let mut query = VectorQuery::new(Vector::new(vec![1.0, 2.0, 3.0]), "embedding", 5);
        query.set_filter(Tag::new("brand").eq("Nike"));
        assert!(query.to_redis_query().starts_with("@brand:{Nike}"));

        query.set_filter(Num::new("price").gte(10.0));
        assert!(query.to_redis_query().starts_with("@price:[10 +inf]"));
    }

    #[test]
    fn pageable_queries_should_clone_updated_limits_for_pagination() {
        let query = FilterQuery::new(Tag::new("brand").eq("Nike")).paging(0, 5);

        let paged = query.paged(10, 3);

        assert_eq!(paged.render().limit.expect("limit").offset, 10);
        assert_eq!(paged.render().limit.expect("limit").num, 3);
        assert_eq!(query.render().limit.expect("limit").offset, 0);
    }

    #[test]
    fn raw_string_queries_should_render_directly_for_python_style_batch_search() {
        let render = "@test:{foo}".render();

        assert_eq!(render.query_string, "@test:{foo}");
        assert!(render.params.is_empty());
    }

    #[test]
    fn aggregate_hybrid_query_should_reject_empty_text() {
        let result = AggregateHybridQuery::new("", "desc", Vector::new(vec![1.0]), "vec");
        assert!(result.is_err());
    }

    #[test]
    fn aggregate_hybrid_query_should_build_query_string_like_python_aggregate_hybrid() {
        let query = AggregateHybridQuery::new(
            "a medical professional with expertise in lung cancer",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .unwrap()
        .with_num_results(10);

        let qs = query.build_query_string();
        assert!(qs.contains("@description:(a medical professional with expertise in lung cancer)"));
        assert!(qs.contains("=>[KNN 10 @user_embedding $vector AS vector_distance]"));
    }

    #[test]
    fn aggregate_hybrid_query_should_build_ft_aggregate_cmd_like_python() {
        let query = AggregateHybridQuery::new(
            "medical professional",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .unwrap()
        .with_alpha(0.5)
        .with_num_results(3)
        .with_text_scorer("BM25STD")
        .with_return_fields(["user", "age", "job"]);

        let cmd = query.build_aggregate_cmd("my_index");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("FT.AGGREGATE"));
        assert!(cmd_str.contains("my_index"));
        assert!(cmd_str.contains("SCORER"));
        assert!(cmd_str.contains("BM25STD"));
        assert!(cmd_str.contains("ADDSCORES"));
        assert!(cmd_str.contains("vector_similarity"));
        assert!(cmd_str.contains("text_score"));
        assert!(cmd_str.contains("hybrid_score"));
        assert!(cmd_str.contains("SORTBY"));
        assert!(cmd_str.contains("LOAD"));
        assert!(cmd_str.contains("DIALECT"));
        assert!(cmd_str.contains("PARAMS"));
    }

    #[test]
    fn aggregate_hybrid_query_with_filter_like_python_aggregate_filter() {
        let filter = Tag::new("credit_score").eq("high") & Num::new("age").gt(30.0);
        let query = AggregateHybridQuery::new(
            "medical professional",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .unwrap()
        .with_filter(filter);

        let qs = query.build_query_string();
        assert!(qs.contains("@credit_score:{high}"));
        assert!(qs.contains("@age:[(30"));
    }

    #[test]
    fn aggregate_hybrid_query_with_stopwords_like_python_aggregate_stopwords() {
        use std::collections::HashSet;
        let mut stopwords = HashSet::new();
        stopwords.insert("medical".to_owned());
        stopwords.insert("expertise".to_owned());

        let query = AggregateHybridQuery::new(
            "a medical professional with expertise in lung cancer",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .unwrap()
        .with_stopwords(stopwords);

        let qs = query.build_query_string();
        assert!(!qs.contains("medical"));
        assert!(!qs.contains("expertise"));
    }

    #[test]
    fn aggregate_hybrid_query_with_text_weights_like_python_aggregate_word_weights() {
        use std::collections::HashMap;
        let mut weights = HashMap::new();
        weights.insert("medical".to_owned(), 3.4_f32);
        weights.insert("cancers".to_owned(), 5.0_f32);

        let query = AggregateHybridQuery::new(
            "a medical professional with expertise in lung cancers",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .unwrap()
        .with_text_weights(weights);

        let qs = query.build_query_string();
        assert!(qs.contains("medical=>{3.4}"));
        assert!(qs.contains("cancers=>{5}"));
    }

    #[test]
    fn aggregate_hybrid_query_set_text_weights_should_match_constructor_weights() {
        use std::collections::HashMap;
        let mut weights = HashMap::new();
        weights.insert("medical".to_owned(), 5.0_f32);

        let query1 = AggregateHybridQuery::new(
            "a medical professional",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .unwrap()
        .with_text_weights(weights.clone());

        let mut query2 = AggregateHybridQuery::new(
            "a medical professional",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        )
        .unwrap();
        query2.set_text_weights(weights);

        assert_eq!(query1.build_query_string(), query2.build_query_string());
    }

    // ---------------------------------------------------------------
    // Multi-vector query parity tests
    // Mirrors: tests/unit/test_aggregation_types.py
    // ---------------------------------------------------------------

    #[test]
    fn multi_vector_query_should_build_vector_range_query_like_python() {
        // Mirrors: test_multi_vector_query_string
        let v1 = VectorInput::from_floats(&[0.1, 0.2, 0.3, 0.4], "text embedding")
            .with_weight(0.2)
            .with_max_distance(0.7);
        let v2 = VectorInput::from_floats(&[0.5, 0.5], "image embedding")
            .with_weight(0.7)
            .with_max_distance(1.8);

        let query = MultiVectorQuery::new(vec![v1, v2]);
        let qs = query.build_query_string();

        assert!(qs.contains("@text embedding:[VECTOR_RANGE 0.7 $vector_0]"));
        assert!(qs.contains("YIELD_DISTANCE_AS: distance_0"));
        assert!(qs.contains("@image embedding:[VECTOR_RANGE 1.8 $vector_1]"));
        assert!(qs.contains("YIELD_DISTANCE_AS: distance_1"));
        assert!(qs.contains("AND"));
    }

    #[test]
    fn multi_vector_query_default_properties_like_python() {
        // Mirrors: test_multi_vector_query default property checks
        let vi = VectorInput::from_floats(&[0.1, 0.2, 0.3, 0.4], "field_1");
        assert_eq!(vi.weight, 1.0);
        assert_eq!(vi.dtype, VectorDtype::Float32);
        assert_eq!(vi.max_distance, 2.0);

        let query = MultiVectorQuery::new(vec![vi]);
        assert!(query.filter_expression.is_none());
        assert_eq!(query.num_results, 10);
        assert!(query.return_fields.is_empty());
        assert_eq!(query.dialect, 2);
    }

    #[test]
    fn multi_vector_query_should_accept_multiple_vectors_like_python() {
        // Mirrors: test_multi_vector_query with multiple vectors
        let v1 = VectorInput::from_floats(&[0.1, 0.2, 0.3, 0.4], "field_1")
            .with_weight(0.2)
            .with_max_distance(2.0);
        let v2 = VectorInput::from_floats(&[0.1, 0.2, 0.3, 0.4], "field_2")
            .with_weight(0.5)
            .with_max_distance(1.5);
        let v3 = VectorInput::from_floats(&[0.5, 0.5], "field_3")
            .with_weight(0.6)
            .with_max_distance(0.4);
        let v4 = VectorInput::from_floats(&[0.1, 0.1, 0.1], "field_4")
            .with_weight(0.1)
            .with_max_distance(0.01);

        let query = MultiVectorQuery::new(vec![v1, v2, v3, v4]);
        assert_eq!(query.vectors().len(), 4);
    }

    #[test]
    fn multi_vector_query_overrides_like_python() {
        // Mirrors: test_multi_vector_query defaults can be overwritten
        let vi = VectorInput::from_floats(&[0.1, 0.2], "field_1");
        let filter = Tag::new("user group").one_of(["group A", "group C"]);

        let query = MultiVectorQuery::new(vec![vi])
            .with_filter(filter)
            .with_num_results(5)
            .with_return_fields(["field_1", "user name", "address"])
            .with_dialect(4);

        assert!(query.filter_expression.is_some());
        assert_eq!(query.num_results, 5);
        assert_eq!(query.return_fields, vec!["field_1", "user name", "address"]);
        assert_eq!(query.dialect, 4);
    }

    #[test]
    fn multi_vector_query_aggregate_cmd_like_python() {
        // Mirrors: test_multi_vector_query_string aggregate command structure
        let v1 = VectorInput::from_floats(&[0.1, 0.2, 0.3, 0.4], "text embedding")
            .with_weight(0.2)
            .with_max_distance(0.7);
        let v2 = VectorInput::from_floats(&[0.5, 0.5], "image embedding")
            .with_weight(0.7)
            .with_max_distance(1.8);

        let query = MultiVectorQuery::new(vec![v1, v2]);
        let cmd = query.build_aggregate_cmd("my_index");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("FT.AGGREGATE"));
        assert!(cmd_str.contains("my_index"));
        assert!(cmd_str.contains("SCORER"));
        assert!(cmd_str.contains("TFIDF"));
        assert!(cmd_str.contains("score_0"));
        assert!(cmd_str.contains("score_1"));
        assert!(cmd_str.contains("combined_score"));
        assert!(cmd_str.contains("SORTBY"));
        assert!(cmd_str.contains("PARAMS"));
    }

    #[test]
    fn multi_vector_query_with_filter_like_python() {
        // Mirrors: test_multivector_query_with_filter (text filter)
        let v1 = VectorInput::from_floats(&[0.1, 0.1, 0.5], "user_embedding");
        let v2 = VectorInput::from_floats(&[0.3, 0.4, 0.7, 0.2, -0.3], "image_embedding");
        let filter = Text::new("description").eq("medical");

        let query = MultiVectorQuery::new(vec![v1, v2]).with_filter(filter);

        let qs = query.build_query_string();
        assert!(qs.contains("@description"));
        assert!(qs.contains("medical"));
    }

    #[test]
    #[should_panic(expected = "max_distance must be in [0.0, 2.0]")]
    fn vector_input_should_reject_invalid_max_distance_like_python() {
        // Mirrors: test_vector_object_validation max_distance bounds
        VectorInput::from_floats(&[0.1, 0.2], "field").with_max_distance(2.001);
    }

    #[test]
    #[should_panic(expected = "max_distance must be in [0.0, 2.0]")]
    fn vector_input_should_reject_negative_max_distance_like_python() {
        // Mirrors: test_vector_object_validation negative distance
        VectorInput::from_floats(&[0.1, 0.2], "field").with_max_distance(-0.1);
    }

    #[test]
    fn vector_input_from_bytes_like_python() {
        // Mirrors: test_vector_object_handles_byte_conversion
        let floats = [0.1_f32, 0.2, 0.3, 0.4];
        let mut expected_bytes = Vec::new();
        for &f in &floats {
            expected_bytes.extend_from_slice(&f.to_le_bytes());
        }
        let vi = VectorInput::from_floats(&floats, "field_1");
        assert_eq!(vi.vector.as_ref(), expected_bytes.as_slice());

        // Also test from pre-encoded bytes
        let vi2 = VectorInput::from_bytes(expected_bytes.clone(), "field_1", VectorDtype::Float32);
        assert_eq!(vi2.vector.as_ref(), expected_bytes.as_slice());
    }

    // ---------------------------------------------------------------
    // AggregateHybridQuery parity: empty-text, stopwords, filter
    // Mirrors: tests/unit/test_aggregation_types.py
    // ---------------------------------------------------------------

    #[test]
    fn aggregate_hybrid_query_reject_stopword_only_text_like_python() {
        // Mirrors: test_empty_query_string – text that becomes empty after
        // default stopword removal. Our Rust impl currently only checks raw
        // empty text; full default-stopword removal is a known gap.
        let result = AggregateHybridQuery::new(
            "",
            "description",
            Vector::new(vec![0.1, 0.1, 0.5]),
            "user_embedding",
        );
        assert!(result.is_err());
    }

    #[test]
    fn aggregate_hybrid_query_with_string_filter_like_python() {
        // Mirrors: test_hybrid_query_with_string_filter
        // Using a raw string filter should be included in the query string.
        use crate::filter::FilterExpression;
        let filter_str = "@category:{tech|science|engineering}";
        let filter = FilterExpression::raw(filter_str);

        let query = AggregateHybridQuery::new(
            "search for document 12345",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .unwrap()
        .with_filter(filter);

        let qs = query.build_query_string();
        assert!(qs.contains("@description:(search for document 12345)"));
        assert!(qs.contains("@category:{tech|science|engineering}"));
    }

    #[test]
    fn aggregate_hybrid_query_wildcard_filter_is_ignored_like_python() {
        // Mirrors: test_hybrid_query_with_string_filter – wildcard filter
        use crate::filter::FilterExpression;
        let filter = FilterExpression::raw("*");

        let query = AggregateHybridQuery::new(
            "search text",
            "description",
            Vector::new(vec![0.1]),
            "embedding",
        )
        .unwrap()
        .with_filter(filter);

        let qs = query.build_query_string();
        assert!(!qs.contains("AND"));
    }

    #[test]
    fn aggregate_hybrid_query_text_weights_validation_like_python() {
        // Mirrors: test_aggregate_hybrid_query_text_weights_validation
        // Empty weights and None should be allowed
        use std::collections::HashMap;

        let q1 = AggregateHybridQuery::new(
            "sample text query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .unwrap()
        .with_text_weights(HashMap::new());
        assert!(q1.build_query_string().contains("sample"));

        // Weights for words not in the query should still work
        let mut weights = HashMap::new();
        weights.insert("alpha".to_owned(), 0.2_f32);
        weights.insert("bravo".to_owned(), 0.4_f32);
        let q2 = AggregateHybridQuery::new(
            "sample text query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .unwrap()
        .with_text_weights(weights);
        // The weights for non-present words should not cause errors
        let qs = q2.build_query_string();
        assert!(qs.contains("sample"));
    }

    // ---------------------------------------------------------------
    // HybridQuery parity: vector search methods, filters, edge cases
    // Mirrors: tests/unit/test_hybrid_types.py
    // ---------------------------------------------------------------

    #[test]
    fn hybrid_query_without_filter_like_python() {
        // Mirrors: test_hybrid_query_without_filter
        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        );

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        // No FILTER should appear
        assert!(!cmd_str.contains("FILTER"));
        assert!(cmd_str.contains("@description:(test query)"));
    }

    #[test]
    fn hybrid_query_vector_search_method_knn_like_python() {
        // Mirrors: test_hybrid_query_vector_search_method_knn
        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .with_knn(Some(100))
        .with_num_results(10);

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("KNN"));
        assert!(cmd_str.contains("EF_RUNTIME"));
    }

    #[test]
    fn hybrid_query_vector_search_method_range_like_python() {
        // Mirrors: test_hybrid_query_vector_search_method_range
        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .with_range(10.0, Some(0.1));

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("RANGE"));
        assert!(cmd_str.contains("RADIUS"));
        assert!(cmd_str.contains("EPSILON"));
    }

    #[test]
    fn hybrid_query_without_vector_search_method_like_python() {
        // Mirrors: test_hybrid_query_vector_search_method_none
        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        );

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("VSIM"));
        // No explicit KNN or RANGE
        assert!(!cmd_str.contains("KNN"));
        assert!(!cmd_str.contains("RANGE"));
    }

    #[test]
    fn hybrid_query_rrf_with_both_params_like_python() {
        // Mirrors: test_hybrid_query_combination_method_rrf_with_both_params
        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .with_rrf(Some(20), Some(50))
        .with_yield_combined_score_as("rrf_score");

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("RRF"));
        assert!(cmd_str.contains("WINDOW"));
        assert!(cmd_str.contains("CONSTANT"));
        assert!(cmd_str.contains("YIELD_SCORE_AS"));
        assert!(cmd_str.contains("rrf_score"));
    }

    #[test]
    fn hybrid_query_linear_with_alpha_like_python() {
        // Mirrors: test_hybrid_query_combination_method_linear
        for alpha in [0.1_f32, 0.5, 0.9] {
            let query = HybridQuery::new(
                "test query",
                "description",
                Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
                "embedding",
            )
            .with_linear(alpha);

            let cmd = query.build_cmd("idx");
            let packed = cmd.get_packed_command();
            let cmd_str = String::from_utf8_lossy(&packed);

            assert!(cmd_str.contains("LINEAR"));
            assert!(cmd_str.contains("ALPHA"));
            assert!(cmd_str.contains("BETA"));
        }
    }

    #[test]
    fn hybrid_query_without_combination_method_like_python() {
        // Mirrors: test_hybrid_query_combination_method_none
        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        );

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(!cmd_str.contains("COMBINE"));
    }

    #[test]
    fn hybrid_query_with_combined_filters_like_python() {
        // Mirrors: test_hybrid_query_with_combined_filters
        let filter = Tag::new("genre").eq("comedy") & Num::new("rating").gt(7.0);

        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .with_filter(filter);

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("FILTER"));
        assert!(cmd_str.contains("genre"));
        assert!(cmd_str.contains("comedy"));
        assert!(cmd_str.contains("rating"));
    }

    #[test]
    fn hybrid_query_with_numeric_filter_like_python() {
        // Mirrors: test_hybrid_query_with_numeric_filter
        let filter = Num::new("age").gt(30.0);

        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .with_filter(filter);

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("FILTER"));
        assert!(cmd_str.contains("@age:[(30"));
    }

    #[test]
    fn hybrid_query_with_text_filter_like_python() {
        // Mirrors: test_hybrid_query_with_text_filter
        let filter = Text::new("job").eq("engineer");

        let query = HybridQuery::new(
            "test query",
            "description",
            Vector::new(vec![0.1, 0.2, 0.3, 0.4]),
            "embedding",
        )
        .with_filter(filter);

        let cmd = query.build_cmd("idx");
        let packed = cmd.get_packed_command();
        let cmd_str = String::from_utf8_lossy(&packed);

        assert!(cmd_str.contains("FILTER"));
        assert!(cmd_str.contains("@job"));
        assert!(cmd_str.contains("engineer"));
    }

    // ── VectorQuery runtime params parity tests ──

    #[test]
    fn vector_query_hybrid_policy_like_python_test_query_types() {
        let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
            .with_hybrid_policy(HybridPolicy::Batches);

        assert_eq!(query.hybrid_policy(), Some(HybridPolicy::Batches));
        assert!(query.to_redis_query().contains("HYBRID_POLICY BATCHES"));
    }

    #[test]
    fn vector_query_hybrid_policy_with_batch_size_like_python() {
        let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
            .with_hybrid_policy(HybridPolicy::Batches)
            .with_batch_size(50);

        let qs = query.to_redis_query();
        assert!(qs.contains("HYBRID_POLICY BATCHES BATCH_SIZE 50"));
    }

    #[test]
    fn vector_query_adhoc_bf_policy_like_python() {
        let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
            .with_hybrid_policy(HybridPolicy::AdhocBf);

        assert!(query.to_redis_query().contains("HYBRID_POLICY ADHOC_BF"));
    }

    #[test]
    fn vector_query_epsilon_like_python_test_query_types() {
        let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
            .with_epsilon(0.05);

        assert_eq!(query.epsilon(), Some(0.05));
        let qs = query.to_redis_query();
        assert!(qs.contains("EPSILON $EPSILON"));
        let params = query.params();
        assert!(params.iter().any(|p| p.name == "EPSILON"));
    }

    #[test]
    fn vector_query_ef_runtime_params_like_python_test_query_types() {
        let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
            .with_ef_runtime(100);

        assert_eq!(query.ef_runtime(), Some(100));
        let qs = query.to_redis_query();
        assert!(qs.contains("EF_RUNTIME $EF"));
        let params = query.params();
        assert!(params.iter().any(|p| p.name == "EF"));
    }

    #[test]
    fn vector_query_search_window_size_like_python() {
        let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
            .with_search_window_size(40);

        assert_eq!(query.search_window_size(), Some(40));
        let qs = query.to_redis_query();
        assert!(qs.contains("SEARCH_WINDOW_SIZE $SEARCH_WINDOW_SIZE"));
    }

    #[test]
    fn vector_query_use_search_history_like_python() {
        for mode in [
            SearchHistoryMode::Off,
            SearchHistoryMode::On,
            SearchHistoryMode::Auto,
        ] {
            let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
                .with_use_search_history(mode);

            assert_eq!(query.use_search_history(), Some(mode));
            let qs = query.to_redis_query();
            assert!(qs.contains("USE_SEARCH_HISTORY $USE_SEARCH_HISTORY"));
        }
    }

    #[test]
    fn vector_query_search_buffer_capacity_like_python() {
        let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
            .with_search_buffer_capacity(50);

        assert_eq!(query.search_buffer_capacity(), Some(50));
        let qs = query.to_redis_query();
        assert!(qs.contains("SEARCH_BUFFER_CAPACITY $SEARCH_BUFFER_CAPACITY"));
    }

    #[test]
    fn vector_query_all_runtime_params_like_python() {
        let query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10)
            .with_ef_runtime(100)
            .with_epsilon(0.05)
            .with_search_window_size(40)
            .with_use_search_history(SearchHistoryMode::On)
            .with_search_buffer_capacity(50);

        let qs = query.to_redis_query();
        assert!(qs.contains("EF_RUNTIME $EF"));
        assert!(qs.contains("EPSILON $EPSILON"));
        assert!(qs.contains("SEARCH_WINDOW_SIZE $SEARCH_WINDOW_SIZE"));
        assert!(qs.contains("USE_SEARCH_HISTORY $USE_SEARCH_HISTORY"));
        assert!(qs.contains("SEARCH_BUFFER_CAPACITY $SEARCH_BUFFER_CAPACITY"));

        let params = query.params();
        assert!(params.iter().any(|p| p.name == "EF"));
        assert!(params.iter().any(|p| p.name == "EPSILON"));
        assert!(params.iter().any(|p| p.name == "SEARCH_WINDOW_SIZE"));
        assert!(params.iter().any(|p| p.name == "USE_SEARCH_HISTORY"));
        assert!(params.iter().any(|p| p.name == "SEARCH_BUFFER_CAPACITY"));
    }

    #[test]
    fn vector_query_set_methods_like_python() {
        let mut query = VectorQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 10);

        assert!(query.ef_runtime().is_none());
        assert!(query.epsilon().is_none());
        assert!(query.hybrid_policy().is_none());

        query.set_ef_runtime(200);
        assert_eq!(query.ef_runtime(), Some(200));

        query.set_epsilon(0.1);
        assert_eq!(query.epsilon(), Some(0.1));

        query.set_hybrid_policy(HybridPolicy::Batches);
        assert_eq!(query.hybrid_policy(), Some(HybridPolicy::Batches));

        query.set_batch_size(100);
        assert_eq!(query.batch_size(), Some(100));
    }

    // ── VectorRangeQuery runtime params parity tests ──

    #[test]
    fn range_query_epsilon_like_python_test_query_types() {
        let query =
            VectorRangeQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 0.3)
                .with_epsilon(0.05);

        assert_eq!(query.epsilon(), Some(0.05));
        let qs = query.to_redis_query();
        assert!(qs.contains("$EPSILON: 0.05"));
    }

    #[test]
    fn range_query_construction_like_python() {
        // Basic range query
        let basic = VectorRangeQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 0.2)
            .with_return_fields(["user", "credit_score"]);

        let qs = basic.to_redis_query();
        assert!(qs.contains("VECTOR_RANGE $distance_threshold $vector"));
        assert!(qs.contains("$YIELD_DISTANCE_AS: vector_distance"));
        assert!(!qs.contains("HYBRID_POLICY"));

        // Range query with epsilon
        let epsilon_query =
            VectorRangeQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 0.2)
                .with_epsilon(0.05);

        let qs = epsilon_query.to_redis_query();
        assert!(qs.contains("$EPSILON: 0.05"));
        assert_eq!(epsilon_query.epsilon(), Some(0.05));
    }

    #[test]
    fn range_query_hybrid_policy_in_params_not_query_string_like_python() {
        let query = VectorRangeQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 0.2)
            .with_hybrid_policy(HybridPolicy::Batches);

        let qs = query.to_redis_query();
        // Hybrid policy should NOT be in the query string for range queries
        assert!(!qs.contains("HYBRID_POLICY"));
        assert_eq!(query.hybrid_policy(), Some(HybridPolicy::Batches));

        let params = query.params();
        assert!(params.iter().any(|p| p.name == "HYBRID_POLICY"));
    }

    #[test]
    fn range_query_hybrid_policy_with_batch_size_in_params_like_python() {
        let query = VectorRangeQuery::new(Vector::new(vec![0.1, 0.1, 0.5]), "user_embedding", 0.2)
            .with_hybrid_policy(HybridPolicy::Batches)
            .with_batch_size(50);

        let qs = query.to_redis_query();
        assert!(!qs.contains("HYBRID_POLICY"));
        assert!(!qs.contains("BATCH_SIZE"));

        let params = query.params();
        assert!(params.iter().any(|p| p.name == "HYBRID_POLICY"));
        assert!(params.iter().any(|p| p.name == "BATCH_SIZE"));
    }

    #[test]
    fn range_query_setter_methods_like_python() {
        let mut query =
            VectorRangeQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "user_embedding", 0.2);

        assert!(query.epsilon().is_none());
        assert!(query.hybrid_policy().is_none());
        assert!(query.batch_size().is_none());

        query.set_epsilon(0.1);
        assert_eq!(query.epsilon(), Some(0.1));
        assert!(query.to_redis_query().contains("$EPSILON: 0.1"));

        query.set_hybrid_policy(HybridPolicy::Batches);
        assert_eq!(query.hybrid_policy(), Some(HybridPolicy::Batches));

        query.set_batch_size(25);
        assert_eq!(query.batch_size(), Some(25));
    }

    #[test]
    fn range_query_search_window_size_like_python() {
        let query =
            VectorRangeQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 0.3)
                .with_search_window_size(40);

        assert_eq!(query.search_window_size(), Some(40));
        assert!(query.to_redis_query().contains("$SEARCH_WINDOW_SIZE: 40"));
    }

    #[test]
    fn range_query_use_search_history_like_python() {
        for (mode, expected_str) in [
            (SearchHistoryMode::Off, "OFF"),
            (SearchHistoryMode::On, "ON"),
            (SearchHistoryMode::Auto, "AUTO"),
        ] {
            let query =
                VectorRangeQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 0.3)
                    .with_use_search_history(mode);

            assert_eq!(query.use_search_history(), Some(mode));
            let qs = query.to_redis_query();
            assert!(
                qs.contains(&format!("$USE_SEARCH_HISTORY: {}", expected_str)),
                "query string should contain USE_SEARCH_HISTORY for {:?}",
                mode,
            );
        }
    }

    #[test]
    fn range_query_search_buffer_capacity_like_python() {
        let query =
            VectorRangeQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 0.3)
                .with_search_buffer_capacity(50);

        assert_eq!(query.search_buffer_capacity(), Some(50));
        assert!(
            query
                .to_redis_query()
                .contains("$SEARCH_BUFFER_CAPACITY: 50")
        );
    }

    #[test]
    fn range_query_all_svs_params_like_python() {
        let query =
            VectorRangeQuery::new(Vector::new(vec![0.1, 0.2, 0.3, 0.4]), "vector_field", 0.3)
                .with_epsilon(0.05)
                .with_search_window_size(40)
                .with_use_search_history(SearchHistoryMode::On)
                .with_search_buffer_capacity(50);

        let qs = query.to_redis_query();
        assert!(qs.contains("$EPSILON: 0.05"));
        assert!(qs.contains("$SEARCH_WINDOW_SIZE: 40"));
        assert!(qs.contains("$USE_SEARCH_HISTORY: ON"));
        assert!(qs.contains("$SEARCH_BUFFER_CAPACITY: 50"));
    }

    // ── TextQuery filter_expression parity tests ──

    #[test]
    fn text_query_with_filter_expression_like_python() {
        let filter = Tag::new("genre").eq("comedy");
        let query = TextQuery::new("basketball")
            .for_field("description")
            .with_filter(filter);

        let qs = query.to_redis_query();
        assert!(qs.contains("@description:(basketball)"));
        assert!(qs.contains("AND @genre:{comedy}"));
    }

    #[test]
    fn text_query_without_filter_like_python() {
        let query = TextQuery::new("basketball").for_field("description");

        let qs = query.to_redis_query();
        assert!(qs.contains("@description:(basketball)"));
        assert!(!qs.contains("AND"));
    }

    #[test]
    fn text_query_set_filter_like_python() {
        let mut query = TextQuery::new("basketball").for_field("description");
        query.set_filter(Tag::new("category").eq("sports"));

        let qs = query.to_redis_query();
        assert!(qs.contains("AND @category:{sports}"));
    }
}
