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

/// Vector nearest-neighbor query.
#[derive(Debug, Clone)]
pub struct VectorQuery<'a> {
    vector: Vector<'a>,
    vector_field_name: String,
    num_results: usize,
    filter_expression: Option<FilterExpression>,
    ef_runtime: Option<usize>,
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

    /// Sets the runtime EF parameter.
    pub fn with_ef_runtime(mut self, ef_runtime: usize) -> Self {
        self.ef_runtime = Some(ef_runtime);
        self
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
            "{}=>[KNN {} @{} $vector AS vector_distance]",
            base, self.num_results, self.vector_field_name
        );
        if let Some(ef_runtime) = self.ef_runtime {
            query.push_str(&format!(" EF_RUNTIME {}", ef_runtime));
        }
        query
    }

    fn params(&self) -> Vec<QueryParam> {
        vec![QueryParam {
            name: "vector".to_owned(),
            value: QueryParamValue::Binary(self.vector.to_bytes().to_vec()),
        }]
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
        format!(
            "@{}:[VECTOR_RANGE {} $vector] {}",
            self.vector_field_name, self.distance_threshold, base
        )
    }

    fn params(&self) -> Vec<QueryParam> {
        vec![QueryParam {
            name: "vector".to_owned(),
            value: QueryParamValue::Binary(self.vector.to_bytes().to_vec()),
        }]
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
    options: QueryOptions,
}

impl TextQuery {
    /// Creates a full-text query.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_field_name: None,
            options: QueryOptions::with_num_results(10),
        }
    }

    /// Restricts the query to a specific text field.
    pub fn for_field(mut self, text_field_name: impl Into<String>) -> Self {
        self.text_field_name = Some(text_field_name.into());
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
        match &self.text_field_name {
            Some(field) => format!("@{}:({})", field, self.text),
            None => self.text.clone(),
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
    fn redis_name(self) -> &'static str {
        match self {
            Self::Linear => "LINEAR",
            Self::Rrf => "RRF",
        }
    }
}

/// Native Redis hybrid query.
#[derive(Debug, Clone)]
pub struct HybridQuery<'a> {
    text: String,
    text_field_name: String,
    vector: Vector<'a>,
    vector_field_name: String,
    num_results: usize,
    combination_method: HybridCombinationMethod,
    options: QueryOptions,
}

impl<'a> HybridQuery<'a> {
    /// Creates a hybrid query.
    pub fn new(
        text: impl Into<String>,
        text_field_name: impl Into<String>,
        vector: Vector<'a>,
        vector_field_name: impl Into<String>,
        num_results: usize,
    ) -> Self {
        Self {
            text: text.into(),
            text_field_name: text_field_name.into(),
            vector,
            vector_field_name: vector_field_name.into(),
            num_results,
            combination_method: HybridCombinationMethod::Linear,
            options: QueryOptions::with_num_results(num_results),
        }
    }

    /// Selects the hybrid combination method.
    pub fn with_combination_method(mut self, combination_method: HybridCombinationMethod) -> Self {
        self.combination_method = combination_method;
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

impl QueryString for HybridQuery<'_> {
    fn to_redis_query(&self) -> String {
        format!(
            "@{}:({})=>[KNN {} @{} $vector AS vector_distance HYBRID_POLICY {}]",
            self.text_field_name,
            self.text,
            self.num_results,
            self.vector_field_name,
            self.combination_method.redis_name()
        )
    }

    fn params(&self) -> Vec<QueryParam> {
        vec![QueryParam {
            name: "vector".to_owned(),
            value: QueryParamValue::Binary(self.vector.to_bytes().to_vec()),
        }]
    }

    fn return_fields(&self) -> Vec<String> {
        self.options.return_fields.clone()
    }

    fn limit(&self) -> Option<QueryLimit> {
        Some(self.options.limit)
    }

    fn dialect(&self) -> u32 {
        self.options.dialect
    }
}

impl PageableQuery for HybridQuery<'_> {
    fn paged(&self, offset: usize, num: usize) -> Self {
        self.clone().paging(offset, num)
    }
}

/// Aggregate-based hybrid query.
#[derive(Debug, Clone)]
pub struct AggregateHybridQuery<'a> {
    inner: HybridQuery<'a>,
}

impl<'a> AggregateHybridQuery<'a> {
    /// Creates an aggregate hybrid query wrapper.
    pub fn new(inner: HybridQuery<'a>) -> Self {
        Self { inner }
    }

    /// Returns the inner vector.
    pub fn vector(&self) -> &Vector<'a> {
        self.inner.vector()
    }
}

impl QueryString for AggregateHybridQuery<'_> {
    fn to_redis_query(&self) -> String {
        self.inner.to_redis_query()
    }

    fn params(&self) -> Vec<QueryParam> {
        self.inner.params()
    }

    fn return_fields(&self) -> Vec<String> {
        self.inner.return_fields()
    }

    fn limit(&self) -> Option<QueryLimit> {
        self.inner.limit()
    }

    fn dialect(&self) -> u32 {
        self.inner.dialect()
    }
}

/// Weighted multi-vector query.
#[derive(Debug, Clone)]
pub struct MultiVectorQuery<'a> {
    vectors: Vec<Vector<'a>>,
    vector_field_name: String,
    weights: Vec<f32>,
}

impl<'a> MultiVectorQuery<'a> {
    /// Creates a multi-vector query.
    pub fn new(
        vectors: Vec<Vector<'a>>,
        vector_field_name: impl Into<String>,
        weights: Vec<f32>,
    ) -> Self {
        Self {
            vectors,
            vector_field_name: vector_field_name.into(),
            weights,
        }
    }

    /// Returns the vectors used by the query.
    pub fn vectors(&self) -> &[Vector<'a>] {
        &self.vectors
    }
}

impl QueryString for MultiVectorQuery<'_> {
    fn to_redis_query(&self) -> String {
        let weights = self
            .weights
            .iter()
            .map(|weight| weight.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "*=>[MULTI_KNN @{} $vectors WEIGHTS {}]",
            self.vector_field_name, weights
        )
    }

    fn params(&self) -> Vec<QueryParam> {
        let mut buffer = Vec::new();
        for vector in &self.vectors {
            buffer.extend_from_slice(vector.to_bytes().as_ref());
        }

        vec![QueryParam {
            name: "vectors".to_owned(),
            value: QueryParamValue::Binary(buffer),
        }]
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
/// SQL-like query representation for later parity work.
#[derive(Debug, Clone)]
pub struct SQLQuery {
    sql: String,
}

#[cfg(feature = "sql")]
impl SQLQuery {
    /// Creates an SQL query wrapper.
    pub fn new(sql: impl Into<String>) -> Self {
        Self { sql: sql.into() }
    }
}

#[cfg(feature = "sql")]
impl QueryString for SQLQuery {
    fn to_redis_query(&self) -> String {
        self.sql.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CountQuery, FilterQuery, HybridQuery, PageableQuery, QueryString, SortDirection, TextQuery,
        Vector, VectorQuery, VectorRangeQuery,
    };
    use crate::filter::{Num, Tag};

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
    fn hybrid_query_should_render_policy() {
        let query = HybridQuery::new(
            "hello",
            "body",
            Vector::new(vec![1.0, 2.0]),
            "embedding",
            10,
        );

        assert!(query.to_redis_query().contains("HYBRID_POLICY"));
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

        assert_eq!(render.params.len(), 1);
        assert_eq!(render.return_fields, vec!["field1", "vector_distance"]);
    }

    #[test]
    fn vector_range_query_should_update_distance_threshold_like_python_integration_test_query() {
        let mut query = VectorRangeQuery::new(Vector::new(vec![1.0, 2.0, 3.0]), "embedding", 0.2);
        assert_eq!(query.distance_threshold(), 0.2);

        query.set_distance_threshold(0.1);

        assert_eq!(query.distance_threshold(), 0.1);
        assert!(query.to_redis_query().contains("VECTOR_RANGE 0.1"));
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
}
