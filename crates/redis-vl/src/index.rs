//! Search index lifecycle helpers and Redis transport adapters.
//!
//! [`SearchIndex`] provides blocking (sync) operations and
//! [`AsyncSearchIndex`] provides Tokio-based async operations. Both manage
//! the full index lifecycle: create, delete, load, fetch, search, query,
//! batch operations, pagination, hybrid search, aggregate queries,
//! multi-vector queries, and `from_existing` reconnection.
//!
//! # Example
//!
//! ```rust,no_run
//! use redis_vl::{IndexSchema, SearchIndex, Vector, VectorQuery};
//!
//! let schema = IndexSchema::from_yaml_file("schema.yaml").unwrap();
//! let index = SearchIndex::new(schema, "redis://127.0.0.1:6379");
//! index.create().unwrap();
//!
//! let result = index.search(&VectorQuery::new(
//!     Vector::new(&[0.1_f32; 128] as &[f32]), "embedding", 5
//! )).unwrap();
//! ```

use std::collections::{HashMap, VecDeque};
use std::ops::Index;

use redis::Commands;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    error::{Error, Result},
    filter::FilterExpression,
    query::{PageableQuery, QueryKind, QueryParamValue, QueryString, SortDirection},
    schema::{FieldKind, IndexDefinition, IndexSchema, StorageType, VectorAlgorithm},
};

/// Redis connection settings for index operations.
#[derive(Debug, Clone)]
pub struct RedisConnectionInfo {
    /// Connection URL for Redis.
    pub redis_url: String,
}

impl RedisConnectionInfo {
    /// Creates a new connection descriptor.
    pub fn new(redis_url: impl Into<String>) -> Self {
        Self {
            redis_url: redis_url.into(),
        }
    }

    pub(crate) fn client(&self) -> Result<redis::Client> {
        Ok(redis::Client::open(self.redis_url.as_str())?)
    }
}

/// A single parsed Redis Search document.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchDocument {
    id: String,
    fields: Map<String, Value>,
}

impl SearchDocument {
    /// Creates a parsed search document.
    pub fn new(id: impl Into<String>, mut fields: Map<String, Value>) -> Self {
        let id = id.into();
        fields.insert("id".to_owned(), Value::String(id.clone()));
        Self { id, fields }
    }

    /// Returns the Redis document identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the projected document fields.
    pub fn fields(&self) -> &Map<String, Value> {
        &self.fields
    }

    /// Returns a field by name.
    pub fn get(&self, field: &str) -> Option<&Value> {
        self.fields.get(field)
    }

    /// Returns the document as a JSON object, including the `id` field.
    pub fn to_map(&self) -> Map<String, Value> {
        self.fields.clone()
    }

    /// Consumes the document and returns it as a JSON object, including the `id` field.
    pub fn into_map(self) -> Map<String, Value> {
        self.fields
    }
}

impl Index<&str> for SearchDocument {
    type Output = Value;

    fn index(&self, index: &str) -> &Self::Output {
        self.fields
            .get(index)
            .unwrap_or_else(|| panic!("field '{index}' not found on search document"))
    }
}

/// Parsed Redis Search results.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Total number of matching documents reported by Redis.
    pub total: usize,
    /// Parsed result documents.
    pub docs: Vec<SearchDocument>,
}

impl SearchResult {
    /// Creates a parsed search result.
    pub fn new(total: usize, docs: Vec<SearchDocument>) -> Self {
        Self { total, docs }
    }
}

/// Processed query output aligned with RedisVL query semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryOutput {
    /// A document-returning query.
    Documents(Vec<Map<String, Value>>),
    /// A count-only query.
    Count(usize),
}

impl QueryOutput {
    /// Returns the contained documents when present.
    pub fn as_documents(&self) -> Option<&[Map<String, Value>]> {
        match self {
            Self::Documents(documents) => Some(documents),
            Self::Count(_) => None,
        }
    }

    /// Returns the contained count when present.
    pub fn as_count(&self) -> Option<usize> {
        match self {
            Self::Count(count) => Some(*count),
            Self::Documents(_) => None,
        }
    }
}

/// Blocking search index client.
#[derive(Debug, Clone)]
pub struct SearchIndex {
    schema: IndexSchema,
    connection: RedisConnectionInfo,
}

impl SearchIndex {
    /// Creates a new blocking search index.
    pub fn new(schema: IndexSchema, redis_url: impl Into<String>) -> Self {
        Self {
            schema,
            connection: RedisConnectionInfo::new(redis_url),
        }
    }

    /// Creates a blocking index from a YAML string.
    pub fn from_yaml_str(yaml: &str, redis_url: impl Into<String>) -> Result<Self> {
        Ok(Self::new(IndexSchema::from_yaml_str(yaml)?, redis_url))
    }

    /// Creates a blocking index from a YAML file.
    pub fn from_yaml_file(
        path: impl AsRef<std::path::Path>,
        redis_url: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self::new(IndexSchema::from_yaml_file(path)?, redis_url))
    }

    /// Creates a blocking index from a JSON value.
    pub fn from_json_value(value: serde_json::Value, redis_url: impl Into<String>) -> Result<Self> {
        Ok(Self::new(IndexSchema::from_json_value(value)?, redis_url))
    }

    /// Returns the schema attached to the index.
    pub fn schema(&self) -> &IndexSchema {
        &self.schema
    }

    /// Returns the Redis index name.
    pub fn name(&self) -> &str {
        &self.schema.index.name
    }

    /// Returns the first (or only) key prefix.
    ///
    /// For multi-prefix indexes, returns the first prefix. Use [`Self::prefixes`]
    /// to access all configured prefixes.
    pub fn prefix(&self) -> &str {
        self.schema.index.prefix.first()
    }

    /// Returns all key prefixes configured for this index.
    pub fn prefixes(&self) -> Vec<&str> {
        self.schema.index.prefix.all()
    }

    /// Returns the separator between prefix and identifier.
    pub fn key_separator(&self) -> &str {
        &self.schema.index.key_separator
    }

    /// Returns the storage type.
    pub fn storage_type(&self) -> StorageType {
        self.schema.index.storage_type
    }

    /// Builds a document key from the index prefix and identifier.
    pub fn key(&self, key_suffix: &str) -> String {
        compose_key(self.prefix(), self.key_separator(), key_suffix)
    }

    /// Builds an `FT.CREATE` command for the current schema.
    pub fn create_cmd(&self) -> redis::Cmd {
        let mut cmd = redis::cmd("FT.CREATE");
        let prefixes = self.schema.index.prefix.all();
        cmd.arg(&self.schema.index.name)
            .arg("ON")
            .arg(self.schema.index.storage_type.redis_name())
            .arg("PREFIX")
            .arg(prefixes.len());
        for pfx in &prefixes {
            cmd.arg(*pfx);
        }

        if !self.schema.index.stopwords.is_empty() {
            cmd.arg("STOPWORDS").arg(self.schema.index.stopwords.len());
            for stopword in &self.schema.index.stopwords {
                cmd.arg(stopword);
            }
        }

        cmd.arg("SCHEMA");
        for arg in self.schema.redis_schema_args() {
            cmd.arg(arg);
        }
        cmd
    }

    /// Creates the Redis search index.
    pub fn create(&self) -> Result<()> {
        self.create_with_options(false, false)
    }

    /// Creates the Redis search index with overwrite controls similar to the
    /// Python RedisVL client.
    pub fn create_with_options(&self, overwrite: bool, drop_documents: bool) -> Result<()> {
        if self.schema.fields.is_empty() {
            return Err(Error::SchemaValidation(
                "No fields defined for index".to_owned(),
            ));
        }

        if self.exists()? {
            if !overwrite {
                return Ok(());
            }
            self.drop(drop_documents)?;
        }

        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let (): () = self.create_cmd().query(&mut connection)?;
        Ok(())
    }

    /// Drops the Redis search index.
    pub fn drop(&self, delete_documents: bool) -> Result<()> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut cmd = redis::cmd("FT.DROPINDEX");
        cmd.arg(&self.schema.index.name);
        if delete_documents {
            cmd.arg("DD");
        }
        let (): () = cmd.query(&mut connection)?;
        Ok(())
    }

    /// Deletes the search index, optionally dropping associated documents.
    pub fn delete(&self, drop_documents: bool) -> Result<()> {
        if !self.exists()? {
            return Err(Error::InvalidInput(format!(
                "index '{}' does not exist",
                self.name()
            )));
        }
        self.drop(drop_documents)
    }

    /// Returns raw Redis index metadata.
    pub fn info(&self) -> Result<Map<String, Value>> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let value = redis::cmd("FT.INFO")
            .arg(&self.schema.index.name)
            .query(&mut connection)?;
        parse_info_response(value)
    }

    /// Lists all Redis search indices.
    pub fn listall(&self) -> Result<Vec<String>> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let value = redis::cmd("FT._LIST").query(&mut connection)?;
        Ok(value)
    }

    /// Returns whether the Redis search index exists.
    pub fn exists(&self) -> Result<bool> {
        Ok(self.listall()?.iter().any(|name| name == self.name()))
    }

    /// Loads a JSON document into Redis.
    pub fn load_json<T>(&self, key_suffix: &str, document: &T) -> Result<()>
    where
        T: Serialize,
    {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let payload = serde_json::to_string(document)?;
        let (): () = redis::cmd("JSON.SET")
            .arg(self.key(key_suffix))
            .arg("$")
            .arg(payload)
            .query(&mut connection)?;
        Ok(())
    }

    /// Loads a Redis hash document.
    pub fn load_hash(&self, key_suffix: &str, values: &HashMap<String, String>) -> Result<()> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut cmd = redis::cmd("HSET");
        cmd.arg(self.key(key_suffix));
        for (field, value) in values {
            cmd.arg(field).arg(value);
        }
        let _: i32 = cmd.query(&mut connection)?;
        Ok(())
    }

    /// Loads documents using the configured storage type and returns the written keys.
    pub fn load(&self, data: &[Value], id_field: &str, ttl: Option<i64>) -> Result<Vec<String>> {
        self.load_with_preprocess(data, id_field, ttl, |record| Ok(record.clone()))
    }

    /// Loads documents after applying a preprocessing callback to each record.
    pub fn load_with_preprocess<F>(
        &self,
        data: &[Value],
        id_field: &str,
        ttl: Option<i64>,
        mut preprocess: F,
    ) -> Result<Vec<String>>
    where
        F: FnMut(&Value) -> Result<Value>,
    {
        let prepared = prepare_load_records(data, &mut preprocess)?;
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut written_keys = Vec::with_capacity(prepared.len());

        for record in &prepared {
            let object = record.as_object().ok_or_else(|| {
                Error::InvalidInput("load expects an array of JSON objects".to_owned())
            })?;
            let id = extract_id(object, id_field)?;
            let key = self.key(id);

            match self.storage_type() {
                StorageType::Json => {
                    let payload = serde_json::to_string(record)?;
                    let (): () = redis::cmd("JSON.SET")
                        .arg(&key)
                        .arg("$")
                        .arg(payload)
                        .query(&mut connection)?;
                }
                StorageType::Hash => {
                    let encoded = encode_hash_record(object, &self.schema)?;
                    let mut cmd = redis::cmd("HSET");
                    cmd.arg(&key);
                    for (field, value) in encoded {
                        cmd.arg(field);
                        match value {
                            EncodedHashValue::String(value) => {
                                cmd.arg(value);
                            }
                            EncodedHashValue::Binary(value) => {
                                cmd.arg(value);
                            }
                        }
                    }
                    let _: i32 = cmd.query(&mut connection)?;
                }
            }

            if let Some(ttl) = ttl {
                let _: bool = redis::cmd("EXPIRE")
                    .arg(&key)
                    .arg(ttl)
                    .query(&mut connection)?;
            }

            written_keys.push(key);
        }

        Ok(written_keys)
    }

    /// Loads documents using caller-supplied Redis keys instead of generating
    /// keys from the index prefix.
    ///
    /// This mirrors the Python `index.load(data, keys=keys)` signature and is
    /// essential for multi-prefix indexes where documents must be written under
    /// different key prefixes.
    ///
    /// `keys` and `data` must have the same length.
    pub fn load_with_keys(
        &self,
        data: &[Value],
        keys: &[String],
        ttl: Option<i64>,
    ) -> Result<Vec<String>> {
        if data.len() != keys.len() {
            return Err(Error::InvalidInput(format!(
                "data length ({}) must equal keys length ({})",
                data.len(),
                keys.len()
            )));
        }

        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;

        for (record, key) in data.iter().zip(keys.iter()) {
            let object = record.as_object().ok_or_else(|| {
                Error::InvalidInput("load expects an array of JSON objects".to_owned())
            })?;

            match self.storage_type() {
                StorageType::Json => {
                    let payload = serde_json::to_string(record)?;
                    let (): () = redis::cmd("JSON.SET")
                        .arg(key)
                        .arg("$")
                        .arg(payload)
                        .query(&mut connection)?;
                }
                StorageType::Hash => {
                    let encoded = encode_hash_record(object, &self.schema)?;
                    let mut cmd = redis::cmd("HSET");
                    cmd.arg(key);
                    for (field, value) in encoded {
                        cmd.arg(field);
                        match value {
                            EncodedHashValue::String(value) => {
                                cmd.arg(value);
                            }
                            EncodedHashValue::Binary(value) => {
                                cmd.arg(value);
                            }
                        }
                    }
                    let _: i32 = cmd.query(&mut connection)?;
                }
            }

            if let Some(ttl) = ttl {
                let _: bool = redis::cmd("EXPIRE")
                    .arg(key)
                    .arg(ttl)
                    .query(&mut connection)?;
            }
        }

        Ok(keys.to_vec())
    }

    /// Fetches a JSON document as raw JSON text.
    pub fn fetch_json_raw(&self, key_suffix: &str) -> Result<String> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let value = redis::cmd("JSON.GET")
            .arg(self.key(key_suffix))
            .arg("$")
            .query(&mut connection)?;
        Ok(value)
    }

    /// Fetches a Redis hash as a string map.
    pub fn fetch_hash(&self, key_suffix: &str) -> Result<HashMap<String, String>> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let value = connection.hgetall(self.key(key_suffix))?;
        Ok(value)
    }

    /// Fetches a document by its logical identifier.
    pub fn fetch(&self, id: &str) -> Result<Option<Value>> {
        match self.storage_type() {
            StorageType::Json => {
                let raw = self.fetch_json_raw(id);
                match raw {
                    Ok(raw) => {
                        let parsed = serde_json::from_str::<Value>(&raw)?;
                        Ok(match parsed {
                            Value::Array(mut values) if values.len() == 1 => values.pop(),
                            other => Some(other),
                        })
                    }
                    Err(Error::Redis(err))
                        if err.kind() == redis::ErrorKind::UnexpectedReturnType =>
                    {
                        Ok(None)
                    }
                    Err(other) => Err(other),
                }
            }
            StorageType::Hash => {
                let map = self.fetch_hash(id)?;
                if map.is_empty() {
                    Ok(None)
                } else {
                    let mut object = Map::new();
                    for (key, value) in map {
                        object.insert(key, Value::String(value));
                    }
                    Ok(Some(Value::Object(object)))
                }
            }
        }
    }

    /// Drops a single Redis key.
    pub fn drop_key(&self, key: &str) -> Result<usize> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let count: usize = redis::cmd("DEL").arg(key).query(&mut connection)?;
        Ok(count)
    }

    /// Drops multiple Redis keys.
    pub fn drop_keys(&self, keys: &[String]) -> Result<usize> {
        if keys.is_empty() {
            return Ok(0);
        }
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut cmd = redis::cmd("DEL");
        for key in keys {
            cmd.arg(key);
        }
        let count: usize = cmd.query(&mut connection)?;
        Ok(count)
    }

    /// Drops a single logical document by identifier.
    pub fn drop_document(&self, id: &str) -> Result<usize> {
        self.drop_key(&self.key(id))
    }

    /// Drops multiple logical documents by identifier.
    pub fn drop_documents(&self, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let keys = ids.iter().map(|id| self.key(id)).collect::<Vec<_>>();
        self.drop_keys(&keys)
    }

    /// Applies a TTL to a single key.
    pub fn expire_key(&self, key: &str, ttl_seconds: i64) -> Result<bool> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let applied: bool = redis::cmd("EXPIRE")
            .arg(key)
            .arg(ttl_seconds)
            .query(&mut connection)?;
        Ok(applied)
    }

    /// Applies a TTL to multiple keys.
    pub fn expire_keys(&self, keys: &[String], ttl_seconds: i64) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            results.push(self.expire_key(key, ttl_seconds)?);
        }
        Ok(results)
    }

    /// Clears keys matching the index prefix while keeping the index itself.
    pub fn clear(&self) -> Result<usize> {
        let mut total_deleted = 0;
        let query = crate::query::FilterQuery::new(FilterExpression::MatchAll).paging(0, 500);

        loop {
            let batch = self.search(&query)?;
            if batch.docs.is_empty() {
                break;
            }

            let keys = batch
                .docs
                .iter()
                .map(|doc| doc.id().to_owned())
                .collect::<Vec<_>>();
            total_deleted += self.drop_keys(&keys)?;
        }

        Ok(total_deleted)
    }

    /// Executes a query and returns the parsed Redis Search response.
    pub fn search<Q>(&self, query: &Q) -> Result<SearchResult>
    where
        Q: QueryString + ?Sized,
    {
        parse_search_result(self.search_raw(query)?)
    }

    /// Executes a query and returns processed documents or a count, depending on
    /// the query type.
    pub fn query<Q>(&self, query: &Q) -> Result<QueryOutput>
    where
        Q: QueryString + ?Sized,
    {
        let results = self.search(query)?;
        process_search_result(results, query, self.storage_type())
    }

    /// Executes multiple queries in order and returns parsed Redis Search responses.
    pub fn batch_search<'a, I, Q>(&self, queries: I) -> Result<Vec<SearchResult>>
    where
        I: IntoIterator<Item = &'a Q>,
        Q: QueryString + ?Sized + 'a,
    {
        self.batch_search_with_size(queries, usize::MAX)
    }

    /// Executes multiple queries in order, processing them in fixed-size chunks.
    pub fn batch_search_with_size<'a, I, Q>(
        &self,
        queries: I,
        batch_size: usize,
    ) -> Result<Vec<SearchResult>>
    where
        I: IntoIterator<Item = &'a Q>,
        Q: QueryString + ?Sized + 'a,
    {
        if batch_size == 0 {
            return Err(Error::InvalidInput(
                "batch_size must be greater than zero".to_owned(),
            ));
        }

        let queries = queries.into_iter().collect::<Vec<_>>();
        let mut results = Vec::with_capacity(queries.len());
        for chunk in queries.chunks(batch_size) {
            for query in chunk {
                results.push(self.search(*query)?);
            }
        }
        Ok(results)
    }

    /// Executes multiple queries in order and processes each result according to
    /// the corresponding query type.
    pub fn batch_query<'a, I, Q>(&self, queries: I) -> Result<Vec<QueryOutput>>
    where
        I: IntoIterator<Item = &'a Q>,
        Q: QueryString + ?Sized + 'a,
    {
        self.batch_query_with_size(queries, usize::MAX)
    }

    /// Executes multiple queries in order, processing them in fixed-size chunks.
    pub fn batch_query_with_size<'a, I, Q>(
        &self,
        queries: I,
        batch_size: usize,
    ) -> Result<Vec<QueryOutput>>
    where
        I: IntoIterator<Item = &'a Q>,
        Q: QueryString + ?Sized + 'a,
    {
        if batch_size == 0 {
            return Err(Error::InvalidInput(
                "batch_size must be greater than zero".to_owned(),
            ));
        }

        let queries = queries.into_iter().collect::<Vec<_>>();
        let mut results = Vec::with_capacity(queries.len());
        for chunk in queries.chunks(batch_size) {
            for query in chunk {
                results.push(self.query(*query)?);
            }
        }
        Ok(results)
    }

    /// Executes a query in successive pages and returns the processed document batches.
    pub fn paginate<Q>(&self, query: &Q, page_size: usize) -> Result<Vec<Vec<Map<String, Value>>>>
    where
        Q: PageableQuery + ?Sized,
    {
        if page_size == 0 {
            return Err(Error::InvalidInput(
                "page_size must be greater than zero".to_owned(),
            ));
        }

        let mut offset = 0;
        let mut batches = Vec::new();
        loop {
            let page = query.paged(offset, page_size);
            let documents = match self.query(&page)? {
                QueryOutput::Documents(documents) => documents,
                QueryOutput::Count(_) => {
                    return Err(Error::InvalidInput(
                        "paginate requires a document-returning query".to_owned(),
                    ));
                }
            };

            if documents.is_empty() {
                break;
            }

            let fetched = documents.len();
            batches.push(documents);
            if fetched < page_size {
                break;
            }
            offset += page_size;
        }

        Ok(batches)
    }

    /// Executes a [`crate::query::HybridQuery`] via `FT.HYBRID` and returns processed
    /// documents.
    ///
    /// Requires Redis 8.4.0+ with the hybrid search capability.
    ///
    /// FT.HYBRID returns a distinct response format (map-like with
    /// `total_results`, `results`, `warnings`, `execution_time`) that differs
    /// from the FT.SEARCH array format. This method uses
    /// `parse_hybrid_result` to decode it.
    pub fn hybrid_query(&self, query: &crate::query::HybridQuery<'_>) -> Result<QueryOutput> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let cmd = query.build_cmd(self.name());
        let value: redis::Value = cmd.query(&mut connection)?;
        let documents = parse_hybrid_result(value)?;
        Ok(QueryOutput::Documents(documents))
    }

    /// Executes an [`crate::query::AggregateHybridQuery`] via `FT.AGGREGATE` and returns
    /// processed documents.
    ///
    /// Mirrors the Python `_aggregate()` code path.
    pub fn aggregate_query(
        &self,
        query: &crate::query::AggregateHybridQuery<'_>,
    ) -> Result<QueryOutput> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let cmd = query.build_aggregate_cmd(self.name());
        let value: redis::Value = cmd.query(&mut connection)?;
        let documents = parse_aggregate_result(value)?;
        Ok(QueryOutput::Documents(documents))
    }

    /// Executes an [`crate::query::SQLQuery`] and automatically dispatches to
    /// `FT.SEARCH` or `FT.AGGREGATE` depending on the SQL statement.
    ///
    /// Aggregate queries (`COUNT`, `SUM`, `GROUP BY`, etc.) are translated to
    /// `FT.AGGREGATE` commands. All other queries use the regular `FT.SEARCH`
    /// path.
    ///
    /// This mirrors the Python `SearchIndex.query(SQLQuery(...))` behavior.
    #[cfg(feature = "sql")]
    pub fn sql_query(&self, query: &crate::query::SQLQuery) -> Result<QueryOutput> {
        // Geo aggregate (geo_distance in SELECT) → FT.AGGREGATE.
        if let Some(cmd) = query.build_geo_aggregate_cmd(self.name()) {
            let client = self.connection.client()?;
            let mut connection = client.get_connection()?;
            let value: redis::Value = cmd.query(&mut connection)?;
            let documents = parse_aggregate_result(value)?;
            return Ok(QueryOutput::Documents(documents));
        }
        // Standard aggregate (COUNT, SUM, GROUP BY, etc.) → FT.AGGREGATE.
        if let Some(cmd) = query.build_aggregate_cmd(self.name()) {
            let client = self.connection.client()?;
            let mut connection = client.get_connection()?;
            let value: redis::Value = cmd.query(&mut connection)?;
            let documents = parse_aggregate_result(value)?;
            return Ok(QueryOutput::Documents(documents));
        }
        // Vector and geo WHERE queries use the regular FT.SEARCH path
        // (QueryString implementation handles KNN + PARAMS / GEOFILTER).
        self.query(query)
    }

    /// Executes a [`crate::query::MultiVectorQuery`] via `FT.AGGREGATE` and returns
    /// processed documents.
    pub fn multi_vector_query(
        &self,
        query: &crate::query::MultiVectorQuery<'_>,
    ) -> Result<QueryOutput> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let cmd = query.build_aggregate_cmd(self.name());
        let value: redis::Value = cmd.query(&mut connection)?;
        let documents = parse_aggregate_result(value)?;
        Ok(QueryOutput::Documents(documents))
    }

    /// Constructs a [`SearchIndex`] from an existing Redis index by reading
    /// `FT.INFO` and reconstructing the schema.
    ///
    /// Mirrors Python `SearchIndex.from_existing(name, redis_url=...)`.
    pub fn from_existing(name: &str, redis_url: impl Into<String>) -> Result<Self> {
        let connection = RedisConnectionInfo::new(redis_url);
        let client = connection.client()?;
        let mut conn = client.get_connection()?;
        let value = redis::cmd("FT.INFO").arg(name).query(&mut conn)?;
        let info = parse_info_response(value)?;
        let schema = schema_from_info(name, &info)?;
        Ok(Self { schema, connection })
    }

    /// Executes a query and returns the raw Redis response.
    pub fn search_raw<Q>(&self, query: &Q) -> Result<redis::Value>
    where
        Q: QueryString + ?Sized,
    {
        self.validate_query(query)?;
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let value = self.search_cmd(query).query(&mut connection)?;
        Ok(value)
    }

    fn search_cmd<Q>(&self, query: &Q) -> redis::Cmd
    where
        Q: QueryString + ?Sized,
    {
        let render = query.render();
        let mut cmd = redis::cmd("FT.SEARCH");
        cmd.arg(&self.schema.index.name).arg(render.query_string);

        if let Some(scorer) = render.scorer {
            cmd.arg("SCORER").arg(scorer);
        }

        if !render.params.is_empty() {
            cmd.arg("PARAMS").arg(render.params.len() * 2);
            for param in render.params {
                cmd.arg(param.name);
                match param.value {
                    QueryParamValue::String(value) => {
                        cmd.arg(value);
                    }
                    QueryParamValue::Binary(value) => {
                        cmd.arg(value);
                    }
                }
            }
        }

        if render.no_content {
            cmd.arg("NOCONTENT");
        }

        if !render.return_fields.is_empty() {
            cmd.arg("RETURN").arg(render.return_fields.len());
            for field in render.return_fields {
                cmd.arg(field);
            }
        }

        if let Some(sort_by) = render.sort_by {
            cmd.arg("SORTBY").arg(sort_by.field);
            cmd.arg(match sort_by.direction {
                SortDirection::Asc => "ASC",
                SortDirection::Desc => "DESC",
            });
        }

        if render.in_order {
            cmd.arg("INORDER");
        }

        if let Some(limit) = render.limit {
            cmd.arg("LIMIT").arg(limit.offset).arg(limit.num);
        }

        if let Some(geofilter) = render.geofilter {
            cmd.arg("GEOFILTER")
                .arg(geofilter.field)
                .arg(geofilter.lon)
                .arg(geofilter.lat)
                .arg(geofilter.radius)
                .arg(geofilter.unit);
        }

        cmd.arg("DIALECT").arg(render.dialect);
        cmd
    }

    fn validate_query<Q>(&self, query: &Q) -> Result<()>
    where
        Q: QueryString + ?Sized,
    {
        let render = query.render();
        if render.query_string.contains("EF_RUNTIME") {
            let supports_ef_runtime = self.schema.fields.iter().any(|field| {
                matches!(
                    &field.kind,
                    FieldKind::Vector { attrs }
                        if matches!(attrs.algorithm, VectorAlgorithm::Hnsw)
                )
            });
            if !supports_ef_runtime {
                return Err(Error::SchemaValidation(
                    "EF_RUNTIME requires an HNSW vector field".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Async search index client.
#[derive(Debug, Clone)]
pub struct AsyncSearchIndex {
    schema: IndexSchema,
    connection: RedisConnectionInfo,
}

impl AsyncSearchIndex {
    /// Creates a new async search index.
    pub fn new(schema: IndexSchema, redis_url: impl Into<String>) -> Self {
        Self {
            schema,
            connection: RedisConnectionInfo::new(redis_url),
        }
    }

    /// Creates an async index from a YAML string.
    pub fn from_yaml_str(yaml: &str, redis_url: impl Into<String>) -> Result<Self> {
        Ok(Self::new(IndexSchema::from_yaml_str(yaml)?, redis_url))
    }

    /// Creates an async index from a YAML file.
    pub fn from_yaml_file(
        path: impl AsRef<std::path::Path>,
        redis_url: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self::new(IndexSchema::from_yaml_file(path)?, redis_url))
    }

    /// Creates an async index from a JSON value.
    pub fn from_json_value(value: serde_json::Value, redis_url: impl Into<String>) -> Result<Self> {
        Ok(Self::new(IndexSchema::from_json_value(value)?, redis_url))
    }

    /// Returns the schema attached to the index.
    pub fn schema(&self) -> &IndexSchema {
        &self.schema
    }

    /// Returns the Redis index name.
    pub fn name(&self) -> &str {
        &self.schema.index.name
    }

    /// Returns the first (or only) key prefix.
    ///
    /// For multi-prefix indexes, returns the first prefix. Use [`Self::prefixes`]
    /// to access all configured prefixes.
    pub fn prefix(&self) -> &str {
        self.schema.index.prefix.first()
    }

    /// Returns all key prefixes configured for this index.
    pub fn prefixes(&self) -> Vec<&str> {
        self.schema.index.prefix.all()
    }

    /// Returns the separator between prefix and identifier.
    pub fn key_separator(&self) -> &str {
        &self.schema.index.key_separator
    }

    /// Returns the storage type.
    pub fn storage_type(&self) -> StorageType {
        self.schema.index.storage_type
    }

    /// Builds a document key from the index prefix and identifier.
    pub fn key(&self, key_suffix: &str) -> String {
        compose_key(self.prefix(), self.key_separator(), key_suffix)
    }

    /// Creates the Redis search index asynchronously.
    pub async fn create(&self) -> Result<()> {
        self.create_with_options(false, false).await
    }

    /// Creates the Redis search index asynchronously with overwrite controls.
    pub async fn create_with_options(&self, overwrite: bool, drop_documents: bool) -> Result<()> {
        if self.schema.fields.is_empty() {
            return Err(Error::SchemaValidation(
                "No fields defined for index".to_owned(),
            ));
        }

        if self.exists().await? {
            if !overwrite {
                return Ok(());
            }
            self.drop(drop_documents).await?;
        }

        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let (): () = SearchIndex::new(self.schema.clone(), self.connection.redis_url.clone())
            .create_cmd()
            .query_async(&mut connection)
            .await?;
        Ok(())
    }

    /// Drops the Redis search index asynchronously.
    pub async fn drop(&self, delete_documents: bool) -> Result<()> {
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let mut cmd = redis::cmd("FT.DROPINDEX");
        cmd.arg(&self.schema.index.name);
        if delete_documents {
            cmd.arg("DD");
        }
        let (): () = cmd.query_async(&mut connection).await?;
        Ok(())
    }

    /// Deletes the search index asynchronously, optionally dropping associated documents.
    pub async fn delete(&self, drop_documents: bool) -> Result<()> {
        if !self.exists().await? {
            return Err(Error::InvalidInput(format!(
                "index '{}' does not exist",
                self.schema.index.name
            )));
        }
        self.drop(drop_documents).await
    }

    /// Returns whether the Redis search index exists.
    pub async fn exists(&self) -> Result<bool> {
        Ok(self
            .listall()
            .await?
            .iter()
            .any(|name| name == &self.schema.index.name))
    }

    /// Lists all Redis search indices asynchronously.
    pub async fn listall(&self) -> Result<Vec<String>> {
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let value = redis::cmd("FT._LIST").query_async(&mut connection).await?;
        Ok(value)
    }

    /// Returns parsed Redis index metadata.
    pub async fn info(&self) -> Result<Map<String, Value>> {
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let value = redis::cmd("FT.INFO")
            .arg(&self.schema.index.name)
            .query_async(&mut connection)
            .await?;
        parse_info_response(value)
    }

    /// Loads documents using the configured storage type and returns the written keys.
    pub async fn load(
        &self,
        data: &[Value],
        id_field: &str,
        ttl: Option<i64>,
    ) -> Result<Vec<String>> {
        self.load_with_preprocess(data, id_field, ttl, |record| Ok(record.clone()))
            .await
    }

    /// Loads documents asynchronously after applying a preprocessing callback to each record.
    pub async fn load_with_preprocess<F>(
        &self,
        data: &[Value],
        id_field: &str,
        ttl: Option<i64>,
        mut preprocess: F,
    ) -> Result<Vec<String>>
    where
        F: FnMut(&Value) -> Result<Value>,
    {
        let prepared = prepare_load_records(data, &mut preprocess)?;
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let mut written_keys = Vec::with_capacity(prepared.len());

        for record in &prepared {
            let object = record.as_object().ok_or_else(|| {
                Error::InvalidInput("load expects an array of JSON objects".to_owned())
            })?;
            let id = extract_id(object, id_field)?;
            let key = self.key(id);

            match self.storage_type() {
                StorageType::Json => {
                    let payload = serde_json::to_string(record)?;
                    let (): () = redis::cmd("JSON.SET")
                        .arg(&key)
                        .arg("$")
                        .arg(payload)
                        .query_async(&mut connection)
                        .await?;
                }
                StorageType::Hash => {
                    let encoded = encode_hash_record(object, &self.schema)?;
                    let mut cmd = redis::cmd("HSET");
                    cmd.arg(&key);
                    for (field, value) in encoded {
                        cmd.arg(field);
                        match value {
                            EncodedHashValue::String(value) => {
                                cmd.arg(value);
                            }
                            EncodedHashValue::Binary(value) => {
                                cmd.arg(value);
                            }
                        }
                    }
                    let _: i32 = cmd.query_async(&mut connection).await?;
                }
            }

            if let Some(ttl) = ttl {
                let _: bool = redis::cmd("EXPIRE")
                    .arg(&key)
                    .arg(ttl)
                    .query_async(&mut connection)
                    .await?;
            }

            written_keys.push(key);
        }

        Ok(written_keys)
    }

    /// Loads documents using caller-supplied Redis keys instead of generating
    /// keys from the index prefix.
    ///
    /// This mirrors the Python `index.load(data, keys=keys)` signature and is
    /// essential for multi-prefix indexes where documents must be written under
    /// different key prefixes.
    ///
    /// `keys` and `data` must have the same length.
    pub async fn load_with_keys(
        &self,
        data: &[Value],
        keys: &[String],
        ttl: Option<i64>,
    ) -> Result<Vec<String>> {
        if data.len() != keys.len() {
            return Err(Error::InvalidInput(format!(
                "data length ({}) must equal keys length ({})",
                data.len(),
                keys.len()
            )));
        }

        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;

        for (record, key) in data.iter().zip(keys.iter()) {
            let object = record.as_object().ok_or_else(|| {
                Error::InvalidInput("load expects an array of JSON objects".to_owned())
            })?;

            match self.storage_type() {
                StorageType::Json => {
                    let payload = serde_json::to_string(record)?;
                    let (): () = redis::cmd("JSON.SET")
                        .arg(key)
                        .arg("$")
                        .arg(payload)
                        .query_async(&mut connection)
                        .await?;
                }
                StorageType::Hash => {
                    let encoded = encode_hash_record(object, &self.schema)?;
                    let mut cmd = redis::cmd("HSET");
                    cmd.arg(key);
                    for (field, value) in encoded {
                        cmd.arg(field);
                        match value {
                            EncodedHashValue::String(value) => {
                                cmd.arg(value);
                            }
                            EncodedHashValue::Binary(value) => {
                                cmd.arg(value);
                            }
                        }
                    }
                    let _: i32 = cmd.query_async(&mut connection).await?;
                }
            }

            if let Some(ttl) = ttl {
                let _: bool = redis::cmd("EXPIRE")
                    .arg(key)
                    .arg(ttl)
                    .query_async(&mut connection)
                    .await?;
            }
        }

        Ok(keys.to_vec())
    }

    /// Fetches a document by its logical identifier.
    pub async fn fetch(&self, id: &str) -> Result<Option<Value>> {
        match self.storage_type() {
            StorageType::Json => {
                let client = self.connection.client()?;
                let mut connection = client.get_multiplexed_async_connection().await?;
                let raw: std::result::Result<String, redis::RedisError> = redis::cmd("JSON.GET")
                    .arg(self.key(id))
                    .arg("$")
                    .query_async(&mut connection)
                    .await;
                match raw {
                    Ok(raw) => {
                        let parsed = serde_json::from_str::<Value>(&raw)?;
                        Ok(match parsed {
                            Value::Array(mut values) if values.len() == 1 => values.pop(),
                            other => Some(other),
                        })
                    }
                    Err(err) if err.kind() == redis::ErrorKind::UnexpectedReturnType => Ok(None),
                    Err(err) => Err(Error::Redis(err)),
                }
            }
            StorageType::Hash => {
                let client = self.connection.client()?;
                let mut connection = client.get_multiplexed_async_connection().await?;
                let map: HashMap<String, String> = redis::cmd("HGETALL")
                    .arg(self.key(id))
                    .query_async(&mut connection)
                    .await?;
                if map.is_empty() {
                    Ok(None)
                } else {
                    let mut object = Map::new();
                    for (key, value) in map {
                        object.insert(key, Value::String(value));
                    }
                    Ok(Some(Value::Object(object)))
                }
            }
        }
    }

    /// Drops a single Redis key.
    pub async fn drop_key(&self, key: &str) -> Result<usize> {
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let count: usize = redis::cmd("DEL")
            .arg(key)
            .query_async(&mut connection)
            .await?;
        Ok(count)
    }

    /// Drops multiple Redis keys.
    pub async fn drop_keys(&self, keys: &[String]) -> Result<usize> {
        if keys.is_empty() {
            return Ok(0);
        }
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let mut cmd = redis::cmd("DEL");
        for key in keys {
            cmd.arg(key);
        }
        let count: usize = cmd.query_async(&mut connection).await?;
        Ok(count)
    }

    /// Drops a single logical document by identifier.
    pub async fn drop_document(&self, id: &str) -> Result<usize> {
        self.drop_key(&self.key(id)).await
    }

    /// Drops multiple logical documents by identifier.
    pub async fn drop_documents(&self, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let keys = ids.iter().map(|id| self.key(id)).collect::<Vec<_>>();
        self.drop_keys(&keys).await
    }

    /// Applies a TTL to a single key.
    pub async fn expire_key(&self, key: &str, ttl_seconds: i64) -> Result<bool> {
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let applied: bool = redis::cmd("EXPIRE")
            .arg(key)
            .arg(ttl_seconds)
            .query_async(&mut connection)
            .await?;
        Ok(applied)
    }

    /// Applies a TTL to multiple keys.
    pub async fn expire_keys(&self, keys: &[String], ttl_seconds: i64) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            results.push(self.expire_key(key, ttl_seconds).await?);
        }
        Ok(results)
    }

    /// Clears keys matching the index prefix while keeping the index itself.
    pub async fn clear(&self) -> Result<usize> {
        let mut total_deleted = 0;
        let query = crate::query::FilterQuery::new(FilterExpression::MatchAll).paging(0, 500);

        loop {
            let batch = self.search(&query).await?;
            if batch.docs.is_empty() {
                break;
            }

            let keys = batch
                .docs
                .iter()
                .map(|doc| doc.id().to_owned())
                .collect::<Vec<_>>();
            total_deleted += self.drop_keys(&keys).await?;
        }

        Ok(total_deleted)
    }

    /// Executes a query asynchronously.
    pub async fn search<Q>(&self, query: &Q) -> Result<SearchResult>
    where
        Q: QueryString + Send + Sync + ?Sized,
    {
        parse_search_result(self.search_raw(query).await?)
    }

    /// Executes a query asynchronously and returns processed documents or a count.
    pub async fn query<Q>(&self, query: &Q) -> Result<QueryOutput>
    where
        Q: QueryString + Send + Sync + ?Sized,
    {
        let results = self.search(query).await?;
        process_search_result(results, query, self.schema.index.storage_type)
    }

    /// Executes multiple queries asynchronously in order.
    pub async fn batch_search<'a, I, Q>(&self, queries: I) -> Result<Vec<SearchResult>>
    where
        I: IntoIterator<Item = &'a Q>,
        Q: QueryString + Send + Sync + ?Sized + 'a,
    {
        self.batch_search_with_size(queries, usize::MAX).await
    }

    /// Executes multiple queries asynchronously in fixed-size chunks.
    pub async fn batch_search_with_size<'a, I, Q>(
        &self,
        queries: I,
        batch_size: usize,
    ) -> Result<Vec<SearchResult>>
    where
        I: IntoIterator<Item = &'a Q>,
        Q: QueryString + Send + Sync + ?Sized + 'a,
    {
        if batch_size == 0 {
            return Err(Error::InvalidInput(
                "batch_size must be greater than zero".to_owned(),
            ));
        }

        let queries = queries.into_iter().collect::<Vec<_>>();
        let mut results = Vec::with_capacity(queries.len());
        for chunk in queries.chunks(batch_size) {
            for query in chunk {
                results.push(parse_search_result(self.search_raw(*query).await?)?);
            }
        }
        Ok(results)
    }

    /// Executes multiple queries asynchronously in order and processes each result.
    pub async fn batch_query<'a, I, Q>(&self, queries: I) -> Result<Vec<QueryOutput>>
    where
        I: IntoIterator<Item = &'a Q>,
        Q: QueryString + Send + Sync + ?Sized + 'a,
    {
        self.batch_query_with_size(queries, usize::MAX).await
    }

    /// Executes multiple queries asynchronously in fixed-size chunks.
    pub async fn batch_query_with_size<'a, I, Q>(
        &self,
        queries: I,
        batch_size: usize,
    ) -> Result<Vec<QueryOutput>>
    where
        I: IntoIterator<Item = &'a Q>,
        Q: QueryString + Send + Sync + ?Sized + 'a,
    {
        if batch_size == 0 {
            return Err(Error::InvalidInput(
                "batch_size must be greater than zero".to_owned(),
            ));
        }

        let queries = queries.into_iter().collect::<Vec<_>>();
        let mut results = Vec::with_capacity(queries.len());
        for chunk in queries.chunks(batch_size) {
            for query in chunk {
                let parsed = parse_search_result(self.search_raw(*query).await?)?;
                results.push(process_search_result(
                    parsed,
                    *query,
                    self.schema.index.storage_type,
                )?);
            }
        }
        Ok(results)
    }

    /// Executes a query asynchronously in successive pages.
    pub async fn paginate<Q>(
        &self,
        query: &Q,
        page_size: usize,
    ) -> Result<Vec<Vec<Map<String, Value>>>>
    where
        Q: PageableQuery + Send + Sync + ?Sized,
    {
        if page_size == 0 {
            return Err(Error::InvalidInput(
                "page_size must be greater than zero".to_owned(),
            ));
        }

        let mut offset = 0;
        let mut batches = Vec::new();
        loop {
            let page = query.paged(offset, page_size);
            let documents = match self.query(&page).await? {
                QueryOutput::Documents(documents) => documents,
                QueryOutput::Count(_) => {
                    return Err(Error::InvalidInput(
                        "paginate requires a document-returning query".to_owned(),
                    ));
                }
            };

            if documents.is_empty() {
                break;
            }

            let fetched = documents.len();
            batches.push(documents);
            if fetched < page_size {
                break;
            }
            offset += page_size;
        }

        Ok(batches)
    }

    /// Executes a query asynchronously and returns the raw Redis response.
    pub async fn search_raw<Q>(&self, query: &Q) -> Result<redis::Value>
    where
        Q: QueryString + Send + Sync + ?Sized,
    {
        let sync_index = SearchIndex::new(self.schema.clone(), self.connection.redis_url.clone());
        sync_index.validate_query(query)?;
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let value = sync_index
            .search_cmd(query)
            .query_async(&mut connection)
            .await?;
        Ok(value)
    }

    /// Executes a [`crate::query::HybridQuery`] asynchronously via `FT.HYBRID` and returns
    /// processed documents.
    ///
    /// Requires Redis 8.4.0+ with the hybrid search capability.
    pub async fn hybrid_query(&self, query: &crate::query::HybridQuery<'_>) -> Result<QueryOutput> {
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let cmd = query.build_cmd(self.name());
        let value: redis::Value = cmd.query_async(&mut connection).await?;
        let documents = parse_hybrid_result(value)?;
        Ok(QueryOutput::Documents(documents))
    }

    /// Executes an [`crate::query::AggregateHybridQuery`] asynchronously via `FT.AGGREGATE`
    /// and returns processed documents.
    pub async fn aggregate_query(
        &self,
        query: &crate::query::AggregateHybridQuery<'_>,
    ) -> Result<QueryOutput> {
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let cmd = query.build_aggregate_cmd(self.name());
        let value: redis::Value = cmd.query_async(&mut connection).await?;
        let documents = parse_aggregate_result(value)?;
        Ok(QueryOutput::Documents(documents))
    }

    /// Executes an [`crate::query::SQLQuery`] asynchronously and automatically dispatches
    /// to `FT.SEARCH` or `FT.AGGREGATE` depending on the SQL statement.
    ///
    /// This mirrors the Python `AsyncSearchIndex.query(SQLQuery(...))` behavior.
    #[cfg(feature = "sql")]
    pub async fn sql_query(&self, query: &crate::query::SQLQuery) -> Result<QueryOutput> {
        // Geo aggregate (geo_distance in SELECT) → FT.AGGREGATE.
        if let Some(cmd) = query.build_geo_aggregate_cmd(self.name()) {
            let client = self.connection.client()?;
            let mut connection = client.get_multiplexed_async_connection().await?;
            let value: redis::Value = cmd.query_async(&mut connection).await?;
            let documents = parse_aggregate_result(value)?;
            return Ok(QueryOutput::Documents(documents));
        }
        // Standard aggregate (COUNT, SUM, GROUP BY, etc.) → FT.AGGREGATE.
        if let Some(cmd) = query.build_aggregate_cmd(self.name()) {
            let client = self.connection.client()?;
            let mut connection = client.get_multiplexed_async_connection().await?;
            let value: redis::Value = cmd.query_async(&mut connection).await?;
            let documents = parse_aggregate_result(value)?;
            return Ok(QueryOutput::Documents(documents));
        }
        // Vector and geo WHERE queries use the regular FT.SEARCH path.
        self.query(query).await
    }

    /// Executes a [`crate::query::MultiVectorQuery`] asynchronously via `FT.AGGREGATE` and
    /// returns processed documents.
    pub async fn multi_vector_query(
        &self,
        query: &crate::query::MultiVectorQuery<'_>,
    ) -> Result<QueryOutput> {
        let client = self.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let cmd = query.build_aggregate_cmd(self.name());
        let value: redis::Value = cmd.query_async(&mut connection).await?;
        let documents = parse_aggregate_result(value)?;
        Ok(QueryOutput::Documents(documents))
    }

    /// Constructs an [`AsyncSearchIndex`] from an existing Redis index by
    /// reading `FT.INFO` and reconstructing the schema.
    ///
    /// Mirrors Python `AsyncSearchIndex.from_existing(name, redis_url=...)`.
    pub async fn from_existing(name: &str, redis_url: impl Into<String>) -> Result<Self> {
        let connection = RedisConnectionInfo::new(redis_url);
        let client = connection.client()?;
        let mut conn = client.get_multiplexed_async_connection().await?;
        let value: redis::Value = redis::cmd("FT.INFO")
            .arg(name)
            .query_async(&mut conn)
            .await?;
        let info = parse_info_response(value)?;
        let schema = schema_from_info(name, &info)?;
        Ok(Self { schema, connection })
    }
}

#[allow(dead_code)]
fn _storage_type_for_load(schema: &IndexSchema) -> StorageType {
    schema.index.storage_type
}

fn extract_id<'a>(object: &'a Map<String, Value>, id_field: &str) -> Result<&'a str> {
    object
        .get(id_field)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::InvalidInput(format!("missing string id field '{id_field}'")))
}

fn compose_key(prefix: &str, key_separator: &str, key_suffix: &str) -> String {
    if prefix.is_empty() {
        return key_suffix.to_owned();
    }

    if key_separator.is_empty() {
        return format!("{prefix}{key_suffix}");
    }

    let normalized_prefix = prefix.trim_end_matches(key_separator);
    if normalized_prefix.is_empty() {
        key_suffix.to_owned()
    } else {
        format!("{normalized_prefix}{key_separator}{key_suffix}")
    }
}

enum EncodedHashValue {
    String(String),
    Binary(Vec<u8>),
}

fn encode_hash_record(
    object: &Map<String, Value>,
    schema: &IndexSchema,
) -> Result<Vec<(String, EncodedHashValue)>> {
    let mut pairs = Vec::with_capacity(object.len());
    for (key, value) in object {
        let encoded_value = match value {
            Value::Array(values)
                if matches!(
                    schema.field(key),
                    Some(crate::schema::Field {
                        kind: FieldKind::Vector { .. },
                        ..
                    })
                ) =>
            {
                EncodedHashValue::Binary(encode_vector_hash_field(key, values, schema)?)
            }
            Value::Null => EncodedHashValue::String("null".to_owned()),
            Value::Bool(value) => EncodedHashValue::String(value.to_string()),
            Value::Number(value) => EncodedHashValue::String(value.to_string()),
            Value::String(value) => EncodedHashValue::String(value.clone()),
            Value::Array(_) | Value::Object(_) => {
                EncodedHashValue::String(serde_json::to_string(value)?)
            }
        };
        pairs.push((key.clone(), encoded_value));
    }
    Ok(pairs)
}

fn encode_vector_hash_field(
    field_name: &str,
    values: &[Value],
    schema: &IndexSchema,
) -> Result<Vec<u8>> {
    let Some(crate::schema::Field {
        kind: FieldKind::Vector { attrs },
        ..
    }) = schema.field(field_name)
    else {
        return Err(Error::SchemaValidation(format!(
            "vector field '{field_name}' not found in schema"
        )));
    };

    if values.len() != attrs.dims {
        return Err(Error::InvalidInput(format!(
            "vector field '{field_name}' expected {} elements, received {}",
            attrs.dims,
            values.len()
        )));
    }

    match attrs.datatype {
        crate::schema::VectorDataType::Bfloat16 => {
            let mut buffer = Vec::with_capacity(values.len() * 2);
            for value in values {
                let number = json_number_to_f64(value, field_name)? as f32;
                // BFloat16: upper 16 bits of f32 (truncate mantissa)
                let bits = number.to_bits();
                let bf16 = (bits >> 16) as u16;
                buffer.extend_from_slice(&bf16.to_le_bytes());
            }
            Ok(buffer)
        }
        crate::schema::VectorDataType::Float16 => {
            let mut buffer = Vec::with_capacity(values.len() * 2);
            for value in values {
                let number = json_number_to_f64(value, field_name)? as f32;
                buffer.extend_from_slice(&f32_to_f16_bytes(number).to_le_bytes());
            }
            Ok(buffer)
        }
        crate::schema::VectorDataType::Float32 => {
            let mut buffer = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
            for value in values {
                let number = json_number_to_f64(value, field_name)? as f32;
                buffer.extend_from_slice(&number.to_le_bytes());
            }
            Ok(buffer)
        }
        crate::schema::VectorDataType::Float64 => {
            let mut buffer = Vec::with_capacity(values.len() * std::mem::size_of::<f64>());
            for value in values {
                let number = json_number_to_f64(value, field_name)?;
                buffer.extend_from_slice(&number.to_le_bytes());
            }
            Ok(buffer)
        }
    }
}

fn json_number_to_f64(value: &Value, field_name: &str) -> Result<f64> {
    value.as_f64().ok_or_else(|| {
        Error::InvalidInput(format!(
            "vector field '{field_name}' must be encoded from numeric JSON values"
        ))
    })
}

/// Converts an f32 value to IEEE 754 half-precision (float16) encoded as u16.
fn f32_to_f16_bytes(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 255 {
        // Infinity or NaN
        let m = if mantissa != 0 { 0x0200 } else { 0 };
        return (sign | 0x7C00 | m) as u16;
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        // Overflow → infinity
        return (sign | 0x7C00) as u16;
    }
    if unbiased < -24 {
        // Underflow → zero
        return sign as u16;
    }
    if unbiased < -14 {
        // Subnormal
        let shift = -14 - unbiased;
        let m = (mantissa | 0x0080_0000) >> (shift + 13);
        return (sign | m) as u16;
    }

    let exp16 = ((unbiased + 15) as u32) << 10;
    let m = mantissa >> 13;
    (sign | exp16 | m) as u16
}

fn prepare_load_records<F>(data: &[Value], preprocess: &mut F) -> Result<Vec<Value>>
where
    F: FnMut(&Value) -> Result<Value>,
{
    let mut prepared = Vec::with_capacity(data.len());
    for record in data {
        let processed = preprocess(record)?;
        if !processed.is_object() {
            return Err(Error::InvalidInput(
                "preprocess must return a JSON object".to_owned(),
            ));
        }
        prepared.push(processed);
    }
    Ok(prepared)
}

fn parse_search_result(value: redis::Value) -> Result<SearchResult> {
    let entries = match value {
        redis::Value::Array(entries) => entries,
        redis::Value::Nil => return Ok(SearchResult::new(0, Vec::new())),
        other => {
            return Err(Error::InvalidInput(format!(
                "expected FT.SEARCH array response, received {other:?}"
            )));
        }
    };

    let mut entries = VecDeque::from(entries);
    let total = entries
        .pop_front()
        .map(redis_value_to_usize)
        .transpose()?
        .unwrap_or(0);

    let mut docs = Vec::new();
    while let Some(id_value) = entries.pop_front() {
        let id = redis_value_to_string(&id_value)?;
        let fields = match entries.front() {
            Some(next) if is_search_payload(next) => {
                let payload = entries.pop_front().expect("front element exists");
                decode_search_payload(payload)?
            }
            _ => Map::new(),
        };
        docs.push(SearchDocument::new(id, fields));
    }

    Ok(SearchResult::new(total, docs))
}

/// Parses an `FT.HYBRID` response.
///
/// FT.HYBRID returns a map-like array:
/// ```text
/// ["total_results", <int>, "results", [[field, value, ...], ...], "warnings", [...], "execution_time", "..."]
/// ```
///
/// Each result document is a flat array of `[field, value, field, value, ...]`
/// with no separate document ID.
fn parse_hybrid_result(value: redis::Value) -> Result<Vec<Map<String, Value>>> {
    let entries = match value {
        redis::Value::Array(entries) => entries,
        redis::Value::Nil => return Ok(Vec::new()),
        other => {
            return Err(Error::InvalidInput(format!(
                "expected FT.HYBRID array response, received {other:?}"
            )));
        }
    };

    // Parse the top-level map: walk key-value pairs.
    let mut results_value: Option<redis::Value> = None;
    let mut iter = entries.into_iter();
    while let Some(key) = iter.next() {
        let key_str = redis_value_to_string(&key).unwrap_or_default();
        let val = iter.next();
        match key_str.as_str() {
            "results" => {
                results_value = val;
            }
            _ => {
                // Skip total_results, warnings, execution_time, etc.
            }
        }
    }

    let results_array = match results_value {
        Some(redis::Value::Array(arr)) => arr,
        Some(redis::Value::Nil) | None => return Ok(Vec::new()),
        Some(other) => {
            return Err(Error::InvalidInput(format!(
                "expected results array in FT.HYBRID response, received {other:?}"
            )));
        }
    };

    let mut documents = Vec::with_capacity(results_array.len());
    for entry in results_array {
        match entry {
            redis::Value::Array(pairs) => {
                let mut map = Map::new();
                let mut pair_iter = pairs.into_iter();
                while let Some(field_val) = pair_iter.next() {
                    let field = redis_value_to_string(&field_val)?;
                    if let Some(value_val) = pair_iter.next() {
                        let json_val = redis_value_to_json(value_val)?;
                        // Skip internal fields like __key, __score
                        if !field.starts_with("__") {
                            map.insert(field, json_val);
                        }
                    }
                }
                documents.push(map);
            }
            _ => {
                // Skip non-array entries
            }
        }
    }

    Ok(documents)
}

fn parse_info_response(value: redis::Value) -> Result<Map<String, Value>> {
    let entries = match value {
        redis::Value::Map(entries) => entries,
        redis::Value::Array(entries) => {
            let mut pairs = VecDeque::from(entries);
            let mut mapped = Vec::new();
            while let Some(key) = pairs.pop_front() {
                let Some(value) = pairs.pop_front() else {
                    return Err(Error::InvalidInput(
                        "FT.INFO response contained an odd number of elements".to_owned(),
                    ));
                };
                mapped.push((key, value));
            }
            mapped
        }
        other => {
            return Err(Error::InvalidInput(format!(
                "expected FT.INFO map response, received {other:?}"
            )));
        }
    };

    let mut info = Map::with_capacity(entries.len());
    for (key, value) in entries {
        info.insert(redis_value_to_string(&key)?, redis_value_to_json(value)?);
    }
    Ok(info)
}

/// Parses an `FT.AGGREGATE` response into a list of document maps.
///
/// Redis `FT.AGGREGATE` returns an array: `[total, [field, value, ...], ...]`.
fn parse_aggregate_result(value: redis::Value) -> Result<Vec<Map<String, Value>>> {
    let entries = match value {
        redis::Value::Array(entries) => entries,
        redis::Value::Nil => return Ok(Vec::new()),
        other => {
            return Err(Error::InvalidInput(format!(
                "expected FT.AGGREGATE array response, received {other:?}"
            )));
        }
    };

    let mut it = entries.into_iter();

    // First element is the total count (we skip it for doc processing)
    let _total = it.next();

    let mut documents = Vec::new();
    for row in it {
        let row_entries = match row {
            redis::Value::Array(entries) => entries,
            redis::Value::Map(entries) => entries
                .into_iter()
                .flat_map(|(k, v)| [k, v])
                .collect::<Vec<_>>(),
            _ => continue,
        };

        let mut pairs = VecDeque::from(row_entries);
        let mut map = Map::new();
        while let Some(key) = pairs.pop_front() {
            let Some(val) = pairs.pop_front() else { break };
            let field = redis_value_to_string(&key)?;
            if field == "__score" {
                continue; // Strip internal score like Python
            }
            map.insert(field, redis_value_to_json(val)?);
        }
        documents.push(map);
    }

    Ok(documents)
}

/// Reconstructs an [`IndexSchema`] from parsed `FT.INFO` output.
///
/// Mirrors Python `convert_index_info_to_schema`.
fn schema_from_info(name: &str, info: &Map<String, Value>) -> Result<IndexSchema> {
    // Extract storage type and prefixes from index_definition
    let index_def = info.get("index_definition").and_then(Value::as_array);

    let mut storage_type = StorageType::Hash;
    let mut prefix = crate::schema::Prefix::default();

    if let Some(def_arr) = index_def {
        // index_definition is a flat array: [key, value, key, value, ...]
        let mut i = 0;
        while i + 1 < def_arr.len() {
            let key = def_arr[i].as_str().unwrap_or("");
            match key {
                "key_type" => {
                    if let Some(v) = def_arr[i + 1].as_str() {
                        storage_type = match v.to_uppercase().as_str() {
                            "JSON" => StorageType::Json,
                            _ => StorageType::Hash,
                        };
                    }
                }
                "prefixes" => {
                    if let Some(arr) = def_arr[i + 1].as_array() {
                        let prefixes: Vec<String> = arr
                            .iter()
                            .filter_map(Value::as_str)
                            .map(String::from)
                            .collect();
                        prefix = if prefixes.len() == 1 {
                            crate::schema::Prefix::Single(prefixes.into_iter().next().unwrap())
                        } else {
                            crate::schema::Prefix::Multi(prefixes)
                        };
                    }
                }
                _ => {}
            }
            i += 2;
        }
    }

    // Parse attributes (fields)
    let attributes = info.get("attributes").and_then(Value::as_array);
    let mut fields = Vec::new();

    if let Some(attrs) = attributes {
        for attr_val in attrs {
            let attr_arr = match attr_val.as_array() {
                Some(arr) => arr,
                None => continue,
            };

            if attr_arr.is_empty() {
                continue;
            }

            // Parse the flat attribute array: [identifier, name, ...]
            let mut field_name = String::new();
            let mut field_type = String::new();
            let mut sortable = false;
            let mut no_index = false;
            let mut case_sensitive = false;
            let mut separator: Option<String> = None;
            let mut weight: Option<f32> = None;
            let mut no_stem = false;
            let mut with_suffix_trie = false;
            let mut phonetic: Option<String> = None;
            // Vector attrs
            let mut algorithm = String::new();
            let mut dims: usize = 0;
            let mut distance_metric = String::new();
            let mut datatype = String::new();

            let mut i = 0;
            while i < attr_arr.len() {
                let key = attr_arr[i].as_str().unwrap_or("");
                match key {
                    "identifier" | "attribute" => {
                        if i + 1 < attr_arr.len() {
                            if let Some(v) = attr_arr[i + 1].as_str() {
                                if key == "attribute" || field_name.is_empty() {
                                    field_name = v.to_owned();
                                }
                            }
                        }
                        i += 2;
                    }
                    "type" => {
                        if i + 1 < attr_arr.len() {
                            if let Some(v) = attr_arr[i + 1].as_str() {
                                field_type = v.to_uppercase();
                            }
                        }
                        i += 2;
                    }
                    "SORTABLE" => {
                        sortable = true;
                        i += 1;
                    }
                    "NOINDEX" => {
                        no_index = true;
                        i += 1;
                    }
                    "CASESENSITIVE" => {
                        case_sensitive = true;
                        i += 1;
                    }
                    "NOSTEM" => {
                        no_stem = true;
                        i += 1;
                    }
                    "WITHSUFFIXTRIE" => {
                        with_suffix_trie = true;
                        i += 1;
                    }
                    "SEPARATOR" => {
                        if i + 1 < attr_arr.len() {
                            separator = attr_arr[i + 1].as_str().map(String::from);
                        }
                        i += 2;
                    }
                    "WEIGHT" => {
                        if i + 1 < attr_arr.len() {
                            weight = attr_arr[i + 1]
                                .as_str()
                                .and_then(|s| s.parse::<f32>().ok())
                                .or_else(|| attr_arr[i + 1].as_f64().map(|v| v as f32));
                        }
                        i += 2;
                    }
                    "PHONETIC" => {
                        if i + 1 < attr_arr.len() {
                            phonetic = attr_arr[i + 1].as_str().map(String::from);
                        }
                        i += 2;
                    }
                    _ if field_type == "VECTOR" => {
                        // Once we hit VECTOR type, remaining entries are vector params
                        // Format: algorithm, param_count, key, value, key, value, ...
                        // Or: key, value, key, value, ...
                        let upper = key.to_uppercase();
                        if upper == "FLAT" || upper == "HNSW" {
                            algorithm = upper.to_lowercase();
                            i += 1;
                            // Next might be a param count, skip it
                            if i < attr_arr.len() {
                                if attr_arr[i]
                                    .as_str()
                                    .and_then(|s| s.parse::<usize>().ok())
                                    .is_some()
                                    || attr_arr[i].as_i64().is_some()
                                {
                                    i += 1; // skip count
                                }
                            }
                        } else if upper == "ALGORITHM" {
                            if i + 1 < attr_arr.len() {
                                algorithm =
                                    attr_arr[i + 1].as_str().unwrap_or("flat").to_lowercase();
                            }
                            i += 2;
                        } else if upper == "DIM" || upper == "DIMS" {
                            if i + 1 < attr_arr.len() {
                                dims = attr_arr[i + 1]
                                    .as_str()
                                    .and_then(|s| s.parse().ok())
                                    .or_else(|| attr_arr[i + 1].as_u64().map(|v| v as usize))
                                    .unwrap_or(0);
                            }
                            i += 2;
                        } else if upper == "DISTANCE_METRIC" {
                            if i + 1 < attr_arr.len() {
                                distance_metric =
                                    attr_arr[i + 1].as_str().unwrap_or("cosine").to_lowercase();
                            }
                            i += 2;
                        } else if upper == "TYPE" || upper == "DATA_TYPE" || upper == "DATATYPE" {
                            if i + 1 < attr_arr.len() {
                                datatype =
                                    attr_arr[i + 1].as_str().unwrap_or("float32").to_lowercase();
                            }
                            i += 2;
                        } else {
                            // Skip unknown vector param
                            i += 2;
                        }
                    }
                    _ => {
                        i += 1;
                    }
                }
            }

            // Strip JSON path prefix from field name
            let field_name = field_name
                .strip_prefix("$.")
                .unwrap_or(&field_name)
                .to_owned();

            // Normalize Redis Search defaults back to None so that
            // schemas reconstructed from FT.INFO compare equal to
            // schemas built from JSON/YAML where optional defaults
            // are omitted.  Redis returns separator="," for TAG fields
            // and weight=1 for TEXT fields even when they were not
            // explicitly set during FT.CREATE.
            let separator = separator.filter(|s| s != ",");
            let weight = weight.filter(|w| (*w - 1.0).abs() > f32::EPSILON);

            let kind = match field_type.as_str() {
                "TAG" => FieldKind::Tag {
                    attrs: crate::schema::TagFieldAttributes {
                        separator,
                        case_sensitive,
                        sortable,
                        no_index,
                        index_missing: false,
                        index_empty: false,
                    },
                },
                "TEXT" => FieldKind::Text {
                    attrs: crate::schema::TextFieldAttributes {
                        weight,
                        sortable,
                        no_stem,
                        no_index,
                        phonetic,
                        with_suffix_trie,
                        index_missing: false,
                        index_empty: false,
                    },
                },
                "NUMERIC" => FieldKind::Numeric {
                    attrs: crate::schema::NumericFieldAttributes {
                        sortable,
                        no_index,
                        index_missing: false,
                        index_empty: false,
                    },
                },
                "GEO" => FieldKind::Geo {
                    attrs: crate::schema::GeoFieldAttributes {
                        sortable,
                        no_index,
                        index_missing: false,
                        index_empty: false,
                    },
                },
                "VECTOR" => {
                    let algo = match algorithm.to_lowercase().as_str() {
                        "hnsw" => crate::schema::VectorAlgorithm::Hnsw,
                        "svs-vamana" | "svs_vamana" => crate::schema::VectorAlgorithm::SvsVamana,
                        _ => crate::schema::VectorAlgorithm::Flat,
                    };
                    let dm = match distance_metric.as_str() {
                        "l2" => crate::schema::VectorDistanceMetric::L2,
                        "ip" => crate::schema::VectorDistanceMetric::Ip,
                        _ => crate::schema::VectorDistanceMetric::Cosine,
                    };
                    let dt = match datatype.to_lowercase().as_str() {
                        "float64" => crate::schema::VectorDataType::Float64,
                        "float16" => crate::schema::VectorDataType::Float16,
                        "bfloat16" => crate::schema::VectorDataType::Bfloat16,
                        _ => crate::schema::VectorDataType::Float32,
                    };
                    FieldKind::Vector {
                        attrs: crate::schema::VectorFieldAttributes {
                            algorithm: algo,
                            dims,
                            distance_metric: dm,
                            datatype: dt,
                            initial_cap: None,
                            block_size: None,
                            m: None,
                            ef_construction: None,
                            ef_runtime: None,
                            epsilon: None,
                            graph_max_degree: None,
                            construction_window_size: None,
                            search_window_size: None,
                            compression: None,
                            reduce: None,
                            training_threshold: None,
                        },
                    }
                }
                _ => continue, // skip unknown field types
            };

            fields.push(crate::schema::Field {
                name: field_name,
                path: None,
                kind,
            });
        }
    }

    Ok(IndexSchema {
        index: IndexDefinition {
            name: name.to_owned(),
            prefix,
            key_separator: ":".to_owned(),
            storage_type,
            stopwords: Vec::new(),
        },
        fields,
    })
}

fn process_search_result<Q>(
    results: SearchResult,
    query: &Q,
    storage_type: StorageType,
) -> Result<QueryOutput>
where
    Q: QueryString + ?Sized,
{
    if query.kind() == QueryKind::Count {
        return Ok(QueryOutput::Count(results.total));
    }

    let unpack_json = matches!(storage_type, StorageType::Json)
        && query.should_unpack_json()
        && query.render().return_fields.is_empty();
    let mut documents = Vec::with_capacity(results.docs.len());

    for document in results.docs {
        let mut map = document.into_map();
        if unpack_json {
            map = unpack_json_document(map)?;
        }
        map.remove("payload");
        documents.push(map);
    }

    Ok(QueryOutput::Documents(documents))
}

fn unpack_json_document(mut document: Map<String, Value>) -> Result<Map<String, Value>> {
    let Some(json_value) = document.remove("json") else {
        return Ok(document);
    };

    let parsed = match json_value {
        Value::String(raw) => serde_json::from_str::<Value>(&raw)?,
        value => value,
    };

    let mut unpacked = Map::new();
    if let Some(id) = document.remove("id") {
        unpacked.insert("id".to_owned(), id);
    }

    match parsed {
        Value::Object(object) => {
            unpacked.extend(object);
            Ok(unpacked)
        }
        other => Err(Error::InvalidInput(format!(
            "expected JSON object payload while unpacking search result, received {other:?}"
        ))),
    }
}

fn is_search_payload(value: &redis::Value) -> bool {
    matches!(
        value,
        redis::Value::Array(_) | redis::Value::Map(_) | redis::Value::Attribute { .. }
    )
}

fn decode_search_payload(value: redis::Value) -> Result<Map<String, Value>> {
    match value {
        redis::Value::Array(entries) => decode_search_pairs(entries),
        redis::Value::Map(entries) => {
            let flat = entries
                .into_iter()
                .flat_map(|(key, value)| [key, value])
                .collect::<Vec<_>>();
            decode_search_pairs(flat)
        }
        redis::Value::Attribute { data, .. } => decode_search_payload(*data),
        other => Err(Error::InvalidInput(format!(
            "expected FT.SEARCH document payload, received {other:?}"
        ))),
    }
}

fn decode_search_pairs(entries: Vec<redis::Value>) -> Result<Map<String, Value>> {
    let mut pairs = VecDeque::from(entries);
    let mut fields = Map::new();
    while let Some(key) = pairs.pop_front() {
        let Some(value) = pairs.pop_front() else {
            return Err(Error::InvalidInput(
                "FT.SEARCH document payload contained an odd number of elements".to_owned(),
            ));
        };
        let field = redis_value_to_string(&key)?;
        let normalized = if field == "$" { "json" } else { field.as_str() };
        fields.insert(normalized.to_owned(), redis_value_to_json(value)?);
    }
    Ok(fields)
}

fn redis_value_to_usize(value: redis::Value) -> Result<usize> {
    let number =
        match value {
            redis::Value::Int(value) => value,
            redis::Value::BulkString(bytes) => String::from_utf8_lossy(&bytes)
                .parse::<i64>()
                .map_err(|_| {
                    Error::InvalidInput("unable to parse integer Redis response".to_owned())
                })?,
            redis::Value::SimpleString(value) => value.parse::<i64>().map_err(|_| {
                Error::InvalidInput("unable to parse integer Redis response".to_owned())
            })?,
            other => {
                return Err(Error::InvalidInput(format!(
                    "expected integer Redis response, received {other:?}"
                )));
            }
        };

    usize::try_from(number)
        .map_err(|_| Error::InvalidInput("redis returned a negative integer".to_owned()))
}

fn redis_value_to_string(value: &redis::Value) -> Result<String> {
    match value {
        redis::Value::BulkString(bytes) => Ok(String::from_utf8_lossy(bytes).into_owned()),
        redis::Value::SimpleString(value) => Ok(value.clone()),
        redis::Value::VerbatimString { text, .. } => Ok(text.clone()),
        redis::Value::Int(value) => Ok(value.to_string()),
        redis::Value::Double(value) => Ok(value.to_string()),
        redis::Value::Boolean(value) => Ok(value.to_string()),
        other => Err(Error::InvalidInput(format!(
            "expected string-like Redis response, received {other:?}"
        ))),
    }
}

fn redis_value_to_json(value: redis::Value) -> Result<Value> {
    match value {
        redis::Value::Nil => Ok(Value::Null),
        redis::Value::Int(value) => Ok(Value::from(value)),
        redis::Value::Double(value) => Ok(Value::from(value)),
        redis::Value::Boolean(value) => Ok(Value::from(value)),
        redis::Value::BulkString(bytes) => {
            Ok(Value::String(String::from_utf8_lossy(&bytes).into_owned()))
        }
        redis::Value::SimpleString(value) => Ok(Value::String(value)),
        redis::Value::Okay => Ok(Value::String("OK".to_owned())),
        redis::Value::VerbatimString { text, .. } => Ok(Value::String(text)),
        redis::Value::Array(values) | redis::Value::Set(values) => {
            let mut array = Vec::with_capacity(values.len());
            for value in values {
                array.push(redis_value_to_json(value)?);
            }
            Ok(Value::Array(array))
        }
        redis::Value::Map(entries) => {
            let mut object = Map::with_capacity(entries.len());
            for (key, value) in entries {
                object.insert(redis_value_to_string(&key)?, redis_value_to_json(value)?);
            }
            Ok(Value::Object(object))
        }
        redis::Value::Attribute { data, .. } => redis_value_to_json(*data),
        redis::Value::BigNumber(number) => Ok(Value::String(number.to_string())),
        redis::Value::Push { .. } | redis::Value::ServerError(_) => {
            Ok(Value::String(format!("{value:?}")))
        }
        _ => Ok(Value::String(format!("{value:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EncodedHashValue, QueryOutput, SearchDocument, SearchIndex, SearchResult, compose_key,
        encode_hash_record, parse_aggregate_result, parse_info_response, parse_search_result,
        prepare_load_records, process_search_result, schema_from_info,
    };
    use crate::{
        filter::Tag,
        query::{CountQuery, FilterQuery},
        schema::{IndexSchema, StorageType},
    };
    use serde_json::{Map, Value, json};

    #[test]
    fn search_index_properties_should_match_python_integration_test_search_index() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": { "name": "my_index" },
                "fields": [
                    { "name": "test", "type": "tag" }
                ]
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");

        assert_eq!(index.name(), "my_index");
        assert_eq!(index.prefix(), "rvl");
        assert_eq!(index.key_separator(), ":");
        assert!(matches!(index.storage_type(), StorageType::Hash));
        assert_eq!(index.key("foo"), "rvl:foo");
    }

    #[test]
    fn search_index_should_honor_empty_prefix_like_python_integration_test_search_index() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": { "name": "my_index", "prefix": "" },
                "fields": [
                    { "name": "test", "type": "tag" }
                ]
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");

        assert_eq!(index.prefix(), "");
        assert_eq!(index.key("foo"), "foo");
    }

    #[test]
    fn search_index_key_should_normalize_trailing_separator_like_python_key_separator_tests() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": {
                    "name": "my_index",
                    "prefix": "user::",
                    "key_separator": ":"
                },
                "fields": [
                    { "name": "test", "type": "tag" }
                ]
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");

        assert_eq!(index.key("456"), "user:456");
        assert!(!index.key("456").contains("::"));
    }

    #[test]
    fn search_index_key_should_use_custom_separator_consistently_like_python_key_separator_tests() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": {
                    "name": "my_index",
                    "prefix": "app:user",
                    "key_separator": "-"
                },
                "fields": [
                    { "name": "test", "type": "tag" }
                ]
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");

        assert_eq!(index.key("999"), "app:user-999");
        assert_eq!(compose_key("routes:", ":", "ref1"), "routes:ref1");
        assert_eq!(compose_key("data", "::", "id"), "data::id");
        assert_eq!(compose_key("data::", "::", "id"), "data::id");
    }

    #[test]
    fn search_index_multi_prefix_should_expose_all_prefixes_like_python_multi_prefix_tests() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": {
                    "name": "multi_pfx",
                    "prefix": ["pfx_a", "pfx_b"]
                },
                "fields": [
                    { "name": "test", "type": "tag" }
                ]
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");

        assert_eq!(index.prefix(), "pfx_a");
        assert_eq!(index.prefixes(), vec!["pfx_a", "pfx_b"]);
        assert_eq!(index.key("doc1"), "pfx_a:doc1");
    }

    #[test]
    fn compose_key_should_handle_special_separators_like_python_key_separator_tests() {
        for sep in &["_", "::", "->", ".", "/"] {
            let result = compose_key("data", sep, "id");
            assert_eq!(result, format!("data{sep}id"));
        }
    }

    /// Tests from Python test_trailing_separator_normalization
    #[test]
    fn trailing_separator_normalization_like_python_key_separator_tests() {
        let cases = [
            ("user:", ":", "123", "user:123"),
            ("user::", ":", "456", "user:456"),
            ("user", ":", "789", "user:789"),
            ("user-", "-", "abc", "user-abc"),
        ];
        for (prefix, sep, id, expected) in &cases {
            let result = compose_key(prefix, sep, id);
            assert_eq!(result, *expected, "prefix={prefix:?} sep={sep:?} id={id:?}");
        }
    }

    /// Tests from Python test_empty_prefix_handled_correctly
    #[test]
    fn empty_prefix_compose_key_like_python_key_separator_tests() {
        let result = compose_key("", ":", "789");
        assert_eq!(result, "789");
    }

    #[test]
    fn hash_load_validation_should_require_string_id_field_like_python_search_index_tests() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": { "name": "my_index" },
                "fields": [
                    { "name": "test", "type": "tag" }
                ]
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");

        let error = index
            .load(
                &[serde_json::json!({ "wrong_key": "1", "value": "test" })],
                "id",
                None,
            )
            .expect_err("missing id field should error before redis usage");

        assert!(error.to_string().contains("missing string id field"));
    }

    #[test]
    fn search_result_parser_should_decode_hash_results_like_python_search() {
        let parsed = parse_search_result(redis::Value::Array(vec![
            redis::Value::Int(2),
            redis::Value::BulkString(b"users:john".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"user".to_vec()),
                redis::Value::BulkString(b"john".to_vec()),
                redis::Value::BulkString(b"age".to_vec()),
                redis::Value::BulkString(b"18".to_vec()),
            ]),
            redis::Value::BulkString(b"users:mary".to_vec()),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"user".to_vec()),
                redis::Value::BulkString(b"mary".to_vec()),
                redis::Value::BulkString(b"vector_distance".to_vec()),
                redis::Value::BulkString(b"0".to_vec()),
            ]),
        ]))
        .expect("result should parse");

        assert_eq!(parsed.total, 2);
        assert_eq!(parsed.docs.len(), 2);
        assert_eq!(parsed.docs[0].id(), "users:john");
        assert_eq!(
            parsed.docs[0].get("user"),
            Some(&Value::String("john".to_owned()))
        );
        assert_eq!(
            parsed.docs[1].to_map().get("vector_distance"),
            Some(&Value::String("0".to_owned()))
        );
    }

    #[test]
    fn process_search_result_should_unpack_json_for_filter_queries_without_projection() {
        let mut fields = Map::new();
        fields.insert(
            "json".to_owned(),
            Value::String(r#"{"user":"john","age":18,"credit_score":"high"}"#.to_owned()),
        );
        let results = SearchResult::new(1, vec![SearchDocument::new("users:john", fields)]);
        let query = FilterQuery::new(Tag::new("credit_score").eq("high"));

        let processed = process_search_result(results, &query, StorageType::Json)
            .expect("query should process");

        assert_eq!(
            processed,
            QueryOutput::Documents(vec![Map::from_iter([
                ("id".to_owned(), Value::String("users:john".to_owned())),
                ("user".to_owned(), Value::String("john".to_owned())),
                ("age".to_owned(), json!(18)),
                ("credit_score".to_owned(), Value::String("high".to_owned())),
            ])])
        );
    }

    #[test]
    fn process_search_result_should_return_count_for_count_queries() {
        let results = SearchResult::new(7, Vec::new());
        let query = CountQuery::new();

        let processed = process_search_result(results, &query, StorageType::Hash)
            .expect("count should process");

        assert_eq!(processed, QueryOutput::Count(7));
    }

    #[test]
    fn paginate_should_reject_zero_page_size_before_redis_usage() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": { "name": "my_index" },
                "fields": [
                    { "name": "brand", "type": "tag" }
                ]
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");
        let query = FilterQuery::new(Tag::new("brand").eq("Nike"));

        let error = index
            .paginate(&query, 0)
            .expect_err("zero page size should fail before redis usage");

        assert!(
            error
                .to_string()
                .contains("page_size must be greater than zero")
        );
    }

    #[test]
    fn create_with_options_should_reject_empty_schema_before_redis_usage() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": { "name": "empty_index" }
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");

        let error = index
            .create_with_options(true, true)
            .expect_err("empty schema should fail before redis usage");

        assert!(error.to_string().contains("No fields defined for index"));
    }

    #[test]
    fn prepare_load_records_should_apply_preprocess_like_python_search_index_tests() {
        let prepared = prepare_load_records(&[json!({"id": "1", "test": "foo"})], &mut |record| {
            let mut record = record.clone();
            let object = record
                .as_object_mut()
                .expect("record remains an object during preprocessing");
            object.insert("test".to_owned(), Value::String("bar".to_owned()));
            Ok(record)
        })
        .expect("preprocess should succeed");

        assert_eq!(prepared[0]["test"], Value::String("bar".to_owned()));
    }

    #[test]
    fn prepare_load_records_should_reject_non_object_preprocess_results() {
        let error = prepare_load_records(&[json!({"id": "1", "test": "foo"})], &mut |_| {
            Ok(Value::String("invalid".to_owned()))
        })
        .expect_err("non-object preprocess output should fail");

        assert!(
            error
                .to_string()
                .contains("preprocess must return a JSON object")
        );
    }

    #[test]
    fn parse_info_response_should_decode_ft_info_shape() {
        let info = parse_info_response(redis::Value::Array(vec![
            redis::Value::BulkString(b"index_name".to_vec()),
            redis::Value::BulkString(b"my_index".to_vec()),
            redis::Value::BulkString(b"num_docs".to_vec()),
            redis::Value::Int(3),
            redis::Value::BulkString(b"hash_indexing_failures".to_vec()),
            redis::Value::Int(0),
        ]))
        .expect("info should parse");

        assert_eq!(info["index_name"], Value::String("my_index".to_owned()));
        assert_eq!(info["num_docs"], json!(3));
        assert_eq!(info["hash_indexing_failures"], json!(0));
    }

    #[test]
    fn search_document_should_expose_id_through_indexing_like_python_results_docs() {
        let document = SearchDocument::new(
            "rvl:1",
            Map::from_iter([("test".to_owned(), Value::String("foo".to_owned()))]),
        );

        assert_eq!(document.id(), "rvl:1");
        assert_eq!(document["id"], Value::String("rvl:1".to_owned()));
        assert_eq!(document["test"], Value::String("foo".to_owned()));
    }

    #[test]
    fn encode_hash_record_should_pack_vector_arrays_for_hash_storage() {
        let schema = IndexSchema::from_json_value(json!({
            "index": { "name": "my_index", "storage_type": "hash" },
            "fields": [
                { "name": "id", "type": "tag" },
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "distance_metric": "COSINE",
                        "algorithm": "FLAT",
                        "datatype": "FLOAT32"
                    }
                }
            ]
        }))
        .expect("schema should parse");

        let encoded = encode_hash_record(
            &json!({
                "id": "1",
                "embedding": [0.1, 0.2, 0.3]
            })
            .as_object()
            .expect("record should be an object")
            .clone(),
            &schema,
        )
        .expect("hash record should encode");

        let embedding = encoded
            .into_iter()
            .find(|(field, _)| field == "embedding")
            .map(|(_, value)| value)
            .expect("embedding should be encoded");

        match embedding {
            EncodedHashValue::Binary(bytes) => assert_eq!(bytes.len(), 12),
            EncodedHashValue::String(_) => panic!("vector field should encode to binary bytes"),
        }
    }

    // ── parse_aggregate_result parity tests ──

    #[test]
    fn parse_aggregate_result_should_produce_document_maps() {
        // Simulate FT.AGGREGATE response: [total, [k, v, ...], [k, v, ...]]
        let value = redis::Value::Array(vec![
            redis::Value::Int(2),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"user".to_vec()),
                redis::Value::BulkString(b"alice".to_vec()),
                redis::Value::BulkString(b"hybrid_score".to_vec()),
                redis::Value::BulkString(b"0.85".to_vec()),
            ]),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"user".to_vec()),
                redis::Value::BulkString(b"bob".to_vec()),
                redis::Value::BulkString(b"hybrid_score".to_vec()),
                redis::Value::BulkString(b"0.72".to_vec()),
            ]),
        ]);

        let docs = parse_aggregate_result(value).expect("should parse");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0]["user"], "alice");
        assert_eq!(docs[0]["hybrid_score"], "0.85");
        assert_eq!(docs[1]["user"], "bob");
    }

    #[test]
    fn parse_aggregate_result_should_strip_internal_score() {
        let value = redis::Value::Array(vec![
            redis::Value::Int(1),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"__score".to_vec()),
                redis::Value::BulkString(b"1.0".to_vec()),
                redis::Value::BulkString(b"user".to_vec()),
                redis::Value::BulkString(b"alice".to_vec()),
            ]),
        ]);

        let docs = parse_aggregate_result(value).expect("should parse");
        assert_eq!(docs.len(), 1);
        assert!(
            !docs[0].contains_key("__score"),
            "internal __score should be stripped"
        );
        assert_eq!(docs[0]["user"], "alice");
    }

    #[test]
    fn parse_aggregate_result_should_handle_nil() {
        let docs = parse_aggregate_result(redis::Value::Nil).expect("should parse");
        assert!(docs.is_empty());
    }

    // ── schema_from_info parity tests ──

    #[test]
    fn schema_from_info_should_reconstruct_basic_schema() {
        let mut info = Map::new();
        info.insert(
            "index_definition".to_owned(),
            json!(["key_type", "HASH", "prefixes", ["rvl"]]),
        );
        info.insert(
            "attributes".to_owned(),
            json!([
                ["identifier", "$.name", "attribute", "name", "type", "TAG"],
                ["identifier", "$.age", "attribute", "age", "type", "NUMERIC"],
            ]),
        );

        let schema = schema_from_info("test_index", &info).expect("should parse");
        assert_eq!(schema.index.name, "test_index");
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.fields[0].name, "name");
        assert_eq!(schema.fields[1].name, "age");
    }

    #[test]
    fn schema_from_info_should_detect_json_storage() {
        let mut info = Map::new();
        info.insert(
            "index_definition".to_owned(),
            json!(["key_type", "JSON", "prefixes", ["myprefix"]]),
        );
        info.insert("attributes".to_owned(), json!([]));

        let schema = schema_from_info("json_idx", &info).expect("should parse");
        assert!(matches!(schema.index.storage_type, StorageType::Json));
    }

    #[test]
    fn schema_from_info_should_parse_vector_fields() {
        let mut info = Map::new();
        info.insert(
            "index_definition".to_owned(),
            json!(["key_type", "HASH", "prefixes", ["rvl"]]),
        );
        info.insert(
            "attributes".to_owned(),
            json!([[
                "identifier",
                "embedding",
                "attribute",
                "embedding",
                "type",
                "VECTOR",
                "HNSW",
                "6",
                "DIM",
                "768",
                "DISTANCE_METRIC",
                "COSINE",
                "TYPE",
                "FLOAT32"
            ]]),
        );

        let schema = schema_from_info("vec_idx", &info).expect("should parse");
        assert_eq!(schema.fields.len(), 1);
        let field = &schema.fields[0];
        assert_eq!(field.name, "embedding");
        match &field.kind {
            crate::schema::FieldKind::Vector { attrs } => {
                assert_eq!(attrs.dims, 768);
                assert!(matches!(
                    attrs.distance_metric,
                    crate::schema::VectorDistanceMetric::Cosine
                ));
                assert!(matches!(
                    attrs.algorithm,
                    crate::schema::VectorAlgorithm::Hnsw
                ));
            }
            _ => panic!("expected vector field kind"),
        }
    }

    #[test]
    fn multi_prefix_index_should_report_correct_prefix_count_in_create_cmd() {
        let index = SearchIndex::from_json_value(
            serde_json::json!({
                "index": {
                    "name": "multi_test",
                    "prefix": ["pfx_a", "pfx_b"]
                },
                "fields": [
                    { "name": "tag", "type": "tag" }
                ]
            }),
            "redis://127.0.0.1:6379",
        )
        .expect("index should parse");

        // Verify the schema round-trips all prefixes — `create_cmd` iterates
        // over `self.schema.index.prefix.all()` so if these pass the command
        // will carry both prefixes.
        assert_eq!(index.prefixes(), vec!["pfx_a", "pfx_b"]);
        assert_eq!(index.prefix(), "pfx_a");
        // create_cmd is exercised end-to-end via integration tests.
        let _cmd = index.create_cmd();
    }

    // ── schema_from_info default-normalization tests ──

    #[test]
    fn schema_from_info_should_normalize_tag_separator_default() {
        // Redis FT.INFO always reports SEPARATOR "," for tag fields even when
        // the field was created without an explicit separator. The reconstructed
        // schema must treat the Redis default (,) as None so that comparison
        // with an original JSON-built schema succeeds.
        let mut info = Map::new();
        info.insert(
            "index_definition".to_owned(),
            json!(["key_type", "HASH", "prefixes", ["test"]]),
        );
        info.insert(
            "attributes".to_owned(),
            json!([[
                "identifier",
                "brand",
                "attribute",
                "brand",
                "type",
                "TAG",
                "SEPARATOR",
                ","
            ]]),
        );

        let schema = schema_from_info("norm_test", &info).expect("should parse");
        match &schema.fields[0].kind {
            crate::schema::FieldKind::Tag { attrs } => {
                assert!(
                    attrs.separator.is_none(),
                    "default separator ',' should be normalized to None, got {:?}",
                    attrs.separator
                );
            }
            other => panic!("expected tag field, got {other:?}"),
        }
    }

    #[test]
    fn schema_from_info_should_preserve_non_default_tag_separator() {
        let mut info = Map::new();
        info.insert(
            "index_definition".to_owned(),
            json!(["key_type", "HASH", "prefixes", ["test"]]),
        );
        info.insert(
            "attributes".to_owned(),
            json!([[
                "identifier",
                "brand",
                "attribute",
                "brand",
                "type",
                "TAG",
                "SEPARATOR",
                "|"
            ]]),
        );

        let schema = schema_from_info("norm_test", &info).expect("should parse");
        match &schema.fields[0].kind {
            crate::schema::FieldKind::Tag { attrs } => {
                assert_eq!(attrs.separator.as_deref(), Some("|"));
            }
            other => panic!("expected tag field, got {other:?}"),
        }
    }

    #[test]
    fn schema_from_info_should_normalize_text_weight_default() {
        // Redis FT.INFO always reports WEIGHT 1 for text fields even when the
        // field was created without an explicit weight.
        let mut info = Map::new();
        info.insert(
            "index_definition".to_owned(),
            json!(["key_type", "HASH", "prefixes", ["test"]]),
        );
        info.insert(
            "attributes".to_owned(),
            json!([[
                "identifier",
                "content",
                "attribute",
                "content",
                "type",
                "TEXT",
                "WEIGHT",
                "1"
            ]]),
        );

        let schema = schema_from_info("norm_test", &info).expect("should parse");
        match &schema.fields[0].kind {
            crate::schema::FieldKind::Text { attrs } => {
                assert!(
                    attrs.weight.is_none(),
                    "default weight 1.0 should be normalized to None, got {:?}",
                    attrs.weight
                );
            }
            other => panic!("expected text field, got {other:?}"),
        }
    }

    #[test]
    fn schema_from_info_should_preserve_non_default_text_weight() {
        let mut info = Map::new();
        info.insert(
            "index_definition".to_owned(),
            json!(["key_type", "HASH", "prefixes", ["test"]]),
        );
        info.insert(
            "attributes".to_owned(),
            json!([[
                "identifier",
                "content",
                "attribute",
                "content",
                "type",
                "TEXT",
                "WEIGHT",
                "2.5"
            ]]),
        );

        let schema = schema_from_info("norm_test", &info).expect("should parse");
        match &schema.fields[0].kind {
            crate::schema::FieldKind::Text { attrs } => {
                assert_eq!(attrs.weight, Some(2.5));
            }
            other => panic!("expected text field, got {other:?}"),
        }
    }

    #[test]
    fn schema_from_info_json_roundtrip_should_match_original_schema() {
        // Simulates the semantic router / message history reconnect scenario:
        // an original schema built from JSON should match a schema reconstructed
        // from FT.INFO output where Redis adds default separator and weight.
        let original = IndexSchema::from_json_value(json!({
            "index": {
                "name": "my_router",
                "prefix": "my_router",
                "storage_type": "hash"
            },
            "fields": [
                { "name": "ref_id", "type": "tag" },
                { "name": "route", "type": "tag" },
                { "name": "reference", "type": "text" },
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "datatype": "float32",
                        "distance_metric": "cosine"
                    }
                }
            ]
        }))
        .expect("original schema should parse");

        // Simulate FT.INFO output with Redis defaults explicitly present
        let mut info = Map::new();
        info.insert(
            "index_definition".to_owned(),
            json!(["key_type", "HASH", "prefixes", ["my_router"]]),
        );
        info.insert(
            "attributes".to_owned(),
            json!([
                [
                    "identifier",
                    "ref_id",
                    "attribute",
                    "ref_id",
                    "type",
                    "TAG",
                    "SEPARATOR",
                    ","
                ],
                [
                    "identifier",
                    "route",
                    "attribute",
                    "route",
                    "type",
                    "TAG",
                    "SEPARATOR",
                    ","
                ],
                [
                    "identifier",
                    "reference",
                    "attribute",
                    "reference",
                    "type",
                    "TEXT",
                    "WEIGHT",
                    "1"
                ],
                [
                    "identifier",
                    "vector",
                    "attribute",
                    "vector",
                    "type",
                    "VECTOR",
                    "FLAT",
                    "6",
                    "TYPE",
                    "FLOAT32",
                    "DIM",
                    "3",
                    "DISTANCE_METRIC",
                    "COSINE"
                ]
            ]),
        );
        let reconstructed =
            schema_from_info("my_router", &info).expect("reconstructed schema should parse");

        let original_json = original.to_json_value().expect("original to_json_value");
        let reconstructed_json = reconstructed
            .to_json_value()
            .expect("reconstructed to_json_value");
        assert_eq!(
            original_json, reconstructed_json,
            "original and reconstructed schemas should match after normalization\n\
             original:      {original_json:#}\n\
             reconstructed: {reconstructed_json:#}"
        );
    }

    #[test]
    fn f32_to_f16_basic_values() {
        use super::f32_to_f16_bytes;

        // Zero → 0x0000
        assert_eq!(f32_to_f16_bytes(0.0), 0x0000);
        // Negative zero → 0x8000
        assert_eq!(f32_to_f16_bytes(-0.0), 0x8000);
        // 1.0 → 0x3C00
        assert_eq!(f32_to_f16_bytes(1.0), 0x3C00);
        // -1.0 → 0xBC00
        assert_eq!(f32_to_f16_bytes(-1.0), 0xBC00);
        // Infinity → 0x7C00
        assert_eq!(f32_to_f16_bytes(f32::INFINITY), 0x7C00);
        // Negative infinity → 0xFC00
        assert_eq!(f32_to_f16_bytes(f32::NEG_INFINITY), 0xFC00);
        // NaN → sign | 0x7C00 | some mantissa bits
        let nan_bits = f32_to_f16_bytes(f32::NAN);
        assert_eq!(nan_bits & 0x7C00, 0x7C00, "NaN exponent should be all ones");
        assert_ne!(nan_bits & 0x03FF, 0, "NaN should have non-zero mantissa");
    }
}
