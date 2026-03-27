//! Search index lifecycle helpers and Redis transport adapters.

use std::collections::{HashMap, VecDeque};
use std::ops::Index;

use redis::Commands;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    error::{Error, Result},
    filter::FilterExpression,
    query::{PageableQuery, QueryKind, QueryParamValue, QueryString, SortDirection},
    schema::{FieldKind, IndexSchema, StorageType, VectorAlgorithm},
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

    /// Returns the key prefix.
    pub fn prefix(&self) -> &str {
        &self.schema.index.prefix
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
        cmd.arg(&self.schema.index.name)
            .arg("ON")
            .arg(self.schema.index.storage_type.redis_name())
            .arg("PREFIX")
            .arg(1)
            .arg(&self.schema.index.prefix);

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
                    Err(Error::Redis(err)) if err.kind() == redis::ErrorKind::TypeError => Ok(None),
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

    /// Returns the key prefix.
    pub fn prefix(&self) -> &str {
        &self.schema.index.prefix
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
                    Err(err) if err.kind() == redis::ErrorKind::TypeError => Ok(None),
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
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EncodedHashValue, QueryOutput, SearchDocument, SearchIndex, SearchResult, compose_key,
        encode_hash_record, parse_info_response, parse_search_result, prepare_load_records,
        process_search_result,
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
}
