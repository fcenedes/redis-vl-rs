//! Semantic and embedding cache extensions.

use std::{collections::HashMap, sync::Arc};

use chrono::Utc;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};

use crate::{
    error::Result,
    filter::FilterExpression,
    index::{AsyncSearchIndex, QueryOutput, RedisConnectionInfo, SearchIndex},
    query::{Vector, VectorRangeQuery},
    vectorizers::Vectorizer,
};

const SEMANTIC_ENTRY_ID_FIELD: &str = "entry_id";
const SEMANTIC_PROMPT_FIELD: &str = "prompt";
const SEMANTIC_RESPONSE_FIELD: &str = "response";
const SEMANTIC_VECTOR_FIELD: &str = "prompt_vector";
const SEMANTIC_INSERTED_AT_FIELD: &str = "inserted_at";
const SEMANTIC_UPDATED_AT_FIELD: &str = "updated_at";
const SEMANTIC_METADATA_FIELD: &str = "metadata";
const SEMANTIC_KEY_FIELD: &str = "key";

/// Shared configuration for cache-backed extensions.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Cache name or key namespace.
    pub name: String,
    /// Redis connection settings.
    pub connection: RedisConnectionInfo,
    /// Optional TTL in seconds.
    pub ttl_seconds: Option<u64>,
}

impl CacheConfig {
    /// Creates a new cache configuration with no default TTL.
    pub fn new(name: impl Into<String>, redis_url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            connection: RedisConnectionInfo::new(redis_url),
            ttl_seconds: None,
        }
    }

    /// Adds a default TTL to the cache configuration.
    #[must_use]
    pub fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.ttl_seconds = Some(ttl_seconds);
        self
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::new("embedcache", "redis://127.0.0.1:6379")
    }
}

/// Entry stored in the embeddings cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingCacheEntry {
    /// Deterministic entry identifier derived from content and model name.
    pub entry_id: String,
    /// Original content that was embedded.
    pub content: String,
    /// Embedding model name.
    pub model_name: String,
    /// Embedding vector payload.
    pub embedding: Vec<f32>,
    /// Optional arbitrary metadata stored alongside the vector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Batch input item for the embeddings cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingCacheItem {
    /// Original content that was embedded.
    pub content: String,
    /// Embedding model name.
    pub model_name: String,
    /// Embedding vector payload.
    pub embedding: Vec<f32>,
    /// Optional arbitrary metadata stored alongside the vector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Semantic cache backed by a Redis Search vector index.
#[derive(Clone)]
pub struct SemanticCache {
    /// Cache configuration.
    pub config: CacheConfig,
    /// Distance threshold used for semantic hits.
    pub distance_threshold: f32,
    /// Prompt embedding dimensions stored in Redis.
    pub vector_dimensions: usize,
    /// Underlying search index.
    pub index: SearchIndex,
    vectorizer: Option<Arc<dyn Vectorizer>>,
    return_fields: Vec<String>,
}

impl SemanticCache {
    /// Creates a new semantic cache with the default reserved schema.
    pub fn new(
        config: CacheConfig,
        distance_threshold: f32,
        vector_dimensions: usize,
    ) -> Result<Self> {
        Self::with_filterable_fields(config, distance_threshold, vector_dimensions, &[])
    }

    /// Creates a new semantic cache with additional filterable schema fields.
    pub fn with_filterable_fields(
        config: CacheConfig,
        distance_threshold: f32,
        vector_dimensions: usize,
        filterable_fields: &[Value],
    ) -> Result<Self> {
        validate_distance_threshold(distance_threshold)?;
        if vector_dimensions == 0 {
            return Err(crate::Error::InvalidInput(
                "vector_dimensions must be greater than zero".to_owned(),
            ));
        }

        let schema = semantic_cache_schema(&config.name, vector_dimensions, filterable_fields);
        let index = SearchIndex::from_json_value(schema, config.connection.redis_url.clone())?;
        if !index.exists().unwrap_or(false) {
            index.create_with_options(false, false)?;
        }

        Ok(Self {
            config,
            distance_threshold,
            vector_dimensions,
            index,
            vectorizer: None,
            return_fields: default_semantic_return_fields(),
        })
    }

    /// Attaches a synchronous vectorizer used when callers pass prompts instead of vectors.
    #[must_use]
    pub fn with_vectorizer<V>(mut self, vectorizer: V) -> Self
    where
        V: Vectorizer + 'static,
    {
        self.vectorizer = Some(Arc::new(vectorizer));
        self
    }

    /// Replaces the vectorizer used for prompt embedding.
    pub fn set_vectorizer<V>(&mut self, vectorizer: V)
    where
        V: Vectorizer + 'static,
    {
        self.vectorizer = Some(Arc::new(vectorizer));
    }

    /// Returns the configured default TTL for semantic cache entries.
    pub fn ttl(&self) -> Option<u64> {
        self.config.ttl_seconds
    }

    /// Sets or clears the default TTL for semantic cache entries.
    pub fn set_ttl(&mut self, ttl_seconds: Option<u64>) {
        self.config.ttl_seconds = ttl_seconds;
    }

    /// Updates the semantic distance threshold.
    pub fn set_threshold(&mut self, distance_threshold: f32) -> Result<()> {
        validate_distance_threshold(distance_threshold)?;
        self.distance_threshold = distance_threshold;
        Ok(())
    }

    /// Stores a semantic cache entry and returns its Redis key.
    pub fn store(
        &self,
        prompt: &str,
        response: &str,
        vector: Option<&[f32]>,
        metadata: Option<Value>,
        filters: Option<Map<String, Value>>,
        ttl_seconds: Option<u64>,
    ) -> Result<String> {
        if let Some(metadata) = metadata.as_ref() {
            validate_metadata(metadata)?;
        }

        let vector = self.resolve_vector(prompt, vector)?;
        let timestamp = current_timestamp();
        let entry_id = semantic_entry_id(prompt, filters.as_ref());
        let mut record = Map::new();
        record.insert(SEMANTIC_ENTRY_ID_FIELD.to_owned(), Value::String(entry_id));
        record.insert(
            SEMANTIC_PROMPT_FIELD.to_owned(),
            Value::String(prompt.to_owned()),
        );
        record.insert(
            SEMANTIC_RESPONSE_FIELD.to_owned(),
            Value::String(response.to_owned()),
        );
        record.insert(
            SEMANTIC_VECTOR_FIELD.to_owned(),
            Value::Array(
                vector
                    .iter()
                    .copied()
                    .map(|value| number_value(f64::from(value)))
                    .collect(),
            ),
        );
        record.insert(
            SEMANTIC_INSERTED_AT_FIELD.to_owned(),
            number_value(timestamp),
        );
        record.insert(
            SEMANTIC_UPDATED_AT_FIELD.to_owned(),
            number_value(timestamp),
        );
        if let Some(metadata) = metadata {
            record.insert(SEMANTIC_METADATA_FIELD.to_owned(), metadata);
        }
        if let Some(filters) = filters {
            for (key, value) in filters {
                record.insert(key, value);
            }
        }

        let keys = self.index.load(
            &[Value::Object(record)],
            SEMANTIC_ENTRY_ID_FIELD,
            ttl_seconds
                .or(self.config.ttl_seconds)
                .map(|value| value as i64),
        )?;
        Ok(keys.into_iter().next().unwrap_or_default())
    }

    /// Stores a semantic cache entry asynchronously and returns its Redis key.
    pub async fn astore(
        &self,
        prompt: &str,
        response: &str,
        vector: Option<&[f32]>,
        metadata: Option<Value>,
        filters: Option<Map<String, Value>>,
        ttl_seconds: Option<u64>,
    ) -> Result<String> {
        if let Some(metadata) = metadata.as_ref() {
            validate_metadata(metadata)?;
        }

        let vector = self.resolve_vector(prompt, vector)?;
        let timestamp = current_timestamp();
        let entry_id = semantic_entry_id(prompt, filters.as_ref());
        let mut record = Map::new();
        record.insert(SEMANTIC_ENTRY_ID_FIELD.to_owned(), Value::String(entry_id));
        record.insert(
            SEMANTIC_PROMPT_FIELD.to_owned(),
            Value::String(prompt.to_owned()),
        );
        record.insert(
            SEMANTIC_RESPONSE_FIELD.to_owned(),
            Value::String(response.to_owned()),
        );
        record.insert(
            SEMANTIC_VECTOR_FIELD.to_owned(),
            Value::Array(
                vector
                    .iter()
                    .copied()
                    .map(|value| number_value(f64::from(value)))
                    .collect(),
            ),
        );
        record.insert(
            SEMANTIC_INSERTED_AT_FIELD.to_owned(),
            number_value(timestamp),
        );
        record.insert(
            SEMANTIC_UPDATED_AT_FIELD.to_owned(),
            number_value(timestamp),
        );
        if let Some(metadata) = metadata {
            record.insert(SEMANTIC_METADATA_FIELD.to_owned(), metadata);
        }
        if let Some(filters) = filters {
            for (key, value) in filters {
                record.insert(key, value);
            }
        }

        let keys = self
            .async_index()
            .load(
                &[Value::Object(record)],
                SEMANTIC_ENTRY_ID_FIELD,
                ttl_seconds
                    .or(self.config.ttl_seconds)
                    .map(|value| value as i64),
            )
            .await?;
        Ok(keys.into_iter().next().unwrap_or_default())
    }

    /// Checks the semantic cache for similar prompts or a supplied vector.
    pub fn check(
        &self,
        prompt: Option<&str>,
        vector: Option<&[f32]>,
        num_results: usize,
        return_fields: Option<&[&str]>,
        filter_expression: Option<FilterExpression>,
        distance_threshold: Option<f32>,
    ) -> Result<Vec<Map<String, Value>>> {
        let vector = self.resolve_query_vector(prompt, vector)?;
        let threshold = distance_threshold.unwrap_or(self.distance_threshold);
        validate_distance_threshold(threshold)?;
        let mut query = VectorRangeQuery::new(
            Vector::new(vector.clone()),
            SEMANTIC_VECTOR_FIELD,
            threshold,
        )
        .paging(0, num_results)
        .with_return_fields(self.return_fields.iter().map(String::as_str));
        if let Some(filter_expression) = filter_expression {
            query = query.with_filter(filter_expression);
        }

        let hits = process_semantic_hits(
            query_output_documents(self.index.query(&query)?)?,
            return_fields,
        )?;
        self.refresh_ttl_sync(&hits)?;
        Ok(hits)
    }

    /// Checks the semantic cache for similar prompts or a supplied vector asynchronously.
    pub async fn acheck(
        &self,
        prompt: Option<&str>,
        vector: Option<&[f32]>,
        num_results: usize,
        return_fields: Option<&[&str]>,
        filter_expression: Option<FilterExpression>,
        distance_threshold: Option<f32>,
    ) -> Result<Vec<Map<String, Value>>> {
        let vector = self.resolve_query_vector(prompt, vector)?;
        let threshold = distance_threshold.unwrap_or(self.distance_threshold);
        validate_distance_threshold(threshold)?;
        let mut query = VectorRangeQuery::new(
            Vector::new(vector.clone()),
            SEMANTIC_VECTOR_FIELD,
            threshold,
        )
        .paging(0, num_results)
        .with_return_fields(self.return_fields.iter().map(String::as_str));
        if let Some(filter_expression) = filter_expression {
            query = query.with_filter(filter_expression);
        }

        let hits = process_semantic_hits(
            query_output_documents(self.async_index().query(&query).await?)?,
            return_fields,
        )?;
        self.refresh_ttl_async(&hits).await?;
        Ok(hits)
    }

    /// Updates cached fields for a stored semantic cache entry and refreshes TTL.
    pub fn update(&self, key: &str, fields: Map<String, Value>) -> Result<()> {
        let mapping = prepare_semantic_update_fields(fields)?;
        let client = self.config.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut cmd = redis::cmd("HSET");
        cmd.arg(key);
        for (field, value) in mapping {
            cmd.arg(field).arg(value);
        }
        let _: usize = cmd.query(&mut connection)?;
        self.expire_key(key, None)
    }

    /// Updates cached fields asynchronously for a stored semantic cache entry and refreshes TTL.
    pub async fn aupdate(&self, key: &str, fields: Map<String, Value>) -> Result<()> {
        let mapping = prepare_semantic_update_fields(fields)?;
        let client = self.config.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let mut cmd = redis::cmd("HSET");
        cmd.arg(key);
        for (field, value) in mapping {
            cmd.arg(field).arg(value);
        }
        let _: usize = cmd.query_async(&mut connection).await?;
        self.aexpire_key(key, None).await
    }

    /// Clears all semantic cache entries while preserving the index.
    pub fn clear(&self) -> Result<usize> {
        self.index.clear()
    }

    /// Clears all semantic cache entries asynchronously while preserving the index.
    pub async fn aclear(&self) -> Result<usize> {
        self.async_index().clear().await
    }

    /// Deletes the semantic cache index and its documents.
    pub fn delete(&self) -> Result<()> {
        self.index.delete(true)
    }

    /// Deletes the semantic cache index asynchronously and its documents.
    pub async fn adelete(&self) -> Result<()> {
        self.async_index().delete(true).await
    }

    /// Drops stored entries by their entry ids.
    pub fn drop_ids(&self, ids: &[String]) -> Result<()> {
        let keys = ids.iter().map(|id| self.index.key(id)).collect::<Vec<_>>();
        self.index.drop_keys(&keys)?;
        Ok(())
    }

    /// Drops stored entries by their Redis keys.
    pub fn drop_keys(&self, keys: &[String]) -> Result<()> {
        self.index.drop_keys(keys)?;
        Ok(())
    }

    /// Drops stored entries asynchronously by their entry ids.
    pub async fn adrop_ids(&self, ids: &[String]) -> Result<()> {
        let keys = ids.iter().map(|id| self.index.key(id)).collect::<Vec<_>>();
        self.async_index().drop_keys(&keys).await?;
        Ok(())
    }

    /// Drops stored entries asynchronously by their Redis keys.
    pub async fn adrop_keys(&self, keys: &[String]) -> Result<()> {
        self.async_index().drop_keys(keys).await?;
        Ok(())
    }

    fn resolve_query_vector(
        &self,
        prompt: Option<&str>,
        vector: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        match (prompt, vector) {
            (_, Some(vector)) => self.validate_vector(vector),
            (Some(prompt), None) => self.resolve_vector(prompt, None),
            (None, None) => Err(crate::Error::InvalidInput(
                "either prompt or vector must be specified".to_owned(),
            )),
        }
    }

    fn resolve_vector(&self, prompt: &str, vector: Option<&[f32]>) -> Result<Vec<f32>> {
        match vector {
            Some(vector) => self.validate_vector(vector),
            None => {
                let Some(vectorizer) = &self.vectorizer else {
                    return Err(crate::Error::InvalidInput(
                        "a vector or configured vectorizer is required".to_owned(),
                    ));
                };
                let vector = vectorizer.embed(prompt)?;
                self.validate_vector(&vector)
            }
        }
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.len() != self.vector_dimensions {
            return Err(crate::Error::InvalidInput(format!(
                "vector dimensions mismatch: expected {}, got {}",
                self.vector_dimensions,
                vector.len()
            )));
        }
        Ok(vector.to_vec())
    }

    fn async_index(&self) -> AsyncSearchIndex {
        AsyncSearchIndex::new(
            self.index.schema().clone(),
            self.config.connection.redis_url.clone(),
        )
    }

    fn refresh_ttl_sync(&self, hits: &[Map<String, Value>]) -> Result<()> {
        if self.config.ttl_seconds.is_none() {
            return Ok(());
        }
        for hit in hits {
            if let Some(key) = hit.get(SEMANTIC_KEY_FIELD).and_then(Value::as_str) {
                self.expire_key(key, None)?;
            }
        }
        Ok(())
    }

    async fn refresh_ttl_async(&self, hits: &[Map<String, Value>]) -> Result<()> {
        if self.config.ttl_seconds.is_none() {
            return Ok(());
        }
        for hit in hits {
            if let Some(key) = hit.get(SEMANTIC_KEY_FIELD).and_then(Value::as_str) {
                self.aexpire_key(key, None).await?;
            }
        }
        Ok(())
    }

    fn expire_key(&self, key: &str, ttl_override: Option<u64>) -> Result<()> {
        if let Some(ttl_seconds) = ttl_override.or(self.config.ttl_seconds) {
            let client = self.config.connection.client()?;
            let mut connection = client.get_connection()?;
            let _: bool = redis::cmd("EXPIRE")
                .arg(key)
                .arg(ttl_seconds)
                .query(&mut connection)?;
        }
        Ok(())
    }

    async fn aexpire_key(&self, key: &str, ttl_override: Option<u64>) -> Result<()> {
        if let Some(ttl_seconds) = ttl_override.or(self.config.ttl_seconds) {
            let client = self.config.connection.client()?;
            let mut connection = client.get_multiplexed_async_connection().await?;
            let _: bool = redis::cmd("EXPIRE")
                .arg(key)
                .arg(ttl_seconds)
                .query_async(&mut connection)
                .await?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for SemanticCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticCache")
            .field("config", &self.config)
            .field("distance_threshold", &self.distance_threshold)
            .field("vector_dimensions", &self.vector_dimensions)
            .field("index_name", &self.index.name())
            .finish()
    }
}

/// Redis-backed cache for deterministic content/model embedding lookups.
#[derive(Debug, Clone)]
pub struct EmbeddingsCache {
    /// Cache configuration.
    pub config: CacheConfig,
}

impl Default for EmbeddingsCache {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

impl EmbeddingsCache {
    /// Creates a new embeddings cache configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self { config }
    }

    /// Generates the deterministic cache entry id for a content/model pair.
    pub fn make_entry_id(&self, content: &str, model_name: &str) -> String {
        hashify(&format!("{content}:{model_name}"))
    }

    /// Generates the full Redis key for a content/model pair.
    pub fn make_cache_key(&self, content: &str, model_name: &str) -> String {
        let entry_id = self.make_entry_id(content, model_name);
        self.key_for_entry(&entry_id)
    }

    /// Retrieves a cached embedding by content and model name.
    pub fn get(&self, content: &str, model_name: &str) -> Result<Option<EmbeddingCacheEntry>> {
        let key = self.make_cache_key(content, model_name);
        self.get_by_key(&key)
    }

    /// Retrieves a cached embedding by its Redis key.
    pub fn get_by_key(&self, key: &str) -> Result<Option<EmbeddingCacheEntry>> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_connection()?;
        let data: HashMap<String, String> =
            redis::cmd("HGETALL").arg(key).query(&mut connection)?;

        if data.is_empty() {
            return Ok(None);
        }

        self.expire_key(key, None)?;
        parse_entry(data)
    }

    /// Retrieves multiple cached embeddings by content and model name.
    pub fn mget<I, S>(
        &self,
        contents: I,
        model_name: &str,
    ) -> Result<Vec<Option<EmbeddingCacheEntry>>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = contents
            .into_iter()
            .map(|content| self.make_cache_key(content.as_ref(), model_name))
            .collect::<Vec<_>>();
        self.mget_by_keys(keys)
    }

    /// Retrieves multiple cached embeddings by Redis key.
    pub fn mget_by_keys<I, S>(&self, keys: I) -> Result<Vec<Option<EmbeddingCacheEntry>>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = collect_strings(keys);
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(keys.len());
        for key in &keys {
            results.push(self.get_by_key(key)?);
        }
        Ok(results)
    }

    /// Stores a cached embedding and returns its Redis key.
    pub fn set(
        &self,
        content: &str,
        model_name: &str,
        embedding: &[f32],
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<String> {
        let entry = self.prepare_entry(content, model_name, embedding, metadata);
        let key = self.key_for_entry(&entry.entry_id);
        self.write_entry(&key, &entry)?;
        self.expire_key(&key, ttl_seconds)?;
        Ok(key)
    }

    /// Stores multiple cached embeddings and returns their Redis keys.
    pub fn mset(
        &self,
        items: &[EmbeddingCacheItem],
        ttl_seconds: Option<u64>,
    ) -> Result<Vec<String>> {
        let mut keys = Vec::with_capacity(items.len());
        for item in items {
            let key = self.set(
                &item.content,
                &item.model_name,
                &item.embedding,
                item.metadata.clone(),
                ttl_seconds,
            )?;
            keys.push(key);
        }
        Ok(keys)
    }

    /// Checks whether a cached embedding exists for a content/model pair.
    pub fn exists(&self, content: &str, model_name: &str) -> Result<bool> {
        let key = self.make_cache_key(content, model_name);
        self.exists_by_key(&key)
    }

    /// Checks whether a cached embedding exists for a Redis key.
    pub fn exists_by_key(&self, key: &str) -> Result<bool> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_connection()?;
        let exists: u64 = redis::cmd("EXISTS").arg(key).query(&mut connection)?;
        Ok(exists > 0)
    }

    /// Checks whether multiple cached embeddings exist for content/model pairs.
    pub fn mexists<I, S>(&self, contents: I, model_name: &str) -> Result<Vec<bool>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = contents
            .into_iter()
            .map(|content| self.make_cache_key(content.as_ref(), model_name))
            .collect::<Vec<_>>();
        self.mexists_by_keys(keys)
    }

    /// Checks whether multiple cached embeddings exist for Redis keys.
    pub fn mexists_by_keys<I, S>(&self, keys: I) -> Result<Vec<bool>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = collect_strings(keys);
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let client = self.config.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            let exists: u64 = redis::cmd("EXISTS").arg(key).query(&mut connection)?;
            results.push(exists > 0);
        }
        Ok(results)
    }

    /// Removes a cached embedding by content and model name.
    pub fn drop(&self, content: &str, model_name: &str) -> Result<()> {
        let key = self.make_cache_key(content, model_name);
        self.drop_by_key(&key)
    }

    /// Removes a cached embedding by Redis key.
    pub fn drop_by_key(&self, key: &str) -> Result<()> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_connection()?;
        let _: usize = redis::cmd("DEL").arg(key).query(&mut connection)?;
        Ok(())
    }

    /// Removes multiple cached embeddings by content and model name.
    pub fn mdrop<I, S>(&self, contents: I, model_name: &str) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = contents
            .into_iter()
            .map(|content| self.make_cache_key(content.as_ref(), model_name))
            .collect::<Vec<_>>();
        self.mdrop_by_keys(keys)
    }

    /// Removes multiple cached embeddings by Redis key.
    pub fn mdrop_by_keys<I, S>(&self, keys: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = collect_strings(keys);
        if keys.is_empty() {
            return Ok(());
        }

        let client = self.config.connection.client()?;
        let mut connection = client.get_connection()?;
        let _: usize = redis::cmd("DEL").arg(keys).query(&mut connection)?;
        Ok(())
    }

    /// Clears every cache entry under this cache namespace.
    pub fn clear(&self) -> Result<usize> {
        let keys = self.all_keys()?;
        if keys.is_empty() {
            return Ok(0);
        }

        let count = keys.len();
        self.mdrop_by_keys(keys)?;
        Ok(count)
    }

    /// Retrieves a cached embedding by content and model name asynchronously.
    pub async fn aget(
        &self,
        content: &str,
        model_name: &str,
    ) -> Result<Option<EmbeddingCacheEntry>> {
        let key = self.make_cache_key(content, model_name);
        self.aget_by_key(&key).await
    }

    /// Retrieves a cached embedding by its Redis key asynchronously.
    pub async fn aget_by_key(&self, key: &str) -> Result<Option<EmbeddingCacheEntry>> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let data: HashMap<String, String> = redis::cmd("HGETALL")
            .arg(key)
            .query_async(&mut connection)
            .await?;

        if data.is_empty() {
            return Ok(None);
        }

        self.aexpire_key(key, None).await?;
        parse_entry(data)
    }

    /// Retrieves multiple cached embeddings by content and model name asynchronously.
    pub async fn amget<I, S>(
        &self,
        contents: I,
        model_name: &str,
    ) -> Result<Vec<Option<EmbeddingCacheEntry>>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = contents
            .into_iter()
            .map(|content| self.make_cache_key(content.as_ref(), model_name))
            .collect::<Vec<_>>();
        self.amget_by_keys(keys).await
    }

    /// Retrieves multiple cached embeddings by Redis key asynchronously.
    pub async fn amget_by_keys<I, S>(&self, keys: I) -> Result<Vec<Option<EmbeddingCacheEntry>>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = collect_strings(keys);
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(keys.len());
        for key in &keys {
            results.push(self.aget_by_key(key).await?);
        }
        Ok(results)
    }

    /// Stores a cached embedding asynchronously and returns its Redis key.
    pub async fn aset(
        &self,
        content: &str,
        model_name: &str,
        embedding: &[f32],
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<String> {
        let entry = self.prepare_entry(content, model_name, embedding, metadata);
        let key = self.key_for_entry(&entry.entry_id);
        self.awrite_entry(&key, &entry).await?;
        self.aexpire_key(&key, ttl_seconds).await?;
        Ok(key)
    }

    /// Stores multiple cached embeddings asynchronously and returns their Redis keys.
    pub async fn amset(
        &self,
        items: &[EmbeddingCacheItem],
        ttl_seconds: Option<u64>,
    ) -> Result<Vec<String>> {
        let mut keys = Vec::with_capacity(items.len());
        for item in items {
            let key = self
                .aset(
                    &item.content,
                    &item.model_name,
                    &item.embedding,
                    item.metadata.clone(),
                    ttl_seconds,
                )
                .await?;
            keys.push(key);
        }
        Ok(keys)
    }

    /// Checks whether a cached embedding exists for a content/model pair asynchronously.
    pub async fn aexists(&self, content: &str, model_name: &str) -> Result<bool> {
        let key = self.make_cache_key(content, model_name);
        self.aexists_by_key(&key).await
    }

    /// Checks whether a cached embedding exists for a Redis key asynchronously.
    pub async fn aexists_by_key(&self, key: &str) -> Result<bool> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        Ok(connection.exists(key).await?)
    }

    /// Checks whether multiple cached embeddings exist for content/model pairs asynchronously.
    pub async fn amexists<I, S>(&self, contents: I, model_name: &str) -> Result<Vec<bool>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = contents
            .into_iter()
            .map(|content| self.make_cache_key(content.as_ref(), model_name))
            .collect::<Vec<_>>();
        self.amexists_by_keys(keys).await
    }

    /// Checks whether multiple cached embeddings exist for Redis keys asynchronously.
    pub async fn amexists_by_keys<I, S>(&self, keys: I) -> Result<Vec<bool>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = collect_strings(keys);
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let client = self.config.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let mut results = Vec::with_capacity(keys.len());
        for key in keys {
            results.push(connection.exists(key).await?);
        }
        Ok(results)
    }

    /// Removes a cached embedding by content and model name asynchronously.
    pub async fn adrop(&self, content: &str, model_name: &str) -> Result<()> {
        let key = self.make_cache_key(content, model_name);
        self.adrop_by_key(&key).await
    }

    /// Removes a cached embedding by Redis key asynchronously.
    pub async fn adrop_by_key(&self, key: &str) -> Result<()> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let _: usize = connection.del(key).await?;
        Ok(())
    }

    /// Removes multiple cached embeddings by content and model name asynchronously.
    pub async fn amdrop<I, S>(&self, contents: I, model_name: &str) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = contents
            .into_iter()
            .map(|content| self.make_cache_key(content.as_ref(), model_name))
            .collect::<Vec<_>>();
        self.amdrop_by_keys(keys).await
    }

    /// Removes multiple cached embeddings by Redis key asynchronously.
    pub async fn amdrop_by_keys<I, S>(&self, keys: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let keys = collect_strings(keys);
        if keys.is_empty() {
            return Ok(());
        }

        let client = self.config.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let _: usize = connection.del(keys).await?;
        Ok(())
    }

    /// Clears every cache entry under this cache namespace asynchronously.
    pub async fn aclear(&self) -> Result<usize> {
        let keys = self.aall_keys().await?;
        if keys.is_empty() {
            return Ok(0);
        }

        let count = keys.len();
        self.amdrop_by_keys(keys).await?;
        Ok(count)
    }

    fn prepare_entry(
        &self,
        content: &str,
        model_name: &str,
        embedding: &[f32],
        metadata: Option<Value>,
    ) -> EmbeddingCacheEntry {
        EmbeddingCacheEntry {
            entry_id: self.make_entry_id(content, model_name),
            content: content.to_owned(),
            model_name: model_name.to_owned(),
            embedding: embedding.to_vec(),
            metadata,
        }
    }

    fn write_entry(&self, key: &str, entry: &EmbeddingCacheEntry) -> Result<()> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut cmd = redis::cmd("HSET");
        cmd.arg(key)
            .arg("entry_id")
            .arg(&entry.entry_id)
            .arg("content")
            .arg(&entry.content)
            .arg("model_name")
            .arg(&entry.model_name)
            .arg("embedding")
            .arg(serde_json::to_string(&entry.embedding)?);

        if let Some(metadata) = &entry.metadata {
            cmd.arg("metadata").arg(serde_json::to_string(metadata)?);
        }

        let _: usize = cmd.query(&mut connection)?;
        Ok(())
    }

    async fn awrite_entry(&self, key: &str, entry: &EmbeddingCacheEntry) -> Result<()> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let mut cmd = redis::cmd("HSET");
        cmd.arg(key)
            .arg("entry_id")
            .arg(&entry.entry_id)
            .arg("content")
            .arg(&entry.content)
            .arg("model_name")
            .arg(&entry.model_name)
            .arg("embedding")
            .arg(serde_json::to_string(&entry.embedding)?);

        if let Some(metadata) = &entry.metadata {
            cmd.arg("metadata").arg(serde_json::to_string(metadata)?);
        }

        let _: usize = cmd.query_async(&mut connection).await?;
        Ok(())
    }

    fn expire_key(&self, key: &str, ttl_override: Option<u64>) -> Result<()> {
        if let Some(ttl_seconds) = ttl_override.or(self.config.ttl_seconds) {
            let client = self.config.connection.client()?;
            let mut connection = client.get_connection()?;
            let _: bool = redis::cmd("EXPIRE")
                .arg(key)
                .arg(ttl_seconds)
                .query(&mut connection)?;
        }
        Ok(())
    }

    async fn aexpire_key(&self, key: &str, ttl_override: Option<u64>) -> Result<()> {
        if let Some(ttl_seconds) = ttl_override.or(self.config.ttl_seconds) {
            let client = self.config.connection.client()?;
            let mut connection = client.get_multiplexed_async_connection().await?;
            let _: bool = redis::cmd("EXPIRE")
                .arg(key)
                .arg(ttl_seconds)
                .query_async(&mut connection)
                .await?;
        }
        Ok(())
    }

    fn all_keys(&self) -> Result<Vec<String>> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_connection()?;
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(format!("{}:*", self.config.name))
            .query(&mut connection)?;
        Ok(keys)
    }

    async fn aall_keys(&self) -> Result<Vec<String>> {
        let client = self.config.connection.client()?;
        let mut connection = client.get_multiplexed_async_connection().await?;
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(format!("{}:*", self.config.name))
            .query_async(&mut connection)
            .await?;
        Ok(keys)
    }

    fn key_for_entry(&self, entry_id: &str) -> String {
        format!("{}:{entry_id}", self.config.name)
    }
}

fn collect_strings<I, S>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    values
        .into_iter()
        .map(|value| value.as_ref().to_owned())
        .collect()
}

fn parse_entry(data: HashMap<String, String>) -> Result<Option<EmbeddingCacheEntry>> {
    if data.is_empty() {
        return Ok(None);
    }

    let entry = EmbeddingCacheEntry {
        entry_id: data.get("entry_id").cloned().unwrap_or_default(),
        content: data.get("content").cloned().unwrap_or_default(),
        model_name: data.get("model_name").cloned().unwrap_or_default(),
        embedding: match data.get("embedding") {
            Some(value) => serde_json::from_str::<Vec<f32>>(value)?,
            None => Vec::new(),
        },
        metadata: data
            .get("metadata")
            .map(|value| serde_json::from_str::<Value>(value))
            .transpose()?,
    };

    Ok(Some(entry))
}

fn hashify(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn semantic_cache_schema(
    name: &str,
    vector_dimensions: usize,
    filterable_fields: &[Value],
) -> Value {
    let mut fields = vec![
        json!({ "name": SEMANTIC_ENTRY_ID_FIELD, "type": "tag" }),
        json!({ "name": SEMANTIC_PROMPT_FIELD, "type": "text" }),
        json!({ "name": SEMANTIC_RESPONSE_FIELD, "type": "text" }),
        json!({ "name": SEMANTIC_INSERTED_AT_FIELD, "type": "numeric" }),
        json!({ "name": SEMANTIC_UPDATED_AT_FIELD, "type": "numeric" }),
        json!({ "name": SEMANTIC_METADATA_FIELD, "type": "text" }),
        json!({
            "name": SEMANTIC_VECTOR_FIELD,
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": vector_dimensions,
                "datatype": "float32",
                "distance_metric": "cosine"
            }
        }),
    ];
    fields.extend(filterable_fields.iter().cloned());
    json!({
        "index": {
            "name": name,
            "prefix": name,
            "storage_type": "hash",
        },
        "fields": fields,
    })
}

fn default_semantic_return_fields() -> Vec<String> {
    vec![
        SEMANTIC_ENTRY_ID_FIELD.to_owned(),
        SEMANTIC_PROMPT_FIELD.to_owned(),
        SEMANTIC_RESPONSE_FIELD.to_owned(),
        "vector_distance".to_owned(),
        SEMANTIC_INSERTED_AT_FIELD.to_owned(),
        SEMANTIC_UPDATED_AT_FIELD.to_owned(),
        SEMANTIC_METADATA_FIELD.to_owned(),
    ]
}

fn current_timestamp() -> f64 {
    Utc::now().timestamp_millis() as f64 / 1000.0
}

fn semantic_entry_id(prompt: &str, filters: Option<&Map<String, Value>>) -> String {
    if let Some(filters) = filters {
        let mut parts = filters
            .iter()
            .map(|(key, value)| format!("{key}{}", value_to_hash_string(value)))
            .collect::<Vec<_>>();
        parts.sort();
        hashify(&format!("{prompt}{}", parts.join("")))
    } else {
        hashify(prompt)
    }
}

fn value_to_hash_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn validate_distance_threshold(distance_threshold: f32) -> Result<()> {
    if !(0.0..=2.0).contains(&distance_threshold) {
        return Err(crate::Error::InvalidInput(format!(
            "distance threshold must be between 0 and 2, got {distance_threshold}"
        )));
    }
    Ok(())
}

fn validate_metadata(metadata: &Value) -> Result<()> {
    if !metadata.is_object() {
        return Err(crate::Error::InvalidInput(
            "metadata must be a JSON object".to_owned(),
        ));
    }
    Ok(())
}

fn query_output_documents(output: QueryOutput) -> Result<Vec<Map<String, Value>>> {
    match output {
        QueryOutput::Documents(documents) => Ok(documents),
        QueryOutput::Count(_) => Err(crate::Error::InvalidInput(
            "semantic cache queries must return documents".to_owned(),
        )),
    }
}

fn process_semantic_hits(
    documents: Vec<Map<String, Value>>,
    return_fields: Option<&[&str]>,
) -> Result<Vec<Map<String, Value>>> {
    let selected = return_fields.map(|fields| {
        fields
            .iter()
            .map(|field| (*field).to_owned())
            .collect::<std::collections::HashSet<_>>()
    });
    let mut hits = Vec::with_capacity(documents.len());
    for mut document in documents {
        let key = document
            .remove("id")
            .unwrap_or_else(|| Value::String(String::new()));
        let mut hit = Map::new();
        hit.insert(SEMANTIC_KEY_FIELD.to_owned(), key);
        for (field, value) in document {
            let include = selected
                .as_ref()
                .is_none_or(|fields| fields.contains(&field));
            if !include {
                continue;
            }
            hit.insert(field.clone(), normalize_semantic_value(&field, value)?);
        }
        hits.push(hit);
    }
    Ok(hits)
}

fn normalize_semantic_value(field: &str, value: Value) -> Result<Value> {
    match (field, value) {
        (SEMANTIC_METADATA_FIELD, Value::String(value)) => {
            Ok(serde_json::from_str(&value).unwrap_or(Value::String(value)))
        }
        (
            "vector_distance" | SEMANTIC_INSERTED_AT_FIELD | SEMANTIC_UPDATED_AT_FIELD,
            Value::String(value),
        ) => {
            let parsed = value.parse::<f64>().map_err(|_| {
                crate::Error::InvalidInput(format!("could not parse numeric field '{field}'"))
            })?;
            Ok(number_value(parsed))
        }
        (_, value) => Ok(value),
    }
}

fn prepare_semantic_update_fields(fields: Map<String, Value>) -> Result<Vec<(String, String)>> {
    let mut mapping = Vec::with_capacity(fields.len() + 1);
    for (field, value) in fields {
        if field == SEMANTIC_VECTOR_FIELD {
            return Err(crate::Error::InvalidInput(
                "updating the stored vector is not supported yet".to_owned(),
            ));
        }
        if field == SEMANTIC_METADATA_FIELD {
            validate_metadata(&value)?;
        }
        let serialized = match value {
            Value::Null => "null".to_owned(),
            Value::Bool(value) => value.to_string(),
            Value::Number(value) => value.to_string(),
            Value::String(value) => value,
            Value::Array(_) | Value::Object(_) => serde_json::to_string(&value)?,
        };
        mapping.push((field, serialized));
    }
    mapping.push((
        SEMANTIC_UPDATED_AT_FIELD.to_owned(),
        current_timestamp().to_string(),
    ));
    Ok(mapping)
}

fn number_value(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::{CacheConfig, EmbeddingsCache, hashify};

    #[test]
    fn hashify_matches_expected_sha256() {
        assert_eq!(
            hashify("Hello world:text-embedding-ada-002"),
            "368dacc611e96e4189a9809faaca1a70b3c3306352bbcfc9ab6291359a5dfca0"
        );
    }

    #[test]
    fn cache_key_is_stable() {
        let cache = EmbeddingsCache::new(CacheConfig::default());
        let key = cache.make_cache_key("Hello world", "text-embedding-ada-002");
        assert_eq!(
            key,
            "embedcache:368dacc611e96e4189a9809faaca1a70b3c3306352bbcfc9ab6291359a5dfca0"
        );
    }
}
