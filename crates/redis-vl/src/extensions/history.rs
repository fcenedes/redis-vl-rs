//! Message history extension types.
//!
//! [`MessageHistory`] stores conversation messages in Redis with ordered
//! retrieval, role-based filtering, and session management. Messages are
//! stored as Redis hashes under session-namespaced keys.
//!
//! [`SemanticMessageHistory`] extends `MessageHistory` with vector-based
//! semantic recall — in addition to ordered retrieval, it finds the most
//! relevant past messages for a given prompt using vector similarity search.

use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value, json};
use ulid::Ulid;

use crate::{
    error::{Error, Result},
    filter::Tag,
    index::{QueryOutput, RedisConnectionInfo, SearchIndex},
    query::{CountQuery, FilterQuery, SortDirection, Vector, VectorRangeQuery},
    schema::VectorDataType,
    vectorizers::Vectorizer,
};

const DEFAULT_TOP_K: usize = 5;
const HISTORY_SCAN_COUNT: usize = 100;
const HISTORY_PAGE_SIZE: usize = 500;

const SEMANTIC_ENTRY_ID_FIELD: &str = "entry_id";
const SEMANTIC_ROLE_FIELD: &str = "role";
const SEMANTIC_CONTENT_FIELD: &str = "content";
const SEMANTIC_TOOL_CALL_ID_FIELD: &str = "tool_call_id";
const SEMANTIC_TIMESTAMP_FIELD: &str = "timestamp";
const SEMANTIC_SESSION_FIELD: &str = "session_tag";
const SEMANTIC_METADATA_FIELD: &str = "metadata";
const SEMANTIC_VECTOR_FIELD: &str = "message_vector";

/// Supported chat roles.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System instruction message.
    System,
    /// End-user message.
    User,
    /// Assistant model message.
    Llm,
    /// Tool result message.
    Tool,
}

impl MessageRole {
    /// Returns the RedisVL role name used for serialization and filtering.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Llm => "llm",
            Self::Tool => "tool",
        }
    }
}

impl TryFrom<&str> for MessageRole {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "llm" => Ok(Self::Llm),
            "tool" => Ok(Self::Tool),
            other => Err(Error::InvalidInput(format!(
                "Invalid role '{other}'. Valid roles: system, user, llm, tool"
            ))),
        }
    }
}

/// Message payload stored by history extensions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// Unique entry identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<String>,
    /// Message role.
    pub role: MessageRole,
    /// Message content.
    pub content: String,
    /// Session tag associated with the message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_tag: Option<String>,
    /// Message timestamp in seconds since the Unix epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<f64>,
    /// Optional tool call identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional arbitrary metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Message {
    /// Creates a new message with the provided role and content.
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            entry_id: None,
            role,
            content: content.into(),
            session_tag: None,
            timestamp: None,
            tool_call_id: None,
            metadata: None,
        }
    }

    fn with_defaults(mut self, session_tag: &str) -> Self {
        let timestamp = self.timestamp.unwrap_or_else(current_timestamp);
        self.timestamp = Some(timestamp);
        self.session_tag
            .get_or_insert_with(|| session_tag.to_owned());
        self.entry_id
            .get_or_insert_with(|| format!("{session_tag}:{timestamp}:{}", Ulid::new()));
        self
    }
}

/// Basic message history handle.
#[derive(Debug, Clone)]
pub struct MessageHistory {
    /// History namespace.
    pub name: String,
    /// Redis connection settings.
    pub connection: RedisConnectionInfo,
    default_session_tag: String,
}

impl MessageHistory {
    /// Creates a new history configuration.
    pub fn new(name: impl Into<String>, redis_url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            connection: RedisConnectionInfo::new(redis_url),
            default_session_tag: Ulid::new().to_string(),
        }
    }

    /// Returns the default session tag used when no explicit session is supplied.
    pub fn default_session_tag(&self) -> &str {
        &self.default_session_tag
    }

    /// Appends a single message under the default session tag.
    pub fn add_message(&self, message: Message) -> Result<()> {
        self.add_message_in_session(message, None)
    }

    /// Appends a single message under the selected session tag.
    pub fn add_message_in_session(
        &self,
        message: Message,
        session_tag: Option<&str>,
    ) -> Result<()> {
        let session_tag = session_tag.unwrap_or(&self.default_session_tag);
        let message = message.with_defaults(session_tag);
        let payload = serde_json::to_string(&message)?;
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let _: usize = redis::cmd("RPUSH")
            .arg(self.session_key(session_tag))
            .arg(payload)
            .query(&mut connection)?;
        Ok(())
    }

    /// Appends multiple messages under the default session tag.
    pub fn add_messages(&self, messages: &[Message]) -> Result<()> {
        self.add_messages_in_session(messages, None)
    }

    /// Appends multiple messages under the selected session tag.
    pub fn add_messages_in_session(
        &self,
        messages: &[Message],
        session_tag: Option<&str>,
    ) -> Result<()> {
        for message in messages {
            self.add_message_in_session(message.clone(), session_tag)?;
        }
        Ok(())
    }

    /// Stores a prompt/response pair under the default session tag.
    pub fn store(&self, prompt: &str, response: &str) -> Result<()> {
        self.store_in_session(prompt, response, None)
    }

    /// Stores a prompt/response pair under the selected session tag.
    pub fn store_in_session(
        &self,
        prompt: &str,
        response: &str,
        session_tag: Option<&str>,
    ) -> Result<()> {
        self.add_messages_in_session(
            &[
                Message::new(MessageRole::User, prompt),
                Message::new(MessageRole::Llm, response),
            ],
            session_tag,
        )
    }

    /// Returns the most recent messages for the selected session, ordered oldest to newest.
    pub fn get_recent(&self, top_k: usize, session_tag: Option<&str>) -> Result<Vec<Message>> {
        self.get_recent_with_roles(top_k, session_tag, None)
    }

    /// Returns the most recent messages for the selected session, optionally
    /// filtered by role, ordered oldest to newest.
    pub fn get_recent_with_roles(
        &self,
        top_k: usize,
        session_tag: Option<&str>,
        roles: Option<&[MessageRole]>,
    ) -> Result<Vec<Message>> {
        validate_top_k(top_k)?;
        let roles = normalize_roles(roles)?;
        let session_tag = session_tag.unwrap_or(&self.default_session_tag);
        let mut messages = self.read_all_messages(session_tag)?;
        apply_role_filter(&mut messages, roles);
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let start = messages.len().saturating_sub(top_k);
        Ok(messages[start..].to_vec())
    }

    /// Returns the most recent messages rendered as plain text.
    pub fn get_recent_as_text(
        &self,
        top_k: usize,
        session_tag: Option<&str>,
    ) -> Result<Vec<String>> {
        self.get_recent_as_text_with_roles(top_k, session_tag, None)
    }

    /// Returns the most recent role-filtered messages rendered as plain text.
    pub fn get_recent_as_text_with_roles(
        &self,
        top_k: usize,
        session_tag: Option<&str>,
        roles: Option<&[MessageRole]>,
    ) -> Result<Vec<String>> {
        Ok(self
            .get_recent_with_roles(top_k, session_tag, roles)?
            .into_iter()
            .map(|message| message.content)
            .collect())
    }

    /// Returns every message for the default session.
    pub fn messages(&self) -> Result<Vec<Message>> {
        self.messages_in_session(None)
    }

    /// Returns every message for the selected session.
    pub fn messages_in_session(&self, session_tag: Option<&str>) -> Result<Vec<Message>> {
        self.read_all_messages(session_tag.unwrap_or(&self.default_session_tag))
    }

    /// Removes the most recent message from the default session, or the specified entry.
    pub fn drop(&self, entry_id: Option<&str>) -> Result<()> {
        match entry_id {
            Some(entry_id) => self.drop_by_id(entry_id),
            None => {
                let client = self.connection.client()?;
                let mut connection = client.get_connection()?;
                let _: Option<String> = redis::cmd("RPOP")
                    .arg(self.session_key(&self.default_session_tag))
                    .query(&mut connection)?;
                Ok(())
            }
        }
    }

    /// Removes every stored session for this history namespace.
    pub fn delete(&self) -> Result<usize> {
        let keys = self.all_session_keys()?;
        if keys.is_empty() {
            return Ok(0);
        }

        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut cmd = redis::cmd("DEL");
        for key in &keys {
            cmd.arg(key);
        }
        let deleted: usize = cmd.query(&mut connection)?;
        Ok(deleted)
    }

    /// Clears all messages for the selected session, returning the number removed.
    pub fn clear_session(&self, session_tag: Option<&str>) -> Result<usize> {
        let session_tag = session_tag.unwrap_or(&self.default_session_tag);
        let count = self.count(session_tag)?;
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let _: usize = redis::cmd("DEL")
            .arg(self.session_key(session_tag))
            .query(&mut connection)?;
        Ok(count)
    }

    /// Clears all messages for the default session.
    pub fn clear(&self) -> Result<usize> {
        self.clear_session(None)
    }

    /// Counts messages in the selected session.
    pub fn count(&self, session_tag: &str) -> Result<usize> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let count: usize = redis::cmd("LLEN")
            .arg(self.session_key(session_tag))
            .query(&mut connection)?;
        Ok(count)
    }

    fn read_all_messages(&self, session_tag: &str) -> Result<Vec<Message>> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let values: Vec<String> = redis::cmd("LRANGE")
            .arg(self.session_key(session_tag))
            .arg(0)
            .arg(-1)
            .query(&mut connection)?;
        values
            .into_iter()
            .map(|value| serde_json::from_str(&value).map_err(Error::from))
            .collect()
    }

    fn drop_by_id(&self, entry_id: &str) -> Result<()> {
        let sessions = self.all_session_keys()?;
        for key in sessions {
            let messages = self.read_messages_from_key(&key)?;
            if let Some(message) = messages
                .into_iter()
                .find(|message| message.entry_id.as_deref() == Some(entry_id))
            {
                let payload = serde_json::to_string(&message)?;
                let client = self.connection.client()?;
                let mut connection = client.get_connection()?;
                let _: usize = redis::cmd("LREM")
                    .arg(&key)
                    .arg(1)
                    .arg(payload)
                    .query(&mut connection)?;
                return Ok(());
            }
        }
        Ok(())
    }

    fn all_session_keys(&self) -> Result<Vec<String>> {
        let pattern = format!("{}:history:*", self.name);
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut cursor = 0_u64;
        let mut keys = Vec::new();

        loop {
            let (next_cursor, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(HISTORY_SCAN_COUNT)
                .query(&mut connection)?;
            keys.extend(batch);
            if next_cursor == 0 {
                break;
            }
            cursor = next_cursor;
        }

        Ok(keys)
    }

    fn read_messages_from_key(&self, key: &str) -> Result<Vec<Message>> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let values: Vec<String> = redis::cmd("LRANGE")
            .arg(key)
            .arg(0)
            .arg(-1)
            .query(&mut connection)?;
        values
            .into_iter()
            .map(|value| serde_json::from_str(&value).map_err(Error::from))
            .collect()
    }

    fn session_key(&self, session_tag: &str) -> String {
        format!("{}:history:{session_tag}", self.name)
    }
}

/// Semantic message history backed by a Redis Search vector index.
#[derive(Clone)]
pub struct SemanticMessageHistory {
    /// Base history handle used for default session management.
    pub history: MessageHistory,
    /// Distance threshold used for semantic recall.
    pub distance_threshold: f32,
    /// Vector element data type used for the index schema.
    pub dtype: VectorDataType,
    /// Underlying search index for semantic lookups.
    pub index: SearchIndex,
    vectorizer: Arc<dyn Vectorizer>,
    vector_dimensions: usize,
}

impl SemanticMessageHistory {
    /// Creates a new semantic message history backed by a hash-based Redis Search index.
    ///
    /// Uses [`VectorDataType::Float32`] by default. For other data types, use
    /// [`Self::new_with_options`].
    pub fn new<V>(
        name: impl Into<String>,
        redis_url: impl Into<String>,
        distance_threshold: f32,
        vector_dimensions: usize,
        vectorizer: V,
    ) -> Result<Self>
    where
        V: Vectorizer + 'static,
    {
        Self::new_with_options(
            name,
            redis_url,
            distance_threshold,
            vector_dimensions,
            vectorizer,
            VectorDataType::Float32,
            false,
        )
    }

    /// Creates a new semantic message history with explicit dtype and overwrite control.
    ///
    /// When `overwrite` is true the existing index is dropped and recreated.
    /// When false and an index with the same name already exists, the existing
    /// index is reused (the schema must match).
    pub fn new_with_options<V>(
        name: impl Into<String>,
        redis_url: impl Into<String>,
        distance_threshold: f32,
        vector_dimensions: usize,
        vectorizer: V,
        dtype: VectorDataType,
        overwrite: bool,
    ) -> Result<Self>
    where
        V: Vectorizer + 'static,
    {
        validate_distance_threshold(distance_threshold)?;
        if vector_dimensions == 0 {
            return Err(Error::InvalidInput(
                "vector_dimensions must be greater than zero".to_owned(),
            ));
        }

        let name = name.into();
        let redis_url = redis_url.into();
        let history = MessageHistory::new(name.clone(), redis_url.clone());
        let index = SearchIndex::from_json_value(
            semantic_message_history_schema(&name, vector_dimensions, dtype),
            redis_url.clone(),
        )?;

        // Validate schema compatibility with existing index (mirrors Python behavior)
        if !overwrite && index.exists().unwrap_or(false) {
            let existing_index = SearchIndex::from_existing(&name, redis_url)?;
            if existing_index.schema().to_json_value()? != index.schema().to_json_value()? {
                return Err(Error::InvalidInput(format!(
                    "Existing index {name} schema does not match the user provided schema for the semantic message history. \
                     If you wish to overwrite the index schema, set overwrite=true during initialization."
                )));
            }
        }

        index.create_with_options(overwrite, false)?;

        Ok(Self {
            history,
            distance_threshold,
            dtype,
            index,
            vectorizer: Arc::new(vectorizer),
            vector_dimensions,
        })
    }

    /// Creates a new semantic message history using the default HuggingFace
    /// local vectorizer (`AllMiniLML6V2`).
    ///
    /// This convenience constructor requires no API key — the model runs
    /// locally via ONNX Runtime and is downloaded from HuggingFace Hub on
    /// first use.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or the index cannot be
    /// created.
    #[cfg(feature = "hf-local")]
    pub fn with_default_vectorizer(
        name: impl Into<String>,
        redis_url: impl Into<String>,
        distance_threshold: f32,
        vector_dimensions: usize,
    ) -> Result<Self> {
        let vectorizer = crate::vectorizers::HuggingFaceTextVectorizer::new(Default::default())?;
        Self::new(
            name,
            redis_url,
            distance_threshold,
            vector_dimensions,
            vectorizer,
        )
    }

    /// Returns the default session tag used when no explicit session is supplied.
    pub fn default_session_tag(&self) -> &str {
        self.history.default_session_tag()
    }

    /// Updates the default semantic distance threshold.
    pub fn set_distance_threshold(&mut self, distance_threshold: f32) -> Result<()> {
        validate_distance_threshold(distance_threshold)?;
        self.distance_threshold = distance_threshold;
        Ok(())
    }

    /// Appends a single message under the default session tag.
    pub fn add_message(&self, message: Message) -> Result<()> {
        self.add_message_in_session(message, None)
    }

    /// Appends a single message under the selected session tag.
    pub fn add_message_in_session(
        &self,
        message: Message,
        session_tag: Option<&str>,
    ) -> Result<()> {
        self.add_messages_in_session(std::slice::from_ref(&message), session_tag)
    }

    /// Appends multiple messages under the default session tag.
    pub fn add_messages(&self, messages: &[Message]) -> Result<()> {
        self.add_messages_in_session(messages, None)
    }

    /// Appends multiple messages under the selected session tag.
    pub fn add_messages_in_session(
        &self,
        messages: &[Message],
        session_tag: Option<&str>,
    ) -> Result<()> {
        let session_tag = session_tag.unwrap_or(self.default_session_tag());
        let mut records = Vec::with_capacity(messages.len());
        for message in messages {
            records.push(self.prepare_message_record(message.clone(), session_tag)?);
        }
        self.index.load(&records, SEMANTIC_ENTRY_ID_FIELD, None)?;
        Ok(())
    }

    /// Stores a prompt/response pair under the default session tag.
    pub fn store(&self, prompt: &str, response: &str) -> Result<()> {
        self.store_in_session(prompt, response, None)
    }

    /// Stores a prompt/response pair under the selected session tag.
    pub fn store_in_session(
        &self,
        prompt: &str,
        response: &str,
        session_tag: Option<&str>,
    ) -> Result<()> {
        self.add_messages_in_session(
            &[
                Message::new(MessageRole::User, prompt),
                Message::new(MessageRole::Llm, response),
            ],
            session_tag,
        )
    }

    /// Returns the recent messages for the selected session, ordered oldest to newest.
    pub fn get_recent(&self, top_k: usize, session_tag: Option<&str>) -> Result<Vec<Message>> {
        self.get_recent_with_roles(top_k, session_tag, None)
    }

    /// Returns the recent messages for the selected session, optionally
    /// filtered by role, ordered oldest to newest.
    pub fn get_recent_with_roles(
        &self,
        top_k: usize,
        session_tag: Option<&str>,
        roles: Option<&[MessageRole]>,
    ) -> Result<Vec<Message>> {
        validate_top_k(top_k)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let documents = self.query_recent_documents(top_k, session_tag, roles)?;
        let mut messages = documents
            .into_iter()
            .map(message_from_document)
            .collect::<Result<Vec<_>>>()?;
        messages.reverse();
        Ok(messages)
    }

    /// Returns the recent messages rendered as plain text.
    pub fn get_recent_as_text(
        &self,
        top_k: usize,
        session_tag: Option<&str>,
    ) -> Result<Vec<String>> {
        self.get_recent_as_text_with_roles(top_k, session_tag, None)
    }

    /// Returns the recent role-filtered messages rendered as plain text.
    pub fn get_recent_as_text_with_roles(
        &self,
        top_k: usize,
        session_tag: Option<&str>,
        roles: Option<&[MessageRole]>,
    ) -> Result<Vec<String>> {
        Ok(self
            .get_recent_with_roles(top_k, session_tag, roles)?
            .into_iter()
            .map(|message| message.content)
            .collect())
    }

    /// Returns every message for the default session.
    pub fn messages(&self) -> Result<Vec<Message>> {
        self.messages_in_session(None)
    }

    /// Returns every message for the selected session.
    pub fn messages_in_session(&self, session_tag: Option<&str>) -> Result<Vec<Message>> {
        self.query_all_documents(session_tag, None)?
            .into_iter()
            .map(message_from_document)
            .collect()
    }

    /// Returns the most semantically relevant messages for the supplied prompt.
    pub fn get_relevant(&self, prompt: &str) -> Result<Vec<Message>> {
        self.get_relevant_with_options(prompt, DEFAULT_TOP_K, None, None, None, false)
    }

    /// Returns the most semantically relevant messages for the supplied prompt
    /// with explicit recall controls.
    pub fn get_relevant_with_options(
        &self,
        prompt: &str,
        top_k: usize,
        session_tag: Option<&str>,
        roles: Option<&[MessageRole]>,
        distance_threshold: Option<f32>,
        fall_back: bool,
    ) -> Result<Vec<Message>> {
        validate_top_k(top_k)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let distance_threshold = distance_threshold.unwrap_or(self.distance_threshold);
        validate_distance_threshold(distance_threshold)?;
        let vector = self.vectorizer.embed(prompt)?;
        self.validate_vector_dimensions(&vector)?;
        let filter_expression =
            semantic_session_role_filter(self.default_session_tag(), session_tag, roles)?;
        let query = VectorRangeQuery::new(
            Vector::new(vector),
            SEMANTIC_VECTOR_FIELD,
            distance_threshold,
        )
        .with_filter(filter_expression)
        .with_return_fields(semantic_return_fields())
        .paging(0, top_k);

        let documents = query_output_documents(self.index.query(&query)?)?;
        if documents.is_empty() && fall_back {
            return self.get_recent_with_roles(top_k, session_tag, roles);
        }

        documents
            .into_iter()
            .map(message_from_document)
            .collect::<Result<Vec<_>>>()
    }

    /// Removes the most recent message from the default session, or the specified entry.
    pub fn drop(&self, entry_id: Option<&str>) -> Result<usize> {
        match entry_id {
            Some(entry_id) => self.index.drop_document(entry_id),
            None => {
                let recent = self.get_recent(1, None)?;
                let Some(entry_id) = recent
                    .first()
                    .and_then(|message| message.entry_id.as_deref())
                else {
                    return Ok(0);
                };
                self.index.drop_document(entry_id)
            }
        }
    }

    /// Counts messages in the selected session.
    pub fn count(&self, session_tag: Option<&str>) -> Result<usize> {
        let filter_expression =
            semantic_session_role_filter(self.default_session_tag(), session_tag, None)?;
        let query = CountQuery::new().with_filter(filter_expression);
        match self.index.query(&query)? {
            QueryOutput::Count(count) => Ok(count),
            QueryOutput::Documents(_) => Err(Error::InvalidInput(
                "semantic message history count query returned documents".to_owned(),
            )),
        }
    }

    /// Clears every stored semantic message while keeping the index itself.
    pub fn clear(&self) -> Result<usize> {
        self.index.clear()
    }

    /// Clears the selected session while keeping the index itself.
    pub fn clear_session(&self, session_tag: Option<&str>) -> Result<usize> {
        let documents = self.query_all_documents(session_tag, None)?;
        let ids = documents
            .into_iter()
            .filter_map(|document| {
                document
                    .get(SEMANTIC_ENTRY_ID_FIELD)
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .collect::<Vec<_>>();
        self.index.drop_documents(&ids)
    }

    /// Removes the semantic history index and all indexed documents.
    pub fn delete(&self) -> Result<()> {
        if self.index.exists()? {
            self.index.delete(true)?;
        }
        Ok(())
    }

    fn prepare_message_record(&self, message: Message, session_tag: &str) -> Result<Value> {
        let message = message.with_defaults(session_tag);
        let vector = self.vectorizer.embed(&message.content)?;
        self.validate_vector_dimensions(&vector)?;

        let entry_id = message.entry_id.clone().ok_or_else(|| {
            Error::InvalidInput("message entry_id was not initialized".to_owned())
        })?;
        let timestamp = message.timestamp.ok_or_else(|| {
            Error::InvalidInput("message timestamp was not initialized".to_owned())
        })?;
        let mut record = Map::new();
        record.insert(SEMANTIC_ENTRY_ID_FIELD.to_owned(), Value::String(entry_id));
        record.insert(
            SEMANTIC_ROLE_FIELD.to_owned(),
            Value::String(message.role.as_str().to_owned()),
        );
        record.insert(
            SEMANTIC_CONTENT_FIELD.to_owned(),
            Value::String(message.content),
        );
        record.insert(
            SEMANTIC_SESSION_FIELD.to_owned(),
            Value::String(session_tag.to_owned()),
        );
        record.insert(
            SEMANTIC_TIMESTAMP_FIELD.to_owned(),
            number_value(timestamp)?,
        );
        if let Some(tool_call_id) = message.tool_call_id {
            record.insert(
                SEMANTIC_TOOL_CALL_ID_FIELD.to_owned(),
                Value::String(tool_call_id),
            );
        }
        if let Some(metadata) = message.metadata {
            record.insert(
                SEMANTIC_METADATA_FIELD.to_owned(),
                Value::String(serde_json::to_string(&metadata)?),
            );
        }
        record.insert(
            SEMANTIC_VECTOR_FIELD.to_owned(),
            serde_json::to_value(vector)?,
        );
        Ok(Value::Object(record))
    }

    fn query_recent_documents(
        &self,
        top_k: usize,
        session_tag: Option<&str>,
        roles: Option<&[MessageRole]>,
    ) -> Result<Vec<Map<String, Value>>> {
        let filter_expression =
            semantic_session_role_filter(self.default_session_tag(), session_tag, roles)?;
        let query = FilterQuery::new(filter_expression)
            .with_return_fields(semantic_return_fields())
            .sort_by(SEMANTIC_TIMESTAMP_FIELD, SortDirection::Desc)
            .paging(0, top_k);
        query_output_documents(self.index.query(&query)?)
    }

    fn query_all_documents(
        &self,
        session_tag: Option<&str>,
        roles: Option<&[MessageRole]>,
    ) -> Result<Vec<Map<String, Value>>> {
        let filter_expression =
            semantic_session_role_filter(self.default_session_tag(), session_tag, roles)?;
        let query = FilterQuery::new(filter_expression)
            .with_return_fields(semantic_return_fields())
            .sort_by(SEMANTIC_TIMESTAMP_FIELD, SortDirection::Asc);
        let batches = self.index.paginate(&query, HISTORY_PAGE_SIZE)?;
        Ok(batches.into_iter().flatten().collect())
    }

    fn validate_vector_dimensions(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.vector_dimensions {
            return Err(Error::InvalidInput(format!(
                "vectorizer produced {} dimensions, expected {}",
                vector.len(),
                self.vector_dimensions
            )));
        }
        Ok(())
    }
}

impl std::fmt::Debug for SemanticMessageHistory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticMessageHistory")
            .field("history", &self.history)
            .field("distance_threshold", &self.distance_threshold)
            .field("vector_dimensions", &self.vector_dimensions)
            .finish()
    }
}

fn current_timestamp() -> f64 {
    Utc::now().timestamp_millis() as f64 / 1000.0
}

fn validate_top_k(top_k: usize) -> Result<()> {
    if top_k > 100_000 {
        return Err(Error::InvalidInput(
            "top_k is unreasonably large".to_owned(),
        ));
    }
    Ok(())
}

fn validate_distance_threshold(distance_threshold: f32) -> Result<()> {
    if !(0.0..=2.0).contains(&distance_threshold) {
        return Err(Error::InvalidInput(format!(
            "distance threshold must be between 0 and 2, got {distance_threshold}"
        )));
    }
    Ok(())
}

fn normalize_roles<'a>(roles: Option<&'a [MessageRole]>) -> Result<Option<&'a [MessageRole]>> {
    match roles {
        Some([]) => Err(Error::InvalidInput("roles cannot be empty".to_owned())),
        other => Ok(other),
    }
}

fn apply_role_filter(messages: &mut Vec<Message>, roles: Option<&[MessageRole]>) {
    if let Some(roles) = roles {
        messages.retain(|message| roles.contains(&message.role));
    }
}

fn semantic_message_history_schema(
    name: &str,
    vector_dimensions: usize,
    dtype: VectorDataType,
) -> Value {
    json!({
        "index": {
            "name": name,
            "prefix": name,
            "storage_type": "hash",
        },
        "fields": [
            { "name": SEMANTIC_ENTRY_ID_FIELD, "type": "tag" },
            { "name": SEMANTIC_ROLE_FIELD, "type": "tag" },
            { "name": SEMANTIC_CONTENT_FIELD, "type": "text" },
            { "name": SEMANTIC_TOOL_CALL_ID_FIELD, "type": "tag" },
            { "name": SEMANTIC_TIMESTAMP_FIELD, "type": "numeric" },
            { "name": SEMANTIC_SESSION_FIELD, "type": "tag" },
            { "name": SEMANTIC_METADATA_FIELD, "type": "text" },
            {
                "name": SEMANTIC_VECTOR_FIELD,
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": vector_dimensions,
                    "datatype": dtype.as_str(),
                    "distance_metric": "cosine"
                }
            }
        ]
    })
}

fn semantic_return_fields() -> Vec<&'static str> {
    vec![
        SEMANTIC_ENTRY_ID_FIELD,
        SEMANTIC_ROLE_FIELD,
        SEMANTIC_CONTENT_FIELD,
        SEMANTIC_TOOL_CALL_ID_FIELD,
        SEMANTIC_TIMESTAMP_FIELD,
        SEMANTIC_SESSION_FIELD,
        SEMANTIC_METADATA_FIELD,
        "vector_distance",
    ]
}

fn semantic_session_role_filter(
    default_session_tag: &str,
    session_tag: Option<&str>,
    roles: Option<&[MessageRole]>,
) -> Result<crate::filter::FilterExpression> {
    let roles = normalize_roles(roles)?;
    let session_filter =
        Tag::new(SEMANTIC_SESSION_FIELD).eq(session_tag.unwrap_or(default_session_tag));
    let Some(roles) = roles else {
        return Ok(session_filter);
    };

    let mut role_filters = roles
        .iter()
        .copied()
        .map(|role| Tag::new(SEMANTIC_ROLE_FIELD).eq(role.as_str()));
    let Some(first) = role_filters.next() else {
        return Err(Error::InvalidInput("roles cannot be empty".to_owned()));
    };
    let role_filter = role_filters.fold(first, |combined, filter| combined | filter);
    Ok(session_filter & role_filter)
}

fn query_output_documents(output: QueryOutput) -> Result<Vec<Map<String, Value>>> {
    match output {
        QueryOutput::Documents(documents) => Ok(documents),
        QueryOutput::Count(_) => Err(Error::InvalidInput(
            "message history queries must return documents".to_owned(),
        )),
    }
}

fn message_from_document(mut document: Map<String, Value>) -> Result<Message> {
    let entry_id = string_field_optional(&mut document, SEMANTIC_ENTRY_ID_FIELD)?;
    let role =
        MessageRole::try_from(string_field_required(&mut document, SEMANTIC_ROLE_FIELD)?.as_str())?;
    let content = string_field_required(&mut document, SEMANTIC_CONTENT_FIELD)?;
    let session_tag = string_field_optional(&mut document, SEMANTIC_SESSION_FIELD)?;
    let timestamp = number_field_optional(&mut document, SEMANTIC_TIMESTAMP_FIELD)?;
    let tool_call_id = string_field_optional(&mut document, SEMANTIC_TOOL_CALL_ID_FIELD)?;
    let metadata = match string_field_optional(&mut document, SEMANTIC_METADATA_FIELD)? {
        Some(metadata) => Some(serde_json::from_str(&metadata)?),
        None => None,
    };

    Ok(Message {
        entry_id,
        role,
        content,
        session_tag,
        timestamp,
        tool_call_id,
        metadata,
    })
}

fn string_field_required(document: &mut Map<String, Value>, field: &str) -> Result<String> {
    string_field_optional(document, field)?
        .ok_or_else(|| Error::InvalidInput(format!("message history document missing '{field}'")))
}

fn string_field_optional(document: &mut Map<String, Value>, field: &str) -> Result<Option<String>> {
    match document.remove(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(other) => Err(Error::InvalidInput(format!(
            "message history field '{field}' expected a string, received {other}"
        ))),
    }
}

fn number_field_optional(document: &mut Map<String, Value>, field: &str) -> Result<Option<f64>> {
    match document.remove(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| Error::InvalidInput(format!("field '{field}' is not a float"))),
        Some(Value::String(value)) => value
            .parse::<f64>()
            .map(Some)
            .map_err(|_| Error::InvalidInput(format!("field '{field}' is not a float"))),
        Some(other) => Err(Error::InvalidInput(format!(
            "message history field '{field}' expected a number, received {other}"
        ))),
    }
}

fn number_value(value: f64) -> Result<Value> {
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| Error::InvalidInput("numeric value must be finite".to_owned()))
}

#[allow(dead_code)]
fn _default_top_k() -> usize {
    DEFAULT_TOP_K
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_role_as_str() {
        assert_eq!(MessageRole::System.as_str(), "system");
        assert_eq!(MessageRole::User.as_str(), "user");
        assert_eq!(MessageRole::Llm.as_str(), "llm");
        assert_eq!(MessageRole::Tool.as_str(), "tool");
    }

    #[test]
    fn message_role_try_from_valid() {
        assert_eq!(
            MessageRole::try_from("system").unwrap(),
            MessageRole::System
        );
        assert_eq!(MessageRole::try_from("user").unwrap(), MessageRole::User);
        assert_eq!(MessageRole::try_from("llm").unwrap(), MessageRole::Llm);
        assert_eq!(MessageRole::try_from("tool").unwrap(), MessageRole::Tool);
    }

    #[test]
    fn message_role_try_from_invalid() {
        assert!(MessageRole::try_from("admin").is_err());
        assert!(MessageRole::try_from("SYSTEM").is_err());
        assert!(MessageRole::try_from("User").is_err());
        assert!(MessageRole::try_from("").is_err());
    }

    #[test]
    fn message_role_serde_roundtrip() {
        let role = MessageRole::Llm;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"llm\"");
        let deserialized: MessageRole = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, role);
    }

    #[test]
    fn message_new_defaults() {
        let message = Message::new(MessageRole::User, "hello");
        assert_eq!(message.role, MessageRole::User);
        assert_eq!(message.content, "hello");
        assert!(message.entry_id.is_none());
        assert!(message.session_tag.is_none());
        assert!(message.timestamp.is_none());
        assert!(message.tool_call_id.is_none());
        assert!(message.metadata.is_none());
    }

    #[test]
    fn message_with_defaults_populates_fields() {
        let message = Message::new(MessageRole::User, "test content");
        let filled = message.with_defaults("my_session");
        assert!(filled.entry_id.is_some());
        assert_eq!(filled.session_tag.as_deref(), Some("my_session"));
        assert!(filled.timestamp.is_some());
        // Entry ID should contain the session tag
        let entry_id = filled.entry_id.unwrap();
        assert!(entry_id.starts_with("my_session:"));
    }

    #[test]
    fn message_with_defaults_preserves_existing() {
        let message = Message {
            entry_id: Some("custom_id".to_owned()),
            session_tag: Some("custom_session".to_owned()),
            timestamp: Some(42.0),
            ..Message::new(MessageRole::Llm, "content")
        };
        let filled = message.with_defaults("other_session");
        assert_eq!(filled.entry_id.as_deref(), Some("custom_id"));
        assert_eq!(filled.session_tag.as_deref(), Some("custom_session"));
        assert_eq!(filled.timestamp, Some(42.0));
    }

    #[test]
    fn message_with_defaults_unique_ids_for_same_session() {
        let session = "test_session";
        let ids: Vec<String> = (0..10)
            .map(|i| {
                let message = Message::new(MessageRole::User, format!("message {i}"));
                message.with_defaults(session).entry_id.unwrap()
            })
            .collect();
        // All IDs should be unique
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "All message IDs should be unique");
    }

    #[test]
    fn message_serde_roundtrip() {
        let message = Message {
            tool_call_id: Some("tool_1".to_owned()),
            metadata: Some(serde_json::json!({"key": "value"})),
            ..Message::new(MessageRole::Tool, "tool result")
        };
        let json = serde_json::to_string(&message).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, message);
    }

    #[test]
    fn message_serde_metadata_types() {
        // Metadata can be any JSON value
        for metadata in [
            serde_json::json!("raw string"),
            serde_json::json!(42),
            serde_json::json!([1, 2, 3]),
            serde_json::json!({"nested": {"key": "value"}}),
        ] {
            let message = Message {
                metadata: Some(metadata.clone()),
                ..Message::new(MessageRole::User, "test")
            };
            let json = serde_json::to_string(&message).unwrap();
            let deserialized: Message = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.metadata, Some(metadata));
        }
    }

    #[test]
    fn validate_top_k_accepts_zero() {
        assert!(validate_top_k(0).is_ok());
    }

    #[test]
    fn validate_top_k_accepts_normal() {
        assert!(validate_top_k(5).is_ok());
        assert!(validate_top_k(100).is_ok());
    }

    #[test]
    fn validate_top_k_rejects_huge() {
        assert!(validate_top_k(200_000).is_err());
    }

    #[test]
    fn validate_distance_threshold_valid() {
        assert!(validate_distance_threshold(0.0).is_ok());
        assert!(validate_distance_threshold(0.5).is_ok());
        assert!(validate_distance_threshold(2.0).is_ok());
    }

    #[test]
    fn validate_distance_threshold_invalid() {
        assert!(validate_distance_threshold(-0.1).is_err());
        assert!(validate_distance_threshold(2.1).is_err());
    }

    #[test]
    fn normalize_roles_none_passes_through() {
        assert!(normalize_roles(None).unwrap().is_none());
    }

    #[test]
    fn normalize_roles_empty_rejects() {
        assert!(normalize_roles(Some(&[])).is_err());
    }

    #[test]
    fn normalize_roles_non_empty_passes() {
        let roles = [MessageRole::User];
        let result = normalize_roles(Some(&roles)).unwrap();
        assert_eq!(result, Some(roles.as_slice()));
    }

    #[test]
    fn apply_role_filter_no_filter() {
        let mut messages = vec![
            Message::new(MessageRole::System, "sys"),
            Message::new(MessageRole::User, "user"),
        ];
        apply_role_filter(&mut messages, None);
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn apply_role_filter_single_role() {
        let mut messages = vec![
            Message::new(MessageRole::System, "sys"),
            Message::new(MessageRole::User, "user"),
            Message::new(MessageRole::Llm, "llm"),
            Message::new(MessageRole::Tool, "tool"),
        ];
        apply_role_filter(&mut messages, Some(&[MessageRole::User]));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::User);
    }

    #[test]
    fn apply_role_filter_multiple_roles() {
        let mut messages = vec![
            Message::new(MessageRole::System, "sys"),
            Message::new(MessageRole::User, "user"),
            Message::new(MessageRole::Llm, "llm"),
            Message::new(MessageRole::Tool, "tool"),
        ];
        apply_role_filter(
            &mut messages,
            Some(&[MessageRole::System, MessageRole::Tool]),
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[1].role, MessageRole::Tool);
    }

    #[test]
    fn semantic_schema_has_expected_fields() {
        let schema = semantic_message_history_schema("test_history", 128, VectorDataType::Float32);
        let fields = schema["fields"].as_array().unwrap();
        let field_names: Vec<&str> = fields.iter().filter_map(|f| f["name"].as_str()).collect();
        assert!(field_names.contains(&"entry_id"));
        assert!(field_names.contains(&"role"));
        assert!(field_names.contains(&"content"));
        assert!(field_names.contains(&"tool_call_id"));
        assert!(field_names.contains(&"timestamp"));
        assert!(field_names.contains(&"session_tag"));
        assert!(field_names.contains(&"metadata"));
        assert!(field_names.contains(&"message_vector"));

        // Vector field should have the correct dimensions
        let vector_field = fields
            .iter()
            .find(|f| f["name"].as_str() == Some("message_vector"))
            .unwrap();
        assert_eq!(vector_field["attrs"]["dims"], 128);
        assert_eq!(vector_field["attrs"]["datatype"], "float32");
    }

    #[test]
    fn message_from_document_basic() {
        let mut doc = Map::new();
        doc.insert("entry_id".to_owned(), Value::String("id_1".to_owned()));
        doc.insert("role".to_owned(), Value::String("user".to_owned()));
        doc.insert(
            "content".to_owned(),
            Value::String("hello world".to_owned()),
        );
        doc.insert(
            "session_tag".to_owned(),
            Value::String("session_1".to_owned()),
        );
        doc.insert(
            "timestamp".to_owned(),
            Value::Number(serde_json::Number::from_f64(1000.0).unwrap()),
        );

        let message = message_from_document(doc).unwrap();
        assert_eq!(message.entry_id.as_deref(), Some("id_1"));
        assert_eq!(message.role, MessageRole::User);
        assert_eq!(message.content, "hello world");
        assert_eq!(message.session_tag.as_deref(), Some("session_1"));
        assert_eq!(message.timestamp, Some(1000.0));
        assert!(message.tool_call_id.is_none());
        assert!(message.metadata.is_none());
    }

    #[test]
    fn message_from_document_with_tool_and_metadata() {
        let mut doc = Map::new();
        doc.insert("role".to_owned(), Value::String("tool".to_owned()));
        doc.insert(
            "content".to_owned(),
            Value::String("tool result".to_owned()),
        );
        doc.insert(
            "tool_call_id".to_owned(),
            Value::String("call_1".to_owned()),
        );
        doc.insert(
            "metadata".to_owned(),
            Value::String(r#"{"key":"value"}"#.to_owned()),
        );

        let message = message_from_document(doc).unwrap();
        assert_eq!(message.role, MessageRole::Tool);
        assert_eq!(message.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(message.metadata, Some(serde_json::json!({"key": "value"})));
    }

    #[test]
    fn message_from_document_string_timestamp() {
        let mut doc = Map::new();
        doc.insert("role".to_owned(), Value::String("llm".to_owned()));
        doc.insert("content".to_owned(), Value::String("response".to_owned()));
        doc.insert("timestamp".to_owned(), Value::String("1234.5".to_owned()));

        let message = message_from_document(doc).unwrap();
        assert_eq!(message.timestamp, Some(1234.5));
    }

    #[test]
    fn message_from_document_missing_content_fails() {
        let mut doc = Map::new();
        doc.insert("role".to_owned(), Value::String("user".to_owned()));
        // Missing "content"

        assert!(message_from_document(doc).is_err());
    }

    #[test]
    fn number_value_finite() {
        let val = number_value(42.5).unwrap();
        assert_eq!(
            val,
            Value::Number(serde_json::Number::from_f64(42.5).unwrap())
        );
    }

    #[test]
    fn number_value_nan_fails() {
        assert!(number_value(f64::NAN).is_err());
    }

    #[test]
    fn number_value_infinity_fails() {
        assert!(number_value(f64::INFINITY).is_err());
    }

    #[test]
    fn semantic_schema_respects_dtype() {
        let schema_f64 =
            semantic_message_history_schema("test_history", 128, VectorDataType::Float64);
        let fields = schema_f64["fields"].as_array().unwrap();
        let vector_field = fields
            .iter()
            .find(|f| f["name"].as_str() == Some("message_vector"))
            .unwrap();
        assert_eq!(vector_field["attrs"]["datatype"], "float64");

        let schema_bf16 =
            semantic_message_history_schema("test_history", 128, VectorDataType::Bfloat16);
        let fields = schema_bf16["fields"].as_array().unwrap();
        let vector_field = fields
            .iter()
            .find(|f| f["name"].as_str() == Some("message_vector"))
            .unwrap();
        assert_eq!(vector_field["attrs"]["datatype"], "bfloat16");

        let schema_f16 =
            semantic_message_history_schema("test_history", 128, VectorDataType::Float16);
        let fields = schema_f16["fields"].as_array().unwrap();
        let vector_field = fields
            .iter()
            .find(|f| f["name"].as_str() == Some("message_vector"))
            .unwrap();
        assert_eq!(vector_field["attrs"]["datatype"], "float16");
    }
}
