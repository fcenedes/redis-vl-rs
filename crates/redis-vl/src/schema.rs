//! Index schema types and Redis Search serialization helpers.
//!
//! An [`IndexSchema`] describes the structure of a Redis Search index including
//! its name, key prefix(es), storage type (Hash or JSON), stopwords, and a
//! list of typed field definitions. Schemas can be loaded from YAML files,
//! YAML strings, or JSON values.
//!
//! # Example
//!
//! ```
//! use redis_vl::IndexSchema;
//!
//! let schema = IndexSchema::from_yaml_str(r#"
//! index:
//!   name: my-index
//!   prefix: doc
//! fields:
//!   - name: title
//!     type: tag
//! "#).unwrap();
//! assert_eq!(schema.index.name, "my-index");
//! ```

use std::{collections::HashSet, fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A prefix specification that accepts either a single string or a list of
/// strings for multi-prefix support.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Prefix {
    /// A single key prefix.
    Single(String),
    /// Multiple key prefixes for multi-prefix indexes.
    Multi(Vec<String>),
}

impl<'de> Deserialize<'de> for Prefix {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de;

        struct PrefixVisitor;

        impl<'de> de::Visitor<'de> for PrefixVisitor {
            type Value = Prefix;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string or a list of strings")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Prefix, E> {
                Ok(Prefix::Single(v.to_owned()))
            }

            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Prefix, E> {
                Ok(Prefix::Single(v))
            }

            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Prefix, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<String>()? {
                    items.push(item);
                }
                Ok(Prefix::Multi(items))
            }
        }

        deserializer.deserialize_any(PrefixVisitor)
    }
}

impl Default for Prefix {
    fn default() -> Self {
        Prefix::Single("rvl".to_owned())
    }
}

impl Prefix {
    /// Returns the first (or only) prefix string.
    pub fn first(&self) -> &str {
        match self {
            Prefix::Single(s) => s,
            Prefix::Multi(v) => v.first().map(String::as_str).unwrap_or(""),
        }
    }

    /// Returns all prefixes as a slice-like view.
    pub fn all(&self) -> Vec<&str> {
        match self {
            Prefix::Single(s) => vec![s.as_str()],
            Prefix::Multi(v) => v.iter().map(String::as_str).collect(),
        }
    }

    /// Returns the number of prefixes.
    pub fn len(&self) -> usize {
        match self {
            Prefix::Single(_) => 1,
            Prefix::Multi(v) => v.len(),
        }
    }

    /// Returns `true` if no prefixes are configured.
    pub fn is_empty(&self) -> bool {
        match self {
            Prefix::Single(s) => s.is_empty(),
            Prefix::Multi(v) => v.is_empty(),
        }
    }
}

/// Complete RedisVL index schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSchema {
    /// Index metadata.
    pub index: IndexDefinition,
    /// Searchable fields in the index.
    #[serde(default)]
    pub fields: Vec<Field>,
}

impl IndexSchema {
    /// Parses an [`IndexSchema`] from a YAML string.
    pub fn from_yaml_str(input: &str) -> Result<Self> {
        let schema: Self = serde_yaml::from_str(input)?;
        schema.validate()?;
        Ok(schema)
    }

    /// Parses an [`IndexSchema`] from a YAML file.
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        Self::from_yaml_str(&contents)
    }

    /// Parses an [`IndexSchema`] from a JSON value.
    pub fn from_json_value(value: serde_json::Value) -> Result<Self> {
        let schema: Self = serde_json::from_value(value)?;
        schema.validate()?;
        Ok(schema)
    }

    /// Serializes the schema into a JSON value.
    pub fn to_json_value(&self) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    /// Serializes the schema into a YAML string.
    pub fn to_yaml_string(&self) -> Result<String> {
        Ok(serde_yaml::to_string(self)?)
    }

    /// Writes the schema to a YAML file.
    pub fn to_yaml_file(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_yaml_string()?)?;
        Ok(())
    }

    /// Validates the schema for common authoring errors.
    pub fn validate(&self) -> Result<()> {
        if self.index.name.trim().is_empty() {
            return Err(Error::SchemaValidation(
                "index name cannot be empty".to_owned(),
            ));
        }
        let mut seen = HashSet::new();
        for field in &self.fields {
            if !seen.insert(field.name.clone()) {
                return Err(Error::SchemaValidation(format!(
                    "duplicate field name '{}'",
                    field.name
                )));
            }

            if field.name.trim().is_empty() {
                return Err(Error::SchemaValidation(
                    "field names cannot be empty".to_owned(),
                ));
            }

            if let FieldKind::Vector { attrs } = &field.kind {
                if attrs.dims == 0 {
                    return Err(Error::SchemaValidation(format!(
                        "vector field '{}' must use dims > 0",
                        field.name
                    )));
                }
                attrs.validate_svs()?;
            }
        }

        Ok(())
    }

    /// Returns the field with the supplied name.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// Adds a single field to the schema.
    ///
    /// Returns an error if a field with the same name already exists or if
    /// validation fails.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SchemaValidation`] when the name is empty, duplicated,
    /// or a vector field has `dims == 0`.
    pub fn add_field(&mut self, field: Field) -> Result<()> {
        if self.fields.iter().any(|f| f.name == field.name) {
            return Err(Error::SchemaValidation(format!(
                "duplicate field name '{}'",
                field.name
            )));
        }
        if field.name.trim().is_empty() {
            return Err(Error::SchemaValidation(
                "field names cannot be empty".to_owned(),
            ));
        }
        if let FieldKind::Vector { attrs } = &field.kind {
            if attrs.dims == 0 {
                return Err(Error::SchemaValidation(format!(
                    "vector field '{}' must use dims > 0",
                    field.name
                )));
            }
        }
        self.fields.push(field);
        Ok(())
    }

    /// Adds multiple fields to the schema.
    ///
    /// This is a convenience wrapper around [`add_field`](Self::add_field) that
    /// stops at the first error.
    pub fn add_fields(&mut self, fields: Vec<Field>) -> Result<()> {
        for field in fields {
            self.add_field(field)?;
        }
        Ok(())
    }

    /// Removes a field by name, returning `true` if it was present.
    pub fn remove_field(&mut self, name: &str) -> bool {
        let before = self.fields.len();
        self.fields.retain(|f| f.name != name);
        self.fields.len() != before
    }

    /// Serializes the schema into `FT.CREATE` arguments after `SCHEMA`.
    pub(crate) fn redis_schema_args(&self) -> Vec<String> {
        self.fields
            .iter()
            .flat_map(|field| field.redis_args(self.index.storage_type))
            .collect()
    }
}

/// Index-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    /// Redis Search index name.
    pub name: String,
    /// Key prefix(es) used when loading documents.
    ///
    /// Accepts either a single string or a list of strings for multi-prefix
    /// support.
    #[serde(default)]
    pub prefix: Prefix,
    /// Separator used between prefix and identifier.
    #[serde(default = "default_key_separator")]
    pub key_separator: String,
    /// Backing storage type for records.
    #[serde(default = "default_storage_type")]
    pub storage_type: StorageType,
    /// Optional list of stop words.
    #[serde(default)]
    pub stopwords: Vec<String>,
}

/// Supported Redis document storage types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageType {
    /// Redis Hash storage.
    Hash,
    /// RedisJSON storage.
    Json,
}

impl StorageType {
    pub(crate) fn redis_name(self) -> &'static str {
        match self {
            Self::Hash => "HASH",
            Self::Json => "JSON",
        }
    }
}

fn default_key_separator() -> String {
    ":".to_owned()
}

fn default_storage_type() -> StorageType {
    StorageType::Hash
}

/// Search field definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    /// Friendly field name and Redis alias.
    pub name: String,
    /// Optional JSON path for JSON-backed indices.
    #[serde(default)]
    pub path: Option<String>,
    /// Field kind and attributes.
    #[serde(flatten)]
    pub kind: FieldKind,
}

impl Field {
    pub(crate) fn redis_args(&self, storage_type: StorageType) -> Vec<String> {
        let mut args = Vec::new();
        match (storage_type, self.path.as_deref()) {
            (StorageType::Json, Some(path)) => {
                args.push(path.to_owned());
                args.push("AS".to_owned());
                args.push(self.name.clone());
            }
            _ => args.push(self.name.clone()),
        }
        self.kind.push_redis_args(&mut args);
        args
    }
}

/// Concrete field kind supported by RedisVL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FieldKind {
    /// Tag field with exact-match semantics.
    Tag {
        /// Tag field attributes.
        #[serde(default)]
        attrs: TagFieldAttributes,
    },
    /// Full text field.
    Text {
        /// Text field attributes.
        #[serde(default)]
        attrs: TextFieldAttributes,
    },
    /// Numeric field.
    Numeric {
        /// Numeric field attributes.
        #[serde(default)]
        attrs: NumericFieldAttributes,
    },
    /// Geospatial field.
    Geo {
        /// Geo field attributes.
        #[serde(default)]
        attrs: GeoFieldAttributes,
    },
    /// Timestamp field represented as a numeric value.
    Timestamp {
        /// Timestamp field attributes.
        #[serde(default)]
        attrs: TimestampFieldAttributes,
    },
    /// Vector similarity field.
    Vector {
        /// Vector field attributes.
        attrs: VectorFieldAttributes,
    },
}

impl FieldKind {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        match self {
            Self::Tag { attrs } => {
                args.push("TAG".to_owned());
                attrs.push_redis_args(args);
            }
            Self::Text { attrs } => {
                args.push("TEXT".to_owned());
                attrs.push_redis_args(args);
            }
            Self::Numeric { attrs } => {
                args.push("NUMERIC".to_owned());
                attrs.push_redis_args(args);
            }
            Self::Geo { attrs } => {
                args.push("GEO".to_owned());
                attrs.push_redis_args(args);
            }
            Self::Timestamp { attrs } => {
                args.push("NUMERIC".to_owned());
                attrs.push_redis_args(args);
            }
            Self::Vector { attrs } => {
                args.push("VECTOR".to_owned());
                args.push(attrs.algorithm.redis_name().to_owned());
                let vector_args = attrs.redis_attribute_pairs();
                args.push(vector_args.len().to_string());
                args.extend(vector_args);
            }
        }
    }
}

/// Attributes for a tag field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TagFieldAttributes {
    /// Separator for multi-value tags.
    pub separator: Option<String>,
    /// Whether to disable case normalization.
    #[serde(default)]
    pub case_sensitive: bool,
    /// Whether the field should be sortable.
    #[serde(default)]
    pub sortable: bool,
    /// Whether indexing should be disabled.
    #[serde(default)]
    pub no_index: bool,
    /// Whether to index missing values so `ismissing(@field)` works.
    #[serde(default)]
    pub index_missing: bool,
    /// Whether to index empty values so `isempty(@field)` works.
    #[serde(default)]
    pub index_empty: bool,
}

impl TagFieldAttributes {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        if let Some(separator) = &self.separator {
            args.push("SEPARATOR".to_owned());
            args.push(separator.clone());
        }
        if self.case_sensitive {
            args.push("CASESENSITIVE".to_owned());
        }
        if self.sortable {
            args.push("SORTABLE".to_owned());
        }
        if self.no_index {
            args.push("NOINDEX".to_owned());
        }
        if self.index_missing {
            args.push("INDEXMISSING".to_owned());
        }
        if self.index_empty {
            args.push("INDEXEMPTY".to_owned());
        }
    }
}

/// Attributes for a text field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextFieldAttributes {
    /// Relative text field weight.
    pub weight: Option<f32>,
    /// Whether the field should be sortable.
    #[serde(default)]
    pub sortable: bool,
    /// Whether stemming should be disabled.
    #[serde(default)]
    pub no_stem: bool,
    /// Whether indexing should be disabled.
    #[serde(default)]
    pub no_index: bool,
    /// Optional phonetic matcher.
    pub phonetic: Option<String>,
    /// Whether suffix trie indexing should be enabled.
    #[serde(default)]
    pub with_suffix_trie: bool,
    /// Whether to index missing values so `ismissing(@field)` works.
    #[serde(default)]
    pub index_missing: bool,
    /// Whether to index empty values so `isempty(@field)` works.
    #[serde(default)]
    pub index_empty: bool,
}

impl TextFieldAttributes {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        if let Some(weight) = self.weight {
            args.push("WEIGHT".to_owned());
            args.push(weight.to_string());
        }
        if self.sortable {
            args.push("SORTABLE".to_owned());
        }
        if self.no_stem {
            args.push("NOSTEM".to_owned());
        }
        if self.no_index {
            args.push("NOINDEX".to_owned());
        }
        if let Some(phonetic) = &self.phonetic {
            args.push("PHONETIC".to_owned());
            args.push(phonetic.clone());
        }
        if self.with_suffix_trie {
            args.push("WITHSUFFIXTRIE".to_owned());
        }
        if self.index_missing {
            args.push("INDEXMISSING".to_owned());
        }
        if self.index_empty {
            args.push("INDEXEMPTY".to_owned());
        }
    }
}

/// Attributes for a numeric field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericFieldAttributes {
    /// Whether the field should be sortable.
    #[serde(default)]
    pub sortable: bool,
    /// Whether indexing should be disabled.
    #[serde(default)]
    pub no_index: bool,
    /// Whether to index missing values so `ismissing(@field)` works.
    #[serde(default)]
    pub index_missing: bool,
    /// Whether to index empty values so `isempty(@field)` works.
    #[serde(default)]
    pub index_empty: bool,
}

impl NumericFieldAttributes {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        if self.sortable {
            args.push("SORTABLE".to_owned());
        }
        if self.no_index {
            args.push("NOINDEX".to_owned());
        }
        if self.index_missing {
            args.push("INDEXMISSING".to_owned());
        }
        if self.index_empty {
            args.push("INDEXEMPTY".to_owned());
        }
    }
}

/// Attributes for a geo field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeoFieldAttributes {
    /// Whether the field should be sortable.
    #[serde(default)]
    pub sortable: bool,
    /// Whether indexing should be disabled.
    #[serde(default)]
    pub no_index: bool,
    /// Whether to index missing values so `ismissing(@field)` works.
    #[serde(default)]
    pub index_missing: bool,
    /// Whether to index empty values so `isempty(@field)` works.
    #[serde(default)]
    pub index_empty: bool,
}

impl GeoFieldAttributes {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        if self.sortable {
            args.push("SORTABLE".to_owned());
        }
        if self.no_index {
            args.push("NOINDEX".to_owned());
        }
        if self.index_missing {
            args.push("INDEXMISSING".to_owned());
        }
        if self.index_empty {
            args.push("INDEXEMPTY".to_owned());
        }
    }
}

/// Attributes for a timestamp field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimestampFieldAttributes {
    /// Whether the field should be sortable.
    #[serde(default)]
    pub sortable: bool,
    /// Whether indexing should be disabled.
    #[serde(default)]
    pub no_index: bool,
    /// Whether to index missing values so `ismissing(@field)` works.
    #[serde(default)]
    pub index_missing: bool,
    /// Whether to index empty values so `isempty(@field)` works.
    #[serde(default)]
    pub index_empty: bool,
}

impl TimestampFieldAttributes {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        if self.sortable {
            args.push("SORTABLE".to_owned());
        }
        if self.no_index {
            args.push("NOINDEX".to_owned());
        }
        if self.index_missing {
            args.push("INDEXMISSING".to_owned());
        }
        if self.index_empty {
            args.push("INDEXEMPTY".to_owned());
        }
    }
}

/// Supported Redis vector index algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorAlgorithm {
    /// Brute-force vector search.
    #[serde(alias = "flat", alias = "FLAT")]
    Flat,
    /// Approximate nearest-neighbor HNSW search.
    #[serde(alias = "hnsw", alias = "HNSW")]
    Hnsw,
    /// SVS-VAMANA graph-based approximate search (Redis 8.2+).
    #[serde(
        alias = "svs-vamana",
        alias = "SVS-VAMANA",
        alias = "svs_vamana",
        alias = "SVS_VAMANA"
    )]
    SvsVamana,
}

impl VectorAlgorithm {
    fn redis_name(self) -> &'static str {
        match self {
            Self::Flat => "FLAT",
            Self::Hnsw => "HNSW",
            Self::SvsVamana => "SVS-VAMANA",
        }
    }
}

/// Compression types for SVS-VAMANA vector fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SvsCompressionType {
    /// 4-bit local vector quantization.
    #[serde(alias = "lvq4", alias = "LVQ4")]
    Lvq4,
    /// 4×4-bit local vector quantization.
    #[serde(alias = "lvq4x4", alias = "LVQ4x4")]
    Lvq4x4,
    /// 4×8-bit local vector quantization.
    #[serde(alias = "lvq4x8", alias = "LVQ4x8")]
    Lvq4x8,
    /// 8-bit local vector quantization.
    #[serde(alias = "lvq8", alias = "LVQ8")]
    Lvq8,
    /// LeanVec 4×8 compression.
    #[serde(alias = "leanvec4x8", alias = "LeanVec4x8")]
    LeanVec4x8,
    /// LeanVec 8×8 compression.
    #[serde(alias = "leanvec8x8", alias = "LeanVec8x8")]
    LeanVec8x8,
}

impl SvsCompressionType {
    fn redis_name(self) -> &'static str {
        match self {
            Self::Lvq4 => "LVQ4",
            Self::Lvq4x4 => "LVQ4x4",
            Self::Lvq4x8 => "LVQ4x8",
            Self::Lvq8 => "LVQ8",
            Self::LeanVec4x8 => "LeanVec4x8",
            Self::LeanVec8x8 => "LeanVec8x8",
        }
    }

    fn is_lean_vec(self) -> bool {
        matches!(self, Self::LeanVec4x8 | Self::LeanVec8x8)
    }
}

/// Supported vector element data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum VectorDataType {
    /// Brain floating point 16-bit vectors.
    Bfloat16,
    /// IEEE 754 half-precision 16-bit vectors.
    Float16,
    /// 32-bit floating point vectors.
    Float32,
    /// 64-bit floating point vectors.
    Float64,
}

impl VectorDataType {
    fn redis_name(self) -> &'static str {
        match self {
            Self::Bfloat16 => "BFLOAT16",
            Self::Float16 => "FLOAT16",
            Self::Float32 => "FLOAT32",
            Self::Float64 => "FLOAT64",
        }
    }

    /// Returns the lowercase string representation (e.g. `"float32"`).
    ///
    /// This matches the convention used by Python RedisVL and is useful
    /// when constructing JSON schema values dynamically.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bfloat16 => "bfloat16",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }
}

impl std::fmt::Display for VectorDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for VectorDataType {
    type Err = crate::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bfloat16" => Ok(Self::Bfloat16),
            "float16" => Ok(Self::Float16),
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            other => Err(crate::Error::InvalidInput(format!(
                "unknown vector data type '{other}'; expected bfloat16, float16, float32, or float64"
            ))),
        }
    }
}

impl Default for VectorDataType {
    fn default() -> Self {
        Self::Float32
    }
}

/// Supported Redis vector distance metrics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum VectorDistanceMetric {
    /// Cosine distance.
    Cosine,
    /// Euclidean distance.
    L2,
    /// Inner product distance.
    Ip,
}

impl VectorDistanceMetric {
    fn redis_name(self) -> &'static str {
        match self {
            Self::Cosine => "COSINE",
            Self::L2 => "L2",
            Self::Ip => "IP",
        }
    }
}

/// Attributes for a vector field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldAttributes {
    /// Index algorithm.
    pub algorithm: VectorAlgorithm,
    /// Vector dimensions.
    pub dims: usize,
    /// Distance metric.
    pub distance_metric: VectorDistanceMetric,
    /// Element data type.
    pub datatype: VectorDataType,
    /// Optional initial capacity.
    pub initial_cap: Option<usize>,
    /// Optional FLAT block size.
    pub block_size: Option<usize>,
    /// Optional HNSW `M`.
    pub m: Option<usize>,
    /// Optional HNSW construction EF.
    pub ef_construction: Option<usize>,
    /// Optional runtime EF hint.
    pub ef_runtime: Option<usize>,
    /// Optional epsilon value.
    pub epsilon: Option<f32>,
    // ── SVS-VAMANA specific ──
    /// Maximum outgoing edges per node (SVS-VAMANA).
    pub graph_max_degree: Option<usize>,
    /// Build-time candidate window (SVS-VAMANA).
    pub construction_window_size: Option<usize>,
    /// Search-time candidate window (SVS-VAMANA).
    pub search_window_size: Option<usize>,
    /// Compression type for SVS-VAMANA.
    pub compression: Option<SvsCompressionType>,
    /// Dimensionality reduction target for LeanVec compression (SVS-VAMANA).
    pub reduce: Option<usize>,
    /// Minimum vectors before compression training (SVS-VAMANA).
    pub training_threshold: Option<usize>,
}

impl VectorFieldAttributes {
    fn redis_attribute_pairs(&self) -> Vec<String> {
        let mut args = vec![
            "TYPE".to_owned(),
            self.datatype.redis_name().to_owned(),
            "DIM".to_owned(),
            self.dims.to_string(),
            "DISTANCE_METRIC".to_owned(),
            self.distance_metric.redis_name().to_owned(),
        ];

        if let Some(initial_cap) = self.initial_cap {
            args.push("INITIAL_CAP".to_owned());
            args.push(initial_cap.to_string());
        }
        // FLAT-specific
        if let Some(block_size) = self.block_size {
            args.push("BLOCK_SIZE".to_owned());
            args.push(block_size.to_string());
        }
        // HNSW-specific
        if let Some(m) = self.m {
            args.push("M".to_owned());
            args.push(m.to_string());
        }
        if let Some(ef_construction) = self.ef_construction {
            args.push("EF_CONSTRUCTION".to_owned());
            args.push(ef_construction.to_string());
        }
        if let Some(ef_runtime) = self.ef_runtime {
            args.push("EF_RUNTIME".to_owned());
            args.push(ef_runtime.to_string());
        }
        if let Some(epsilon) = self.epsilon {
            args.push("EPSILON".to_owned());
            args.push(epsilon.to_string());
        }
        // SVS-VAMANA specific
        if let Some(graph_max_degree) = self.graph_max_degree {
            args.push("GRAPH_MAX_DEGREE".to_owned());
            args.push(graph_max_degree.to_string());
        }
        if let Some(construction_window_size) = self.construction_window_size {
            args.push("CONSTRUCTION_WINDOW_SIZE".to_owned());
            args.push(construction_window_size.to_string());
        }
        if let Some(search_window_size) = self.search_window_size {
            args.push("SEARCH_WINDOW_SIZE".to_owned());
            args.push(search_window_size.to_string());
        }
        if let Some(compression) = self.compression {
            args.push("COMPRESSION".to_owned());
            args.push(compression.redis_name().to_owned());
        }
        if let Some(reduce) = self.reduce {
            args.push("REDUCE".to_owned());
            args.push(reduce.to_string());
        }
        if let Some(training_threshold) = self.training_threshold {
            args.push("TRAINING_THRESHOLD".to_owned());
            args.push(training_threshold.to_string());
        }

        args
    }

    /// Validates SVS-VAMANA specific constraints.
    ///
    /// Call after construction when `algorithm == SvsVamana` to ensure the
    /// data-type restriction (only Float16/Float32) and LeanVec `reduce`
    /// constraints are met.
    pub fn validate_svs(&self) -> Result<()> {
        if self.algorithm != VectorAlgorithm::SvsVamana {
            return Ok(());
        }
        // SVS-VAMANA only supports Float16 and Float32
        if !matches!(
            self.datatype,
            VectorDataType::Float16 | VectorDataType::Float32
        ) {
            return Err(Error::SchemaValidation(format!(
                "SVS-VAMANA only supports FLOAT16 and FLOAT32 datatypes, got {}",
                self.datatype
            )));
        }
        // `reduce` requires a LeanVec compression type
        if let Some(reduce) = self.reduce {
            match self.compression {
                None => {
                    return Err(Error::SchemaValidation(
                        "reduce parameter requires compression to be set".to_owned(),
                    ));
                }
                Some(c) if !c.is_lean_vec() => {
                    return Err(Error::SchemaValidation(format!(
                        "reduce parameter is only supported with LeanVec compression types, got {:?}",
                        c
                    )));
                }
                _ => {}
            }
            if reduce >= self.dims {
                return Err(Error::SchemaValidation(format!(
                    "reduce ({reduce}) must be less than dims ({})",
                    self.dims
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IndexSchema, Prefix, StorageType};

    #[test]
    fn schema_from_yaml_should_parse_json_storage() {
        let schema = IndexSchema::from_yaml_str(
            r#"
index:
  name: docs
  prefix: doc
  storage_type: json
fields:
  - name: title
    path: $.title
    type: text
  - name: embedding
    path: $.embedding
    type: vector
    attrs:
      algorithm: HNSW
      dims: 3
      datatype: FLOAT32
      distance_metric: COSINE
"#,
        )
        .expect("schema should parse");

        assert!(matches!(schema.index.storage_type, StorageType::Json));
        assert_eq!(schema.fields.len(), 2);
    }

    #[test]
    fn schema_should_apply_defaults_like_python_unit_tests() {
        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test" }
        }))
        .expect("schema should parse");

        assert_eq!(schema.index.prefix.first(), "rvl");
        assert_eq!(schema.index.key_separator, ":");
        assert!(matches!(schema.index.storage_type, StorageType::Hash));
        assert!(schema.fields.is_empty());
    }

    #[test]
    fn schema_should_accept_multi_prefix_list_like_python_multi_prefix_tests() {
        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": {
                "name": "test",
                "prefix": ["pfx_a", "pfx_b"]
            }
        }))
        .expect("schema should parse");

        assert_eq!(schema.index.prefix.len(), 2);
        assert_eq!(schema.index.prefix.first(), "pfx_a");
        assert_eq!(schema.index.prefix.all(), vec!["pfx_a", "pfx_b"]);
        assert!(matches!(schema.index.prefix, Prefix::Multi(_)));
    }

    #[test]
    fn schema_should_accept_single_string_prefix_like_python_tests() {
        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": {
                "name": "test",
                "prefix": "my_prefix"
            }
        }))
        .expect("schema should parse");

        assert_eq!(schema.index.prefix.first(), "my_prefix");
        assert_eq!(schema.index.prefix.len(), 1);
        assert_eq!(schema.index.prefix.all(), vec!["my_prefix"]);
        assert!(matches!(schema.index.prefix, Prefix::Single(_)));
    }

    #[test]
    fn schema_multi_prefix_yaml_should_parse() {
        let schema = IndexSchema::from_yaml_str(
            r#"
index:
  name: multi
  prefix:
    - alpha
    - beta
fields:
  - name: tag
    type: tag
"#,
        )
        .expect("schema should parse");

        assert_eq!(schema.index.prefix.len(), 2);
        assert_eq!(schema.index.prefix.all(), vec!["alpha", "beta"]);
    }

    // ── index_missing / index_empty parity tests (upstream: test_fields.py) ──

    #[test]
    fn tag_field_index_missing_should_render_indexmissing_arg() {
        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test_missing" },
            "fields": [
                { "name": "brand", "type": "tag", "attrs": { "index_missing": true } }
            ]
        }))
        .expect("schema should parse");

        let args = schema.fields[0].redis_args(StorageType::Hash);
        assert!(args.contains(&"INDEXMISSING".to_owned()));
    }

    #[test]
    fn numeric_field_index_empty_should_render_indexempty_arg() {
        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test_empty" },
            "fields": [
                { "name": "price", "type": "numeric", "attrs": { "index_empty": true } }
            ]
        }))
        .expect("schema should parse");

        let args = schema.fields[0].redis_args(StorageType::Hash);
        assert!(args.contains(&"INDEXEMPTY".to_owned()));
    }

    #[test]
    fn text_field_both_index_missing_and_index_empty() {
        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test_both" },
            "fields": [
                { "name": "description", "type": "text", "attrs": { "index_missing": true, "index_empty": true } }
            ]
        }))
        .expect("schema should parse");

        let args = schema.fields[0].redis_args(StorageType::Hash);
        assert!(args.contains(&"INDEXMISSING".to_owned()));
        assert!(args.contains(&"INDEXEMPTY".to_owned()));
    }

    #[test]
    fn fields_default_to_no_index_missing_or_empty() {
        let schema = IndexSchema::from_yaml_str(
            r#"
index:
  name: test_defaults
fields:
  - name: brand
    type: tag
"#,
        )
        .expect("schema should parse");

        let args = schema.fields[0].redis_args(StorageType::Hash);
        assert!(!args.contains(&"INDEXMISSING".to_owned()));
        assert!(!args.contains(&"INDEXEMPTY".to_owned()));
    }

    #[test]
    fn vector_data_type_from_str_roundtrip() {
        use super::VectorDataType;
        use std::str::FromStr;

        for (input, expected) in [
            ("bfloat16", VectorDataType::Bfloat16),
            ("float16", VectorDataType::Float16),
            ("float32", VectorDataType::Float32),
            ("float64", VectorDataType::Float64),
            ("BFLOAT16", VectorDataType::Bfloat16),
            ("FLOAT16", VectorDataType::Float16),
            ("FLOAT32", VectorDataType::Float32),
            ("FLOAT64", VectorDataType::Float64),
            ("Float32", VectorDataType::Float32),
        ] {
            let parsed = VectorDataType::from_str(input)
                .unwrap_or_else(|_| panic!("should parse '{input}'"));
            assert_eq!(parsed, expected, "mismatch for input '{input}'");
        }

        assert!(VectorDataType::from_str("int8").is_err());
        assert!(VectorDataType::from_str("").is_err());
    }

    #[test]
    fn vector_data_type_as_str_and_display() {
        use super::VectorDataType;

        assert_eq!(VectorDataType::Bfloat16.as_str(), "bfloat16");
        assert_eq!(VectorDataType::Float16.as_str(), "float16");
        assert_eq!(VectorDataType::Float32.as_str(), "float32");
        assert_eq!(VectorDataType::Float64.as_str(), "float64");

        assert_eq!(VectorDataType::Float32.to_string(), "float32");
        assert_eq!(VectorDataType::Bfloat16.to_string(), "bfloat16");
    }

    #[test]
    fn vector_data_type_default_is_float32() {
        use super::VectorDataType;
        assert_eq!(VectorDataType::default(), VectorDataType::Float32);
    }

    #[test]
    fn vector_data_type_serde_uppercase() {
        use super::VectorDataType;

        let json = serde_json::to_string(&VectorDataType::Bfloat16).unwrap();
        assert_eq!(json, "\"BFLOAT16\"");

        let json = serde_json::to_string(&VectorDataType::Float16).unwrap();
        assert_eq!(json, "\"FLOAT16\"");

        let deserialized: VectorDataType = serde_json::from_str("\"FLOAT64\"").unwrap();
        assert_eq!(deserialized, VectorDataType::Float64);
    }

    #[test]
    fn schema_from_yaml_bfloat16_vector() {
        use super::{FieldKind, VectorDataType};
        let schema = IndexSchema::from_yaml_str(
            r#"
index:
  name: bf16test
  prefix: bf16
fields:
  - name: vec
    type: vector
    attrs:
      algorithm: FLAT
      dims: 4
      datatype: BFLOAT16
      distance_metric: COSINE
"#,
        )
        .expect("schema with BFLOAT16 should parse");

        assert_eq!(schema.index.name, "bf16test");
        let vec_field = &schema.fields[0];
        if let FieldKind::Vector { ref attrs } = vec_field.kind {
            assert_eq!(attrs.datatype, VectorDataType::Bfloat16);
        } else {
            panic!("expected vector field");
        }
    }

    #[test]
    fn schema_from_yaml_float16_vector() {
        use super::{FieldKind, VectorDataType};
        let schema = IndexSchema::from_yaml_str(
            r#"
index:
  name: f16test
  prefix: f16
fields:
  - name: vec
    type: vector
    attrs:
      algorithm: HNSW
      dims: 8
      datatype: FLOAT16
      distance_metric: L2
"#,
        )
        .expect("schema with FLOAT16 should parse");

        let vec_field = &schema.fields[0];
        if let FieldKind::Vector { ref attrs } = vec_field.kind {
            assert_eq!(attrs.datatype, VectorDataType::Float16);
        } else {
            panic!("expected vector field");
        }
    }

    // ── add_field / remove_field parity tests (upstream: test_schema.py) ──

    #[test]
    fn add_field_should_append_and_validate() {
        use super::{Field, FieldKind, TagFieldAttributes};

        let mut schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test" },
            "fields": [
                { "name": "title", "type": "text" }
            ]
        }))
        .expect("schema should parse");

        assert_eq!(schema.fields.len(), 1);

        let field = Field {
            name: "brand".to_owned(),
            path: None,
            kind: FieldKind::Tag {
                attrs: TagFieldAttributes::default(),
            },
        };
        schema.add_field(field).expect("add_field should succeed");
        assert_eq!(schema.fields.len(), 2);
        assert!(schema.field("brand").is_some());
    }

    #[test]
    fn add_field_duplicate_should_error() {
        let mut schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test" },
            "fields": [
                { "name": "title", "type": "text" }
            ]
        }))
        .expect("schema should parse");

        let field = super::Field {
            name: "title".to_owned(),
            path: None,
            kind: super::FieldKind::Text {
                attrs: super::TextFieldAttributes::default(),
            },
        };
        assert!(schema.add_field(field).is_err());
    }

    #[test]
    fn remove_field_should_drop_by_name() {
        let mut schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test" },
            "fields": [
                { "name": "title", "type": "text" },
                { "name": "brand", "type": "tag" }
            ]
        }))
        .expect("schema should parse");

        assert_eq!(schema.fields.len(), 2);
        assert!(schema.remove_field("title"));
        assert_eq!(schema.fields.len(), 1);
        assert!(schema.field("title").is_none());
        // removing again returns false
        assert!(!schema.remove_field("title"));
    }

    // ── SVS-VAMANA parity tests (upstream: test_validation.py) ──

    #[test]
    fn svs_vamana_schema_with_float32_should_parse() {
        use super::{FieldKind, VectorAlgorithm};

        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test-svs-index" },
            "fields": [{
                "name": "vec",
                "type": "vector",
                "attrs": {
                    "algorithm": "SvsVamana",
                    "dims": 128,
                    "distance_metric": "COSINE",
                    "datatype": "FLOAT32"
                }
            }]
        }))
        .expect("SVS-VAMANA with float32 should parse");

        if let FieldKind::Vector { ref attrs } = schema.fields[0].kind {
            assert_eq!(attrs.algorithm, VectorAlgorithm::SvsVamana);
        } else {
            panic!("expected vector field");
        }
    }

    #[test]
    fn svs_vamana_with_float64_should_fail_validation() {
        let result = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test-svs-index" },
            "fields": [{
                "name": "vec",
                "type": "vector",
                "attrs": {
                    "algorithm": "SvsVamana",
                    "dims": 128,
                    "distance_metric": "COSINE",
                    "datatype": "FLOAT64"
                }
            }]
        }));
        assert!(result.is_err(), "SVS-VAMANA should reject FLOAT64");
    }

    #[test]
    fn svs_vamana_with_compression_and_reduce() {
        use super::{FieldKind, SvsCompressionType};

        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test-svs-index" },
            "fields": [{
                "name": "vec",
                "type": "vector",
                "attrs": {
                    "algorithm": "SvsVamana",
                    "dims": 128,
                    "distance_metric": "COSINE",
                    "datatype": "FLOAT32",
                    "compression": "LeanVec4x8",
                    "reduce": 64
                }
            }]
        }))
        .expect("SVS-VAMANA with LeanVec + reduce should parse");

        if let FieldKind::Vector { ref attrs } = schema.fields[0].kind {
            assert_eq!(attrs.compression, Some(SvsCompressionType::LeanVec4x8));
            assert_eq!(attrs.reduce, Some(64));
        } else {
            panic!("expected vector field");
        }
    }

    #[test]
    fn svs_vamana_reduce_without_compression_should_fail() {
        let result = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test-svs-index" },
            "fields": [{
                "name": "vec",
                "type": "vector",
                "attrs": {
                    "algorithm": "SvsVamana",
                    "dims": 128,
                    "distance_metric": "COSINE",
                    "datatype": "FLOAT32",
                    "reduce": 64
                }
            }]
        }));
        assert!(
            result.is_err(),
            "SVS-VAMANA reduce without compression should fail"
        );
    }

    #[test]
    fn svs_vamana_reduce_with_lvq4_should_fail() {
        let result = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test-svs-index" },
            "fields": [{
                "name": "vec",
                "type": "vector",
                "attrs": {
                    "algorithm": "SvsVamana",
                    "dims": 128,
                    "distance_metric": "COSINE",
                    "datatype": "FLOAT32",
                    "compression": "Lvq4",
                    "reduce": 64
                }
            }]
        }));
        assert!(result.is_err(), "SVS-VAMANA reduce with LVQ4 should fail");
    }

    #[test]
    fn svs_vamana_reduce_gte_dims_should_fail() {
        let result = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test-svs-index" },
            "fields": [{
                "name": "vec",
                "type": "vector",
                "attrs": {
                    "algorithm": "SvsVamana",
                    "dims": 128,
                    "distance_metric": "COSINE",
                    "datatype": "FLOAT32",
                    "compression": "LeanVec4x8",
                    "reduce": 128
                }
            }]
        }));
        assert!(result.is_err(), "SVS-VAMANA reduce >= dims should fail");
    }

    #[test]
    fn svs_vamana_redis_args_include_svs_params() {
        let schema = IndexSchema::from_json_value(serde_json::json!({
            "index": { "name": "test-svs-index" },
            "fields": [{
                "name": "vec",
                "type": "vector",
                "attrs": {
                    "algorithm": "SvsVamana",
                    "dims": 128,
                    "distance_metric": "COSINE",
                    "datatype": "FLOAT32",
                    "graph_max_degree": 40,
                    "construction_window_size": 250,
                    "search_window_size": 20,
                    "compression": "Lvq8",
                    "training_threshold": 10000
                }
            }]
        }))
        .expect("SVS schema should parse");

        let args = schema.fields[0].redis_args(StorageType::Hash);
        assert!(args.contains(&"VECTOR".to_owned()));
        assert!(args.contains(&"SVS-VAMANA".to_owned()));
        assert!(args.contains(&"GRAPH_MAX_DEGREE".to_owned()));
        assert!(args.contains(&"40".to_owned()));
        assert!(args.contains(&"CONSTRUCTION_WINDOW_SIZE".to_owned()));
        assert!(args.contains(&"SEARCH_WINDOW_SIZE".to_owned()));
        assert!(args.contains(&"COMPRESSION".to_owned()));
        assert!(args.contains(&"LVQ8".to_owned()));
        assert!(args.contains(&"TRAINING_THRESHOLD".to_owned()));
    }
}
