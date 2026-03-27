//! Index schema types and Redis Search serialization helpers.

use std::{collections::HashSet, fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

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
            }
        }

        Ok(())
    }

    /// Returns the field with the supplied name.
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|field| field.name == name)
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
    /// Key prefix used when loading documents.
    #[serde(default = "default_prefix")]
    pub prefix: String,
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

fn default_prefix() -> String {
    "rvl".to_owned()
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
    pub case_sensitive: bool,
    /// Whether the field should be sortable.
    pub sortable: bool,
    /// Whether indexing should be disabled.
    pub no_index: bool,
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
    }
}

/// Attributes for a text field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextFieldAttributes {
    /// Relative text field weight.
    pub weight: Option<f32>,
    /// Whether the field should be sortable.
    pub sortable: bool,
    /// Whether stemming should be disabled.
    pub no_stem: bool,
    /// Whether indexing should be disabled.
    pub no_index: bool,
    /// Optional phonetic matcher.
    pub phonetic: Option<String>,
    /// Whether suffix trie indexing should be enabled.
    pub with_suffix_trie: bool,
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
    }
}

/// Attributes for a numeric field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericFieldAttributes {
    /// Whether the field should be sortable.
    pub sortable: bool,
    /// Whether indexing should be disabled.
    pub no_index: bool,
}

impl NumericFieldAttributes {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        if self.sortable {
            args.push("SORTABLE".to_owned());
        }
        if self.no_index {
            args.push("NOINDEX".to_owned());
        }
    }
}

/// Attributes for a geo field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeoFieldAttributes {
    /// Whether the field should be sortable.
    pub sortable: bool,
    /// Whether indexing should be disabled.
    pub no_index: bool,
}

impl GeoFieldAttributes {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        if self.sortable {
            args.push("SORTABLE".to_owned());
        }
        if self.no_index {
            args.push("NOINDEX".to_owned());
        }
    }
}

/// Attributes for a timestamp field.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimestampFieldAttributes {
    /// Whether the field should be sortable.
    pub sortable: bool,
    /// Whether indexing should be disabled.
    pub no_index: bool,
}

impl TimestampFieldAttributes {
    fn push_redis_args(&self, args: &mut Vec<String>) {
        if self.sortable {
            args.push("SORTABLE".to_owned());
        }
        if self.no_index {
            args.push("NOINDEX".to_owned());
        }
    }
}

/// Supported Redis vector index algorithms.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum VectorAlgorithm {
    /// Brute-force vector search.
    Flat,
    /// Approximate nearest-neighbor HNSW search.
    Hnsw,
}

impl VectorAlgorithm {
    fn redis_name(self) -> &'static str {
        match self {
            Self::Flat => "FLAT",
            Self::Hnsw => "HNSW",
        }
    }
}

/// Supported vector element data types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum VectorDataType {
    /// 32-bit floating point vectors.
    Float32,
    /// 64-bit floating point vectors.
    Float64,
}

impl VectorDataType {
    fn redis_name(self) -> &'static str {
        match self {
            Self::Float32 => "FLOAT32",
            Self::Float64 => "FLOAT64",
        }
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
        if let Some(block_size) = self.block_size {
            args.push("BLOCK_SIZE".to_owned());
            args.push(block_size.to_string());
        }
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

        args
    }
}

#[cfg(test)]
mod tests {
    use super::{IndexSchema, StorageType};

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

        assert_eq!(schema.index.prefix, "rvl");
        assert_eq!(schema.index.key_separator, ":");
        assert!(matches!(schema.index.storage_type, StorageType::Hash));
        assert!(schema.fields.is_empty());
    }
}
