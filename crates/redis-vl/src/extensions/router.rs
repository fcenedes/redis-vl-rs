//! Semantic router extension types.
//!
//! [`SemanticRouter`] classifies input text against predefined [`Route`]s using
//! vector similarity. Each route is defined by a set of reference utterances;
//! incoming text is embedded and matched against the closest references.
//!
//! The router supports per-route distance thresholds, configurable aggregation
//! methods ([`DistanceAggregationMethod`]), serialization to/from YAML and JSON,
//! and `from_existing` reconnection to a previously created router index.

use std::{collections::HashMap, path::Path, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    error::{Error, Result},
    index::{QueryOutput, RedisConnectionInfo, SearchIndex},
    query::{Vector, VectorRangeQuery},
    schema::VectorDataType,
    vectorizers::Vectorizer,
};

const ROUTER_REFERENCE_ID_FIELD: &str = "reference_id";
const ROUTER_ROUTE_NAME_FIELD: &str = "route_name";
const ROUTER_REFERENCE_FIELD: &str = "reference";
const ROUTER_VECTOR_FIELD: &str = "vector";
const DEFAULT_ROUTE_DISTANCE_THRESHOLD: f32 = 0.5;

/// Aggregation method used to combine reference distances into a route score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceAggregationMethod {
    /// Average the distances of all matched route references.
    Avg,
    /// Use the minimum matched reference distance.
    Min,
    /// Sum all matched reference distances.
    Sum,
}

/// Route definition used by the semantic router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    /// Route name.
    pub name: String,
    /// Reference utterances used to anchor the route.
    pub references: Vec<String>,
    /// Optional route metadata.
    #[serde(default)]
    pub metadata: Map<String, Value>,
    /// Optional per-route threshold.
    #[serde(default)]
    pub distance_threshold: Option<f32>,
}

impl Route {
    /// Creates a route with the provided name and references.
    pub fn new(name: impl Into<String>, references: Vec<String>) -> Self {
        Self {
            name: name.into(),
            references,
            metadata: Map::new(),
            distance_threshold: None,
        }
    }

    fn effective_threshold(&self) -> f32 {
        self.distance_threshold
            .unwrap_or(DEFAULT_ROUTE_DISTANCE_THRESHOLD)
    }
}

/// Route match result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteMatch {
    /// Matched route name.
    pub name: Option<String>,
    /// Calculated route distance.
    pub distance: Option<f32>,
}

impl RouteMatch {
    fn no_match() -> Self {
        Self {
            name: None,
            distance: None,
        }
    }
}

/// Runtime router configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Maximum number of routes returned by `route_many`.
    pub max_k: usize,
    /// Aggregation method used to score a route from its matched references.
    pub aggregation_method: DistanceAggregationMethod,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            max_k: 1,
            aggregation_method: DistanceAggregationMethod::Avg,
        }
    }
}

/// Semantic router backed by a Redis Search vector index.
#[derive(Clone)]
pub struct SemanticRouter {
    /// Router name.
    pub name: String,
    /// Redis connection settings.
    pub connection: RedisConnectionInfo,
    /// Registered routes.
    pub routes: Vec<Route>,
    /// Runtime routing configuration.
    pub routing_config: RoutingConfig,
    /// Vector element data type used for the index schema.
    pub dtype: VectorDataType,
    /// Underlying search index used for route references.
    pub index: SearchIndex,
    vectorizer: Arc<dyn Vectorizer>,
    vector_dimensions: usize,
}

impl SemanticRouter {
    /// Creates a new semantic router and loads the provided routes into Redis.
    ///
    /// Uses [`VectorDataType::Float32`] by default. For other data types, use
    /// [`Self::new_with_options`].
    pub fn new<V>(
        name: impl Into<String>,
        redis_url: impl Into<String>,
        routes: Vec<Route>,
        routing_config: RoutingConfig,
        vectorizer: V,
    ) -> Result<Self>
    where
        V: Vectorizer + 'static,
    {
        Self::new_with_options(
            name,
            redis_url,
            routes,
            routing_config,
            vectorizer,
            VectorDataType::Float32,
            false,
        )
    }

    /// Creates a new semantic router using the default HuggingFace local
    /// vectorizer (`AllMiniLML6V2`).
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
        routes: Vec<Route>,
        routing_config: RoutingConfig,
    ) -> Result<Self> {
        let vectorizer = crate::vectorizers::HuggingFaceTextVectorizer::new(Default::default())?;
        Self::new(name, redis_url, routes, routing_config, vectorizer)
    }

    /// Creates a new semantic router with explicit dtype and overwrite control.
    pub fn new_with_options<V>(
        name: impl Into<String>,
        redis_url: impl Into<String>,
        routes: Vec<Route>,
        routing_config: RoutingConfig,
        vectorizer: V,
        dtype: VectorDataType,
        overwrite: bool,
    ) -> Result<Self>
    where
        V: Vectorizer + 'static,
    {
        if routes.is_empty() {
            return Err(Error::InvalidInput(
                "semantic router requires at least one route".to_owned(),
            ));
        }

        let Some(first_reference) = routes
            .iter()
            .flat_map(|route| route.references.iter())
            .next()
            .cloned()
        else {
            return Err(Error::InvalidInput(
                "semantic router routes require at least one reference".to_owned(),
            ));
        };

        let vectorizer = Arc::new(vectorizer);
        let probe_vector = vectorizer.embed(&first_reference)?;
        if probe_vector.is_empty() {
            return Err(Error::InvalidInput(
                "router vectorizer produced an empty embedding".to_owned(),
            ));
        }

        let name = name.into();
        let connection = RedisConnectionInfo::new(redis_url);
        let schema = router_schema(&name, probe_vector.len(), dtype);
        let index = SearchIndex::from_json_value(schema, connection.redis_url.clone())?;
        let existed = index.exists().unwrap_or(false);

        // Validate schema compatibility with existing index (mirrors Python behavior)
        if !overwrite && existed {
            let existing_index = SearchIndex::from_existing(&name, connection.redis_url.clone())?;
            if existing_index.schema().to_json_value()? != index.schema().to_json_value()? {
                return Err(Error::InvalidInput(format!(
                    "Existing index {name} schema does not match the user provided schema for the semantic router. \
                     If you wish to overwrite the index schema, set overwrite=true during initialization."
                )));
            }
        }

        index.create_with_options(overwrite, false)?;

        let router = Self {
            name,
            connection,
            routes,
            routing_config,
            dtype,
            index,
            vectorizer,
            vector_dimensions: probe_vector.len(),
        };

        if !existed || overwrite {
            router.load_routes()?;
        }
        router.persist_config()?;
        Ok(router)
    }

    /// Reconnects to an existing semantic router stored in Redis.
    ///
    /// The router configuration (name, routes, routing config) must have been
    /// previously persisted by a [`SemanticRouter::new`] call.  The vectorizer
    /// must be supplied by the caller since vectorizers cannot be serialized.
    ///
    /// The `dtype` parameter must match the data type used when the router was
    /// originally created; otherwise the schema comparison will fail.
    pub fn from_existing<V>(
        name: impl Into<String>,
        redis_url: impl Into<String>,
        vectorizer: V,
        dtype: VectorDataType,
    ) -> Result<Self>
    where
        V: Vectorizer + 'static,
    {
        let name = name.into();
        let connection = RedisConnectionInfo::new(redis_url);
        let config_key = router_config_key(&name);

        let client = connection.client()?;
        let mut conn = client.get_connection()?;

        // Read persisted config
        let raw: Option<String> = redis::cmd("JSON.GET")
            .arg(&config_key)
            .arg(".")
            .query(&mut conn)?;
        let raw = raw.ok_or_else(|| {
            Error::InvalidInput(format!(
                "No valid router config found for {name}. No persisted configuration exists at key '{config_key}'."
            ))
        })?;
        let config_value: Value = serde_json::from_str(&raw)?;
        let config_obj = config_value
            .as_object()
            .ok_or_else(|| Error::InvalidInput("Router config is not an object".to_owned()))?;

        let routes_value = config_obj
            .get("routes")
            .ok_or_else(|| Error::InvalidInput("Router config missing 'routes'".to_owned()))?;
        let routes: Vec<Route> = serde_json::from_value(routes_value.clone())?;
        let routing_config_value = config_obj.get("routing_config").ok_or_else(|| {
            Error::InvalidInput("Router config missing 'routing_config'".to_owned())
        })?;
        let routing_config: RoutingConfig = serde_json::from_value(routing_config_value.clone())?;

        let vectorizer = Arc::new(vectorizer);

        // Probe dimensions from the first route reference
        let first_ref = routes
            .iter()
            .flat_map(|r| r.references.iter())
            .next()
            .ok_or_else(|| Error::InvalidInput("Persisted routes have no references".to_owned()))?;
        let probe = vectorizer.embed(first_ref)?;
        let vector_dimensions = probe.len();

        let schema = router_schema(&name, vector_dimensions, dtype);
        let index = SearchIndex::from_json_value(schema, connection.redis_url.clone())?;
        // The index should already exist
        if !index.exists().unwrap_or(false) {
            return Err(Error::InvalidInput(format!(
                "Index '{name}' does not exist in Redis"
            )));
        }

        Ok(Self {
            name,
            connection,
            routes,
            routing_config,
            dtype,
            index,
            vectorizer,
            vector_dimensions,
        })
    }

    /// Returns the names of the configured routes.
    pub fn route_names(&self) -> Vec<String> {
        self.routes.iter().map(|route| route.name.clone()).collect()
    }

    /// Returns the configured per-route thresholds.
    pub fn route_thresholds(&self) -> HashMap<String, f32> {
        self.routes
            .iter()
            .map(|route| (route.name.clone(), route.effective_threshold()))
            .collect()
    }

    /// Returns a route by name when present.
    pub fn get(&self, route_name: &str) -> Option<&Route> {
        self.routes.iter().find(|route| route.name == route_name)
    }

    /// Updates the runtime routing configuration.
    pub fn update_routing_config(&mut self, routing_config: RoutingConfig) {
        self.routing_config = routing_config;
    }

    /// Updates route thresholds in place for any supplied route names.
    pub fn update_route_thresholds(&mut self, route_thresholds: &HashMap<String, f32>) {
        for route in &mut self.routes {
            if let Some(distance_threshold) = route_thresholds.get(&route.name) {
                route.distance_threshold = Some(*distance_threshold);
            }
        }
    }

    /// Routes a statement or vector to the best matching route.
    pub fn route(&self, statement: Option<&str>, vector: Option<&[f32]>) -> Result<RouteMatch> {
        Ok(self
            .route_many(statement, vector, None, None)?
            .into_iter()
            .next()
            .unwrap_or_else(RouteMatch::no_match))
    }

    /// Routes a statement or vector to the top matching routes.
    pub fn route_many(
        &self,
        statement: Option<&str>,
        vector: Option<&[f32]>,
        max_k: Option<usize>,
        aggregation_method: Option<DistanceAggregationMethod>,
    ) -> Result<Vec<RouteMatch>> {
        let vector = self.resolve_vector(statement, vector)?;
        let max_threshold = self
            .routes
            .iter()
            .map(Route::effective_threshold)
            .fold(DEFAULT_ROUTE_DISTANCE_THRESHOLD, f32::max);
        let reference_count = self
            .routes
            .iter()
            .map(|route| route.references.len())
            .sum::<usize>()
            .max(1);
        let query = VectorRangeQuery::new(Vector::new(vector), ROUTER_VECTOR_FIELD, max_threshold)
            .paging(0, reference_count)
            .with_return_fields([ROUTER_ROUTE_NAME_FIELD, ROUTER_REFERENCE_ID_FIELD]);

        let documents = query_output_documents(self.index.query(&query)?)?;
        let mut grouped: HashMap<String, Vec<f32>> = HashMap::new();
        for document in documents {
            let Some(route_name) = document
                .get(ROUTER_ROUTE_NAME_FIELD)
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            let Some(distance) = parse_distance(document.get("vector_distance")) else {
                continue;
            };
            grouped.entry(route_name).or_default().push(distance);
        }

        let aggregation_method =
            aggregation_method.unwrap_or(self.routing_config.aggregation_method);
        let mut matches = self
            .routes
            .iter()
            .filter_map(|route| {
                let distances = grouped.get(&route.name)?;
                let distance = aggregate_distances(distances, aggregation_method);
                (distance <= route.effective_threshold()).then(|| RouteMatch {
                    name: Some(route.name.clone()),
                    distance: Some(distance),
                })
            })
            .collect::<Vec<_>>();

        matches.sort_by(|left, right| {
            let left = left.distance.unwrap_or(f32::INFINITY);
            let right = right.distance.unwrap_or(f32::INFINITY);
            left.total_cmp(&right)
        });
        matches.truncate(max_k.unwrap_or(self.routing_config.max_k));
        Ok(matches)
    }

    /// Adds routes to the router and loads their references into Redis.
    pub fn add_routes(&mut self, routes: &[Route]) -> Result<()> {
        for route in routes {
            validate_route(route)?;
        }
        self.load_route_batch(routes)?;
        for route in routes {
            if self.get(&route.name).is_none() {
                self.routes.push(route.clone());
            }
        }
        Ok(())
    }

    /// Removes a route and its references from Redis.
    pub fn remove_route(&mut self, route_name: &str) -> Result<()> {
        let Some(route) = self.get(route_name).cloned() else {
            return Ok(());
        };

        let keys = route
            .references
            .iter()
            .map(|reference| self.index.key(&route_reference_id(&route.name, reference)))
            .collect::<Vec<_>>();
        self.index.drop_keys(&keys)?;
        self.routes.retain(|route| route.name != route_name);
        self.persist_config()?;
        Ok(())
    }

    /// Clears all route references while preserving the index.
    pub fn clear(&mut self) -> Result<usize> {
        let deleted = self.index.clear()?;
        self.routes.clear();
        Ok(deleted)
    }

    /// Deletes the router index, its documents, and the persisted config key.
    pub fn delete(&self) -> Result<()> {
        // Remove persisted config
        let config_key = router_config_key(&self.name);
        let client = self.connection.client()?;
        let mut conn = client.get_connection()?;
        let _: usize = redis::cmd("DEL").arg(&config_key).query(&mut conn)?;
        // Tolerate the index already being deleted (e.g. by another router
        // instance pointing at the same name after a YAML/dict round-trip).
        match self.index.delete(true) {
            Ok(()) => Ok(()),
            Err(Error::InvalidInput(msg)) if msg.contains("does not exist") => Ok(()),
            Err(other) => Err(other),
        }
    }

    /// Adds new references to an existing route and loads them into Redis.
    ///
    /// Returns the list of Redis keys created for the added references.
    pub fn add_route_references(
        &mut self,
        route_name: &str,
        references: &[String],
    ) -> Result<Vec<String>> {
        if references.is_empty() {
            return Ok(Vec::new());
        }

        // Validate route exists before doing any work
        if self.get(route_name).is_none() {
            return Err(crate::Error::InvalidInput(format!(
                "Route '{route_name}' not found in the SemanticRouter"
            )));
        }

        let refs_str: Vec<&str> = references.iter().map(String::as_str).collect();
        let embeddings = self.vectorizer.embed_many(&refs_str)?;
        let mut records = Vec::with_capacity(references.len());
        let mut keys = Vec::with_capacity(references.len());

        for (reference, embedding) in references.iter().zip(embeddings) {
            if embedding.len() != self.vector_dimensions {
                return Err(crate::Error::InvalidInput(format!(
                    "router vector dimensions mismatch: expected {}, got {}",
                    self.vector_dimensions,
                    embedding.len()
                )));
            }
            let ref_id = route_reference_id(route_name, reference);
            keys.push(self.index.key(&ref_id));
            records.push(json!({
                ROUTER_REFERENCE_ID_FIELD: ref_id,
                ROUTER_ROUTE_NAME_FIELD: route_name,
                ROUTER_REFERENCE_FIELD: reference,
                ROUTER_VECTOR_FIELD: embedding,
            }));
        }

        if !records.is_empty() {
            let _: Vec<String> = self.index.load(&records, ROUTER_REFERENCE_ID_FIELD, None)?;
        }

        // Update the in-memory route with the new references
        if let Some(route) = self.routes.iter_mut().find(|r| r.name == route_name) {
            route.references.extend(references.iter().cloned());
        }
        self.persist_config()?;

        Ok(keys)
    }

    /// Retrieves reference metadata for a route by name or by specific reference IDs.
    ///
    /// At least one of `route_name` or `reference_ids` must be provided.
    pub fn get_route_references(
        &self,
        route_name: Option<&str>,
        reference_ids: Option<&[String]>,
    ) -> Result<Vec<Map<String, Value>>> {
        let ids_to_query: Vec<String> = if let Some(ref_ids) = reference_ids {
            ref_ids.to_vec()
        } else if let Some(route_name) = route_name {
            let pattern = self.route_pattern(route_name);
            let scanned = self.scan_keys(&pattern)?;
            let sep = self.index.key_separator();
            let prefix = self.index.prefix();
            let prefix_with_sep = if prefix.ends_with(sep) {
                prefix.to_owned()
            } else {
                format!("{prefix}{sep}")
            };
            // Strip the prefix to recover the reference_id ("route_name:hash")
            scanned
                .into_iter()
                .map(|key| {
                    key.strip_prefix(&prefix_with_sep)
                        .unwrap_or(&key)
                        .to_owned()
                })
                .collect()
        } else {
            return Err(crate::Error::InvalidInput(
                "Must provide a route name, reference ids, or keys to get references".to_owned(),
            ));
        };

        let queries: Vec<crate::query::FilterQuery> = ids_to_query
            .iter()
            .map(|id| {
                let filter = crate::filter::Tag::new(ROUTER_REFERENCE_ID_FIELD).eq(id.as_str());
                crate::query::FilterQuery::new(filter).with_return_fields([
                    ROUTER_REFERENCE_ID_FIELD,
                    ROUTER_ROUTE_NAME_FIELD,
                    ROUTER_REFERENCE_FIELD,
                ])
            })
            .collect();

        let results = self.index.batch_query(queries.iter())?;
        let mut refs = Vec::new();
        for result in results {
            if let QueryOutput::Documents(docs) = result {
                for doc in docs {
                    refs.push(doc);
                }
            }
        }
        Ok(refs)
    }

    /// Deletes route references by route name, reference IDs, or explicit keys.
    ///
    /// Returns the number of references deleted.
    pub fn delete_route_references(
        &mut self,
        route_name: Option<&str>,
        reference_ids: Option<&[String]>,
        keys: Option<&[String]>,
    ) -> Result<usize> {
        let keys_to_delete: Vec<String> = if let Some(explicit_keys) = keys {
            explicit_keys.to_vec()
        } else if let Some(ref_ids) = reference_ids {
            // Query to find the full keys for these reference IDs
            let queries: Vec<crate::query::FilterQuery> = ref_ids
                .iter()
                .map(|id| {
                    let filter = crate::filter::Tag::new(ROUTER_REFERENCE_ID_FIELD).eq(id.as_str());
                    crate::query::FilterQuery::new(filter).with_return_fields([
                        ROUTER_REFERENCE_ID_FIELD,
                        ROUTER_ROUTE_NAME_FIELD,
                        ROUTER_REFERENCE_FIELD,
                    ])
                })
                .collect();

            let results = self.index.batch_query(queries.iter())?;
            let mut found_keys = Vec::new();
            for result in results {
                if let QueryOutput::Documents(docs) = result {
                    for doc in docs {
                        // reference_id already contains "route_name:hash"
                        // (produced by route_reference_id), so use it directly
                        if let Some(ref_id) =
                            doc.get(ROUTER_REFERENCE_ID_FIELD).and_then(Value::as_str)
                        {
                            found_keys.push(self.index.key(ref_id));
                        }
                    }
                }
            }
            found_keys
        } else if let Some(route_name) = route_name {
            let pattern = self.route_pattern(route_name);
            self.scan_keys(&pattern)?
        } else {
            return Err(crate::Error::InvalidInput(
                "Must provide route_name, reference_ids, or keys to delete references".to_owned(),
            ));
        };

        if keys_to_delete.is_empty() {
            return Ok(0);
        }

        // Remove matching references from in-memory routes.
        //
        // We derive the route name and reference text from the key structure
        // rather than fetching from Redis, because hash documents contain
        // binary vector data that cannot be decoded as UTF-8 strings.
        //
        // Key format: {prefix}{sep}{route_name}{sep}{sha256_hash}
        // Reference ID format: {route_name}:{sha256_hash}
        let sep = self.index.key_separator();
        let prefix_raw = self.index.prefix().trim_end_matches(sep);
        let prefix_with_sep = if prefix_raw.is_empty() {
            String::new()
        } else {
            format!("{prefix_raw}{sep}")
        };
        for key in &keys_to_delete {
            let id = key.strip_prefix(&prefix_with_sep).unwrap_or(key);
            // id is "route_name:hash" — extract route_name from the first ':'
            if let Some((rname, _hash)) = id.split_once(':') {
                if let Some(route) = self.routes.iter_mut().find(|r| r.name == rname) {
                    // Find which reference produces this reference_id
                    route
                        .references
                        .retain(|ref_text| route_reference_id(rname, ref_text) != id);
                }
            }
        }

        let deleted = self.index.drop_keys(&keys_to_delete)?;
        self.persist_config()?;
        Ok(deleted)
    }

    /// Serializes the router to a JSON value (equivalent to Python `to_dict()`).
    ///
    /// The `vectorizer` field contains a `{"type": "custom"}` marker since
    /// Rust vectorizer implementations cannot be serialized.  Use this for
    /// round-tripping via [`Self::from_dict`].
    pub fn to_json_value(&self) -> Result<Value> {
        Ok(json!({
            "name": self.name,
            "routes": self.routes,
            "routing_config": self.routing_config,
            "vectorizer": {
                "type": "custom"
            }
        }))
    }

    /// Alias for [`Self::to_json_value`] matching the Python `to_dict()` name.
    pub fn to_dict(&self) -> Result<Value> {
        self.to_json_value()
    }

    /// Writes the router configuration to a YAML file.
    ///
    /// If `overwrite` is false and the file already exists, returns an error.
    pub fn to_yaml(&self, file_path: impl AsRef<Path>, overwrite: bool) -> Result<()> {
        let path = file_path.as_ref();
        if path.exists() && !overwrite {
            return Err(Error::InvalidInput(format!(
                "Schema file {} already exists.",
                path.display()
            )));
        }
        let dict = self.to_json_value()?;
        let file = std::fs::File::create(path)
            .map_err(|e| Error::InvalidInput(format!("Cannot create file: {e}")))?;
        serde_yaml::to_writer(file, &dict)
            .map_err(|e| Error::InvalidInput(format!("YAML serialization error: {e}")))?;
        Ok(())
    }

    /// Creates a semantic router from a YAML file.
    ///
    /// The caller must supply a vectorizer since vectorizer implementations
    /// cannot be deserialized from YAML.
    pub fn from_yaml<V>(
        file_path: impl AsRef<Path>,
        redis_url: impl Into<String>,
        vectorizer: V,
        dtype: VectorDataType,
        overwrite: bool,
    ) -> Result<Self>
    where
        V: Vectorizer + 'static,
    {
        let path = file_path.as_ref();
        if !path.exists() {
            return Err(Error::InvalidInput(format!(
                "File {} does not exist",
                path.display()
            )));
        }
        let file = std::fs::File::open(path)
            .map_err(|e| Error::InvalidInput(format!("Cannot open file: {e}")))?;
        let dict: Value = serde_yaml::from_reader(file)
            .map_err(|e| Error::InvalidInput(format!("YAML deserialization error: {e}")))?;
        Self::from_dict(dict, redis_url, vectorizer, dtype, overwrite)
    }

    /// Creates a semantic router from a previously serialized JSON value.
    ///
    /// The caller must supply a vectorizer since vectorizer implementations
    /// cannot be deserialized from JSON.
    pub fn from_dict<V>(
        data: Value,
        redis_url: impl Into<String>,
        vectorizer: V,
        dtype: VectorDataType,
        overwrite: bool,
    ) -> Result<Self>
    where
        V: Vectorizer + 'static,
    {
        let obj = data
            .as_object()
            .ok_or_else(|| Error::InvalidInput("Router dict must be a JSON object".to_owned()))?;

        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::InvalidInput(
                    "Unable to load semantic router from dict: missing 'name'".to_owned(),
                )
            })?
            .to_owned();

        let routes_value = obj.get("routes").ok_or_else(|| {
            Error::InvalidInput(
                "Unable to load semantic router from dict: missing 'routes'".to_owned(),
            )
        })?;
        let routes: Vec<Route> = serde_json::from_value(routes_value.clone())?;

        let routing_config_value = obj.get("routing_config").ok_or_else(|| {
            Error::InvalidInput(
                "Unable to load semantic router from dict: missing 'routing_config'".to_owned(),
            )
        })?;
        let routing_config: RoutingConfig = serde_json::from_value(routing_config_value.clone())?;

        Self::new_with_options(
            name,
            redis_url,
            routes,
            routing_config,
            vectorizer,
            dtype,
            overwrite,
        )
    }

    /// Persists the current router configuration to Redis as a JSON document.
    fn persist_config(&self) -> Result<()> {
        let config_key = router_config_key(&self.name);
        let dict = self.to_json_value()?;
        let json_str = serde_json::to_string(&dict)?;
        let client = self.connection.client()?;
        let mut conn = client.get_connection()?;
        let _: () = redis::cmd("JSON.SET")
            .arg(&config_key)
            .arg(".")
            .arg(&json_str)
            .query(&mut conn)?;
        Ok(())
    }

    /// Scans Redis keys matching a glob pattern.
    fn scan_keys(&self, pattern: &str) -> Result<Vec<String>> {
        let client = self.connection.client()?;
        let mut connection = client.get_connection()?;
        let mut cursor = 0_u64;
        let mut keys = Vec::new();
        loop {
            let (next_cursor, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(pattern)
                .arg("COUNT")
                .arg(100)
                .query(&mut connection)?;
            keys.extend(batch);
            if next_cursor == 0 {
                break;
            }
            cursor = next_cursor;
        }
        Ok(keys)
    }

    /// Builds a scan pattern for a route's references.
    fn route_pattern(&self, route_name: &str) -> String {
        let sep = self.index.key_separator();
        let prefix = self.index.prefix().trim_end_matches(sep);
        if prefix.is_empty() {
            format!("{route_name}{sep}*")
        } else {
            format!("{prefix}{sep}{route_name}{sep}*")
        }
    }

    fn load_routes(&self) -> Result<()> {
        self.load_route_batch(&self.routes)
    }

    fn load_route_batch(&self, routes: &[Route]) -> Result<()> {
        for route in routes {
            validate_route(route)?;
        }

        let mut records = Vec::new();
        for route in routes {
            let refs = route
                .references
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let embeddings = self.vectorizer.embed_many(&refs)?;
            for (reference, embedding) in route.references.iter().zip(embeddings) {
                if embedding.len() != self.vector_dimensions {
                    return Err(crate::Error::InvalidInput(format!(
                        "router vector dimensions mismatch: expected {}, got {}",
                        self.vector_dimensions,
                        embedding.len()
                    )));
                }
                records.push(json!({
                    ROUTER_REFERENCE_ID_FIELD: route_reference_id(&route.name, reference),
                    ROUTER_ROUTE_NAME_FIELD: route.name,
                    ROUTER_REFERENCE_FIELD: reference,
                    ROUTER_VECTOR_FIELD: embedding,
                }));
            }
        }

        if !records.is_empty() {
            let _: Vec<String> = self.index.load(&records, ROUTER_REFERENCE_ID_FIELD, None)?;
        }
        Ok(())
    }

    fn resolve_vector(&self, statement: Option<&str>, vector: Option<&[f32]>) -> Result<Vec<f32>> {
        match (statement, vector) {
            (_, Some(vector)) => {
                if vector.len() != self.vector_dimensions {
                    return Err(crate::Error::InvalidInput(format!(
                        "router vector dimensions mismatch: expected {}, got {}",
                        self.vector_dimensions,
                        vector.len()
                    )));
                }
                Ok(vector.to_vec())
            }
            (Some(statement), None) => {
                let vector = self.vectorizer.embed(statement)?;
                if vector.len() != self.vector_dimensions {
                    return Err(crate::Error::InvalidInput(format!(
                        "router vector dimensions mismatch: expected {}, got {}",
                        self.vector_dimensions,
                        vector.len()
                    )));
                }
                Ok(vector)
            }
            (None, None) => Err(crate::Error::InvalidInput(
                "must provide a statement or vector to the router".to_owned(),
            )),
        }
    }
}

impl std::fmt::Debug for SemanticRouter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticRouter")
            .field("name", &self.name)
            .field("routes", &self.routes)
            .field("routing_config", &self.routing_config)
            .field("vector_dimensions", &self.vector_dimensions)
            .finish()
    }
}

fn validate_route(route: &Route) -> Result<()> {
    if route.name.trim().is_empty() {
        return Err(crate::Error::InvalidInput(
            "route name must not be empty".to_owned(),
        ));
    }
    if route.references.is_empty() {
        return Err(crate::Error::InvalidInput(
            "route references must not be empty".to_owned(),
        ));
    }
    if route
        .references
        .iter()
        .any(|reference| reference.trim().is_empty())
    {
        return Err(crate::Error::InvalidInput(
            "route references must not contain empty strings".to_owned(),
        ));
    }
    let threshold = route.effective_threshold();
    if !(0.0..=2.0).contains(&threshold) {
        return Err(crate::Error::InvalidInput(format!(
            "route distance threshold must be between 0 and 2, got {threshold}"
        )));
    }
    Ok(())
}

fn router_config_key(name: &str) -> String {
    format!("{name}:route_config")
}

fn router_schema(name: &str, vector_dimensions: usize, dtype: VectorDataType) -> Value {
    json!({
        "index": {
            "name": name,
            "prefix": name,
            "storage_type": "hash",
        },
        "fields": [
            { "name": ROUTER_REFERENCE_ID_FIELD, "type": "tag" },
            { "name": ROUTER_ROUTE_NAME_FIELD, "type": "tag" },
            { "name": ROUTER_REFERENCE_FIELD, "type": "text" },
            {
                "name": ROUTER_VECTOR_FIELD,
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

fn route_reference_id(route_name: &str, reference: &str) -> String {
    format!("{route_name}:{}", hashify(reference))
}

fn hashify(content: &str) -> String {
    use sha2::{Digest, Sha256};

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

fn query_output_documents(output: QueryOutput) -> Result<Vec<Map<String, Value>>> {
    match output {
        QueryOutput::Documents(documents) => Ok(documents),
        QueryOutput::Count(_) => Err(crate::Error::InvalidInput(
            "router queries must return documents".to_owned(),
        )),
    }
}

fn parse_distance(value: Option<&Value>) -> Option<f32> {
    match value {
        Some(Value::Number(number)) => number.as_f64().map(|value| value as f32),
        Some(Value::String(value)) => value.parse::<f32>().ok(),
        _ => None,
    }
}

fn aggregate_distances(distances: &[f32], aggregation_method: DistanceAggregationMethod) -> f32 {
    match aggregation_method {
        DistanceAggregationMethod::Avg => distances.iter().sum::<f32>() / distances.len() as f32,
        DistanceAggregationMethod::Min => distances.iter().copied().fold(f32::INFINITY, f32::min),
        DistanceAggregationMethod::Sum => distances.iter().sum::<f32>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_new_defaults() {
        let route = Route::new("test", vec!["ref1".to_owned()]);
        assert_eq!(route.name, "test");
        assert_eq!(route.references, vec!["ref1".to_owned()]);
        assert!(route.metadata.is_empty());
        assert!(route.distance_threshold.is_none());
    }

    #[test]
    fn route_effective_threshold_default() {
        let route = Route::new("test", vec!["ref1".to_owned()]);
        assert_eq!(
            route.effective_threshold(),
            DEFAULT_ROUTE_DISTANCE_THRESHOLD
        );
    }

    #[test]
    fn route_effective_threshold_custom() {
        let route = Route {
            distance_threshold: Some(0.3),
            ..Route::new("test", vec!["ref1".to_owned()])
        };
        assert_eq!(route.effective_threshold(), 0.3);
    }

    #[test]
    fn route_match_no_match() {
        let rm = RouteMatch::no_match();
        assert!(rm.name.is_none());
        assert!(rm.distance.is_none());
    }

    #[test]
    fn routing_config_default() {
        let config = RoutingConfig::default();
        assert_eq!(config.max_k, 1);
        assert_eq!(config.aggregation_method, DistanceAggregationMethod::Avg);
    }

    #[test]
    fn routing_config_serde_roundtrip() {
        let config = RoutingConfig {
            max_k: 5,
            aggregation_method: DistanceAggregationMethod::Min,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RoutingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_k, 5);
        assert_eq!(
            deserialized.aggregation_method,
            DistanceAggregationMethod::Min
        );
    }

    #[test]
    fn validate_route_ok() {
        let route = Route {
            distance_threshold: Some(0.3),
            ..Route::new("greeting", vec!["hello".to_owned()])
        };
        assert!(validate_route(&route).is_ok());
    }

    #[test]
    fn validate_route_empty_name() {
        let route = Route::new("", vec!["hello".to_owned()]);
        assert!(validate_route(&route).is_err());
    }

    #[test]
    fn validate_route_whitespace_name() {
        let route = Route::new("  ", vec!["hello".to_owned()]);
        assert!(validate_route(&route).is_err());
    }

    #[test]
    fn validate_route_empty_references() {
        let route = Route::new("test", vec![]);
        assert!(validate_route(&route).is_err());
    }

    #[test]
    fn validate_route_empty_reference_string() {
        let route = Route::new("test", vec!["".to_owned()]);
        assert!(validate_route(&route).is_err());
    }

    #[test]
    fn validate_route_bad_threshold() {
        let route = Route {
            distance_threshold: Some(-0.1),
            ..Route::new("test", vec!["hello".to_owned()])
        };
        assert!(validate_route(&route).is_err());

        let route = Route {
            distance_threshold: Some(2.5),
            ..Route::new("test", vec!["hello".to_owned()])
        };
        assert!(validate_route(&route).is_err());
    }

    #[test]
    fn route_reference_id_deterministic() {
        let id1 = route_reference_id("greeting", "hello");
        let id2 = route_reference_id("greeting", "hello");
        assert_eq!(id1, id2);
    }

    #[test]
    fn route_reference_id_different_for_different_refs() {
        let id1 = route_reference_id("greeting", "hello");
        let id2 = route_reference_id("greeting", "hi");
        assert_ne!(id1, id2);
    }

    #[test]
    fn route_reference_id_different_for_different_routes() {
        let id1 = route_reference_id("greeting", "hello");
        let id2 = route_reference_id("farewell", "hello");
        assert_ne!(id1, id2);
    }

    #[test]
    fn hashify_deterministic() {
        assert_eq!(hashify("test"), hashify("test"));
        assert_ne!(hashify("test"), hashify("other"));
    }

    #[test]
    fn aggregate_distances_avg() {
        let result = aggregate_distances(&[0.1, 0.3], DistanceAggregationMethod::Avg);
        assert!((result - 0.2).abs() < 1e-6);
    }

    #[test]
    fn aggregate_distances_min() {
        let result = aggregate_distances(&[0.3, 0.1, 0.5], DistanceAggregationMethod::Min);
        assert!((result - 0.1).abs() < 1e-6);
    }

    #[test]
    fn aggregate_distances_sum() {
        let result = aggregate_distances(&[0.1, 0.2, 0.3], DistanceAggregationMethod::Sum);
        assert!((result - 0.6).abs() < 1e-6);
    }

    #[test]
    fn parse_distance_number() {
        let val = Value::Number(serde_json::Number::from_f64(0.5).unwrap());
        assert_eq!(parse_distance(Some(&val)), Some(0.5));
    }

    #[test]
    fn parse_distance_string() {
        let val = Value::String("0.25".to_owned());
        assert_eq!(parse_distance(Some(&val)), Some(0.25));
    }

    #[test]
    fn parse_distance_none() {
        assert_eq!(parse_distance(None), None);
    }

    #[test]
    fn parse_distance_invalid() {
        let val = Value::String("not_a_number".to_owned());
        assert_eq!(parse_distance(Some(&val)), None);
    }

    #[test]
    fn router_schema_structure() {
        let schema = router_schema("my_router", 64, VectorDataType::Float32);
        assert_eq!(schema["index"]["name"], "my_router");
        assert_eq!(schema["index"]["prefix"], "my_router");
        assert_eq!(schema["index"]["storage_type"], "hash");

        let fields = schema["fields"].as_array().unwrap();
        let field_names: Vec<&str> = fields.iter().filter_map(|f| f["name"].as_str()).collect();
        assert!(field_names.contains(&"reference_id"));
        assert!(field_names.contains(&"route_name"));
        assert!(field_names.contains(&"reference"));
        assert!(field_names.contains(&"vector"));

        let vector_field = fields
            .iter()
            .find(|f| f["name"].as_str() == Some("vector"))
            .unwrap();
        assert_eq!(vector_field["attrs"]["dims"], 64);
        assert_eq!(vector_field["attrs"]["datatype"], "float32");
    }

    #[test]
    fn distance_aggregation_method_serde() {
        let json = serde_json::to_string(&DistanceAggregationMethod::Min).unwrap();
        assert_eq!(json, "\"min\"");
        let deserialized: DistanceAggregationMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, DistanceAggregationMethod::Min);
    }

    #[test]
    fn route_serde_roundtrip() {
        let route = Route {
            name: "test".to_owned(),
            references: vec!["hello".to_owned(), "hi".to_owned()],
            metadata: serde_json::Map::from_iter([("type".to_owned(), json!("greeting"))]),
            distance_threshold: Some(0.3),
        };
        let json = serde_json::to_string(&route).unwrap();
        let deserialized: Route = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test");
        assert_eq!(deserialized.references, vec!["hello", "hi"]);
        assert_eq!(deserialized.distance_threshold, Some(0.3));
    }

    #[test]
    fn router_config_key_format() {
        assert_eq!(router_config_key("my_router"), "my_router:route_config");
    }

    #[test]
    fn router_schema_respects_dtype() {
        let schema_f64 = router_schema("my_router", 64, VectorDataType::Float64);
        let fields = schema_f64["fields"].as_array().unwrap();
        let vector_field = fields
            .iter()
            .find(|f| f["name"].as_str() == Some("vector"))
            .unwrap();
        assert_eq!(vector_field["attrs"]["datatype"], "float64");

        let schema_bf16 = router_schema("my_router", 64, VectorDataType::Bfloat16);
        let fields = schema_bf16["fields"].as_array().unwrap();
        let vector_field = fields
            .iter()
            .find(|f| f["name"].as_str() == Some("vector"))
            .unwrap();
        assert_eq!(vector_field["attrs"]["datatype"], "bfloat16");

        let schema_f16 = router_schema("my_router", 64, VectorDataType::Float16);
        let fields = schema_f16["fields"].as_array().unwrap();
        let vector_field = fields
            .iter()
            .find(|f| f["name"].as_str() == Some("vector"))
            .unwrap();
        assert_eq!(vector_field["attrs"]["datatype"], "float16");
    }
}
