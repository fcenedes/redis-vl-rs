//! Semantic router extension types.

use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    error::Result,
    index::{QueryOutput, RedisConnectionInfo, SearchIndex},
    query::{Vector, VectorRangeQuery},
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
    /// Underlying search index used for route references.
    pub index: SearchIndex,
    vectorizer: Arc<dyn Vectorizer>,
    vector_dimensions: usize,
}

impl SemanticRouter {
    /// Creates a new semantic router and loads the provided routes into Redis.
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
        if routes.is_empty() {
            return Err(crate::Error::InvalidInput(
                "semantic router requires at least one route".to_owned(),
            ));
        }

        let Some(first_reference) = routes
            .iter()
            .flat_map(|route| route.references.iter())
            .next()
            .cloned()
        else {
            return Err(crate::Error::InvalidInput(
                "semantic router routes require at least one reference".to_owned(),
            ));
        };

        let vectorizer = Arc::new(vectorizer);
        let probe_vector = vectorizer.embed(&first_reference)?;
        if probe_vector.is_empty() {
            return Err(crate::Error::InvalidInput(
                "router vectorizer produced an empty embedding".to_owned(),
            ));
        }

        let name = name.into();
        let connection = RedisConnectionInfo::new(redis_url);
        let schema = router_schema(&name, probe_vector.len());
        let index = SearchIndex::from_json_value(schema, connection.redis_url.clone())?;
        if !index.exists().unwrap_or(false) {
            index.create_with_options(false, false)?;
        }

        let router = Self {
            name,
            connection,
            routes,
            routing_config,
            index,
            vectorizer,
            vector_dimensions: probe_vector.len(),
        };
        router.load_routes()?;
        Ok(router)
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
        Ok(())
    }

    /// Clears all route references while preserving the index.
    pub fn clear(&mut self) -> Result<usize> {
        let deleted = self.index.clear()?;
        self.routes.clear();
        Ok(deleted)
    }

    /// Deletes the router index and its documents.
    pub fn delete(&self) -> Result<()> {
        self.index.delete(true)
    }

    /// Serializes the router to a JSON value.
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

fn router_schema(name: &str, vector_dimensions: usize) -> Value {
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
                    "datatype": "float32",
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
