//! Criterion benchmarks for core `redis-vl` operations that don't require a
//! Redis connection: schema parsing, filter rendering, and query building.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use redis_vl::{
    BetweenInclusivity, IndexSchema, Vector, VectorQuery,
    filter::{Num, Tag, Text},
    query::QueryString,
};

// ---------------------------------------------------------------------------
// Schema parsing
// ---------------------------------------------------------------------------

const SCHEMA_YAML: &str = r#"
index:
  name: bench-index
  prefix: doc
  storage_type: hash
fields:
  - name: title
    type: tag
  - name: content
    type: text
  - name: score
    type: numeric
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 128
      distance_metric: cosine
      datatype: float32
"#;

fn bench_schema_from_yaml(c: &mut Criterion) {
    c.bench_function("schema_from_yaml", |b| {
        b.iter(|| {
            let _ = IndexSchema::from_yaml_str(black_box(SCHEMA_YAML)).unwrap();
        });
    });
}

fn bench_schema_from_json(c: &mut Criterion) {
    let schema = IndexSchema::from_yaml_str(SCHEMA_YAML).unwrap();
    let json_value = schema.to_json_value().unwrap();
    c.bench_function("schema_from_json", |b| {
        b.iter(|| {
            let _ = IndexSchema::from_json_value(black_box(json_value.clone())).unwrap();
        });
    });
}

fn bench_schema_to_yaml(c: &mut Criterion) {
    let schema = IndexSchema::from_yaml_str(SCHEMA_YAML).unwrap();
    c.bench_function("schema_to_yaml", |b| {
        b.iter(|| {
            let _ = black_box(&schema).to_yaml_string().unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// Filter rendering
// ---------------------------------------------------------------------------

fn bench_simple_tag_filter(c: &mut Criterion) {
    c.bench_function("filter_tag_simple", |b| {
        b.iter(|| {
            let filter = Tag::new("color").eq(black_box("red"));
            let _ = filter.to_redis_syntax();
        });
    });
}

fn bench_compound_filter(c: &mut Criterion) {
    c.bench_function("filter_compound", |b| {
        b.iter(|| {
            let filter = Tag::new("color").eq("red")
                & Num::new("price").between(10.0, 100.0, BetweenInclusivity::Both)
                & Text::new("description").eq("basketball");
            let _ = filter.to_redis_syntax();
        });
    });
}

fn bench_negated_filter(c: &mut Criterion) {
    c.bench_function("filter_negated", |b| {
        b.iter(|| {
            let filter = !(Tag::new("status").eq("deleted") | Tag::new("status").eq("archived"));
            let _ = filter.to_redis_syntax();
        });
    });
}

// ---------------------------------------------------------------------------
// Query building
// ---------------------------------------------------------------------------

fn bench_vector_query_render(c: &mut Criterion) {
    let data = vec![0.1_f32; 128];
    c.bench_function("vector_query_render", |b| {
        b.iter(|| {
            let vector = Vector::new(black_box(data.as_slice()));
            let query = VectorQuery::new(vector, "embedding", 10);
            let _ = query.to_redis_query();
        });
    });
}

fn bench_vector_query_with_filter_render(c: &mut Criterion) {
    let data = vec![0.1_f32; 128];
    c.bench_function("vector_query_with_filter_render", |b| {
        b.iter(|| {
            let vector = Vector::new(black_box(data.as_slice()));
            let filter = Tag::new("color").eq("red") & Num::new("price").lt(50.0);
            let query = VectorQuery::new(vector, "embedding", 10).with_filter(filter);
            let _ = query.to_redis_query();
        });
    });
}

criterion_group!(
    benches,
    bench_schema_from_yaml,
    bench_schema_from_json,
    bench_schema_to_yaml,
    bench_simple_tag_filter,
    bench_compound_filter,
    bench_negated_filter,
    bench_vector_query_render,
    bench_vector_query_with_filter_render,
);
criterion_main!(benches);
