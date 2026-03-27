# Benchmarks

Criterion benchmarks for the `redis-vl` library.

## Running

```bash
cargo bench -p redis-vl
```

## Current benchmarks

| Benchmark | Description |
| --- | --- |
| schema_parsing | YAML/JSON schema parsing and validation |
| filter_rendering | Filter expression compilation to Redis query syntax |
| query_building | Query construction and parameter compilation |

## Python comparison

A Python comparison harness is planned but not yet implemented. The goal is tomeasure Rust vs Python performance for schema parsing, query compilation, andRedis round-trip operations.