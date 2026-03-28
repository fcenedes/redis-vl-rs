# Benchmarks

`redis-vl` includes [Criterion](https://bheisler.github.io/criterion.rs/book/)
micro-benchmarks that measure the performance of core operations.

## Running benchmarks

```bash
# Pure-Rust benchmarks (no Redis required)
cargo bench -p redis-vl

# Include Redis-backed benchmarks (requires running Redis instance)
REDISVL_RUN_INTEGRATION=1 cargo bench -p redis-vl
```

## Benchmark inventory

### Pure-Rust benchmarks (`core_benchmarks`)

| Benchmark | What it measures |
| --- | --- |
| Schema parsing | YAML and JSON schema parsing and validation |
| Filter rendering | Filter expression compilation to Redis query syntax |
| Query building | Query construction and parameter compilation |

### Redis-backed benchmarks (`redis_benchmarks`)

These require `REDISVL_RUN_INTEGRATION=1` and a running Redis instance.

| Benchmark | What it measures |
| --- | --- |
| Index create/exists/info | Index lifecycle operations |
| Index load/fetch | Document loading and retrieval |
| Vector search | Vector similarity search latency |
| Filter search | Filter-only query latency |
| Count query | Document counting latency |
| Batch search | Multi-query batch execution |
| Paginated search | Paginated query execution |
| Embeddings cache | Cache get/set/exists operations |
| Semantic cache | Semantic cache store/check operations |
| Message history | History add/get_recent operations |
| Semantic message history | Semantic history add/get_relevant operations |

## Interpreting results

Criterion generates HTML reports in `target/criterion/`. Open the report index
to see statistical analysis, regression detection, and comparison plots:

```bash
open target/criterion/report/index.html
```

## Adding new benchmarks

Benchmarks live in `crates/redis-vl/benches/`:

- `core_benchmarks.rs` – pure-Rust benchmarks (always runnable)
- `redis_benchmarks.rs` – Redis-backed benchmarks (environment-gated)

To add a new benchmark:

1. Add a benchmark function to the appropriate file
2. Register it in the criterion group
3. Run `cargo bench -p redis-vl` to verify

## Future: Rust vs Python comparison

A Rust-vs-Python comparison harness is planned but not yet implemented. The
goal is to measure relative performance for schema parsing, query compilation,
and Redis round-trip operations across both libraries.
