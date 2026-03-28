# Benchmarks

[Criterion](https://bheisler.github.io/criterion.rs/book/) benchmarks for the
`redis-vl` library. Benchmarks are defined in `crates/redis-vl/benches/`.

## Running

```bash
# Pure-Rust benchmarks (no Redis required)
cargo bench -p redis-vl

# Include Redis-backed benchmarks (requires running Redis 8+ or Redis Stack)
REDISVL_RUN_INTEGRATION=1 cargo bench -p redis-vl
```

## Benchmark files

| File | Description |
| --- | --- |
| `core_benchmarks.rs` | Pure-Rust benchmarks (always runnable) |
| `redis_benchmarks.rs` | Redis-backed benchmarks (environment-gated) |

## Pure-Rust benchmarks (`core_benchmarks`)

| Benchmark | What it measures |
| --- | --- |
| Schema parsing | YAML and JSON schema parsing and validation |
| Filter rendering | Filter expression compilation to Redis query syntax |
| Query building | Query construction and parameter compilation |

## Redis-backed benchmarks (`redis_benchmarks`)

Require `REDISVL_RUN_INTEGRATION=1` and a running Redis instance.

| Benchmark | What it measures |
| --- | --- |
| Index create/exists/info | Index lifecycle round-trip latency |
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

## Viewing results

Criterion generates HTML reports in `target/criterion/`:

```bash
open target/criterion/report/index.html
```

## Rust-vs-Python Comparison

A comparison harness runs equivalent benchmarks in both Rust (Criterion) and
Python (redisvl) and generates a side-by-side Markdown report.

### Prerequisites

- Python 3.10+ with redisvl installed:

```bash
pip install -r benches/python/requirements.txt
```

### Running

```bash
# Core benchmarks only (no Redis required)
benches/compare.sh

# Include Redis-backed benchmarks (requires Redis 8+ / Redis Stack)
benches/compare.sh --redis

# With progress output
benches/compare.sh --verbose
```

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `REDIS_URL` | `redis://127.0.0.1:6379` | Redis connection URL |
| `PYTHON` | `python3` | Python interpreter path |
| `CARGO_BENCH_ARGS` | *(empty)* | Extra flags passed to `cargo bench` |

### Output

The runner saves raw JSON results in `target/bench-compare/` and prints a
Markdown comparison table to stdout. The table shows each benchmark's mean
time for Rust and Python, plus the ratio (Py÷Rust; values > 1× mean Rust is
faster).

### Covered operations

| Category | Benchmarks |
| --- | --- |
| Schema | YAML parse, JSON parse, YAML serialize |
| Filters | Simple tag, compound, negated |
| Queries | Vector query build, vector+filter query build |
| Index lifecycle | Create, exists, info |
| Load/fetch | Single load, batch load (100), single fetch |
| Search | Vector k=10, vector+filter, filter-only, count |
| Semantic cache | Store, check hit, check miss |

### Python benchmark files

| File | Description |
| --- | --- |
| `python/bench_harness.py` | Shared timing/reporting framework |
| `python/bench_core.py` | Pure-Python benchmarks (schema, filter, query) |
| `python/bench_redis.py` | Redis-backed benchmarks |
| `python/compare_report.py` | Report generator |

### CI

The `bench.yml` workflow provides a manual `workflow_dispatch` trigger with
optional `run_comparison` and `run_redis` inputs.