# RedisVL Rust Benchmarks

This directory contains Criterion benchmarks for the `redis-vl` crate.

## Benchmark Files

### `core_benchmarks.rs`

**In-memory construction benchmarks** that measure performance of pure Rust operations without requiring Redis:

- **Schema parsing**: YAML → schema, JSON → schema, schema → YAML serialization
- **Filter rendering**: Tag/Text/Num filter construction and Redis syntax rendering
- **Query building**: Vector query construction with and without filters

**Run these benchmarks:**
```bash
cargo bench --bench core_benchmarks
```

These benchmarks always run and provide a baseline for construction performance.

### `redis_benchmarks.rs`

**Redis-backed operation benchmarks** that measure real I/O performance:

- **Index lifecycle**: create, exists, info
- **Index data operations**: load (single, batch), fetch (single, batch)
- **Search operations**: 
  - Vector search (K-NN, with filters)
  - Filter queries (simple, compound)
  - Count queries
  - Batch search
  - Pagination
- **Cache extensions**:
  - Embeddings cache: set, get (hit/miss)
  - Semantic cache: store, check (hit/miss)
- **Message history**:
  - Standard history: add, get_recent
  - Semantic history: add, get_recent, get_relevant

**Run these benchmarks:**
```bash
# Requires Redis to be running
REDISVL_RUN_BENCHMARKS=1 cargo bench --bench redis_benchmarks
```

These benchmarks are **opt-in** via the `REDISVL_RUN_BENCHMARKS` environment variable.

## Prerequisites

### For `core_benchmarks.rs`

No prerequisites - benchmarks run entirely in-memory.

### For `redis_benchmarks.rs`

1. **Redis instance** running locally or remotely
2. Set `REDIS_URL` environment variable (default: `redis://127.0.0.1:6379`)
3. Set `REDISVL_RUN_BENCHMARKS=1` to enable the benchmarks

**Example:**
```bash
# Using default local Redis
REDISVL_RUN_BENCHMARKS=1 cargo bench --bench redis_benchmarks

# Using custom Redis URL
REDIS_URL=redis://192.168.1.100:6379 REDISVL_RUN_BENCHMARKS=1 cargo bench --bench redis_benchmarks
```

## Benchmark Organization

The Redis benchmarks are organized into criterion groups:

- `index_lifecycle` - Index create, exists, info
- `index_data` - Load and fetch operations
- `search_vector` - Vector similarity search
- `search_filter_count` - Filter and count queries
- `search_batch_pagination` - Batch and paginated queries
- `embeddings_cache` - Embeddings cache operations
- `semantic_cache` - Semantic cache operations
- `message_history` - Basic message history
- `semantic_history` - Semantic message history

You can run individual groups:
```bash
REDISVL_RUN_BENCHMARKS=1 cargo bench --bench redis_benchmarks -- index_lifecycle
REDISVL_RUN_BENCHMARKS=1 cargo bench --bench redis_benchmarks -- search_vector
```

## Understanding the Results

Criterion generates:
- **Console output**: Summary statistics (mean, median, std dev)
- **HTML reports**: Located in `target/criterion/` with detailed plots and comparisons
- **Baseline comparison**: Re-running benchmarks compares against previous runs

**Example output:**
```
search_vector_k10_n100 time:   [1.234 ms 1.245 ms 1.256 ms]
                       change: [-2.3% +0.5% +3.2%] (p = 0.12 > 0.05)
                       No change in performance detected.
```

## Benchmark Design Principles

1. **Separation of concerns**: Construction benchmarks vs. I/O benchmarks
2. **Opt-in Redis testing**: Mirrors integration test pattern
3. **Realistic scenarios**: Benchmark actual usage patterns
4. **Proper cleanup**: Each benchmark cleans up its Redis data
5. **Unique naming**: Benchmarks use atomic counters to avoid conflicts
6. **Throughput metrics**: Batch operations report elements/second

## Adding New Benchmarks

When adding a new benchmark:

1. **Choose the right file**:
   - Pure construction → `core_benchmarks.rs`
   - Requires Redis → `redis_benchmarks.rs`

2. **For Redis benchmarks**:
   - Guard with `if !benchmarks_enabled() { return; }`
   - Use `unique_name()` for index/cache names
   - Clean up resources at the end
   - Consider adding to an existing criterion group

3. **Add to criterion groups**:
```rust
criterion_group!(
    my_group,
    bench_my_operation,
    bench_my_other_operation,
);
criterion_main!(my_group, /* other groups */);
```

## Comparison with Python RedisVL

The Rust implementation aims for parity with Python RedisVL performance while maintaining Rust's safety guarantees. Key areas of comparison:

- **Schema parsing**: Rust typically faster due to zero-copy parsing
- **Filter rendering**: Similar performance, both generate Redis Search syntax
- **Vector search**: Limited by Redis performance, similar throughput
- **Cache operations**: Rust may have lower overhead on serialization
- **Message history**: Comparable performance for Redis-backed operations

A formal Python-vs-Rust comparison harness is tracked in the parity roadmap but not yet implemented.

## Performance Tips

1. **Baseline runs**: Run benchmarks at least twice to establish baseline
2. **System load**: Close other applications for accurate results
3. **Redis tuning**: Use local Redis for lowest latency
4. **Batch sizes**: Adjust `batch_size` parameters in benchmarks to match your use case
5. **Warm-up**: Criterion automatically warms up before measuring

## CI/CD Integration

The benchmarks can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run construction benchmarks
  run: cargo bench --bench core_benchmarks

- name: Run Redis benchmarks
  run: |
    docker run -d -p 6379:6379 redis:latest
    sleep 5
    REDISVL_RUN_BENCHMARKS=1 cargo bench --bench redis_benchmarks
```

For performance regression detection, use Criterion's `--save-baseline` and `--baseline` flags.
