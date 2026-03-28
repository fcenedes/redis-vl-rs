# Examples

Runnable examples for the `redis-vl` library. Examples live in
`crates/redis-vl/examples/` and are run via Cargo.

## Available examples

| Example | Redis required? | Feature flags | Description |
| --- | --- | --- | --- |
| `schema_basics` | No | — | Parse and validate an index schema from YAML |
| `filter_basics` | No | — | Build and compose filter expressions |
| `vector_search` | **Yes** | — | Create an index, load data, and run vector/filter/count queries |
| `semantic_cache_basics` | **Yes** *(partial)* | — | Set up a semantic LLM response cache |
| `message_history_basics` | **Yes** | — | Store and retrieve conversation messages |
| `semantic_router_basics` | No *(config only)* | — | Define routes and routing configuration |
| `sql_query_basics` | No *(build only)* | `sql` | Build SQL queries and inspect their translations |

## Running

Most examples that interact with Redis require a running Redis instance with the
Search module (Redis 8+ or Redis Stack). Set `REDIS_URL` if your instance is not
at the default `redis://127.0.0.1:6379`.

```bash
# No Redis required
cargo run -p redis-vl --example schema_basics
cargo run -p redis-vl --example filter_basics

# Requires Redis
cargo run -p redis-vl --example vector_search
cargo run -p redis-vl --example message_history_basics

# Requires feature flag
cargo run -p redis-vl --features sql --example sql_query_basics
```

## Writing new examples

1. Create a new `.rs` file in `crates/redis-vl/examples/`
2. Add a module doc comment explaining what the example demonstrates
3. Include the run command in the doc comment
4. Update this README with the new example

