# Examples

Runnable examples for the `redis-vl` library.

## Available examples

| Example | Description |
| --- | --- |
| `schema_basics` | Parse and validate an index schema from YAML |

## Running

Most examples that interact with Redis require a running Redis instance with the
Search module (Redis 8+ or Redis Stack). Set `REDIS_URL` if your instance is not
at the default `redis://127.0.0.1:6379`.

```bash
cargo run --example schema_basics
```

