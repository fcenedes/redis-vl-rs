# CLI

The `rvl` binary provides command-line access to Redis Search index management.

## Installation

```bash
cargo install --path crates/rvl
```

## Commands

### Version

```bash
rvl version
```

### Index management

All index commands require a `--schema` flag pointing to a YAML schema file.

```bash
# Create an index
rvl index create --schema schema.yaml

# Show index metadata
rvl index info --schema schema.yaml

# List all indices
rvl index listall --schema schema.yaml

# Delete an index
rvl index delete --schema schema.yaml

# Delete an index and its documents
rvl index delete --schema schema.yaml --delete-documents

# Alias for delete
rvl index destroy --schema schema.yaml
```

### Stats

```bash
rvl stats --schema schema.yaml
```

## Configuration

| Option | Environment variable | Default |
| --- | --- | --- |
| `--redis-url` | `REDIS_URL` | `redis://127.0.0.1:6379` |

