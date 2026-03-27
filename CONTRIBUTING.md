# Contributing

## Local Checks

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

## Principles

- Keep public APIs documented.
- Prefer borrowed inputs over eager clones.
- Treat parity gaps as tracked work, not hidden behavior differences.

