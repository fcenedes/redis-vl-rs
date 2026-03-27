# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Core library: schema, filter, query, index, vectorizers, and extensions
  modules with sync and async support.
- CLI (`rvl`): `version`, `index create/delete/destroy/info/listall`, and
  `stats` commands.
- Extensions: `EmbeddingsCache`, `SemanticCache`, `MessageHistory`,
  `SemanticMessageHistory`, and `SemanticRouter`.
- Vectorizers: `OpenAITextVectorizer`, `LiteLLMTextVectorizer`, and
  `CustomTextVectorizer`.
- Integration test suite mirroring upstream Python test behaviors.
- Criterion benchmark harness for schema, filter, and query operations.
- CI, security, docs, and release workflows.
- `deny.toml` for cargo-deny license and advisory checks.
- Publication-quality README with accurate API examples.
- mdBook documentation with getting-started, schema, queries, extensions, and
  CLI guides.

### Not yet implemented

- SQL queries, rerankers, additional vectorizer providers.
- Full hybrid/aggregate query runtime parity.
- Multi-prefix index support.
- Implemented core schema, filter, query, index, vectorizer, cache, history,
  and router modules with sync and async support.
- Added publication-quality README with accurate API examples.
- Added MIT LICENSE file.
- Added mdBook documentation: getting started, schema, queries and filters,
  extensions, and CLI guides.
- Added `schema_basics` runnable example.
- Added Criterion benchmark harness (`core_benchmarks`) for schema parsing,
  filter rendering, and query building.
- Added CLI smoke tests for version, help, and subcommand validation.
- Added `deny.toml` for `cargo deny` security workflow.
- Improved crate metadata for crates.io readiness (keywords, categories,
  readme references).

