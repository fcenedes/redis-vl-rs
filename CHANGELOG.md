# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Core library: schema, filter, query, index, vectorizers, and extensions
  modules with sync and async support.
- CLI (`rvl`): `version` (with `--short`), `index create/delete/destroy/info/listall`,
  and `stats` commands with `--index`/`--schema` alternatives and `REDIS_URL`
  env support.
- Extensions: `EmbeddingsCache`, `SemanticCache`, `MessageHistory`,
  `SemanticMessageHistory`, and `SemanticRouter`.
- Vectorizers: `OpenAITextVectorizer`, `LiteLLMTextVectorizer`,
  `CustomTextVectorizer`, `AzureOpenAITextVectorizer`, `CohereTextVectorizer`,
  `VoyageAITextVectorizer`, `MistralAITextVectorizer`,
  `AnthropicTextVectorizer` (Voyage AI-backed adapter), and
  `HuggingFaceTextVectorizer` (local ONNX via `fastembed`, behind `hf-local`).
- Rerankers: `CohereReranker` behind the `rerankers` feature flag.
- SQL queries (`SQLQuery` behind `sql` feature): non-aggregate SELECT,
  aggregate functions (`COUNT`, `SUM`, `AVG`, `GROUP BY`, etc.),
  vector search functions (`vector_distance()`, `cosine_distance()`),
  and geo functions (`geo_distance()` in WHERE and SELECT clauses).
  Auto-dispatch between `FT.SEARCH` and `FT.AGGREGATE`.
- Hybrid/aggregate/multi-vector query command builders (`HybridQuery`,
  `AggregateHybridQuery`, `MultiVectorQuery`) for Redis 8.4+.
- Multi-prefix index support.
- `with_default_vectorizer()` convenience methods on `SemanticCache`,
  `SemanticMessageHistory`, and `SemanticRouter` (behind `hf-local`).
- Integration test suite mirroring upstream Python test behaviors.
- Criterion benchmark harness for schema, filter, query, and Redis-backed
  operations (index lifecycle, search, cache, history).
- CI, security, docs, and release workflows.
- `deny.toml` for cargo-deny license and advisory checks.
- Publication-quality README with accurate API examples.
- mdBook documentation with getting-started, schema, queries, vectorizers,
  extensions, and CLI guides.

### Not yet implemented

- Vertex AI and Bedrock vectorizer providers (source exists, not yet wired).
- SQL date functions (`YEAR()`), `IS NULL`/`IS NOT NULL`, `HAVING`.
- CLI `load` and query commands.
