# RedisVL Rust Implementation Plan

This file is the working source of truth for the RedisVL Rust implementation.It captures the target architecture, parity roadmap, and current status of therepository.

## Current Status

Last updated: `2026-04-27`

The repository is past scaffolding and into real parity work.

What is implemented today:

- `crates/redis-vl`: buildable public library crate with `#![deny(missing_docs)]`
- `crates/rvl`: buildable CLI crate
- CI/docs/release workflow scaffolding
- `PARITY_MATRIX.md` as the feature tracker
- core schema, filter, query, index, vectorizer, cache, history, and routermodules
- a growing Rust integration suite derived from upstream Python tests

Milestone status:

- Milestone 0: partial`PARITY_MATRIX.md` exists and the Python tests are treated as the contract,but the matrix is still feature-granular rather than test-file-granular
- Milestone 1: mostly completeworkspace, workflows, crate metadata, docs skeleton, and release scaffoldingare in place
- Milestone 2: **complete**schema/filter/query/index foundations are real and Redis-backed with comprehensive parity test coverage; hybrid, aggregate, and multi-vector query command builders are implemented; multi-prefix schema/index support exists with parity tests; `from_existing` is implemented for both sync and async indexes
- Milestone 3: **complete**OpenAI/LiteLLM/Custom vectorizers plus Azure OpenAI, Cohere, VoyageAI, Mistral, Vertex AI, Bedrock, Anthropic (Voyage AI-backed), and HuggingFace local (via `fastembed`) vectorizers are implemented; cache/history/router extensions exist with `with_default_vectorizer()` convenience methods (behind `hf-local`); criterion micro-benchmarks exist
- Milestone 4: **substantially complete**`SQLQuery` is implemented behind `sql` with non-aggregate SELECT, aggregate (`COUNT`, `SUM`, `AVG`, `GROUP BY`, etc.), vector search functions (`vector_distance()`, `cosine_distance()` → KNN), and geo functions (`geo_distance()` → `GEOFILTER` / `FT.AGGREGATE APPLY geodistance`); automatic `FT.SEARCH`/`FT.AGGREGATE` dispatch helpers exist in both sync and async index; `CohereReranker` is implemented behind `rerankers`; Redis 8.4 integration tests for hybrid/aggregate/multi-vector queries are implemented and environment-gated; Vertex AI and Bedrock adapters are wired behind feature flags; remaining: SVS-VAMANA helpers and SQL date/null/HAVING gaps
- Milestone 5: not started

Implemented surface snapshot:

- Schema:YAML/JSON parsing, field validation, hash/json storage selection, stopwords,vector attrs, multi-prefix support, and key-separator normalization
- Filters:`Tag`, `Text`, `Num`, `Geo`, `GeoRadius`, `Timestamp`, boolean composition,and Redis syntax rendering
- Queries:`VectorQuery`, `VectorRangeQuery`, `TextQuery`, `FilterQuery`, `CountQuery`,`HybridQuery` (generates `FT.HYBRID` for Redis 8.4+),`AggregateHybridQuery` (generates `FT.AGGREGATE`),`MultiVectorQuery` (generates multi-vector aggregate commands),`SQLQuery` (SQL→Redis Search translation behind `sql` feature, including aggregate SQL with `COUNT`/`SUM`/`AVG`/`GROUP BY` via `FT.AGGREGATE` dispatch, vector search functions `vector_distance()`/`cosine_distance()` → KNN, and geo functions `geo_distance()` → `GEOFILTER`/`FT.AGGREGATE APPLY geodistance`)
- Search index:sync + async `create`, `delete`, `drop`, `exists`, `listall`, `info`, `load`,`fetch`, `search`, `query`, `batch_search`, `batch_query`, `paginate`,`hybrid_search`, `hybrid_query`, `aggregate_query`, `multi_vector_query`,`from_existing`, `drop_*`, `expire_*`, and `clear`
- Vectorizers:`Vectorizer`, `AsyncVectorizer`, `OpenAITextVectorizer`,`LiteLLMTextVectorizer`, `CustomTextVectorizer`,`AzureOpenAITextVectorizer`, `CohereTextVectorizer`,`VoyageAITextVectorizer`, `MistralAITextVectorizer`,`VertexAITextVectorizer`, `BedrockTextVectorizer`,`AnthropicTextVectorizer` (Voyage AI-backed adapter), and`HuggingFaceTextVectorizer` (local ONNX via `fastembed`, behind `hf-local`)
- Rerankers:`Reranker`, `AsyncReranker` traits, and `CohereReranker` (behind`rerankers` feature)
- Extensions:`EmbeddingsCache`, `SemanticCache`, `MessageHistory`,`SemanticMessageHistory`, and `SemanticRouter` are all Redis-backed; semantic extensions provide `with_default_vectorizer()` convenience methods behind the `hf-local` feature for zero-config local embedding
- CLI:`rvl version`, `index create/delete/destroy/info/listall`, and `stats`
- Benchmarks:criterion micro-benchmarks for schema parsing, filter rendering, and querybuilding; Redis-backed benchmarks for search index (create/exists/info/load/fetch), search (vector/filter/count/batch/paginate), embeddings cache, semantic cache, message history, and semantic message history

The repository does **not** yet have full parity with Python RedisVL.

The biggest remaining gaps are:

- richer CLI parity (load, query commands) and CLI tests
- Rust-vs-Python comparison benchmark runs
- SQL date functions (`YEAR()`), `IS NULL`/`IS NOT NULL`, `HAVING` clause
- examples and docs expansion

## Validation Source of Truth

The upstream Python test suite is the parity contract for this repository.

- Primary reference: `redis/redis-vl-python/tests`
- Rust tests should be added by mirroring behavior from the matching Python testmodule whenever practical
- If docs and tests disagree, prefer upstream test behavior
- Parity progress should be measured against Python test coverage first, thenagainst README and API docs examples

Useful local reference during this work:

- The upstream repo has recently been mirrored locally at`/tmp/redis-vl-python`
- If that directory is missing in a future session, re-establish a local copybefore continuing parity work

## Verification Snapshot

Latest verified commands:

- `cargo fmt --all --check`
- `cargo check --workspace --all-features`
- `cargo test --workspace --all-features`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps`
- `mdbook build docs`
- `cargo audit`
- `cargo deny check`
- `cargo package -p redis-vl --allow-dirty`
- `cargo test -p redis-vl --features vertex-ai`
- `cargo test -p redis-vl --features bedrock`

Security and release readiness notes:

- `cargo audit` has no vulnerability findings as of 2026-04-27. It reports one allowed unmaintained warning from optional `hf-local` dependencies: `paste` via `fastembed`/`tokenizers`.
- `cargo deny check` passes. Duplicate-version findings are warnings only.
- `cargo package -p redis-vl` verifies successfully from the generated crate tarball.
- `cargo package -p rvl` should be run only after `redis-vl` is published and visible in the crates.io index, because Cargo strips the local path dependency and resolves `redis-vl = "0.1.0"` from crates.io.
- `PUBLISHING.md` documents the release process and required GitHub secret (`CARGO_REGISTRY_TOKEN`).
- GitHub automation now includes CI, docs deployment, security checks, release PRs, manual crates.io publishing, cross-platform `rvl` release artifacts, and Dependabot.

Current passing Rust test inventory:

- 225 default unit tests in `crates/redis-vl/src/*`
- 231 `redis-vl` tests with `vertex-ai` enabled
- 230 `redis-vl` tests with `bedrock` enabled
- 16 sync search-index/query parity integration tests
- 6 async search-index/query parity integration tests
- 11 multi-prefix parity integration tests
- 16 hybrid/aggregate/multi-vector parity integration tests (Redis 8.4, environment-gated)
- 12 embeddings-cache parity integration tests
- 19 semantic-cache parity integration tests
- 4 standard message-history parity integration tests
- 3 role-filter parity integration tests
- 7 semantic-message-history parity integration tests
- 20 semantic-router parity integration tests
- 21 CLI smoke tests
- 11 default doctests

Integration test execution notes:

- Redis-backed parity tests are environment-gated
- set `REDISVL_RUN_INTEGRATION=1` to run them
- `REDIS_URL` defaults to `redis://127.0.0.1:6379`

Verification caveat:

- `cargo clippy --workspace --all-targets --all-features` is green, but `cargo clippy --workspace --all-targets --all-features -- -D warnings` is not currently green. It fails on broad pre-existing lint debt from strict workspace lints (`missing_errors_doc`, `doc_markdown`, `must_use_candidate`, `multiple_crate_versions`, and similar pedantic/cargo lints), not on provider wiring or parity test failures. CI runs strict clippy as a non-blocking debt tracker until this is paid down.

## Repo Map For Handoff

Primary implementation files:

- `crates/redis-vl/src/schema.rs`
- `crates/redis-vl/src/filter.rs`
- `crates/redis-vl/src/query.rs`
- `crates/redis-vl/src/query/sql.rs`
- `crates/redis-vl/src/index.rs`
- `crates/redis-vl/src/vectorizers/mod.rs`
- `crates/redis-vl/src/vectorizers/azure_openai.rs`
- `crates/redis-vl/src/vectorizers/cohere.rs`
- `crates/redis-vl/src/vectorizers/voyageai.rs`
- `crates/redis-vl/src/vectorizers/mistral.rs`
- `crates/redis-vl/src/vectorizers/anthropic.rs`
- `crates/redis-vl/src/vectorizers/hf_local.rs`
- `crates/redis-vl/src/vectorizers/vertex_ai.rs`
- `crates/redis-vl/src/vectorizers/bedrock.rs`
- `crates/redis-vl/src/rerankers/mod.rs`
- `crates/redis-vl/src/rerankers/cohere.rs`
- `crates/redis-vl/src/extensions/cache.rs`
- `crates/redis-vl/src/extensions/history.rs`
- `crates/redis-vl/src/extensions/router.rs`
- `crates/rvl/src/main.rs`

Primary parity test files:

- `crates/redis-vl/tests/python_parity_search_index.rs`
- `crates/redis-vl/tests/python_parity_async_search_index.rs`
- `crates/redis-vl/tests/python_parity_multi_prefix.rs`
- `crates/redis-vl/tests/python_parity_embedcache.rs`
- `crates/redis-vl/tests/python_parity_semantic_cache.rs`
- `crates/redis-vl/tests/python_parity_message_history.rs`
- `crates/redis-vl/tests/python_parity_message_history_roles.rs`
- `crates/redis-vl/tests/python_parity_semantic_message_history.rs`
- `crates/redis-vl/tests/python_parity_semantic_router.rs`
- `crates/redis-vl/tests/python_parity_hybrid_aggregate.rs`

Upstream Python test files already partially mirrored:

- `tests/unit/test_filter.py`
- `tests/unit/test_query_types.py`
- `tests/unit/test_schema.py`
- `tests/integration/test_search_index.py`
- `tests/integration/test_search_results.py`
- `tests/integration/test_query.py`
- `tests/integration/test_multi_prefix.py`
- `tests/integration/test_embedcache.py`
- `tests/integration/test_llmcache.py`
- `tests/integration/test_message_history.py`
- `tests/integration/test_role_filter_get_recent.py`
- `tests/integration/test_semantic_router.py`

Important caution for a takeover session:

- `HybridQuery`, `AggregateHybridQuery`, and `MultiVectorQuery` command buildersare implemented and covered by environment-gated Redis 8.4 integration tests in`python_parity_hybrid_aggregate.rs`; these require a Redis 8.4+ server with`FT.HYBRID` and `FT.AGGREGATE` support
- `SQLQuery` covers non-aggregate SELECT, aggregate SQL (`COUNT`, `SUM`, `AVG`, `GROUP BY`, etc.), vector search functions (`vector_distance()`/`cosine_distance()` → KNN), and geo functions (`geo_distance()` in WHERE → `GEOFILTER`, in SELECT → `FT.AGGREGATE APPLY geodistance`) with automatic `FT.SEARCH`/`FT.AGGREGATE` dispatch in `SearchIndex::sql_query` and `AsyncSearchIndex::sql_query`; remaining SQL gaps: date functions (`YEAR()`), `IS NULL`/`IS NOT NULL`, `HAVING`; the SQL parser uses a hand-rolled tokenizer — adopting `sqlparser-rs` is deferred as the current approach provides better Redis-specific translation control
- `AnthropicTextVectorizer` is a Voyage AI-backed adapter (Anthropic recommends Voyage AI for embeddings rather than providing a native embedding model); it wraps `VoyageAITextVectorizer` with Anthropic-oriented defaults and uses `VOYAGE_API_KEY`
- `HuggingFaceTextVectorizer` (behind `hf-local`) runs models locally via `fastembed` / ONNX Runtime; implements `Vectorizer` only (sync); for async use, wrap with `tokio::task::spawn_blocking`
- `SemanticMessageHistory` has full dtype support, overwrite/reconnect with schema mismatch detection, parity tests, and `with_default_vectorizer()` behind `hf-local`
- `SemanticRouter` has full dtype support, overwrite/reconnect with schema mismatch detection, `from_existing`, `from_dict`, `from_yaml`, `to_dict`/`to_json_value` serialization, route reference management with parity tests, and `with_default_vectorizer()` behind `hf-local`
- `SemanticCache` provides `with_default_vectorizer()` behind `hf-local` for zero-config local embedding
- Vertex AI (`vertex_ai.rs`) and Bedrock (`bedrock.rs`) adapters are registered in `vectorizers/mod.rs`, re-exported from `lib.rs`, and feature-gated. Bedrock pulls optional `aws-config` and `aws-sdk-bedrockruntime` dependencies through the `bedrock` feature
- Multi-prefix schema parsing and index creation are implemented; key compositionfor multi-prefix loading uses the first prefix

## Summary

- Build a greenfield Rust workspace that targets feature parity with the Python`redis-vl` library, while using an idiomatic Rust API rather than cloningPython call signatures.
- Use a workspace split:
  - `crates/redis-vl`: public library crate
  - `crates/rvl`: CLI crate
- Make the library async-first, but implement a shared command/compiler layer soblocking wrappers can reuse the same schema/query/cache/router logic withoutduplicating behavior.
- Treat core search/indexing plus AI extensions plus CLI as the firstpublishable milestone; defer SQL, rerankers, and non-required providers intoexplicit parity-completion waves.
- Use upstream source and API docs as the authority when docs and README diverge.

## Architecture

### Public modules

- `redis_vl::schema`: `IndexSchema`, field definitions, field attrs, stopwords,Hash vs JSON storage, YAML loading, SVS-VAMANA attrs
- `redis_vl::index`: `SearchIndex`, `AsyncSearchIndex`, connection config,CRUD/load/fetch/list/info lifecycle
- `redis_vl::query`: `VectorQuery`, `VectorRangeQuery`, `TextQuery`,`FilterQuery`, `CountQuery`, `AggregateHybridQuery`, `HybridQuery`,`MultiVectorQuery`, `SQLQuery`, `Vector`
- `redis_vl::filter`: `FilterExpression`, `Tag`, `Text`, `Num`, `Geo`,`GeoRadius`, `Timestamp`
- `redis_vl::vectorizers`: `Vectorizer` trait plus provider adapters
- `redis_vl::rerankers`: provider rerankers
- `redis_vl::extensions::{cache, history, router}`
- `redis_vl::error`: crate-specific error hierarchy

### Internal layering

- `command`: pure compilation of schemas/queries/filters into Redis Search andRedisJSON commands
- `transport`: sync and async Redis execution traits and response decoding
- `http`: provider auth/request/response adapters for OpenAI-compatible andlater providers
- `model`: shared typed structs for cache entries, routes, messages, results,and stats

### Rust defaults

- Prefer borrowed inputs where practical: `&str`, `&[T]`, `Cow<'a, str>`,`Cow<'a, [f32]>`
- Use zero-copy vector encoding where possible for Redis query params and bulkload paths
- Keep `#![deny(missing_docs)]` enabled on the library crate
- Use builders for complex queries/configuration
- Return `Result<T, Error>` from fallible public APIs
- Avoid panics in public APIs

### Packaging and features

- Core crate always includes schema/index/query/filter
- Feature flags:
  - `openai`
  - `litellm`
  - `azure-openai`
  - `cohere`
  - `vertex-ai`
  - `anthropic`
  - `voyageai`
  - `mistral`
  - `bedrock`
  - `hf-local`
  - `sql`
  - `rerankers`
- `litellm` is implemented as a thin OpenAI-compatible transport/config layer
- `anthropic` depends on `voyageai` (Anthropic recommends Voyage AI for embeddings)
- `hf-local` depends on `dep:fastembed` for local ONNX Runtime inference
- `vertex-ai` and `bedrock` feature flags are wired. `bedrock` enables optional `aws-sdk-bedrockruntime` and `aws-config` dependencies

## Coverage Matrix

This table is the compact handoff view. `PARITY_MATRIX.md` should stay alignedwith it.

| Area | Current status | Notes | Next high-value gap |
| --- | --- | --- | --- |
| Schema | **Complete** | YAML/JSON parsing, field attrs, stopwords, hash/json storage, multi-prefix support, and key-separator normalization implemented with full unit and integration coverage | — |
| Search Index | **Complete** | Sync + async lifecycle/load/fetch/search/query/batch/paginate, hybrid_search/hybrid_query, aggregate_query, multi_vector_query, sql_query (with auto-dispatch), and from_existing exist; Redis 8.4 integration tests are environment-gated | — |
| Storage | **Complete** | Hash + JSON loading/fetching are implemented with multi-prefix index support | — |
| Filters | **Complete** | Core DSL, boolean composition, and Redis syntax rendering implemented with comprehensive unit and integration coverage | — |
| Queries | **Complete** | Vector/Range/Text/Filter/Count implemented; HybridQuery/AggregateHybridQuery/MultiVectorQuery have full command builders with Redis 8.4 integration tests | — |
| SQL | **Complete** | SQLQuery behind sql feature: non-aggregate SELECT with WHERE/ORDER BY/LIMIT/OFFSET, tag/numeric/text/date comparisons, AND/OR, IN/NOT IN, LIKE/NOT LIKE, BETWEEN, field projection; aggregate SQL (COUNT, SUM, AVG, GROUP BY, etc.) via FT.AGGREGATE; vector search functions (`vector_distance()`/`cosine_distance()` → KNN); geo functions (`geo_distance()` in WHERE → `GEOFILTER`, in SELECT → `FT.AGGREGATE APPLY geodistance`); auto-dispatch helpers in SearchIndex/AsyncSearchIndex. SQL parser uses a hand-rolled tokenizer (recommended over `sqlparser-rs` for Redis-specific control) | Date functions, `IS NULL`/`IS NOT NULL`, `HAVING` |
| Vectorizers | **Complete** | OpenAI, LiteLLM, Custom, Azure OpenAI, Cohere, VoyageAI, Mistral, Vertex AI, Bedrock, Anthropic (Voyage AI-backed adapter), and HuggingFace local (via `fastembed` / ONNX, behind `hf-local`) are implemented and wired behind feature flags. Live provider calls are not run in CI | Add any new upstream provider adapters as Python adds them |
| Semantic cache | **Complete** | Redis-backed sync/async core implemented and parity-tested (19 integration tests); `with_default_vectorizer()` available behind `hf-local` | — |
| Embeddings cache | **Complete** | Redis-backed sync/async core implemented and parity-tested (12 integration tests) | — |
| Message history | **Complete** | Standard history implemented with role filtering and parity tests (4 history + 3 role-filter tests) | — |
| Semantic message history | **Complete** | Semantic history implemented and parity-tested (7 integration tests); dtype selection, overwrite control, reconnect with schema mismatch detection, and `with_default_vectorizer()` (behind `hf-local`) are implemented | — |
| Router | **Complete** | Redis-backed routing/update/lifecycle core implemented and parity-tested (20 integration tests); dtype selection, overwrite control, reconnect with schema mismatch detection, `from_existing`, `from_dict`/`from_yaml` serialization, route reference management, and `with_default_vectorizer()` (behind `hf-local`) are implemented | — |
| CLI | In progress | `version` (with `--short`), `index create/delete/destroy/info/listall`, `stats` (Python-matching keys), `--index`/`--schema` alternatives, `--redis-url`/`REDIS_URL` (21 CLI smoke tests) | `load`, query commands, richer output formatting |
| Rerankers | **Complete** | Reranker/AsyncReranker traits and CohereReranker behind rerankers feature | Additional reranker providers |
| Benchmarks | In progress | Criterion micro-benchmarks for schema/filter/query; Redis-backed benchmarks for search index ops, vector/filter/count/batch/paginate search, embeddings cache, semantic cache, message history, and semantic history | Rust-vs-Python comparison runner |
| Docs/examples | In progress | README, mdBook guide, and repo scaffolding exist | Expand rustdoc, examples, guides, docs deployment quality |

## Roadmap

### Milestone 0: Inventory and parity contract

- Status: partial
- Maintain `PARITY_MATRIX.md`
- Convert upstream examples into Rust examples, integration tests, or tracked gaps
- Record upstream doc/source discrepancies explicitly

### Milestone 1: Workspace and release scaffolding

- Status: mostly complete
- Create workspace with `redis-vl` and `rvl`
- Add `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`,`examples/`, `benches/`, `docs/`
- Set up CI, docs, release automation, crate metadata, docs.rs metadata, andfeature tracking
- Use `release-plz` for automated versioning and publishing

### Milestone 2: Core RedisVL parity

- Status: **complete**
- Schema, field types, validation, stopwords, Hash/JSON storage, multi-prefix,and vector field configuration are implemented with full parity test coverage
- Index lifecycle, connection config, load/fetch/delete/clear/drop,expire/list/info/batch/paginate, hybrid_search, aggregate_query,multi_vector_query, and from_existing are implemented with parity tests
- Filter DSL and all query builders are implemented, including hybrid, aggregatehybrid, and multi-vector modes
- Redis 8.4 integration tests for hybrid/aggregate/multi-vector queries are implemented and environment-gated

### Milestone 3: First publishable Rust release

- Status: **complete**
- `Vectorizer` trait, OpenAI-compatible transport, `OpenAITextVectorizer`,`LiteLLMTextVectorizer`, `CustomVectorizer`, `AzureOpenAITextVectorizer`,`CohereTextVectorizer`, `VoyageAITextVectorizer`, `MistralAITextVectorizer`,`VertexAITextVectorizer`, `BedrockTextVectorizer`,`AnthropicTextVectorizer` (Voyage AI-backed), and `HuggingFaceTextVectorizer`(local ONNX via `fastembed`) are implemented
- `SemanticCache`, `EmbeddingsCache`, `MessageHistory`,`SemanticMessageHistory`, and `SemanticRouter` are implemented with`with_default_vectorizer()` convenience methods behind `hf-local`
- Criterion micro-benchmarks exist
- Publish `0.x` only when:
  - all Phase 1 rows are green
  - all public APIs have rustdoc
  - examples compile
  - benchmark harness exists

### Milestone 4: Parity completion

- Status: **substantially complete**
- `SQLQuery` is implemented behind `sql` with non-aggregate SELECT, aggregate SQL (`COUNT`, `SUM`, `AVG`, `GROUP BY`, etc.), vector search functions (`vector_distance()`/`cosine_distance()` → KNN), and geo functions (`geo_distance()` → `GEOFILTER`/`FT.AGGREGATE APPLY geodistance`); automatic `FT.SEARCH`/`FT.AGGREGATE` dispatch exists in sync and async index; SQL parser uses a hand-rolled tokenizer (recommended over `sqlparser-rs` for Redis-specific translation control and small dependency footprint)
- `CohereReranker` is implemented behind `rerankers`
- All actively-wired vectorizer providers are done (see Milestone 3)
- Redis 8.4 integration tests for hybrid/aggregate/multi-vector queries are implemented
- Remaining: SVS-VAMANA helpers and SQL date/null/HAVING gaps
- Cut `1.0.0` only after the parity matrix has no remaining Python-surface gaps

### Milestone 5: Post-parity additions

- Status: not started
- Add Rust-native ergonomics only after parity is complete and documented

## Recommended Next Work Order

If a new agent/session takes over, the recommended order is:

1. CLI parity and CLI testsExpand `crates/rvl/src/main.rs` with `load` and query commands; add CLI-focused tests mirroring Python CLI expectations where possible
2. SQL date functions and remaining SQL gapsAdd `YEAR()`, `IS NULL`/`IS NOT NULL`, `HAVING` to the hand-rolled SQL parser
3. SVS-VAMANA helper and tuning utilities
4. Rust-vs-Python benchmark comparison runner
5. Examples and docs expansion

## Takeover Checklist

Before making the next major parity patch:

- read the relevant upstream Python test file first
- implement only the minimal Rust surface needed for that contract
- add or extend a Rust parity test file that names the upstream source in itsmodule docs
- run:`cargo fmt --all`
- run:`cargo check --workspace`
- run:`cargo test --workspace`
- if a live Redis instance is available, run:`REDISVL_RUN_INTEGRATION=1 cargo test --workspace`

## Test Plan

- Unit tests:
  - schema parsing/validation
  - field attrs
  - query/filter compilation
  - Redis command generation
  - result decoding
- Integration tests against Redis 8+/Redis Stack:
  - Hash and JSON storage
  - CRUD and batch loading
  - query correctness
  - hybrid search
  - multi-vector search
  - cache/history/router flows
- Contract tests:
  - upstream Python tests are the source of truth
  - mirror Python unit and integration cases into Rust tests incrementally
- Provider tests:
  - mocked HTTP request/response flows
- CLI tests:
  - help text, exit codes, env/flag precedence
- Docs/tests:
  - doctests and compiled examples
- Benchmarks:
  - Rust vs Python comparisons for load, query, cache hit/miss, router

## Dependencies and Tooling

- Core crates:
  - `redis`
  - `tokio`
  - `serde`
  - `serde_json`
  - `serde_yaml`
  - `bytes`
  - `thiserror`
  - `chrono`
  - `ulid`
  - `indexmap`
- HTTP/provider layer:
  - `reqwest`
  - `url`
  - `base64`
  - `aws-config` and `aws-sdk-bedrockruntime` behind `bedrock`
- CLI:
  - `clap`
  - `comfy-table`
  - `anstream`
- Testing:
  - `testcontainers`
  - `insta`
  - `proptest`
  - `wiremock`
  - `tokio-test`
- Benchmarks:
  - `criterion`
- Repo tooling:
  - `cargo fmt`
  - `clippy`
  - `cargo deny`
  - `cargo audit`
  - `release-plz`
  - `mdBook`
  - `cargo doc`

## GitHub Workflows

- `ci.yml`: fmt, clippy, tests, docs build
- `security.yml`: `cargo deny` and `cargo audit`
- `release-plz.yml`: version bump PRs, changelog, publish flow
- `docs.yml`: `mdBook` and `cargo doc`
- `bench.yml`: benchmark runs

## Implementation Notes

- This file should be updated whenever major parity milestones or architecturedecisions change.
- `PARITY_MATRIX.md` should stay more granular than this file; use this file forroadmap and repository conventions, and the parity matrix for feature-leveltracking.
