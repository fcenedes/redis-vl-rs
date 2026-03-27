# RedisVL Rust Implementation Plan

This file is the working source of truth for the RedisVL Rust implementation.It captures the target architecture, parity roadmap, and current status of therepository.

## Current Status

Last updated: `2026-03-27`

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
- Milestone 3: substantially completeOpenAI/LiteLLM/Custom vectorizers plus Azure OpenAI, Cohere, VoyageAI, andMistral vectorizers are implemented; cache/history/router extensions exist;criterion micro-benchmarks exist; first publishable release criteria areapproaching but not yet fully met
- Milestone 4: substantially complete`SQLQuery` is implemented behind `sql` with both non-aggregate SELECT and aggregate (`COUNT`, `SUM`, `AVG`, `GROUP BY`, etc.) support; automatic `FT.SEARCH`/`FT.AGGREGATE` dispatch helpers exist in both sync and async index; `CohereReranker` is implemented behind `rerankers`; Azure OpenAI, Cohere, VoyageAI, and Mistral vectorizers are done; Redis 8.4 integration tests for hybrid/aggregate/multi-vector queries are implemented and environment-gated; remaining: Vertex AI, Bedrock, HF local, SVS-VAMANA helpers
- Milestone 5: not started

Implemented surface snapshot:

- Schema:YAML/JSON parsing, field validation, hash/json storage selection, stopwords,vector attrs, multi-prefix support, and key-separator normalization
- Filters:`Tag`, `Text`, `Num`, `Geo`, `GeoRadius`, `Timestamp`, boolean composition,and Redis syntax rendering
- Queries:`VectorQuery`, `VectorRangeQuery`, `TextQuery`, `FilterQuery`, `CountQuery`,`HybridQuery` (generates `FT.HYBRID` for Redis 8.4+),`AggregateHybridQuery` (generates `FT.AGGREGATE`),`MultiVectorQuery` (generates multi-vector aggregate commands),`SQLQuery` (SQL→Redis Search translation behind `sql` feature, including aggregate SQL with `COUNT`/`SUM`/`AVG`/`GROUP BY` via `FT.AGGREGATE` dispatch)
- Search index:sync + async `create`, `delete`, `drop`, `exists`, `listall`, `info`, `load`,`fetch`, `search`, `query`, `batch_search`, `batch_query`, `paginate`,`hybrid_search`, `hybrid_query`, `aggregate_query`, `multi_vector_query`,`from_existing`, `drop_*`, `expire_*`, and `clear`
- Vectorizers:`Vectorizer`, `AsyncVectorizer`, `OpenAITextVectorizer`,`LiteLLMTextVectorizer`, `CustomTextVectorizer`,`AzureOpenAITextVectorizer`, `CohereTextVectorizer`,`VoyageAITextVectorizer`, and `MistralAITextVectorizer`
- Rerankers:`Reranker`, `AsyncReranker` traits, and `CohereReranker` (behind`rerankers` feature)
- Extensions:`EmbeddingsCache`, `SemanticCache`, `MessageHistory`,`SemanticMessageHistory`, and `SemanticRouter` are all Redis-backed
- CLI:`rvl version`, `index create/delete/destroy/info/listall`, and `stats`
- Benchmarks:criterion micro-benchmarks for schema parsing, filter rendering, and querybuilding; Redis-backed benchmarks for search index (create/exists/info/load/fetch), search (vector/filter/count/batch/paginate), embeddings cache, semantic cache, message history, and semantic message history

The repository does **not** yet have full parity with Python RedisVL.

The biggest remaining gaps are:

- Vertex AI, Bedrock, and HuggingFace local vectorizer providers (deferred)
- Anthropic vectorizer adapter (deferred)
- richer CLI parity and CLI tests
- Rust-vs-Python comparison benchmark runs
- provider-dependent dtype/default-vectorizer/from-existing parity for semantic extensions (deferred pending provider work)
- vector/geo aggregate SQL functions (deferred)
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

- `cargo fmt --all`
- `cargo check --workspace`
- `cargo test --workspace`

Current passing Rust test inventory:

- 217 unit tests in `crates/redis-vl/src/*`
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
- 5 CLI smoke tests
- 3 doctests

Integration test execution notes:

- Redis-backed parity tests are environment-gated
- set `REDISVL_RUN_INTEGRATION=1` to run them
- `REDIS_URL` defaults to `redis://127.0.0.1:6379`

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
- `SQLQuery` covers both non-aggregate SELECT queries and aggregate SQL(`COUNT`, `SUM`, `AVG`, `GROUP BY`, etc.) with automatic `FT.SEARCH`/`FT.AGGREGATE`dispatch in `SearchIndex::sql_query` and `AsyncSearchIndex::sql_query`;vector/geo aggregate functions are not yet supported
- `SemanticMessageHistory` exists, but Python features around dtype/defaultvectorizers/reconnection/from-existing are still missing
- `SemanticRouter` exists, but Python features around serialization helpers,route reference management, dtype/default vectorizers, and from-existing stylebehavior are still missing
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
- `anthropic` is a later additive adapter, not a parity blocker

## Coverage Matrix

This table is the compact handoff view. `PARITY_MATRIX.md` should stay alignedwith it.

| Area | Current status | Notes | Next high-value gap |
| --- | --- | --- | --- |
| Schema | **Complete** | YAML/JSON parsing, field attrs, stopwords, hash/json storage, multi-prefix support, and key-separator normalization implemented with full unit and integration coverage | — |
| Search Index | **Complete** | Sync + async lifecycle/load/fetch/search/query/batch/paginate, hybrid_search/hybrid_query, aggregate_query, multi_vector_query, sql_query (with auto-dispatch), and from_existing exist; Redis 8.4 integration tests are environment-gated | — |
| Storage | **Complete** | Hash + JSON loading/fetching are implemented with multi-prefix index support | — |
| Filters | **Complete** | Core DSL, boolean composition, and Redis syntax rendering implemented with comprehensive unit and integration coverage | — |
| Queries | **Complete** | Vector/Range/Text/Filter/Count implemented; HybridQuery/AggregateHybridQuery/MultiVectorQuery have full command builders with Redis 8.4 integration tests | — |
| SQL | **Complete** | SQLQuery behind sql feature: non-aggregate SELECT with WHERE/ORDER BY/LIMIT/OFFSET, tag/numeric/text/date comparisons, AND/OR, IN/NOT IN, LIKE/NOT LIKE, BETWEEN, field projection; aggregate SQL (COUNT, SUM, AVG, GROUP BY, etc.) via FT.AGGREGATE with auto-dispatch helpers in SearchIndex/AsyncSearchIndex. Vector/geo aggregate functions are deferred | Vector/geo aggregate functions (deferred) |
| Vectorizers | In progress | OpenAI, LiteLLM, Custom, Azure OpenAI, Cohere, VoyageAI, and Mistral implemented | Vertex AI, Bedrock, HF local (deferred) |
| Semantic cache | **Complete** | Redis-backed sync/async core implemented and parity-tested (19 integration tests) | — |
| Embeddings cache | **Complete** | Redis-backed sync/async core implemented and parity-tested (12 integration tests) | — |
| Message history | **Complete** | Standard history implemented with role filtering and parity tests (4 history + 3 role-filter tests) | — |
| Semantic message history | **Complete** | Semantic history implemented and parity-tested (7 integration tests). Provider-dependent dtype/default-vectorizer/from-existing parity is deferred | Provider-dependent defaults (deferred) |
| Router | **Complete** | Redis-backed routing/update/lifecycle core implemented and parity-tested (20 integration tests). Provider-dependent dtype/default-vectorizer/from-existing parity and serialization helpers are deferred | Provider-dependent defaults (deferred) |
| CLI | In progress | Basic command surface implemented (5 CLI smoke tests) | More Python CLI parity, tests, and UX polishing |
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

- Status: substantially complete
- `Vectorizer` trait, OpenAI-compatible transport, `OpenAITextVectorizer`,`LiteLLMTextVectorizer`, `CustomVectorizer`, `AzureOpenAITextVectorizer`,`CohereTextVectorizer`, `VoyageAITextVectorizer`, and`MistralAITextVectorizer` are implemented
- `SemanticCache`, `EmbeddingsCache`, `MessageHistory`,`SemanticMessageHistory`, and `SemanticRouter` are implemented
- Criterion micro-benchmarks exist
- Publish `0.x` only when:
  - all Phase 1 rows are green
  - all public APIs have rustdoc
  - examples compile
  - benchmark harness exists

### Milestone 4: Parity completion

- Status: substantially complete
- `SQLQuery` is implemented behind `sql` with both non-aggregate SELECT and aggregate SQL (`COUNT`, `SUM`, `AVG`, `GROUP BY`, etc.) support; automatic `FT.SEARCH`/`FT.AGGREGATE` dispatch exists in sync and async index
- `CohereReranker` is implemented behind `rerankers`
- Azure OpenAI, Cohere, VoyageAI, and Mistral vectorizers are implemented
- Redis 8.4 integration tests for hybrid/aggregate/multi-vector queries are implemented
- Remaining: Vertex AI, Bedrock, HF local, SVS-VAMANA helpers, vector/geo aggregate SQL functions
- Cut `1.0.0` only after the parity matrix has no remaining Python-surface gaps

### Milestone 5: Post-parity additions

- Status: not started
- Add Anthropic and any other OpenAI-compatible or ecosystem-driven adapters
- Add Rust-native ergonomics only after parity is complete and documented

## Recommended Next Work Order

If a new agent/session takes over, the recommended order is:

1. Remaining vectorizer providersAdd Vertex AI, Bedrock, and HuggingFace local vectorizers
2. Search-index/key-construction edge casesContinue from `tests/integration/test_key_separator_handling.py` and relatedschema/index tests
3. CLI parity and CLI testsExpand `crates/rvl/src/main.rs` and add CLI-focused tests mirroring Python CLIexpectations where possible
4. Semantic extension parityAdd dtype/default-vectorizer/from-existing/reconnection support to`SemanticMessageHistory` and `SemanticRouter`
5. Vector/geo aggregate SQL functionsExtend aggregate SQL support in `crates/redis-vl/src/query/sql.rs` to covervector distance and geo functions
6. Rust-vs-Python benchmark comparison runner
7. Examples and docs expansion

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