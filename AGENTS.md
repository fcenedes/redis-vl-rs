# RedisVL Rust Implementation Plan

This file is the working source of truth for the RedisVL Rust implementation.
It captures the target architecture, parity roadmap, and current status of the
repository.

## Current Status

Last updated: `2026-03-27`

The repository is past scaffolding and into real parity work.

What is implemented today:

- `crates/redis-vl`: buildable public library crate with `#![deny(missing_docs)]`
- `crates/rvl`: buildable CLI crate
- CI/docs/release workflow scaffolding
- `PARITY_MATRIX.md` as the feature tracker
- core schema, filter, query, index, vectorizer, cache, history, and router
  modules
- a growing Rust integration suite derived from upstream Python tests

Milestone status:

- Milestone 0: partial
  `PARITY_MATRIX.md` exists and the Python tests are treated as the contract,
  but the matrix is still feature-granular rather than test-file-granular
- Milestone 1: mostly complete
  workspace, workflows, crate metadata, docs skeleton, and release scaffolding
  are in place
- Milestone 2: partially complete
  schema/filter/query/index foundations are real and Redis-backed, but hybrid,
  aggregate, multi-prefix, and broader search/runtime parity are still open
- Milestone 3: partially complete
  OpenAI/LiteLLM/Custom vectorizers plus cache/history/router extensions exist,
  but the first publishable release criteria are not yet met
- Milestone 4: not started in substance
  SQL, rerankers, secondary providers, and broader advanced-query parity remain
  open
- Milestone 5: not started

Implemented surface snapshot:

- Schema:
  YAML/JSON parsing, field validation, hash/json storage selection, stopwords,
  vector attrs, and key-separator normalization
- Filters:
  `Tag`, `Text`, `Num`, `Geo`, `GeoRadius`, `Timestamp`, boolean composition,
  and Redis syntax rendering
- Queries:
  `VectorQuery`, `VectorRangeQuery`, `TextQuery`, `FilterQuery`, `CountQuery`,
  `HybridQuery`, `AggregateHybridQuery`, and `MultiVectorQuery` types exist
- Search index:
  sync + async `create`, `delete`, `drop`, `exists`, `listall`, `info`, `load`,
  `fetch`, `search`, `query`, `batch_search`, `batch_query`, `paginate`,
  `drop_*`, `expire_*`, and `clear`
- Vectorizers:
  `Vectorizer`, `AsyncVectorizer`, `OpenAITextVectorizer`,
  `LiteLLMTextVectorizer`, and `CustomTextVectorizer`
- Extensions:
  `EmbeddingsCache`, `SemanticCache`, `MessageHistory`,
  `SemanticMessageHistory`, and `SemanticRouter` are all Redis-backed
- CLI:
  `rvl version`, `index create/delete/destroy/info/listall`, and `stats`

The repository does **not** yet have full parity with Python RedisVL.

The biggest remaining gaps are:

- hybrid and aggregate query runtime parity
- multi-prefix index support
- SQL query support
- rerankers
- provider waves beyond OpenAI/LiteLLM/Custom
- richer CLI parity and CLI tests
- benchmark harness and Rust-vs-Python comparison runs
- broader upstream Python test mirroring for advanced integrations

## Validation Source of Truth

The upstream Python test suite is the parity contract for this repository.

- Primary reference: `redis/redis-vl-python/tests`
- Rust tests should be added by mirroring behavior from the matching Python test
  module whenever practical
- If docs and tests disagree, prefer upstream test behavior
- Parity progress should be measured against Python test coverage first, then
  against README and API docs examples

Useful local reference during this work:

- The upstream repo has recently been mirrored locally at
  `/tmp/redis-vl-python`
- If that directory is missing in a future session, re-establish a local copy
  before continuing parity work

## Verification Snapshot

Latest verified commands:

- `cargo fmt --all`
- `cargo check --workspace`
- `cargo test --workspace`

Current passing Rust test inventory:

- 38 unit tests in `crates/redis-vl/src/*`
- 16 sync search-index/query parity integration tests
- 6 async search-index/query parity integration tests
- 8 embeddings-cache parity integration tests
- 5 semantic-cache parity integration tests
- 4 standard message-history parity integration tests
- 3 role-filter parity integration tests
- 4 semantic-message-history parity integration tests
- 4 semantic-router parity integration tests

Integration test execution notes:

- Redis-backed parity tests are environment-gated
- set `REDISVL_RUN_INTEGRATION=1` to run them
- `REDIS_URL` defaults to `redis://127.0.0.1:6379`

## Repo Map For Handoff

Primary implementation files:

- `crates/redis-vl/src/schema.rs`
- `crates/redis-vl/src/filter.rs`
- `crates/redis-vl/src/query.rs`
- `crates/redis-vl/src/index.rs`
- `crates/redis-vl/src/vectorizers/mod.rs`
- `crates/redis-vl/src/extensions/cache.rs`
- `crates/redis-vl/src/extensions/history.rs`
- `crates/redis-vl/src/extensions/router.rs`
- `crates/rvl/src/main.rs`

Primary parity test files:

- `crates/redis-vl/tests/python_parity_search_index.rs`
- `crates/redis-vl/tests/python_parity_async_search_index.rs`
- `crates/redis-vl/tests/python_parity_embedcache.rs`
- `crates/redis-vl/tests/python_parity_semantic_cache.rs`
- `crates/redis-vl/tests/python_parity_message_history.rs`
- `crates/redis-vl/tests/python_parity_message_history_roles.rs`
- `crates/redis-vl/tests/python_parity_semantic_message_history.rs`
- `crates/redis-vl/tests/python_parity_semantic_router.rs`

Upstream Python test files already partially mirrored:

- `tests/unit/test_filter.py`
- `tests/unit/test_query_types.py`
- `tests/unit/test_schema.py`
- `tests/integration/test_search_index.py`
- `tests/integration/test_search_results.py`
- `tests/integration/test_query.py`
- `tests/integration/test_embedcache.py`
- `tests/integration/test_llmcache.py`
- `tests/integration/test_message_history.py`
- `tests/integration/test_role_filter_get_recent.py`
- `tests/integration/test_semantic_router.py`

Important caution for a takeover session:

- `HybridQuery`, `AggregateHybridQuery`, and `MultiVectorQuery` types exist, but
  that does **not** mean Python-level runtime parity is complete
- `SemanticMessageHistory` exists, but Python features around dtype/default
  vectorizers/reconnection/from-existing are still missing
- `SemanticRouter` exists, but Python features around serialization helpers,
  route reference management, dtype/default vectorizers, and from-existing style
  behavior are still missing
- key-separator normalization is implemented for single-prefix key composition,
  but full multi-prefix support is not

## Summary

- Build a greenfield Rust workspace that targets feature parity with the Python
  `redis-vl` library, while using an idiomatic Rust API rather than cloning
  Python call signatures.
- Use a workspace split:
  - `crates/redis-vl`: public library crate
  - `crates/rvl`: CLI crate
- Make the library async-first, but implement a shared command/compiler layer so
  blocking wrappers can reuse the same schema/query/cache/router logic without
  duplicating behavior.
- Treat core search/indexing plus AI extensions plus CLI as the first
  publishable milestone; defer SQL, rerankers, and non-required providers into
  explicit parity-completion waves.
- Use upstream source and API docs as the authority when docs and README diverge.

## Architecture

### Public modules

- `redis_vl::schema`: `IndexSchema`, field definitions, field attrs, stopwords,
  Hash vs JSON storage, YAML loading, SVS-VAMANA attrs
- `redis_vl::index`: `SearchIndex`, `AsyncSearchIndex`, connection config,
  CRUD/load/fetch/list/info lifecycle
- `redis_vl::query`: `VectorQuery`, `VectorRangeQuery`, `TextQuery`,
  `FilterQuery`, `CountQuery`, `AggregateHybridQuery`, `HybridQuery`,
  `MultiVectorQuery`, `SQLQuery`, `Vector`
- `redis_vl::filter`: `FilterExpression`, `Tag`, `Text`, `Num`, `Geo`,
  `GeoRadius`, `Timestamp`
- `redis_vl::vectorizers`: `Vectorizer` trait plus provider adapters
- `redis_vl::rerankers`: provider rerankers
- `redis_vl::extensions::{cache, history, router}`
- `redis_vl::error`: crate-specific error hierarchy

### Internal layering

- `command`: pure compilation of schemas/queries/filters into Redis Search and
  RedisJSON commands
- `transport`: sync and async Redis execution traits and response decoding
- `http`: provider auth/request/response adapters for OpenAI-compatible and
  later providers
- `model`: shared typed structs for cache entries, routes, messages, results,
  and stats

### Rust defaults

- Prefer borrowed inputs where practical: `&str`, `&[T]`, `Cow<'a, str>`,
  `Cow<'a, [f32]>`
- Use zero-copy vector encoding where possible for Redis query params and bulk
  load paths
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

This table is the compact handoff view. `PARITY_MATRIX.md` should stay aligned
with it.

| Area | Current status | Notes | Next high-value gap |
| --- | --- | --- | --- |
| Schema | In progress | YAML/JSON parsing, field attrs, stopwords, hash/json storage, and key-separator normalization implemented | Multi-prefix support, more advanced attrs, upstream schema edge cases |
| Search Index | In progress | Sync + async lifecycle/load/fetch/search/query/batch/paginate exist | Aggregate APIs, multi-prefix, `from_existing`-style parity, more runtime behaviors |
| Storage | In progress | Hash + JSON loading/fetching are implemented | Broader parity around multi-prefix and advanced storage edge cases |
| Filters | In progress | Core DSL implemented and covered by unit/integration tests | More integration parity and advanced combinations from upstream tests |
| Queries | In progress | Vector/Range/Text/Filter/Count implemented; hybrid types exist | Real hybrid/aggregate runtime parity, multi-vector execution parity, SQL |
| SQL | Not started | No `SQLQuery` implementation yet | Add `sql` feature and mirror upstream SQL tests |
| Vectorizers | In progress | OpenAI, LiteLLM, and Custom implemented | Azure/Cohere/Vertex/VoyageAI/Mistral/Bedrock/HF local |
| Semantic cache | In progress | Redis-backed sync/async core implemented and parity-tested | Broader LangCache attribute/config parity |
| Embeddings cache | In progress | Redis-backed sync/async core implemented and parity-tested | Warning/client-mode parity and broader edge cases |
| Message history | In progress | Standard + semantic history implemented with role filtering and parity tests | dtype/default-vectorizer/from-existing/reconnection parity |
| Router | In progress | Redis-backed routing/update/lifecycle core implemented and parity-tested | Serialization helpers, route reference APIs, dtype/default-vectorizer/from-existing parity |
| CLI | In progress | Basic command surface implemented | More Python CLI parity, tests, and UX polishing |
| Rerankers | Not started | No reranker implementation yet | Add `rerankers` feature and mirror upstream tests |
| Benchmarks | Not started | No Rust-vs-Python benchmark harness yet | Add `criterion` + Python comparison runner |
| Docs/examples | In progress | README and repo scaffolding exist | Expand rustdoc, examples, guides, docs deployment quality |

## Roadmap

### Milestone 0: Inventory and parity contract

- Status: partial
- Maintain `PARITY_MATRIX.md`
- Convert upstream examples into Rust examples, integration tests, or tracked gaps
- Record upstream doc/source discrepancies explicitly

### Milestone 1: Workspace and release scaffolding

- Status: mostly complete
- Create workspace with `redis-vl` and `rvl`
- Add `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`,
  `examples/`, `benches/`, `docs/`
- Set up CI, docs, release automation, crate metadata, docs.rs metadata, and
  feature tracking
- Use `release-plz` for automated versioning and publishing

### Milestone 2: Core RedisVL parity

- Status: partial
- Implement schema, field types, validation, stopwords, Hash/JSON storage,
  nested JSON paths, and vector field configuration
- Implement index lifecycle, connection config, load/fetch/delete/clear/drop,
  expire/list/info/batch/paginate
- Implement filter DSL and query builders, including hybrid modes and
  multi-vector weighting
- Match the Python CLI command surface

### Milestone 3: First publishable Rust release

- Status: partial
- Implement `Vectorizer` trait, OpenAI-compatible transport,
  `OpenAITextVectorizer`, `LiteLLMTextVectorizer`, and `CustomVectorizer`
- Implement `SemanticCache`, `EmbeddingsCache`, `MessageHistory`,
  `SemanticMessageHistory`, and `SemanticRouter`
- Publish `0.x` only when:
  - all Phase 1 rows are green
  - all public APIs have rustdoc
  - examples compile
  - benchmark harness exists

### Milestone 4: Parity completion

- Status: not started in substance
- Add `SQLQuery` behind `sql`
- Add rerankers behind `rerankers`
- Add secondary providers: Cohere, Vertex AI/Gemini, Azure OpenAI, VoyageAI,
  Mistral, Bedrock, HF local
- Add SVS-VAMANA helper/tuning utilities
- Cut `1.0.0` only after the parity matrix has no remaining Python-surface gaps

### Milestone 5: Post-parity additions

- Status: not started
- Add Anthropic and any other OpenAI-compatible or ecosystem-driven adapters
- Add Rust-native ergonomics only after parity is complete and documented

## Recommended Next Work Order

If a new agent/session takes over, the recommended order is:

1. Hybrid and aggregate query parity
   Start in `crates/redis-vl/src/query.rs`, `crates/redis-vl/src/index.rs`,
   then mirror `tests/integration/test_hybrid.py` and
   `tests/integration/test_aggregation.py`
2. Multi-prefix support
   Start in `crates/redis-vl/src/schema.rs` and `crates/redis-vl/src/index.rs`,
   then mirror `tests/integration/test_multi_prefix.py`
3. Search-index/key-construction edge cases
   Continue from `tests/integration/test_key_separator_handling.py` and related
   schema/index tests
4. CLI parity and CLI tests
   Expand `crates/rvl/src/main.rs` and add CLI-focused tests mirroring Python CLI
   expectations where possible
5. SQLQuery feature
   Add `sql`-gated implementation and mirror
   `tests/integration/test_sql_redis_hash.py` and
   `tests/integration/test_sql_redis_json.py`
6. Rerankers and additional providers
7. Benchmarks, examples, and docs expansion

## Takeover Checklist

Before making the next major parity patch:

- read the relevant upstream Python test file first
- implement only the minimal Rust surface needed for that contract
- add or extend a Rust parity test file that names the upstream source in its
  module docs
- run:
  `cargo fmt --all`
- run:
  `cargo check --workspace`
- run:
  `cargo test --workspace`
- if a live Redis instance is available, run:
  `REDISVL_RUN_INTEGRATION=1 cargo test --workspace`

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

- This file should be updated whenever major parity milestones or architecture
  decisions change.
- `PARITY_MATRIX.md` should stay more granular than this file; use this file for
  roadmap and repository conventions, and the parity matrix for feature-level
  tracking.
