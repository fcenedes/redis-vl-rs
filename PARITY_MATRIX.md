# RedisVL Parity Matrix

This document tracks the Rust implementation against the current Python
`redis-vl` surface.

Validation policy: upstream Python tests under `redis-vl-python/tests` are the
source of truth for behavior parity.

| Area | Target | Status | Notes |
| --- | --- | --- | --- |
| Schema | `IndexSchema`, YAML loading, typed field attrs | In progress | YAML/JSON parsing, field validation, stopwords, hash/json storage, and single-prefix key-separator normalization are implemented; multi-prefix and more advanced schema parity are still open |
| Filters | Tag/Text/Num/Geo/GeoRadius/Timestamp | In progress | Core DSL and rendering are implemented with unit/integration coverage; broader integration parity is still pending |
| Queries | Vector/Range/Text/Filter/Count/Hybrid/MultiVector | In progress | Vector/Range/Text/Filter/Count work today; `HybridQuery`, `AggregateHybridQuery`, and `MultiVectorQuery` types exist but runtime parity is incomplete; SQL is still absent |
| Search Index | Sync + async lifecycle/load/info | In progress | Sync/async create/delete/load/fetch/search/query/batch/paginate are implemented; aggregate/multi-prefix/from-existing style parity is still missing |
| Vectorizers | OpenAI/LiteLLM/Custom | In progress | `Vectorizer`, `AsyncVectorizer`, OpenAI-compatible adapters, and custom vectorizers are implemented; secondary providers are still missing |
| Cache Extensions | Semantic + embeddings cache | In progress | `EmbeddingsCache` and `SemanticCache` are Redis-backed and parity-tested for core sync/async flows; broader LangCache attribute/config parity is still open |
| Message History | MessageHistory/SemanticMessageHistory | In progress | Standard and semantic history are Redis-backed and parity-tested for ordered retrieval, semantic recall, scope, drop/count, and role filtering; dtype/default-vectorizer/from-existing parity is still missing |
| Router | SemanticRouter/Route/RouteMatch | In progress | Redis-backed semantic router is implemented and parity-tested for routing, updates, and lifecycle behavior; serialization helpers, route reference APIs, and from-existing/provider parity are still open |
| CLI | `rvl version/index/stats` | In progress | Basic command scaffold is implemented; richer command/flag parity and CLI tests are still missing |
| SQL | `SQLQuery` | Planned | Future `sql` feature |
| Rerankers | Provider rerankers | Planned | Future `rerankers` feature |
| Benchmarks | Rust vs Python benchmark harness | Planned | No benchmark harness exists yet |
| Docs/Examples | README/examples/API docs | In progress | Repo/doc scaffolding exists, but example coverage and publication-quality docs are still incomplete |
