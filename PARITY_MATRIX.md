# RedisVL Parity Matrix

This document tracks the Rust implementation against the current Python
`redis-vl` surface.

Validation policy: upstream Python tests under `redis-vl-python/tests` are the
source of truth for behavior parity.

| Area | Target | Status | Notes |
| --- | --- | --- | --- |
| Schema | `IndexSchema`, YAML loading, typed field attrs | In progress | YAML/JSON parsing, field validation, stopwords, hash/json storage, multi-prefix support, and key-separator normalization are implemented; some advanced schema edge cases remain |
| Filters | Tag/Text/Num/Geo/GeoRadius/Timestamp | In progress | Core DSL and rendering are implemented with unit/integration coverage; broader integration parity is still pending |
| Queries | Vector/Range/Text/Filter/Count/Hybrid/AggregateHybrid/MultiVector | In progress | All query types are implemented with command builders. `HybridQuery` generates `FT.HYBRID` commands (requires Redis 8.4+). `AggregateHybridQuery` generates `FT.AGGREGATE` commands. `MultiVectorQuery` generates multi-vector aggregate commands. Redis 8.4 integration tests are implemented and environment-gated in `python_parity_hybrid_aggregate.rs` |
| Search Index | Sync + async lifecycle/load/info | In progress | Sync/async create/delete/load/fetch/search/query/batch/paginate, `hybrid_search`/`hybrid_query`, `aggregate_query`, `multi_vector_query`, `sql_query` (auto-dispatch to `FT.SEARCH`/`FT.AGGREGATE`), and `from_existing` are implemented; multi-prefix index creation is supported; Redis 8.4 integration tests are environment-gated |
| SQL | `SQLQuery` behind `sql` feature | Implemented | SQL→Redis Search translation for non-aggregate `SELECT` queries: `WHERE` (tag/numeric/text/date comparisons, `IN`/`NOT IN`, `LIKE`/`NOT LIKE`, `BETWEEN`, `AND`/`OR`), `ORDER BY`, `LIMIT`/`OFFSET`, field projection. Aggregate SQL (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `STDDEV`, `COUNT_DISTINCT`, `QUANTILE`, `GROUP BY`) is translated to `FT.AGGREGATE`; `SearchIndex::sql_query`/`AsyncSearchIndex::sql_query` auto-dispatch between `FT.SEARCH` and `FT.AGGREGATE`. Vector/geo aggregate functions are not yet supported |
| Vectorizers | OpenAI/LiteLLM/Custom + Azure/Cohere/VoyageAI/Mistral | In progress | `OpenAITextVectorizer`, `LiteLLMTextVectorizer`, `CustomTextVectorizer` plus `AzureOpenAITextVectorizer`, `CohereTextVectorizer`, `VoyageAITextVectorizer`, and `MistralAITextVectorizer` are implemented. Remaining: Vertex AI, Bedrock, HuggingFace local, Anthropic |
| Rerankers | `CohereReranker` behind `rerankers` feature | Implemented | `Reranker`/`AsyncReranker` traits and `CohereReranker` with sync/async support are implemented. Additional reranker providers can be added incrementally |
| Cache Extensions | Semantic + embeddings cache | In progress | `EmbeddingsCache` and `SemanticCache` are Redis-backed and parity-tested for core sync/async flows; broader LangCache attribute/config parity is still open |
| Message History | MessageHistory/SemanticMessageHistory | In progress | Standard and semantic history are Redis-backed and parity-tested for ordered retrieval, semantic recall, scope, drop/count, and role filtering; dtype/default-vectorizer/reconnection parity is still missing |
| Router | SemanticRouter/Route/RouteMatch | In progress | Redis-backed semantic router is implemented and parity-tested for routing, updates, and lifecycle behavior; serialization helpers, route reference APIs, and from-existing/provider parity are still open |
| CLI | `rvl version/index/stats` | In progress | Basic command scaffold is implemented; richer command/flag parity and CLI tests are still missing |
| Benchmarks | Criterion micro-benchmarks + Redis-backed benchmarks | In progress | `criterion` benchmarks for schema parsing, filter rendering, and query building exist. Redis-backed benchmarks cover search index operations (create/exists/info/load/fetch), search queries (vector/filter/count/batch/paginate), embeddings cache (set/get hit/miss), semantic cache (store/check hit/miss), message history (add/get_recent), and semantic message history (add/get_recent/get_relevant). Rust-vs-Python comparison harness is not yet implemented |
| Docs/Examples | README/examples/API docs | In progress | Repo scaffolding, mdBook guide, and rustdoc exist but example coverage and publication-quality docs are still incomplete |
