#!/usr/bin/env python3
"""Redis-backed Python benchmarks.

Mirrors the operations in ``crates/redis-vl/benches/redis_benchmarks.rs``:
  - Index lifecycle (create / exists / info)
  - Load and fetch
  - Vector search, filter search, count query
  - Batch search, paginated search
  - Embeddings cache, semantic cache
  - Message history, semantic message history

Requires a running Redis 8+ / Redis Stack instance.

Usage::

    REDIS_URL=redis://127.0.0.1:6379 python benches/python/bench_redis.py
    REDIS_URL=redis://127.0.0.1:6379 python benches/python/bench_redis.py --verbose
"""

from __future__ import annotations

import os
import random
import sys
import time
import uuid

import numpy as np
import redis as _redis_lib

from bench_harness import BenchSuite
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.index import SearchIndex
from redisvl.query import CountQuery, FilterQuery, VectorQuery
from redisvl.query.filter import Num, Tag
from redisvl.schema import IndexSchema

REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")


def _wait_for_redis(url: str = REDIS_URL, retries: int = 15, delay: float = 2.0) -> None:
    """Block until a Redis PING succeeds, retrying with back-off."""
    for attempt in range(1, retries + 1):
        try:
            r = _redis_lib.from_url(url, socket_connect_timeout=5)
            r.ping()
            r.close()
            return
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Redis at {url} not reachable after {retries} attempts: {exc}"
                ) from exc
            print(
                f"  Redis not ready (attempt {attempt}/{retries}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


SCHEMA_DICT = {
    "index": {"name": "placeholder", "prefix": "doc", "storage_type": "hash"},
    "fields": [
        {"name": "title", "type": "tag"},
        {"name": "content", "type": "text"},
        {"name": "score", "type": "numeric"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": 128,
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        },
    ],
}


def _make_index(name: str) -> SearchIndex:
    d = {**SCHEMA_DICT, "index": {**SCHEMA_DICT["index"], "name": name}}
    schema = IndexSchema.from_dict(d)
    return SearchIndex(schema, redis_url=REDIS_URL)


def _random_vec(dim: int = 128) -> list[float]:
    return [random.random() for _ in range(dim)]


def _docs(n: int, random_embed: bool = False) -> list[dict]:
    return [
        {
            "id": f"doc:{i}",
            "title": f"title-{i}",
            "content": "benchmark document with searchable content",
            "score": str(i),
            "embedding": np.array(
                _random_vec() if random_embed else [0.1] * 128, dtype=np.float32
            ).tobytes(),
        }
        for i in range(n)
    ]



# ---------------------------------------------------------------------------
# Index lifecycle benchmarks
# ---------------------------------------------------------------------------

def add_index_benchmarks(suite: BenchSuite) -> None:
    indices_to_cleanup: list[SearchIndex] = []

    def bench_create():
        idx = _make_index(_uid("idx_create"))
        idx.create(overwrite=True)
        indices_to_cleanup.append(idx)

    suite.add("index_create", bench_create, warmup_iters=3, sample_iters=20,
              teardown=lambda: [i.delete() for i in indices_to_cleanup])

    idx_e = _make_index(_uid("idx_exists"))
    idx_e.create(overwrite=True)
    suite.add("index_exists", lambda: idx_e.exists(),
              teardown=lambda: idx_e.delete())

    idx_i = _make_index(_uid("idx_info"))
    idx_i.create(overwrite=True)
    suite.add("index_info", lambda: idx_i.info(),
              teardown=lambda: idx_i.delete())


# ---------------------------------------------------------------------------
# Load / fetch benchmarks
# ---------------------------------------------------------------------------

def add_load_fetch_benchmarks(suite: BenchSuite) -> None:
    idx = _make_index(_uid("idx_load"))
    idx.create(overwrite=True)
    doc = _docs(1)[0]

    suite.add("index_load_single", lambda: idx.load([doc], id_field="id"),
              sample_iters=50)

    docs100 = _docs(100)
    suite.add("index_load_batch_100", lambda: idx.load(docs100, id_field="id"),
              sample_iters=30)

    idx.load(docs100, id_field="id")
    suite.add("index_fetch_single", lambda: idx.fetch("doc:50"),
              teardown=lambda: idx.delete())


# ---------------------------------------------------------------------------
# Search benchmarks
# ---------------------------------------------------------------------------

def add_search_benchmarks(suite: BenchSuite) -> None:
    idx = _make_index(_uid("idx_search"))
    idx.create(overwrite=True)
    idx.load(_docs(100, random_embed=True), id_field="id")

    qvec = _random_vec()
    q = VectorQuery(vector=qvec, vector_field_name="embedding", num_results=10)
    suite.add("search_vector_k10_n100", lambda: idx.search(q.query, q.params))

    filt = (Tag("title") == "title-5") & (Num("score") >= 50)
    qf = VectorQuery(
        vector=qvec, vector_field_name="embedding",
        num_results=10, filter_expression=filt,
    )
    suite.add("search_vector_with_filter", lambda: idx.search(qf.query, qf.params))

    ff = Tag("title") == "title-5"
    fq = FilterQuery(filter_expression=ff)
    suite.add("search_filter_simple", lambda: idx.search(fq.query, fq.params))

    cq = CountQuery(filter_expression=ff)
    suite.add("search_count", lambda: idx.search(cq.query, cq.params),
              teardown=lambda: idx.delete())


# ---------------------------------------------------------------------------
# Semantic cache benchmarks
# ---------------------------------------------------------------------------

def _simple_vectorizer(text: str) -> list[float]:
    vec = [0.0] * 128
    for i, b in enumerate(text.encode()):
        vec[i % 128] += b / 255.0
    return vec


def add_cache_benchmarks(suite: BenchSuite) -> None:
    from redisvl.utils.vectorize import CustomVectorizer

    vectorizer = CustomVectorizer(embed=_simple_vectorizer)
    cache = SemanticCache(
        name=_uid("semcache"),
        redis_url=REDIS_URL,
        distance_threshold=0.3,
        vectorizer=vectorizer,
    )

    suite.add("semcache_store",
              lambda: cache.store("what is the capital of France?", "Paris"),
              sample_iters=50)

    cache.store("what is the capital of France?", "Paris")
    suite.add("semcache_check_hit",
              lambda: cache.check("what is the capital of France?"),
              sample_iters=50)

    suite.add("semcache_check_miss",
              lambda: cache.check("completely different query"),
              sample_iters=50,
              teardown=lambda: cache.delete())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    verbose = "--verbose" in sys.argv

    # Wait until Redis is actually accepting commands (Docker may be slow)
    if verbose:
        print("  Checking Redis readiness …", file=sys.stderr, flush=True)
    _wait_for_redis()

    suite = BenchSuite("redis")

    add_index_benchmarks(suite)
    add_load_fetch_benchmarks(suite)
    add_search_benchmarks(suite)
    add_cache_benchmarks(suite)

    suite.run(verbose=verbose)
    suite.emit_json()


if __name__ == "__main__":
    main()