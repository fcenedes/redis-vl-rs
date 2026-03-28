#!/usr/bin/env python3
"""Pure-Python benchmarks (no Redis required).

Mirrors the operations in ``crates/redis-vl/benches/core_benchmarks.rs``:
  - Schema parsing (YAML / JSON / to-YAML)
  - Filter rendering (simple tag, compound, negated)
  - Query building (vector query, vector query with filter)

Usage::

    python benches/python/bench_core.py          # JSON to stdout
    python benches/python/bench_core.py --verbose # progress on stderr
"""

from __future__ import annotations

import json
import sys

import numpy as np
import yaml

from bench_harness import BenchSuite
from redisvl.query import VectorQuery
from redisvl.query.filter import Num, Tag, Text
from redisvl.schema import IndexSchema

# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------

SCHEMA_YAML = """
index:
  name: bench-index
  prefix: doc
  storage_type: hash
fields:
  - name: title
    type: tag
  - name: content
    type: text
  - name: score
    type: numeric
  - name: embedding
    type: vector
    attrs:
      algorithm: flat
      dims: 128
      distance_metric: cosine
      datatype: float32
"""

SCHEMA_DICT = yaml.safe_load(SCHEMA_YAML)
SCHEMA_JSON_STR = json.dumps(SCHEMA_DICT)


def bench_schema_from_yaml():
    IndexSchema.from_dict(yaml.safe_load(SCHEMA_YAML))


def bench_schema_from_json():
    IndexSchema.from_dict(json.loads(SCHEMA_JSON_STR))


def bench_schema_to_yaml():
    schema = IndexSchema.from_dict(SCHEMA_DICT)
    # mode="json" ensures enum values (e.g. FieldTypes.TEXT) are serialized as
    # plain strings so PyYAML's SafeDumper can handle them.
    yaml.dump(schema.model_dump(mode="json"), Dumper=yaml.SafeDumper)


# ---------------------------------------------------------------------------
# Filter rendering
# ---------------------------------------------------------------------------

def bench_filter_tag_simple():
    f = Tag("color") == "red"
    str(f)


def bench_filter_compound():
    f = (Tag("color") == "red") & (Num("price") >= 10) & (Num("price") <= 100) & (Text("description") == "basketball")
    str(f)


def bench_filter_negated():
    f = (Tag("status") != "deleted") & (Tag("status") != "archived")
    str(f)


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

_VEC_128 = [0.1] * 128


def bench_vector_query_render():
    q = VectorQuery(
        vector=_VEC_128,
        vector_field_name="embedding",
        num_results=10,
    )
    q.query


def bench_vector_query_with_filter_render():
    filt = (Tag("color") == "red") & (Num("price") < 50)
    q = VectorQuery(
        vector=_VEC_128,
        vector_field_name="embedding",
        num_results=10,
        filter_expression=filt,
    )
    q.query


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    verbose = "--verbose" in sys.argv
    suite = BenchSuite("core")

    suite.add("schema_from_yaml", bench_schema_from_yaml)
    suite.add("schema_from_json", bench_schema_from_json)
    suite.add("schema_to_yaml", bench_schema_to_yaml)
    suite.add("filter_tag_simple", bench_filter_tag_simple)
    suite.add("filter_compound", bench_filter_compound)
    suite.add("filter_negated", bench_filter_negated)
    suite.add("vector_query_render", bench_vector_query_render)
    suite.add("vector_query_with_filter_render", bench_vector_query_with_filter_render)

    suite.run(verbose=verbose)
    suite.emit_json()


if __name__ == "__main__":
    main()
