# Typed Connectors Guide

Typed connectors are the storage-aware adapters at the heart of SDOL. Each connector knows how to translate declarative intents into native queries for a specific data paradigm, execute them, and normalize results with full provenance metadata.

---

## Table of Contents

1. [What Is a Typed Connector?](#what-is-a-typed-connector)
2. [Three-Tier Architecture](#three-tier-architecture)
3. [The Four-Stage Pipeline](#the-four-stage-pipeline)
4. [Built-In Connectors](#built-in-connectors)
   - [OLAP Paradigm](#olap-paradigm)
   - [OLTP Paradigm](#oltp-paradigm)
   - [Document Paradigm](#document-paradigm)
5. [The QueryExecutor Protocol](#the-queryexecutor-protocol)
6. [Capability Declaration](#capability-declaration)
7. [Registering Connectors](#registering-connectors)
8. [Building a Custom Connector](#building-a-custom-connector)
9. [Suitability Scoring](#suitability-scoring)

---

## What Is a Typed Connector?

A typed connector is a storage-aware adapter that:

1. **Declares capabilities** — which intent types it supports, what entities it has access to, latency profile, feature flags (aggregation, windowing, similarity, etc.)
2. **Translates intents to native queries** — a `PointLookupIntent` becomes a `SELECT ... WHERE id = ?`; an `AggregateAnalysisIntent` becomes `SELECT ... GROUP BY ...`
3. **Executes queries** — via a pluggable `QueryExecutor` (protocol-based, so you can swap mock/real backends)
4. **Normalizes results** — raw database rows become `ConnectorResult` with provenance envelope, trust-relevant metadata, slot typing, and entity key extraction

The Semantic Router uses the Capability Registry to pick the best connector for each intent automatically.

---

## Three-Tier Architecture

Connectors are organized in three layers:

```
┌────────────────────────────────────────────────────────────┐
│  Layer 1: Foundation (static — rarely changes)              │
│  BaseConnector (4-stage pipeline ABC)                       │
│  QueryExecutor (protocol), sql_utils (shared helpers)       │
├────────────────────────────────────────────────────────────┤
│  Layer 2: Paradigm Bases (semi-stable)                      │
│  BaseOLAPConnector, BaseOLTPConnector, BaseDocumentConnector│
│  Shared: interpret_intent, normalize_result, capabilities   │
├────────────────────────────────────────────────────────────┤
│  Layer 3: Provider Extensions (highly extensible)           │
│  Core: GenericOLAP, GenericOLTP, GenericDocument (src/)     │
│  Databricks: DBSQL, Lakebase, VectorSearch (extensions/)   │
│  Only implements: synthesize_query + get_performance        │
└────────────────────────────────────────────────────────────┘
```

**Adding a new provider** (e.g., BigQuery OLAP, Postgres OLTP) requires:
1. One new file under `extensions/<provider>/<paradigm>/`
2. One query builder with native query dataclass + builder functions
3. Subclass the paradigm base, implement `synthesize_query()` + `get_performance()`
4. Register it — routing, trust scoring, and context compilation work automatically

Provider extensions live in `extensions/` alongside `src/`, keeping core clean. Import path: `sdol.extensions.<provider>.<paradigm>.<module>`.

---

## The Four-Stage Pipeline

Every connector's `execute(intent)` method runs four stages in order:

```
Intent ──▶ interpret_intent ──▶ synthesize_query ──▶ execute_query ──▶ normalize_result ──▶ ConnectorResult
              (stage 1)            (stage 2)           (stage 3)          (stage 4)
```

| Stage | Method | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| 1 | `interpret_intent(intent)` | `BaseIntent` | Typed intent subclass | Validate that the connector can handle this intent type; extract typed parameters |
| 2 | `synthesize_query(params)` | Typed intent | Native query object | Build a platform-specific query (SQL, vector search request, graph traversal, etc.) |
| 3 | `execute_query(query)` | Native query | Raw dict `{"records": [...], "meta": {...}}` | Run the query against the data source via `QueryExecutor` |
| 4 | `normalize_result(raw, intent, execution_ms)` | Raw dict + intent + timing | `ConnectorResult` | Attach provenance, determine slot type, extract entity keys |

The `BaseConnector.execute()` method orchestrates these stages and measures wall-clock time automatically.

---

## Built-In Connectors

### OLAP Paradigm

**Base class:** `BaseOLAPConnector` (`src/sdol/connectors/olap/base.py`)
**Shared logic:** `interpret_intent` (validates `AggregateAnalysisIntent | TemporalTrendIntent`), `normalize_result` (STRUCTURED/TEMPORAL slot types, COMPUTED_AGGREGATE retrieval, STRONG consistency default)

| Provider | Class | Default Source | Staleness | Special Features |
|----------|-------|---------------|-----------|-----------------|
| Generic | `GenericOLAPConnector` | `snowflake.analytics` | 3600s | PostgreSQL-style `$n` params |
| Databricks DBSQL | `DatabricksDBSQLConnector` | `databricks.sql_warehouse` | 600s | Unity Catalog, `DATE_TRUNC`, `PERCENTILE_APPROX`, named `:p` params, Photon tracking |

```python
from sdol import GenericOLAPConnector, DatabricksDBSQLConnector
from sdol.connectors.executor import MockQueryExecutor

generic = GenericOLAPConnector(
    executor=MockQueryExecutor(records=[...]),
    connector_id="olap.snowflake",
    source_system="snowflake.analytics",
    available_entities=["orders", "revenue"],
)

dbsql = DatabricksDBSQLConnector(
    executor=MockQueryExecutor(records=[...]),
    connector_id="databricks.analytics",
    catalog="prod_catalog",
    schema="analytics",
    available_entities=["orders", "revenue_daily"],
)
```

See [Databricks Guide](databricks-guide.md) for full DBSQL details.

### OLTP Paradigm

**Base class:** `BaseOLTPConnector` (`src/sdol/connectors/oltp/base.py`)
**Shared logic:** `interpret_intent` (validates `PointLookupIntent | AggregateAnalysisIntent`), `normalize_result` (STRUCTURED slot type, DIRECT_QUERY/COMPUTED_AGGREGATE retrieval, READ_COMMITTED consistency default)

| Provider | Class | Default Source | Staleness | Special Features |
|----------|-------|---------------|-----------|-----------------|
| Generic | `GenericOLTPConnector` | `postgres.production` | 60s | Batch lookups, `$n` positional params |
| Databricks Lakebase | `DatabricksLakebaseConnector` | `databricks.lakebase` | 30s | Unity Catalog, row index, batch lookups, named `:p` params |

```python
from sdol import GenericOLTPConnector, DatabricksLakebaseConnector
from sdol.connectors.executor import MockQueryExecutor

generic = GenericOLTPConnector(
    executor=MockQueryExecutor(records=[...]),
    connector_id="oltp.postgres",
    source_system="postgres.production",
    available_entities=["customers", "tickets"],
)

lakebase = DatabricksLakebaseConnector(
    executor=MockQueryExecutor(records=[...]),
    connector_id="databricks.serving",
    catalog="prod_catalog",
    schema="serving",
    available_entities=["customers", "products"],
)
```

See [Databricks Guide](databricks-guide.md) for full Lakebase details.

### Document Paradigm

**Base class:** `BaseDocumentConnector` (`src/sdol/connectors/document/base.py`)
**Shared logic:** `interpret_intent` (validates `SemanticSearchIntent`), `normalize_result` (UNSTRUCTURED slot type, VECTOR_SIMILARITY retrieval, EVENTUAL consistency default)

| Provider | Class | Default Source | Staleness | Special Features |
|----------|-------|---------------|-----------|-----------------|
| Generic | `GenericDocumentConnector` | `pinecone.vectors` | 300s | Hybrid retrieval (vector + keyword), reranking, score-based truncation |
| Databricks Vector Search | `DatabricksVectorSearchConnector` | `databricks.vector_search` | 180s | Unity Catalog, ANN/HYBRID search, Delta Sync auto-update, metadata filter pushdown |

```python
from sdol import GenericDocumentConnector, DatabricksVectorSearchConnector
from sdol.connectors.executor import MockQueryExecutor

generic = GenericDocumentConnector(
    executor=MockQueryExecutor(records=[...]),
    connector_id="doc.pinecone",
    source_system="pinecone.vectors",
    available_entities=["knowledge_base"],
)

databricks_vs = DatabricksVectorSearchConnector(
    executor=MockQueryExecutor(records=[...]),
    connector_id="databricks.vs",
    catalog="prod_catalog",
    schema="ml",
    index_name="prod_catalog.ml.kb_index",
    available_entities=["knowledge_base"],
)
```

See [Databricks Guide](databricks-guide.md) for full Vector Search details.

---

## The QueryExecutor Protocol

Connectors don't talk to databases directly — they delegate to a `QueryExecutor`. This is a Python protocol (structural typing) with a single method:

```python
class QueryExecutor(Protocol):
    async def execute(self, query: Any) -> dict[str, Any]:
        """Returns {"records": [...], "meta": {...}}"""
        ...
```

The `query` argument is whatever the connector's `synthesize_query()` produces — an `OLAPQuery`, `OLTPQuery`, `DBSQLQuery`, `LakebaseQuery`, `DocumentQuery`, or your own type.

The return value must be a dict with:
- `records` — list of result dicts
- `meta` — optional dict with `native_query` (string) and `total_available` (int)

### MockQueryExecutor

For testing, SDOL provides `MockQueryExecutor`:

```python
from sdol.connectors.executor import MockQueryExecutor

executor = MockQueryExecutor(
    records=[{"id": "C-1", "name": "Alice"}],
    meta={"native_query": "SELECT ...", "total_available": 1},
)

result = await executor.execute(any_query)
assert executor.last_query == any_query  # inspect what was sent
```

---

## Capability Declaration

Every connector declares its capabilities via `get_capabilities() -> ConnectorCapability`:

```python
ConnectorCapability(
    connector_id="my.connector",          # unique ID (matches connector.id)
    connector_type="olap",                # paradigm: olap, oltp, document, etc.
    supported_intent_types=[              # which intents this connector handles
        "aggregate_analysis",
        "temporal_trend",
    ],
    capabilities=ConnectorCapabilities(   # feature flags
        supports_aggregation=True,
        supports_windowing=True,
        supports_temporal_bucketing=True,
        supports_traversal=False,
        supports_similarity=False,
        supports_inference=False,
        supports_full_text_search=False,
    ),
    performance=ConnectorPerformance(     # latency and scale profile
        estimated_latency_ms=300,
        max_result_cardinality=10_000_000,
        supports_batch_lookup=False,
    ),
    available_entities=["orders", "revenue"],  # tables/collections this connector has
)
```

The registry uses these declarations for:
- **Filtering** — only connectors whose `supported_intent_types` include the intent type are candidates
- **Scoring** — entity match, latency, and capability alignment determine ranking

---

## Registering Connectors

```python
from sdol import CapabilityRegistry

registry = CapabilityRegistry()
registry.register(olap_connector)
registry.register(oltp_connector)
registry.register(document_connector)
registry.register(databricks_dbsql_connector)
registry.register(databricks_lakebase_connector)

# Inspect what's registered
for cap in registry.list_capabilities():
    print(f"{cap.connector_id}: {cap.supported_intent_types}")

# Look up a specific connector
connector = registry.get_connector("databricks.dbsql")

# Find best candidates for an intent
candidates = registry.find_candidates(some_intent)
for c in candidates:
    print(f"  {c.capability.connector_id} — score: {c.suitability_score:.2f}")

# Remove a connector
registry.unregister("old.connector")
```

---

## Building a Custom Connector

The recommended approach is to extend the appropriate **paradigm base class** rather than raw `BaseConnector`. This gives you shared `interpret_intent`, `normalize_result`, `get_capabilities`, and `check_health` for free — you only implement `synthesize_query()` and `get_performance()`.

### Example: Redis OLTP Connector

```python
from __future__ import annotations
from typing import Any
from dataclasses import dataclass

from sdol.connectors.oltp.base import BaseOLTPConnector
from sdol.connectors.executor import QueryExecutor
from sdol.types.capability import ConnectorPerformance
from sdol.types.intent import PointLookupIntent
from sdol.types.provenance import ConsistencyGuarantee


@dataclass
class RedisQuery:
    key: str
    fields: list[str] | None


class RedisConnector(BaseOLTPConnector):
    """Redis-backed OLTP connector for low-latency point lookups."""

    def __init__(
        self,
        executor: QueryExecutor,
        connector_id: str = "redis.cache",
        available_entities: list[str] | None = None,
    ) -> None:
        super().__init__(
            executor=executor,
            connector_id=connector_id,
            source_system="redis.cache",
            available_entities=available_entities,
        )

    @property
    def default_staleness_sec(self) -> float:
        return 5.0

    @property
    def default_consistency(self) -> ConsistencyGuarantee:
        return ConsistencyGuarantee.EVENTUAL

    def get_performance(self) -> ConnectorPerformance:
        return ConnectorPerformance(
            estimated_latency_ms=2,
            max_result_cardinality=1,
        )

    def synthesize_query(self, params: Any) -> RedisQuery:
        intent = params  # PointLookupIntent or AggregateAnalysisIntent
        if isinstance(params, PointLookupIntent):
            key_parts = [f"{k}:{v}" for k, v in intent.identifier.items()]
            return RedisQuery(
                key=f"{intent.entity}:{':'.join(key_parts)}",
                fields=intent.fields,
            )
        return RedisQuery(key=intent.entity, fields=None)
```

Compare this to the old approach of extending `BaseConnector` directly — `interpret_intent`, `normalize_result`, `get_capabilities`, `check_health`, `execute_query`, `connector_type`, and `id` are all provided by `BaseOLTPConnector`. The Redis connector only implements what's truly provider-specific.

### When to extend `BaseConnector` directly

Use `BaseConnector` directly only for paradigms that don't yet have a base class (e.g., a graph connector or an ontology connector). For OLAP, OLTP, and document paradigms, always extend the paradigm base.

---

## Suitability Scoring

When the router receives an intent, it asks the registry to score all matching connectors:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Entity match** | 40% | Does the connector's `available_entities` include the requested entity/collection? Full match = 0.4, no entities declared = 0.2, mismatch = 0.0 |
| **Latency** | 30% | `1 - (estimated_latency_ms / 5000)` — lower latency scores higher |
| **Capability alignment** | 30% | Fraction of intent-relevant feature flags that are `True` (e.g., `aggregate_analysis` checks `supports_aggregation`) |

The connector with the highest composite score handles the intent. For composite intents, each sub-intent is scored and routed independently.
