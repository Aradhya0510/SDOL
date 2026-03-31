# SDOL — Semantic Data Orchestration Layer

**Storage-aware agent middleware that replaces flat context windows with structured, provenance-enriched context frames.**

SDOL sits between AI agents and their data sources. When an agent needs data, it declares *what* it wants (an intent) — never *how* to get it. SDOL handles routing, execution, provenance tracking, trust scoring, conflict resolution, and epistemic context generation automatically.

---

## The Problem

Today's AI agents consume data through flat context windows — raw text dumps with no metadata about where the data came from, how fresh it is, whether sources agree, or how much to trust any given fact. This leads to:

- **Hallucination amplification** — stale or low-quality data treated with the same confidence as authoritative sources
- **Source blindness** — the agent can't distinguish a cached estimate from an exact database query
- **Conflict ignorance** — when two sources disagree, the agent has no framework to reason about which to trust
- **Paradigm lock-in** — agents hard-coded to one data source can't seamlessly query across OLAP, OLTP, document stores, and more

## How SDOL Solves This

SDOL introduces a semantic layer that:

1. **Typed intents** — 8 declarative intent types (`point_lookup`, `aggregate_analysis`, `temporal_trend`, `semantic_search`, `graph_traversal`, `ontology_query`, `composite`, `escape_hatch`) that describe what the agent needs, not how to retrieve it
2. **Typed connectors** — storage-aware adapters that translate intents into native queries. Each declares its capabilities (supported intents, latency, features) so the router can pick the best backend automatically
3. **Provenance tracking** — every data element carries a `ProvenanceEnvelope` with source system, retrieval method, consistency guarantee, precision class, and freshness
4. **Trust scoring** — four-dimensional scoring (authority, consistency, freshness, precision) produces a composite 0–1 confidence signal per element
5. **Conflict resolution** — when sources disagree on the same entity, heuristic resolution picks the freshest, most authoritative, or most consistent source — or defers to the agent
6. **Epistemic context** — a generated summary injected into the agent's prompt that warns about low-trust data, unresolved conflicts, and overall confidence levels

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           Agent / LLM                                │
│                                                                      │
│   formulate intent ──▶  SDOL.query(intent)  ──▶  ContextFrame       │
│                                                  + epistemic prompt   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                          SDOL SDK                                     │
│                                                                      │
│   IntentFormulator ──▶ SemanticRouter ──▶ ContextCompiler            │
│                              │                    │                   │
│                        QueryPlanner          TrustScorer              │
│                         │        │          ConflictDetector          │
│                  IntentDecomposer CostEstimator  ConflictResolver     │
│                         │                                            │
│                  CapabilityRegistry                                   │
│                    ┌────┼────┬──────────┐                            │
│                    ▼    ▼    ▼          │  Paradigm Bases            │
│              BaseOLAP BaseOLTP BaseDoc  │  (shared logic)            │
│               ┌──┴──┐ ┌──┴──┐   │      │                            │
│               ▼     ▼ ▼     ▼   ▼      │  Provider Extensions       │
│           Generic DBSQL Generic Lakebase Generic DatabricksVS        │
│            OLAP       OLTP              Document                     │
│               │     │ │     │   │                                    │
│               ▼     ▼ ▼     ▼   ▼                                    │
│              QueryExecutor (protocol) ── pluggable backends          │
│                                                                      │
│   EpistemicTracker ◀── ContextFrame ── trust + conflicts + slots     │
└──────────────────────────────────────────────────────────────────────┘
```

Each layer is independent and composable:
- **Intent layer** — Pydantic v2 models with discriminated unions, validated at construction time
- **Router layer** — decomposes composites, plans execution topology, estimates cost, routes to connectors
- **Connector layer** — three-tier model: foundation (`BaseConnector`), paradigm bases (`BaseOLAPConnector`, `BaseOLTPConnector`, `BaseDocumentConnector`), and provider extensions (in `extensions/`) with pluggable executors
- **Context layer** — assembles typed slots, detects cross-source conflicts, resolves via heuristics
- **Epistemic layer** — aggregates trust signals into an LLM-injectable prompt

For the full technical deep-dive, see [Architecture](docs/architecture.md).

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v   # 254 tests
```

```python
import asyncio
from sdol import SDOL, CapabilityRegistry, ContextCompiler, GenericOLTPConnector, SemanticRouter, TrustScorer
from sdol.connectors.executor import MockQueryExecutor
from sdol.core.router.cost_estimator import CostEstimator
from sdol.core.router.intent_decomposer import IntentDecomposer
from sdol.core.router.query_planner import QueryPlanner

async def main():
    executor = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "name": "Alice", "email": "alice@example.com"},
    ])
    registry = CapabilityRegistry()
    registry.register(GenericOLTPConnector(executor=executor))

    sdol = SDOL(SemanticRouter(
        QueryPlanner(registry, IntentDecomposer(), CostEstimator()),
        ContextCompiler(TrustScorer()),
        registry,
    ))

    frame = await sdol.query(
        sdol.formulator.point_lookup("customer", {"customer_id": "C-1042"})
    )

    for slot in frame.slots:
        for elem in slot.elements:
            print(f"{elem.data}  trust={elem.trust.composite:.2f}  source={elem.provenance.source_system}")

asyncio.run(main())
```

---

## Built-In Typed Connectors

Connectors follow a three-tier architecture: **Foundation** (`BaseConnector`) → **Paradigm bases** (`BaseOLAPConnector`, `BaseOLTPConnector`, `BaseDocumentConnector`) → **Provider extensions**.

Core paradigm bases and generic connectors live in `src/sdol/connectors/`. Provider-specific extensions (Databricks) live in `extensions/databricks/` — keeping the core clean and unchanged as new providers are added.

| Paradigm | Base Class | Provider | Class | Location | Intent Types |
|----------|-----------|----------|-------|----------|-------------|
| OLAP | `BaseOLAPConnector` | Generic (Snowflake, etc.) | `GenericOLAPConnector` | `src/sdol/connectors/olap/` | `aggregate_analysis`, `temporal_trend` |
| OLAP | `BaseOLAPConnector` | Databricks SQL Warehouse | `DatabricksDBSQLConnector` | `extensions/databricks/olap/` | `aggregate_analysis`, `temporal_trend` |
| OLTP | `BaseOLTPConnector` | Generic (PostgreSQL, etc.) | `GenericOLTPConnector` | `src/sdol/connectors/oltp/` | `point_lookup`, `aggregate_analysis` |
| OLTP | `BaseOLTPConnector` | Databricks Lakebase | `DatabricksLakebaseConnector` | `extensions/databricks/oltp/` | `point_lookup`, `aggregate_analysis` |
| Document | `BaseDocumentConnector` | Generic (Pinecone, etc.) | `GenericDocumentConnector` | `src/sdol/connectors/document/` | `semantic_search` |
| Document | `BaseDocumentConnector` | Databricks Vector Search | `DatabricksVectorSearchConnector` | `extensions/databricks/document/` | `semantic_search` |

All connectors use the `QueryExecutor` protocol — swap `MockQueryExecutor` for a real implementation to connect to production databases. To add a new provider, subclass the appropriate paradigm base and implement only `synthesize_query()` + `get_performance()`.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Intent** | Declarative "what I want to know" — 8 typed intents, Pydantic-validated, never prescribes retrieval strategy |
| **Typed Connector** | Storage-aware adapter with a 4-stage pipeline: interpret → synthesize → execute → normalize |
| **Provenance Envelope** | Source system, retrieval method, consistency guarantee, precision class, staleness window |
| **Trust Score** | Composite 0–1 confidence from four dimensions: authority, consistency, freshness, precision |
| **Context Frame** | Typed slots (STRUCTURED, TEMPORAL, UNSTRUCTURED, etc.) + detected conflicts + aggregate stats |
| **Capability Registry** | Routes intents to connectors via suitability scoring (entity match, latency, capability alignment) |
| **Epistemic Tracker** | Generates LLM-injectable summaries of data confidence, low-trust warnings, and unresolved conflicts |

---

## Documentation

All documentation lives in the [`docs/`](docs/) directory. See the [documentation index](docs/README.md) for a complete guide.

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, setup, and basic working examples |
| [Architecture](docs/architecture.md) | Detailed system design — layers, data flow, type system |
| [Typed Connectors Guide](docs/typed-connectors-guide.md) | Using built-in connectors and building custom ones |
| [Databricks Guide](docs/databricks-guide.md) | DBSQL, Lakebase, and Vector Search integration with Unity Catalog |
| [Implementation Spec](docs/implementation-spec.md) | Original milestone-based implementation specification |

---

## Example Scripts

```bash
python examples/basic_query.py          # single-source point lookup
python examples/cross_source_query.py   # multi-paradigm composite query
python examples/with_mcp_server.py      # MCP adapter integration
```

---

## Project Stats

- 58 Python source files in `src/sdol/` (core only, no provider code)
- 10 Python source files in `extensions/databricks/`
- 23 test files, 254 tests passing
- 3 example scripts
- 5 benchmark scripts in `databricks_test/`
- Zero external runtime dependencies beyond Pydantic v2
