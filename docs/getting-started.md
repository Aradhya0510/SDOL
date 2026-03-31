# Getting Started with SDOL

This guide covers installation, setup, and basic working examples to get you up and running with SDOL (Semantic Data Orchestration Layer).

For deeper topics, see:
- [Architecture](architecture.md) — detailed system design
- [Typed Connectors Guide](typed-connectors-guide.md) — connector usage and building custom connectors
- [Databricks Guide](databricks-guide.md) — Databricks-specific integration via DBSQL, Lakebase, and Vector Search

---

## Prerequisites

- Python 3.11+
- `pydantic` 2.x
- `pytest` and `pytest-asyncio` (for running tests)

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify everything works:

```bash
python -m pytest tests/ -v
```

You should see all 254 tests passing.

---

## Core Concepts at a Glance

SDOL sits between your AI agent and its data sources. Instead of dumping raw query results into a flat context window, SDOL gives your agent a **structured, provenance-enriched context frame** — so it always knows where data came from, how fresh it is, and how much to trust it.

The flow is:

```
Your Agent  →  formulate intent  →  SDOL routes & executes  →  ContextFrame back to agent
```

Key pieces:

| Concept | One-liner |
|---------|-----------|
| **Intent** | Declarative "what I want to know" — never "how to get it" |
| **Typed Connector** | Storage-aware adapter (OLAP, OLTP, Document, etc.) |
| **Provenance Envelope** | Where data came from, its consistency, precision, freshness |
| **Trust Score** | Composite 0–1 confidence derived from provenance |
| **Context Frame** | Typed slots + conflicts + stats — replaces flat context |

---

## Example 1: Point Lookup

The simplest query — fetch a single entity by its identifier.

```python
import asyncio
from sdol import (
    SDOL, CapabilityRegistry, ContextCompiler,
    GenericOLTPConnector, SemanticRouter, TrustScorer,
)
from sdol.connectors.executor import MockQueryExecutor
from sdol.core.router.cost_estimator import CostEstimator
from sdol.core.router.intent_decomposer import IntentDecomposer
from sdol.core.router.query_planner import QueryPlanner

async def main():
    # MockQueryExecutor returns pre-configured data (swap for real DB in production)
    executor = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "name": "Alice", "email": "alice@example.com"},
    ])

    # Wire up the SDOL pipeline
    registry = CapabilityRegistry()
    registry.register(GenericOLTPConnector(executor=executor))

    compiler = ContextCompiler(TrustScorer())
    planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
    router = SemanticRouter(planner, compiler, registry)
    sdol = SDOL(router)

    # Build and execute the intent
    intent = sdol.formulator.point_lookup(
        "customer", {"customer_id": "C-1042"}, fields=["name", "email"]
    )
    frame = await sdol.query(intent)

    # Inspect the result
    for slot in frame.slots:
        for elem in slot.elements:
            print(f"Data:   {elem.data}")
            print(f"Trust:  {elem.trust.composite:.2f} ({elem.trust.label})")
            print(f"Source: {elem.provenance.source_system}")

asyncio.run(main())
```

Output:

```
Data:   {'customer_id': 'C-1042', 'name': 'Alice', 'email': 'alice@example.com'}
Trust:  0.68 (medium)
Source: postgres.production
```

---

## Example 2: Aggregate Analysis

Ask for statistical summaries across dimensions — routed to an OLAP connector.

```python
import asyncio
from sdol import (
    SDOL, CapabilityRegistry, ContextCompiler,
    GenericOLAPConnector, SemanticRouter, TrustScorer,
)
from sdol.connectors.executor import MockQueryExecutor
from sdol.core.router.cost_estimator import CostEstimator
from sdol.core.router.intent_decomposer import IntentDecomposer
from sdol.core.router.query_planner import QueryPlanner

async def main():
    executor = MockQueryExecutor(records=[
        {"region": "west", "sum_revenue": 2_500_000},
        {"region": "east", "sum_revenue": 1_800_000},
    ])

    registry = CapabilityRegistry()
    registry.register(GenericOLAPConnector(executor=executor))

    compiler = ContextCompiler(TrustScorer())
    planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
    router = SemanticRouter(planner, compiler, registry)
    sdol = SDOL(router)

    intent = sdol.formulator.aggregate_analysis(
        entity="orders",
        measures=[{"field": "revenue", "aggregation": "sum"}],
        dimensions=["region"],
        filters=[{"field": "status", "operator": "eq", "value": "active"}],
        order_by=[{"field": "sum_revenue", "direction": "desc"}],
    )
    frame = await sdol.query(intent)

    print(f"Elements: {frame.stats.total_elements}")
    print(f"Avg trust: {frame.stats.avg_trust_score:.2f}")
    for slot in frame.slots:
        for elem in slot.elements:
            print(f"  {elem.data}")

asyncio.run(main())
```

---

## Example 3: Temporal Trend

Retrieve change patterns over a time window.

```python
intent = sdol.formulator.temporal_trend(
    entity="api_metrics",
    metric="request_count",
    window={"relative": "last_90d"},
    granularity="1d",
)
frame = await sdol.query(intent)
```

---

## Example 4: Cross-Source Composite Query

Combine multiple intents into a single query. SDOL decomposes the composite, routes each sub-intent to the best connector, executes them (in parallel where possible), and assembles a unified context frame.

```python
import asyncio
from sdol import (
    SDOL, CapabilityRegistry, ContextCompiler,
    GenericOLAPConnector, GenericOLTPConnector, SemanticRouter, TrustScorer,
)
from sdol.connectors.executor import MockQueryExecutor
from sdol.core.router.cost_estimator import CostEstimator
from sdol.core.router.intent_decomposer import IntentDecomposer
from sdol.core.router.query_planner import QueryPlanner

async def main():
    olap_exec = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "churn_probability": 0.89, "region": "west"},
    ])
    oltp_exec = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "ticket_id": "T-501", "status": "unresolved"},
    ])

    registry = CapabilityRegistry()
    registry.register(GenericOLAPConnector(executor=olap_exec))
    registry.register(GenericOLTPConnector(executor=oltp_exec))

    compiler = ContextCompiler(TrustScorer())
    planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
    router = SemanticRouter(planner, compiler, registry)
    sdol = SDOL(router)

    intent = sdol.formulator.composite(
        sub_intents=[
            sdol.formulator.aggregate_analysis(
                entity="churn_scores",
                measures=[{"field": "churn_probability", "aggregation": "max"}],
                dimensions=["customer_id", "region"],
                having=[{"field": "churn_probability", "operator": "gt", "value": 0.7}],
            ),
            sdol.formulator.point_lookup("support_tickets", {"status": "unresolved"}),
        ],
        fusion_operator="intersect",
        fusion_key="customer_id",
    )

    frame = await sdol.query(intent)

    print(f"Elements: {frame.stats.total_elements}")
    print(f"Avg trust: {frame.stats.avg_trust_score:.2f}")
    print(f"Conflicts: {len(frame.conflicts)}")

asyncio.run(main())
```

---

## Example 5: Epistemic Context for LLM Prompts

After querying, SDOL generates an epistemic summary you can inject into your LLM's system prompt. This tells the model about data confidence, conflicts, and uncertainty.

```python
frame = await sdol.query(intent)

# Generate and inject
epistemic_context = sdol.get_epistemic_context()
system_prompt = f"""You are a data analyst assistant.

{epistemic_context}

Answer the user's question based on the data above. Flag any uncertainty."""
```

The output includes warnings like:

```
## Data Confidence Summary
- Total elements: 3
- Average trust: 0.72

LOW TRUST ELEMENTS:
  - Element from "staging.warehouse" — trust 0.42 (low)

UNRESOLVED CONFLICTS:
  - Field "revenue": $2.5M vs $2.4M — deferred to agent
```

---

## Working with the Context Frame

Every `sdol.query()` returns a `ContextFrame`. Here's how to work with it:

```python
frame = await sdol.query(intent)

# Typed slots group data by semantic category
for slot in frame.slots:
    print(f"Slot type: {slot.type}")       # STRUCTURED, TEMPORAL, UNSTRUCTURED, etc.
    print(f"Note: {slot.interpretation_notes}")
    for elem in slot.elements:
        print(f"  Data: {elem.data}")
        print(f"  Trust: {elem.trust.composite:.2f} ({elem.trust.label})")
        print(f"  Source: {elem.provenance.source_system}")
        print(f"  Precision: {elem.provenance.precision}")

# Conflicts detected across sources
for conflict in frame.conflicts:
    print(f"Conflict on '{conflict.field}': {conflict.value_a} vs {conflict.value_b}")
    print(f"  Strategy: {conflict.resolution.strategy}")

# Aggregate stats
print(f"Total elements: {frame.stats.total_elements}")
print(f"Average trust: {frame.stats.avg_trust_score:.2f}")
print(f"Slot counts: {frame.stats.slot_counts}")
```

---

## Running the Example Scripts

The `examples/` directory has ready-to-run scripts:

```bash
source .venv/bin/activate
python examples/basic_query.py          # single-source point lookup
python examples/cross_source_query.py   # multi-paradigm composite
python examples/with_mcp_server.py      # MCP adapter integration
```

---

## Next Steps

- [Architecture](architecture.md) — understand how the layers interact
- [Typed Connectors Guide](typed-connectors-guide.md) — use built-in connectors or build your own
- [Databricks Guide](databricks-guide.md) — DBSQL, Lakebase, and Vector Search integration
