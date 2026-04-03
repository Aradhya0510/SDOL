# Provena Architecture

This document provides a detailed technical walkthrough of Provena's internal architecture — layers, components, data flow, type system, and design decisions.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [End-to-End Data Flow](#end-to-end-data-flow)
3. [Layer 1: Type System](#layer-1-type-system)
   - [Intent Types](#intent-types)
   - [Provenance Types](#provenance-types)
   - [Context Types](#context-types)
   - [Capability Types](#capability-types)
   - [Error Hierarchy](#error-hierarchy)
4. [Layer 2: Agent SDK](#layer-2-agent-sdk)
   - [Provena Class](#provena-class)
   - [IntentFormulator](#intentformulator)
5. [Layer 3: Semantic Router and Query Planner](#layer-3-semantic-router-and-query-planner)
   - [SemanticRouter](#semanticrouter)
   - [QueryPlanner](#queryplanner)
   - [IntentDecomposer](#intentdecomposer)
   - [CostEstimator](#costestimator)
6. [Layer 4: Typed Connectors](#layer-4-typed-connectors)
   - [BaseConnector Pipeline](#baseconnector-pipeline)
   - [QueryExecutor Protocol](#queryexecutor-protocol)
   - [Capability Registry](#capability-registry)
7. [Layer 5: Context Assembly](#layer-5-context-assembly)
   - [ContextCompiler](#contextcompiler)
   - [ConflictDetector](#conflictdetector)
   - [ConflictResolver](#conflictresolver)
8. [Layer 6: Provenance and Trust](#layer-6-provenance-and-trust)
   - [TrustScorer](#trustscorer)
9. [Layer 7: Epistemic Tracking](#layer-7-epistemic-tracking)
10. [Layer 8: MCP Integration](#layer-8-mcp-integration)
11. [Layer 9: Cross-Source Join Optimizer](#layer-9-cross-source-join-optimizer)
12. [Directory Structure](#directory-structure)

---

## System Overview

Provena is organized into nine composable layers, each with a single responsibility:

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 2: Agent SDK                                              │
│  Provena (public API) + IntentFormulator (builders)              │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3: Semantic Router                                        │
│  SemanticRouter → QueryPlanner → IntentDecomposer                │
│                                → CostEstimator                   │
├──────────────────────────────────────────────────────────────────┤
│  Layer 4: Typed Connectors (three-tier)                          │
│  Foundation: BaseConnector (4-stage pipeline)                    │
│  Paradigm:   BaseOLAPConnector │ BaseOLTPConnector │ BaseDocConn │
│  Providers:  Generic OLAP/OLTP/Doc (core) + Databricks (extensions) │
│  CapabilityRegistry (routing + scoring)                          │
├──────────────────────────────────────────────────────────────────┤
│  Layer 5: Context Assembly                                       │
│  ContextCompiler → ConflictDetector → ConflictResolver           │
├──────────────────────────────────────────────────────────────────┤
│  Layer 6: Provenance & Trust                                     │
│  ProvenanceEnvelope + TrustScorer (4-dimension scoring)          │
├──────────────────────────────────────────────────────────────────┤
│  Layer 7: Epistemic Tracking                                     │
│  EpistemicTracker → generate_epistemic_prompt()                  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 8: MCP Integration                                        │
│  MCPAdapter + ResponseWrapper + ProtocolExtensions               │
├──────────────────────────────────────────────────────────────────┤
│  Layer 9: Join Optimizer                                         │
│  JoinOptimizer (push-down, hash-materialize, etc.)               │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1: Type System (foundation)                               │
│  Pydantic v2 models: Intent, Provenance, Context, Caps          │
└──────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Data Flow

A complete query traverses five phases:

```
Phase 1: FORMULATE
  Agent → IntentFormulator.point_lookup() → PointLookupIntent (Pydantic-validated)

Phase 2: PLAN
  Provena.query(intent)
    → SemanticRouter.route(intent)
      → QueryPlanner.plan(intent)
        → IntentDecomposer.decompose(intent)         # flatten composites
        → CapabilityRegistry.find_candidates(sub)     # score connectors
        → CostEstimator.estimate_latency/tokens       # budget estimates
        → ExecutionPlan (steps + topology + dependencies)

Phase 3: EXECUTE
  SemanticRouter._execute_plan(plan)
    → group steps into topological levels
    → per level: asyncio.gather(connector.execute(intent) for each step)
    → ConnectorResult per step (records + provenance + slot_type + entity_keys)

Phase 4: COMPILE
  SemanticRouter → ContextCompiler
    → for each record: TrustScorer.score(provenance) → TrustScore
    → ContextCompiler.add_element() → ContextElement
    → ConflictDetector.detect() → ContextConflict[]
    → ConflictResolver.resolve() per conflict
    → ContextCompiler.compile() → ContextFrame

Phase 5: EPISTEMIC
  Provena.query()
    → EpistemicTracker.ingest(frame)
    → Agent calls Provena.get_epistemic_context()
    → LLM-injectable prompt with confidence, warnings, conflicts
```

---

## Layer 1: Type System

All models use Pydantic v2 for combined type safety and runtime validation. The type system is the foundation — every other layer depends on it.

### Intent Types

**Location:** `src/provena/types/intent.py`

Eight intent types represent ontological categories of questions:

| Intent Type | Key Fields | Use Case |
|-------------|-----------|----------|
| `PointLookupIntent` | `entity`, `identifier`, `fields` | "Get customer C-1042" |
| `TemporalTrendIntent` | `entity`, `metric`, `window`, `granularity` | "Show API calls over 90 days" |
| `AggregateAnalysisIntent` | `entity`, `measures`, `dimensions`, `filters` | "Revenue by region" |
| `GraphTraversalIntent` | `start_node`, `edge_types`, `max_depth` | "Friends of friends of Alice" |
| `SemanticSearchIntent` | `query`, `collection`, `hybrid_weight` | "Find docs about caching" |
| `OntologyQueryIntent` | `subject`, `predicate`, `object` | "What is a mammal?" |
| `CompositeIntent` | `sub_intents`, `fusion_operator`, `fusion_key` | Combine any of the above |
| `EscapeHatchIntent` | `target_connector`, `raw_parameters` | Bypass the type system |

All intents share `BaseIntent` fields: `id`, `max_results`, `budget_ms`, `priority`.

The `Intent` discriminated union uses Pydantic's `Field(discriminator="type")` for runtime deserialization from dicts (e.g., LLM-generated JSON).

**Supporting models:**
- `TimeWindow` — at least one of `start`, `end`, `relative` (model validator)
- `FilterClause` — `field`, `operator` (9 operators), `value`
- `MeasureSpec` — `field`, `aggregation` (9 aggregation types including percentiles)
- `OrderSpec`, `NodeSpec`, `FusionOperator` (5 fusion strategies)

### Provenance Types

**Location:** `src/provena/types/provenance.py`

Every data element carries a `ProvenanceEnvelope`:

| Field | Type | Purpose |
|-------|------|---------|
| `source_system` | string | Identifies the origin (e.g., `databricks.sql_warehouse.prod`) |
| `retrieval_method` | enum | How data was obtained: `DIRECT_QUERY`, `COMPUTED_AGGREGATE`, `VECTOR_SIMILARITY`, `CACHE_HIT`, `ML_PREDICTION`, `GRAPH_TRAVERSAL`, `INFERENCE_ENGINE`, `MCP_PASSTHROUGH` |
| `consistency` | enum | Guarantee level: `STRONG`, `READ_COMMITTED`, `EVENTUAL`, `BEST_EFFORT` |
| `precision` | enum | Accuracy class: `EXACT`, `EXACT_AGGREGATE`, `ESTIMATED`, `PREDICTED`, `HEURISTIC`, `SIMILARITY_RANKED`, `LOGICALLY_ENTAILED` |
| `retrieved_at` | ISO string | When the data was fetched |
| `staleness_window_sec` | float | How long before data should be considered stale |
| `execution_ms` | float | Wall-clock query time |
| `result_truncated` | bool | Whether results were cut off by a limit |
| `total_available` | int | Total matching rows (before truncation) |

**Trust types:**
- `TrustDimensions` — four 0–1 scores: `source_authority`, `consistency_score`, `freshness_score`, `precision_score`
- `TrustScore` — `composite` (0–1), `dimensions`, `label` (`high` / `medium` / `low` / `uncertain`)

### Context Types

**Location:** `src/provena/types/context.py`

| Type | Purpose |
|------|---------|
| `ContextSlotType` | Semantic category: `STRUCTURED`, `RELATIONAL`, `TEMPORAL`, `UNSTRUCTURED`, `INFERRED` |
| `ContextElement` | Atomic unit: `data` + `provenance` + `trust` + `source_intent_id` + optional `entity_key` |
| `ContextSlot` | Groups elements by slot type with `interpretation_notes` |
| `ContextConflict` | Two elements disagreeing on a field, with a `ConflictResolution` |
| `ConflictResolution` | Strategy: `prefer_freshest`, `prefer_authoritative`, `prefer_strongest_consistency`, `defer_to_agent` |
| `ContextFrame` | The complete output: `slots` + `conflicts` + `stats` + `assembled_at` |

### Capability Types

**Location:** `src/provena/types/capability.py`

- `ConnectorCapabilities` — feature flags (7 booleans: aggregation, windowing, traversal, similarity, inference, temporal bucketing, full-text search)
- `ConnectorPerformance` — `estimated_latency_ms`, `max_result_cardinality`, `supports_batch_lookup`
- `ConnectorCapability` — complete declaration: `connector_id`, `connector_type`, `supported_intent_types`, `capabilities`, `performance`, `available_entities`

### Error Hierarchy

**Location:** `src/provena/types/errors.py`

All errors extend `ProvenaError(Exception)` with a typed `ProvenaErrorCode` and structured `context` dict:

| Error | Code | When |
|-------|------|------|
| `InvalidIntentError` | `INVALID_INTENT` | Connector receives an intent type it can't handle |
| `NoCapableConnectorError` | `NO_CAPABLE_CONNECTOR` | Registry has no connector for the intent type |
| `ConnectorTimeoutError` | `CONNECTOR_TIMEOUT` | Execution exceeds `budget_ms` |
| `MCPTransportError` | `MCP_TRANSPORT_ERROR` | MCP server communication failure |

---

## Layer 2: Agent SDK

### Provena Class

**Location:** `src/provena/agent/agent_sdk.py`

The public entry point. Intentionally thin — delegates everything to the router.

| Member | Description |
|--------|-------------|
| `__init__(router: SemanticRouter)` | Creates `IntentFormulator` and `EpistemicTracker` |
| `formulator` | Access to intent builders (`point_lookup`, `aggregate_analysis`, etc.) |
| `async query(intent) -> ContextFrame` | Delegates to `router.route(intent)`, then `tracker.ingest(frame)` |
| `get_epistemic_context() -> str` | Returns `tracker.generate_epistemic_prompt()` |
| `reset()` | Clears the epistemic tracker for a new session |

### IntentFormulator

**Location:** `src/provena/agent/intent_formulator.py`

Builder methods for all 8 intent types. Each method:
1. Generates a unique ID (`intent-{counter}-{timestamp_ms}`)
2. Converts raw dicts to Pydantic models (`MeasureSpec`, `FilterClause`, etc.)
3. Returns a fully validated intent (Pydantic raises `ValidationError` on bad data)

---

## Layer 3: Semantic Router and Query Planner

### SemanticRouter

**Location:** `src/provena/core/router/semantic_router.py`

The main orchestrator. `route(intent)` does three things:

1. **Plan** — `QueryPlanner.plan(intent)` → `ExecutionPlan`
2. **Execute** — `_execute_plan(plan)` → run connector steps in topological order with `asyncio.gather` for parallelism within each level
3. **Compile** — feed `ConnectorResult`s into `ContextCompiler` → `ContextFrame`

**Topological execution:** Steps are grouped by dependency level. Steps with no dependencies run in parallel at level 0. Steps depending on level-0 steps run at level 1, etc. This naturally handles `CompositeIntent` with `sequence` fusion.

**Error handling:** If a connector is missing or throws, the step is recorded as an `ExecutionError` but other steps continue (partial results).

### QueryPlanner

**Location:** `src/provena/core/router/query_planner.py`

Builds an `ExecutionPlan` from an intent:

1. **Decompose** — `IntentDecomposer.decompose(intent)` flattens composites into a list of atomic intents
2. **Route** — for each sub-intent, `CapabilityRegistry.find_candidates()` → pick the top-scoring candidate
3. **Estimate** — `CostEstimator.estimate_latency()` and `estimate_tokens()` per step
4. **Wire dependencies** — for composites, `IntentDecomposer.analyze_dependencies()` determines execution order
5. **Detect parallelism** — `has_parallel_steps = True` if any step has no dependencies beyond the first

Output: `ExecutionPlan` with `steps`, `estimated_total_ms`, `has_parallel_steps`, optional `fusion_strategy`.

### IntentDecomposer

**Location:** `src/provena/core/router/intent_decomposer.py`

Two responsibilities:

1. **`decompose(intent)`** — non-composite returns `[intent]`; composite recursively flattens `sub_intents` (depth-first)
2. **`analyze_dependencies(sub_intents, fusion_operator, fusion_key)`** — builds a dependency map:
   - `SEQUENCE` fusion: linear chain (`step[i]` depends on `step[i-1]`)
   - With `fusion_key`: intents with matching `join_key` are "providers"; others depend on all providers
   - Default: no dependencies (fully parallel)

### CostEstimator

**Location:** `src/provena/core/router/cost_estimator.py`

Estimates execution cost at plan time:

| Method | Logic |
|--------|-------|
| `estimate_latency(capability)` | `estimated_latency_ms + overhead_ms`, optionally averaged with rolling historical actuals |
| `estimate_tokens(capability)` | `min(max_result_cardinality, 1000) * tokens_per_record` |
| `estimate_total_ms(steps)` | Groups steps by topological level; sums the **max** latency per level (parallelism-aware) |
| `record_actual(actual_ms)` | Appends to history (last 100) for future estimates |

---

## Layer 4: Typed Connectors

Connectors follow a three-tier architecture:

1. **Foundation** — `BaseConnector` defines the 4-stage pipeline contract and `QueryExecutor` protocol. These rarely change.
2. **Paradigm bases** — `BaseOLAPConnector`, `BaseOLTPConnector`, `BaseDocumentConnector` encode what it *means* to be a connector of that paradigm: supported intent types, shared `interpret_intent` validation, shared `normalize_result` with paradigm-appropriate provenance defaults. Providers only override `synthesize_query()` and `get_performance()`.
3. **Provider extensions** — Generic connectors (`GenericOLAPConnector`, `GenericOLTPConnector`, `GenericDocumentConnector`) live in `src/provena/connectors/` under their paradigm directories. Databricks-specific connectors (`DatabricksDBSQLConnector`, `DatabricksLakebaseConnector`, `DatabricksVectorSearchConnector`) live in `src/provena/extensions/databricks/` and are installed via `pip install provena[databricks]`.

### BaseConnector Pipeline (Foundation)

**Location:** `src/provena/connectors/base_connector.py`

Abstract base class with a concrete `execute(intent)` method that orchestrates the 4-stage pipeline:

```
execute(intent):
    start = time.monotonic()
    params = self.interpret_intent(intent)      # stage 1: validate + extract
    query = self.synthesize_query(params)        # stage 2: build native query
    raw = await self.execute_query(query)        # stage 3: run via executor
    elapsed_ms = (time.monotonic() - start) * 1000
    return self.normalize_result(raw, intent, elapsed_ms)  # stage 4: attach provenance
```

Abstract methods every connector must implement:
- `id` (property), `connector_type` (property)
- `get_capabilities() -> ConnectorCapability`
- `check_health() -> ConnectorHealth`
- `interpret_intent`, `synthesize_query`, `execute_query`, `normalize_result`

Concrete method `can_handle(intent)` checks `intent.type in get_capabilities().supported_intent_types`.

### Paradigm Base Classes

**Locations:** `src/provena/connectors/olap/base.py`, `src/provena/connectors/oltp/base.py`, `src/provena/connectors/document/base.py`

Each paradigm base:
- Locks down `connector_type` (e.g., `"olap"`)
- Provides shared `interpret_intent()` — validates intent types, rejects unsupported ones
- Provides shared `normalize_result()` — builds `ConnectorResult` with paradigm-appropriate provenance defaults (consistency, staleness, retrieval method, precision)
- Provides shared `get_capabilities()` with an abstract `get_performance()` hook for provider-specific latency/cardinality
- Declares overridable properties: `default_staleness_sec`, `default_consistency`, `source_system`
- Leaves `synthesize_query()` abstract — this is the true provider extension point

Shared utilities (`OPERATOR_MAP`, `qualify_table`, `extract_entity_keys`) live in `src/provena/connectors/sql_utils.py`.

### Provider Extensions

Provider-specific connectors live in `src/provena/extensions/` and use the **extras/optional dependencies** pattern — install with `pip install provena[<provider>]`.

Adding a new provider (e.g., BigQuery for OLAP, Postgres for OLTP) requires:
1. One new file under `src/provena/extensions/<provider>/<paradigm>/` (e.g., `src/provena/extensions/bigquery/olap/bigquery.py`)
2. One query builder file with the native query dataclass + builder functions
3. Subclass the paradigm base, implement `synthesize_query()` + `get_performance()`
4. Add an optional-dependency group to `pyproject.toml` (e.g., `bigquery = ["google-cloud-bigquery>=3.0"]`)
5. Register it — routing, trust scoring, context compilation, and MCP wrapping work automatically

Import path: `provena.extensions.<provider>.<paradigm>.<module>` (e.g., `provena.extensions.databricks.olap.dbsql`). All Databricks connectors are also re-exported from `provena.__init__` for convenience.

### QueryExecutor Protocol

**Location:** `src/provena/connectors/executor.py`

```python
class QueryExecutor(Protocol):
    async def execute(self, query: Any) -> dict[str, Any]:
        """Returns {"records": [...], "meta": {...}}"""
```

Connectors accept any `QueryExecutor` — this is the seam where you swap mock backends for real databases. The protocol uses structural typing (no inheritance required).

`MockQueryExecutor` stores pre-configured `records` and `meta`, exposes `last_query` for test assertions.

### Capability Registry

**Location:** `src/provena/connectors/capability_registry.py`

| Method | Description |
|--------|-------------|
| `register(connector)` | Reads capabilities, indexes by `connector_id` |
| `find_candidates(intent)` | Filters by `supported_intent_types`, scores via `_compute_suitability`, sorts descending |
| `get_connector(id)` | Direct lookup |
| `list_capabilities()` | All registered capabilities |
| `unregister(id)` | Remove a connector |

**Suitability scoring** (three factors, each 0–1):
- **Entity match (40%):** intent entity in `available_entities` → 0.4; empty entities list → 0.2; mismatch → 0.0
- **Latency (30%):** `max(0, 1 - latency_ms / 5000) * 0.3`
- **Capability alignment (30%):** fraction of intent-relevant feature flags that are True (mapped per intent type)

---

## Layer 5: Context Assembly

### ContextCompiler

**Location:** `src/provena/core/context/context_compiler.py`

Assembles raw connector results into a `ContextFrame`:

1. **`add_element(input: CompilerInput)`** — scores trust via `TrustScorer`, builds `ContextElement` with generated ID
2. **`compile()`** — groups elements into `ContextSlot`s by `ContextSlotType`, assigns interpretation notes per slot type, runs conflict detection/resolution, computes stats

**Interpretation notes** per slot type:
- `STRUCTURED` → "Tabular data. Numbers are precise within query scope. NULLs mean absent, not unknown. Trust provenance precision class for confidence."
- `TEMPORAL` → "Time-series data. Always check window and granularity. Trends are computed within the stated window — do not extrapolate beyond it."
- `UNSTRUCTURED` → "Natural language text. Apply standard NLP-level caution. Claims are assertions, not verified facts. Cross-reference with structured data."
- `RELATIONAL` → "Relationship data. Absence of an edge means not-found, not does-not-exist. Check traversal depth limits before assuming completeness."
- `INFERRED` → "Derived by formal reasoning. Validity depends on ontology correctness. Uses open-world assumption: absence of entailment is not negation."

### ConflictDetector

**Location:** `src/provena/core/context/conflict_detector.py`

Groups elements by `entity_key`. For elements from **different source systems** with the same entity key, compares dict `data` shared keys. Any value mismatch creates a `ContextConflict` with initial resolution `defer_to_agent`.

### ConflictResolver

**Location:** `src/provena/core/context/conflict_resolver.py`

Applies heuristic resolution to each conflict in priority order:

1. **Freshness** — if age gap > 10x the minimum staleness window → `prefer_freshest`
2. **Authority** — if one side has source authority > 0.9 and the other < 0.7 → `prefer_authoritative`
3. **Consistency** — if consistency level gap >= 2 (e.g., `strong` vs `eventual`) → `prefer_strongest_consistency`
4. **Fallback** — `defer_to_agent` (flagged for epistemic tracking)

---

## Layer 6: Provenance and Trust

### TrustScorer

**Location:** `src/provena/core/provenance/trust_scorer.py`

Produces a `TrustScore` from a `ProvenanceEnvelope` across four dimensions:

| Dimension | How It's Computed | Default Weight |
|-----------|-------------------|----------------|
| **Authority** | `source_authority_map.get(source_system, 0.5)` | 0.2 |
| **Consistency** | Lookup: `strong → 1.0`, `read_committed → 0.8`, `eventual → 0.5`, `best_effort → 0.2` | 0.3 |
| **Freshness** | `max(0, 1 - (age / staleness_window) / 2)` — null staleness → 0.5 | 0.2 |
| **Precision** | Lookup: `exact → 1.0`, `exact_aggregate → 0.95`, `logically_entailed → 0.9`, `estimated → 0.6`, `similarity_ranked → 0.55`, `predicted → 0.5`, `heuristic → 0.3` | 0.3 |

**Composite** = weighted sum, clamped to [0, 1].

**Label** thresholds: >= 0.8 → `high`, >= 0.55 → `medium`, >= 0.3 → `low`, else `uncertain`.

`TrustScorerConfig` allows custom weights and authority maps per source system.

---

## Layer 7: Epistemic Tracking

**Location:** `src/provena/core/epistemic/epistemic_tracker.py`

The `EpistemicTracker` maintains a session-level view of data quality across multiple queries:

| Method | Description |
|--------|-------------|
| `ingest(frame)` | Appends a `ContextFrame` to the session |
| `get_trust(element_id)` | Look up trust for a specific element across all frames |
| `get_low_trust_elements(threshold=0.4)` | Elements below the trust threshold |
| `get_unresolved_conflicts()` | Conflicts with `defer_to_agent` strategy |
| `generate_epistemic_prompt()` | Markdown-formatted summary for LLM injection |

The generated prompt includes:
- Total element count and average trust
- Up to 5 low-trust elements with source details
- Up to 3 unresolved conflicts with field/value details

---

## Layer 8: MCP Integration

**Location:** `src/provena/mcp/`

Provena can wrap results for Model Context Protocol (MCP) transport.

### MCPAdapter

Manages MCP server registrations and dispatches tool calls:
- `register_server(config)` — register an MCP server with declared consistency/precision defaults
- `call(server_id, tool_call)` — send a tool call via the transport protocol

### ResponseWrapper

Converts MCP responses into `ProvenanceEnvelope` with a three-level fallback:
1. **Provena metadata** in the response (highest priority — server explicitly provides provenance)
2. **Server defaults** from `MCPServerConfig` (declared consistency, precision, staleness)
3. **Conservative defaults** (`BEST_EFFORT` consistency, `ESTIMATED` precision, no staleness guarantee)

---

## Layer 9: Cross-Source Join Optimizer

**Location:** `src/provena/core/router/join_optimizer.py`

The `JoinOptimizer` provides strategies for joining results across different connectors. It is available as a library component (not automatically invoked by `SemanticRouter.route()`).

### Join Strategies

| Strategy | When Used | Description |
|----------|-----------|-------------|
| `PUSH_DOWN` | Same connector for both sides | Let the database handle the join natively |
| `CONTEXT_WINDOW_JOIN` | Both sides < 50 rows | Small enough to join in the agent's context window |
| `HASH_MATERIALIZE` | Different connectors, larger result sets | Build a hash table from the smaller side, probe with the larger side |
| `NESTED_LOOKUP` | Reserved | Nested loop with per-key lookups |

### Strategy Selection

```
Same connector?          → PUSH_DOWN
Both < 50 rows?          → CONTEXT_WINDOW_JOIN
Different connectors?    → HASH_MATERIALIZE (build on smaller side)
```

`execute_hash_materialize` collects join keys from the build result, injects them as an `IN` filter on the probe intent, then executes the probe step.

---

## Directory Structure

```
src/provena/                                 # Core package + provider extensions
├── __init__.py                           # public exports (re-exports extensions for convenience)
├── extensions/
│   ├── __init__.py                       # namespace marker
│   └── databricks/                       # pip install provena[databricks]
│       ├── olap/dbsql.py                 # DatabricksDBSQLConnector
│       ├── oltp/lakebase.py              # DatabricksLakebaseConnector
│       └── document/vector_search.py     # DatabricksVectorSearchConnector
├── agent/
│   ├── agent_sdk.py                      # Provena class (public API)
│   └── intent_formulator.py              # intent builders
├── connectors/
│   ├── base_connector.py                 # BaseConnector ABC (foundation)
│   ├── capability_registry.py            # routing + scoring (foundation)
│   ├── executor.py                       # QueryExecutor protocol + mock (foundation)
│   ├── sql_utils.py                      # shared OPERATOR_MAP, qualify_table, extract_entity_keys
│   ├── olap/
│   │   ├── base.py                       # BaseOLAPConnector (paradigm base)
│   │   ├── generic.py                    # GenericOLAPConnector
│   │   └── query.py                      # generic OLAP SQL builder
│   ├── oltp/
│   │   ├── base.py                       # BaseOLTPConnector (paradigm base)
│   │   ├── generic.py                    # GenericOLTPConnector
│   │   └── query.py                      # generic OLTP SQL builder
│   └── document/
│       ├── base.py                       # BaseDocumentConnector (paradigm base)
│       ├── generic.py                    # GenericDocumentConnector
│       └── query.py                      # hybrid retrieval builder
├── core/
│   ├── context/
│   │   ├── context_compiler.py           # assembles ContextFrame
│   │   ├── conflict_detector.py          # cross-source conflict detection
│   │   ├── conflict_resolver.py          # heuristic resolution
│   │   └── typed_slot.py                # interpretation notes per slot type
│   ├── epistemic/
│   │   └── epistemic_tracker.py          # session-level confidence tracking
│   ├── provenance/
│   │   ├── envelope.py                   # ProvenanceEnvelope factory functions
│   │   └── trust_scorer.py               # 4-dimension trust scoring
│   └── router/
│       ├── semantic_router.py            # main orchestrator
│       ├── query_planner.py              # execution plan builder
│       ├── intent_decomposer.py          # composite flattener
│       ├── cost_estimator.py             # latency/token estimation
│       └── join_optimizer.py             # cross-source join strategies
├── mcp/
│   ├── mcp_adapter.py                    # MCP server management
│   ├── response_wrapper.py               # MCP → ProvenanceEnvelope
│   └── protocol_extensions.py            # Provena metadata envelope for MCP
├── types/
│   ├── intent.py                         # 8 intent types + discriminated union
│   ├── provenance.py                     # provenance enums + envelope + trust
│   ├── context.py                        # context frame, slots, conflicts
│   ├── capability.py                     # connector capability declarations
│   ├── connector.py                      # connector result + health types
│   ├── errors.py                         # typed error hierarchy
│   └── router.py                         # execution plan + step types
└── utils/
    ├── hashing.py                        # deterministic hashing
    ├── timer.py                          # execution timer context manager
    └── logger.py                         # structured logger

tests/                                    # Unit + integration tests
├── unit/                                # per-module unit tests
└── integration/                         # end-to-end pipeline tests
```

---

## Design Principles

1. **Declare what, not how** — intents are ontological questions, never execution plans. The same intent routes differently depending on registered connectors.

2. **Provenance is not optional** — every piece of data entering the context carries its full lineage. This is the foundation for trust scoring, conflict resolution, and epistemic tracking.

3. **Pluggable at every seam** — the connector layer uses a three-tier architecture (foundation → paradigm base → provider extension) so adding a new storage backend requires only `synthesize_query()` and `get_performance()`; trust scoring has configurable weights and authority maps; conflict resolution strategies are composable; MCP integration is optional.

4. **Parallel by default** — the router groups execution steps into topological levels and runs each level with `asyncio.gather`. Independent sub-intents naturally parallelize.

5. **Graceful degradation** — if one connector fails in a composite query, the others still contribute results. Errors are recorded in `ExecutionResult`, not propagated as exceptions.

6. **Type safety end-to-end** — Pydantic v2 validates intents at construction, discriminated unions handle deserialization from LLM output, and typed errors carry structured context for debugging.
