# Provena (formerly SDOL) — Implementation Specification
## Complete Implementation Specification for Coding Agents (Python Edition)

> **What this document is:** A step-by-step implementation plan for building Provena, a middleware layer that sits between AI agents and data sources. It adds storage-aware query planning, typed connectors, and provenance-enriched context assembly on top of the existing MCP (Model Context Protocol) transport.
>
> **How to use this document:** Implement each milestone in order. Each milestone is self-contained and testable. Do NOT skip ahead. Complete all files in a milestone, run its tests, then proceed.
>
> **Tech stack:** Python 3.12+, Pydantic v2 for validation and schemas, pytest for testing, asyncio for concurrent execution, no framework dependencies.
>
> **Why Python:** The agent ecosystem (LangChain, LlamaIndex, CrewAI, AutoGen) is Python-first. Building Provena in Python means the Agent SDK can be embedded directly inside agent processes — no cross-language serialization boundary at the critical point where epistemic context enters the agent's prompt. The data engineering community that will contribute typed connectors also lives in Python.

---

## TABLE OF CONTENTS

1. [Project Structure](#1-project-structure)
2. [Milestone 0: Project Bootstrap](#2-milestone-0-project-bootstrap)
3. [Milestone 1: Core Type System & Intent Schema](#3-milestone-1-core-type-system--intent-schema)
4. [Milestone 2: Provenance Envelope & Trust Scoring](#4-milestone-2-provenance-envelope--trust-scoring)
5. [Milestone 3: Context Compiler](#5-milestone-3-context-compiler)
6. [Milestone 4: Typed Connector Interface & Base Class](#6-milestone-4-typed-connector-interface--base-class)
7. [Milestone 5: Reference Connectors (OLAP, OLTP, Document)](#7-milestone-5-reference-connectors)
8. [Milestone 6: Capability Registry](#8-milestone-6-capability-registry)
9. [Milestone 7: Semantic Router & Query Planner](#9-milestone-7-semantic-router--query-planner)
10. [Milestone 8: MCP Integration Layer](#10-milestone-8-mcp-integration-layer)
11. [Milestone 9: Epistemic Tracker](#11-milestone-9-epistemic-tracker)
12. [Milestone 10: Agent SDK & Intent Formulator](#12-milestone-10-agent-sdk--intent-formulator)
13. [Milestone 11: Cross-Source Join Optimizer](#13-milestone-11-cross-source-join-optimizer)
14. [Milestone 12: End-to-End Integration & CLI](#14-milestone-12-end-to-end-integration--cli)
15. [Appendix A: Full Type Reference](#appendix-a-full-type-reference)
16. [Appendix B: Error Handling Strategy](#appendix-b-error-handling-strategy)
17. [Appendix C: Testing Strategy](#appendix-c-testing-strategy)

---

## 1. Project Structure

```
provena/
├── pyproject.toml
├── README.md
├── src/
│   └── provena/
│       ├── __init__.py                         # Public API exports
│       ├── py.typed                            # PEP 561 marker
│       ├── types/
│       │   ├── __init__.py
│       │   ├── intent.py                       # Intent model definitions
│       │   ├── provenance.py                   # Provenance envelope models
│       │   ├── context.py                      # Context slot & frame models
│       │   ├── connector.py                    # Connector interface models
│       │   ├── capability.py                   # Capability registry models
│       │   ├── router.py                       # Router & query plan models
│       │   └── errors.py                       # Error class hierarchy
│       ├── core/
│       │   ├── __init__.py
│       │   ├── provenance/
│       │   │   ├── __init__.py
│       │   │   ├── envelope.py                 # Envelope factory functions
│       │   │   └── trust_scorer.py             # TrustScorer implementation
│       │   ├── context/
│       │   │   ├── __init__.py
│       │   │   ├── context_compiler.py         # ContextCompiler class
│       │   │   ├── typed_slot.py               # TypedSlot factory logic
│       │   │   ├── conflict_detector.py        # ConflictDetector class
│       │   │   └── conflict_resolver.py        # ConflictResolver class
│       │   ├── router/
│       │   │   ├── __init__.py
│       │   │   ├── semantic_router.py          # SemanticRouter class
│       │   │   ├── intent_decomposer.py        # IntentDecomposer class
│       │   │   ├── query_planner.py            # QueryPlanner class
│       │   │   ├── cost_estimator.py           # CostEstimator class
│       │   │   └── join_optimizer.py           # CrossSourceJoinOptimizer
│       │   └── epistemic/
│       │       ├── __init__.py
│       │       └── epistemic_tracker.py        # EpistemicTracker class
│       ├── connectors/
│       │   ├── __init__.py
│       │   ├── base_connector.py               # Abstract BaseConnector (ABC)
│       │   ├── capability_registry.py          # CapabilityRegistry class
│       │   ├── olap/
│       │   │   ├── __init__.py
│       │   │   ├── olap_connector.py           # OLAPConnector class
│       │   │   └── query_builder.py            # SQL/analytical query builder
│       │   ├── oltp/
│       │   │   ├── __init__.py
│       │   │   ├── oltp_connector.py           # OLTPConnector class
│       │   │   └── query_builder.py            # Transactional query builder
│       │   ├── document/
│       │   │   ├── __init__.py
│       │   │   ├── document_connector.py       # DocumentConnector class
│       │   │   └── query_builder.py            # Search query builder
│       │   ├── graph/
│       │   │   ├── __init__.py
│       │   │   ├── graph_connector.py          # GraphConnector class
│       │   │   └── query_builder.py            # Cypher/Gremlin builder
│       │   ├── timeseries/
│       │   │   ├── __init__.py
│       │   │   ├── timeseries_connector.py     # TimeSeriesConnector class
│       │   │   └── query_builder.py            # InfluxQL/PromQL builder
│       │   └── ontology/
│       │       ├── __init__.py
│       │       ├── ontology_connector.py       # OntologyConnector class
│       │       └── query_builder.py            # SPARQL builder
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── mcp_adapter.py                  # MCPAdapter wraps MCP client
│       │   ├── response_wrapper.py             # Adds metadata to MCP responses
│       │   └── protocol_extensions.py          # Provena metadata envelope for MCP
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── intent_formulator.py            # IntentFormulator class
│       │   └── agent_sdk.py                    # High-level SDK for agent frameworks
│       └── utils/
│           ├── __init__.py
│           ├── logger.py                       # Structured logger
│           ├── timer.py                        # Execution timer context manager
│           └── hashing.py                      # Deterministic hashing for entity resolution
├── tests/
│   ├── __init__.py
│   ├── conftest.py                             # Shared fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── types/
│   │   │   ├── __init__.py
│   │   │   └── test_intent_schema.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── provenance/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test_envelope.py
│   │   │   │   └── test_trust_scorer.py
│   │   │   ├── context/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test_context_compiler.py
│   │   │   │   ├── test_typed_slot.py
│   │   │   │   └── test_conflict_detector.py
│   │   │   ├── router/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test_semantic_router.py
│   │   │   │   ├── test_intent_decomposer.py
│   │   │   │   ├── test_query_planner.py
│   │   │   │   └── test_join_optimizer.py
│   │   │   └── epistemic/
│   │   │       ├── __init__.py
│   │   │       └── test_epistemic_tracker.py
│   │   ├── connectors/
│   │   │   ├── __init__.py
│   │   │   ├── test_olap_connector.py
│   │   │   ├── test_oltp_connector.py
│   │   │   └── test_document_connector.py
│   │   ├── mcp/
│   │   │   ├── __init__.py
│   │   │   ├── test_mcp_adapter.py
│   │   │   └── test_response_wrapper.py
│   │   └── agent/
│   │       ├── __init__.py
│   │       ├── test_intent_formulator.py
│   │       └── test_agent_sdk.py
│   └── integration/
│       ├── __init__.py
│       └── test_end_to_end.py
├── examples/
│   ├── basic_query.py                          # Simple single-source query
│   ├── cross_source_query.py                   # Multi-paradigm composite query
│   └── with_mcp_server.py                      # Integration with real MCP server
└── docs/
    ├── CONNECTOR_GUIDE.md                      # How to build a new connector
    └── INTENT_REFERENCE.md                     # Complete intent type reference
```

---

## 2. Milestone 0: Project Bootstrap

### Goal
Initialize the project with Python packaging, testing, linting, and type checking configured.

### Step 0.1: Create `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "provena"
version = "0.2.0"
description = "Provena — Epistemic provenance for AI agents"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "mypy>=1.10.0",
    "ruff>=0.5.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/provena"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
plugins = ["pydantic.mypy"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "TCH"]
```

### Step 0.2: Create directory structure

```bash
mkdir -p src/provena/{types,core/{provenance,context,router,epistemic},connectors/{olap,oltp,document,graph,timeseries,ontology},mcp,agent,utils}
mkdir -p tests/{unit/{types,core/{provenance,context,router,epistemic},connectors,mcp,agent},integration}
touch src/provena/py.typed

# Create all __init__.py files
find src/provena -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;
```

### Step 0.3: Create virtual environment and install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Step 0.4: Verify toolchain

```bash
pytest              # should run 0 tests, exit 0
mypy src/provena       # should pass with no errors
ruff check src/     # should pass with no issues
```

### Completion Check
- `pip install -e ".[dev]"` succeeds
- `pytest` exits cleanly with 0 tests collected
- `mypy src/provena` passes strict mode
- All directories and `__init__.py` files exist

---

## 3. Milestone 1: Core Type System & Intent Schema

### Goal
Define all core Pydantic models for the intent system. These models serve as BOTH the type definitions AND runtime validators — Pydantic v2 gives us both in one place.

### File: `src/provena/types/intent.py`

```python
"""
Intent types represent WHAT the agent wants to know, never HOW to retrieve it.
Each intent type maps to a different ontological category of question.

All models use Pydantic v2 for combined type safety + runtime validation.
LLM-generated intents WILL be malformed sometimes — Pydantic catches that.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


class IntentType(StrEnum):
    POINT_LOOKUP = "point_lookup"
    TEMPORAL_TREND = "temporal_trend"
    AGGREGATE_ANALYSIS = "aggregate_analysis"
    GRAPH_TRAVERSAL = "graph_traversal"
    SEMANTIC_SEARCH = "semantic_search"
    ONTOLOGY_QUERY = "ontology_query"
    COMPOSITE = "composite"
    ESCAPE_HATCH = "escape_hatch"


# ── Supporting models ──


class TimeWindow(BaseModel):
    """At least one of start, end, or relative must be provided."""

    start: str | None = None  # ISO 8601 or relative like "-90d"
    end: str | None = None
    relative: str | None = None  # shorthand: "last_90d", "last_7d", "today"

    @model_validator(mode="after")
    def at_least_one_field(self) -> TimeWindow:
        if not any([self.start, self.end, self.relative]):
            raise ValueError("TimeWindow must have at least one of: start, end, relative")
        return self


class FilterClause(BaseModel):
    field: str = Field(min_length=1)
    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "in", "contains", "exists"]
    value: Any


class MeasureSpec(BaseModel):
    field: str = Field(min_length=1)
    aggregation: Literal[
        "sum", "avg", "min", "max", "count", "count_distinct", "p50", "p95", "p99"
    ]
    alias: str | None = None


class OrderSpec(BaseModel):
    field: str
    direction: Literal["asc", "desc"]


class NodeSpec(BaseModel):
    type: str | None = None
    identifier: dict[str, str | int] | None = None
    filters: list[FilterClause] | None = None


class FusionOperator(StrEnum):
    INTERSECT = "intersect"
    UNION = "union"
    SEQUENCE = "sequence"
    LEFT_JOIN = "left_join"
    CROSS_ENRICH = "cross_enrich"


# ── Intent models ──


class BaseIntent(BaseModel):
    """Base fields shared by all intents."""

    id: str = Field(min_length=1)
    max_results: int | None = Field(default=None, gt=0)
    budget_ms: int | None = Field(default=None, gt=0)
    priority: float | None = None


class PointLookupIntent(BaseIntent):
    """Retrieve current state of a specific entity by identifier."""

    type: Literal["point_lookup"] = "point_lookup"
    entity: str = Field(min_length=1)
    identifier: dict[str, str | int]
    fields: list[str] | None = None


class TemporalTrendIntent(BaseIntent):
    """Retrieve change patterns over a time window."""

    type: Literal["temporal_trend"] = "temporal_trend"
    entity: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    window: TimeWindow
    granularity: str | None = None  # e.g. "1h", "1d", "1w"
    filters: list[FilterClause] | None = None
    direction: Literal["rising", "falling", "any"] | None = None
    join_key: str | None = None


class AggregateAnalysisIntent(BaseIntent):
    """Retrieve statistical summaries across dimensions."""

    type: Literal["aggregate_analysis"] = "aggregate_analysis"
    entity: str = Field(min_length=1)
    measures: list[MeasureSpec] = Field(min_length=1)
    dimensions: list[str] = Field(min_length=1)
    filters: list[FilterClause] | None = None
    order_by: list[OrderSpec] | None = None
    having: list[FilterClause] | None = None


class GraphTraversalIntent(BaseIntent):
    """Retrieve entity relationships within depth and filter constraints."""

    type: Literal["graph_traversal"] = "graph_traversal"
    start_node: NodeSpec
    edge_types: list[str] | None = None
    max_depth: int = Field(ge=1, le=10)
    direction: Literal["outbound", "inbound", "both"] | None = None
    node_filters: list[FilterClause] | None = None
    return_paths: bool = False


class SemanticSearchIntent(BaseIntent):
    """Retrieve information by meaning similarity."""

    type: Literal["semantic_search"] = "semantic_search"
    query: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    filters: list[FilterClause] | None = None
    hybrid_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    rerank: bool = False


class OntologyQueryIntent(BaseIntent):
    """Retrieve entailments and class-based inferences."""

    type: Literal["ontology_query"] = "ontology_query"
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    inference_depth: int | None = Field(default=None, ge=0)
    include_entailments: bool = True


class CompositeIntent(BaseIntent):
    """Combines sub-intents with fusion operators."""

    type: Literal["composite"] = "composite"
    sub_intents: list[Intent]
    fusion_operator: FusionOperator
    fusion_key: str | None = None


class EscapeHatchIntent(BaseIntent):
    """Bypass for queries that don't fit the type system."""

    type: Literal["escape_hatch"] = "escape_hatch"
    target_connector: str
    raw_parameters: dict[str, Any]
    description: str  # human-readable intent description for logging


# ── Discriminated union ──

Intent = Annotated[
    Union[
        PointLookupIntent,
        TemporalTrendIntent,
        AggregateAnalysisIntent,
        GraphTraversalIntent,
        SemanticSearchIntent,
        OntologyQueryIntent,
        CompositeIntent,
        EscapeHatchIntent,
    ],
    Field(discriminator="type"),
]

# Allow CompositeIntent to reference the union
CompositeIntent.model_rebuild()


def validate_intent(data: dict[str, Any]) -> BaseIntent:
    """
    Validate raw dict (e.g. from LLM output) into a typed Intent.
    Raises pydantic.ValidationError on invalid data.
    """
    from pydantic import TypeAdapter

    adapter = TypeAdapter(Intent)
    return adapter.validate_python(data)
```

### File: `src/provena/types/errors.py`

```python
"""
Error hierarchy for Provena.
Every public method that can fail throws a typed ProvenaError subclass.
Errors always carry context for debugging.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ProvenaErrorCode(StrEnum):
    INVALID_INTENT = "INVALID_INTENT"
    NO_CAPABLE_CONNECTOR = "NO_CAPABLE_CONNECTOR"
    CONNECTOR_TIMEOUT = "CONNECTOR_TIMEOUT"
    CONNECTOR_ERROR = "CONNECTOR_ERROR"
    DECOMPOSITION_FAILED = "DECOMPOSITION_FAILED"
    CONFLICT_UNRESOLVABLE = "CONFLICT_UNRESOLVABLE"
    MCP_TRANSPORT_ERROR = "MCP_TRANSPORT_ERROR"
    TRUST_BELOW_THRESHOLD = "TRUST_BELOW_THRESHOLD"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class ProvenaError(Exception):
    """Base error class for all Provena errors."""

    def __init__(
        self,
        message: str,
        code: ProvenaErrorCode,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}


class InvalidIntentError(ProvenaError):
    def __init__(self, message: str, validation_errors: list[Any]) -> None:
        super().__init__(message, ProvenaErrorCode.INVALID_INTENT, {"validation_errors": validation_errors})
        self.validation_errors = validation_errors


class NoCapableConnectorError(ProvenaError):
    def __init__(self, intent_type: str) -> None:
        super().__init__(
            f"No registered connector can handle intent type: {intent_type}",
            ProvenaErrorCode.NO_CAPABLE_CONNECTOR,
            {"intent_type": intent_type},
        )


class ConnectorTimeoutError(ProvenaError):
    def __init__(self, connector_id: str, budget_ms: int, actual_ms: float) -> None:
        super().__init__(
            f"Connector {connector_id} timed out: {actual_ms:.0f}ms > {budget_ms}ms budget",
            ProvenaErrorCode.CONNECTOR_TIMEOUT,
            {"connector_id": connector_id, "budget_ms": budget_ms, "actual_ms": actual_ms},
        )


class MCPTransportError(ProvenaError):
    def __init__(self, server_id: str, detail: str) -> None:
        super().__init__(
            f"MCP transport error for server {server_id}: {detail}",
            ProvenaErrorCode.MCP_TRANSPORT_ERROR,
            {"server_id": server_id, "detail": detail},
        )
```

### Tests: `tests/unit/types/test_intent_schema.py`

```python
"""Tests for intent model validation."""

import pytest
from pydantic import ValidationError

from provena.types.intent import (
    AggregateAnalysisIntent,
    PointLookupIntent,
    TemporalTrendIntent,
    TimeWindow,
    validate_intent,
)


class TestPointLookupIntent:
    def test_valid_point_lookup(self) -> None:
        intent = PointLookupIntent(
            id="intent-001",
            entity="customer",
            identifier={"customer_id": "C-1042"},
        )
        assert intent.type == "point_lookup"
        assert intent.entity == "customer"

    def test_rejects_empty_entity(self) -> None:
        with pytest.raises(ValidationError):
            PointLookupIntent(
                id="intent-001",
                entity="",
                identifier={"customer_id": "C-1042"},
            )

    def test_rejects_missing_identifier(self) -> None:
        with pytest.raises(ValidationError):
            PointLookupIntent(id="intent-001", entity="customer")  # type: ignore[call-arg]

    def test_optional_fields(self) -> None:
        intent = PointLookupIntent(
            id="intent-001",
            entity="customer",
            identifier={"customer_id": "C-1042"},
            fields=["name", "email"],
            max_results=10,
            budget_ms=500,
        )
        assert intent.fields == ["name", "email"]
        assert intent.max_results == 10


class TestTemporalTrendIntent:
    def test_valid_with_relative_window(self) -> None:
        intent = TemporalTrendIntent(
            id="intent-002",
            entity="usage",
            metric="api_calls",
            window=TimeWindow(relative="last_90d"),
            granularity="1d",
        )
        assert intent.type == "temporal_trend"

    def test_rejects_empty_time_window(self) -> None:
        with pytest.raises(ValidationError):
            TemporalTrendIntent(
                id="intent-002",
                entity="usage",
                metric="api_calls",
                window=TimeWindow(),
            )

    def test_valid_with_absolute_window(self) -> None:
        intent = TemporalTrendIntent(
            id="intent-002",
            entity="usage",
            metric="api_calls",
            window=TimeWindow(start="2025-01-01T00:00:00Z", end="2025-03-01T00:00:00Z"),
        )
        assert intent.window.start is not None


class TestValidateIntent:
    def test_discriminates_by_type_field(self) -> None:
        result = validate_intent({
            "id": "intent-001",
            "type": "point_lookup",
            "entity": "customer",
            "identifier": {"customer_id": "C-1042"},
        })
        assert isinstance(result, PointLookupIntent)

    def test_discriminates_aggregate(self) -> None:
        result = validate_intent({
            "id": "intent-003",
            "type": "aggregate_analysis",
            "entity": "orders",
            "measures": [{"field": "revenue", "aggregation": "sum"}],
            "dimensions": ["region"],
        })
        assert isinstance(result, AggregateAnalysisIntent)

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValidationError):
            validate_intent({
                "id": "intent-bad",
                "type": "nonexistent_type",
                "entity": "x",
            })

    def test_rejects_malformed_data(self) -> None:
        with pytest.raises(ValidationError):
            validate_intent({"garbage": True})
```

### Completion Check
- All type files pass `mypy src/provena --strict`
- `pytest tests/unit/types/` passes all tests
- `validate_intent()` correctly discriminates all 8 intent types
- Export everything from `src/provena/types/__init__.py`

---

## 4. Milestone 2: Provenance Envelope & Trust Scoring

### Goal
Implement provenance metadata and trust scoring. Every retrieved data element gets tagged with source, consistency, precision, and freshness — and a composite trust score.

### File: `src/provena/types/provenance.py`

```python
"""Provenance and trust types — the epistemic foundation of Provena."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ConsistencyGuarantee(StrEnum):
    STRONG = "strong"                   # linearizable / serializable
    READ_COMMITTED = "read_committed"   # read committed isolation
    EVENTUAL = "eventual"               # eventually consistent
    BEST_EFFORT = "best_effort"         # no guarantees (cache, scrape, etc.)


class PrecisionClass(StrEnum):
    EXACT = "exact"                     # deterministic computation
    EXACT_AGGREGATE = "exact_aggregate" # exact within aggregation scope
    ESTIMATED = "estimated"             # statistical estimate
    PREDICTED = "predicted"             # ML model output
    HEURISTIC = "heuristic"             # rule-of-thumb
    SIMILARITY_RANKED = "similarity_ranked"   # relevance-scored
    LOGICALLY_ENTAILED = "logically_entailed" # derived by inference


class RetrievalMethod(StrEnum):
    DIRECT_QUERY = "direct_query"
    CACHE_HIT = "cache_hit"
    COMPUTED_AGGREGATE = "computed_aggregate"
    ML_PREDICTION = "ml_prediction"
    VECTOR_SIMILARITY = "vector_similarity"
    GRAPH_TRAVERSAL = "graph_traversal"
    INFERENCE_ENGINE = "inference_engine"
    MCP_PASSTHROUGH = "mcp_passthrough"


class ProvenanceEnvelope(BaseModel):
    """Metadata attached to every data element entering the context."""

    source_system: str = Field(min_length=1)
    retrieval_method: RetrievalMethod
    consistency: ConsistencyGuarantee
    precision: PrecisionClass
    retrieved_at: str  # ISO 8601
    staleness_window_sec: float | None = None
    execution_ms: float | None = None
    result_truncated: bool | None = None
    total_available: int | None = None


class TrustDimensions(BaseModel):
    source_authority: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    precision_score: float = Field(ge=0.0, le=1.0)


class TrustScore(BaseModel):
    composite: float = Field(ge=0.0, le=1.0)
    dimensions: TrustDimensions
    label: Literal["high", "medium", "low", "uncertain"]
```

### File: `src/provena/core/provenance/envelope.py`

```python
"""Factory functions for creating ProvenanceEnvelopes."""

from datetime import datetime, timezone

from provena.types.provenance import (
    ConsistencyGuarantee,
    PrecisionClass,
    ProvenanceEnvelope,
    RetrievalMethod,
)


def create_envelope(
    source_system: str,
    retrieval_method: RetrievalMethod,
    consistency: ConsistencyGuarantee,
    precision: PrecisionClass,
    staleness_window_sec: float | None = None,
    execution_ms: float | None = None,
    result_truncated: bool | None = None,
    total_available: int | None = None,
) -> ProvenanceEnvelope:
    """Create a validated ProvenanceEnvelope with current timestamp."""
    return ProvenanceEnvelope(
        source_system=source_system,
        retrieval_method=retrieval_method,
        consistency=consistency,
        precision=precision,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        staleness_window_sec=staleness_window_sec,
        execution_ms=execution_ms,
        result_truncated=result_truncated,
        total_available=total_available,
    )


def create_default_envelope(source_system: str) -> ProvenanceEnvelope:
    """
    Conservative defaults for legacy MCP responses
    that don't provide provenance metadata.
    """
    return ProvenanceEnvelope(
        source_system=source_system,
        retrieval_method=RetrievalMethod.MCP_PASSTHROUGH,
        consistency=ConsistencyGuarantee.BEST_EFFORT,
        precision=PrecisionClass.ESTIMATED,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        staleness_window_sec=None,
    )
```

### File: `src/provena/core/provenance/trust_scorer.py`

```python
"""
TrustScorer computes composite trust signals from provenance metadata.
Trust scores are ADVISORY — the agent retains full autonomy over weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from provena.types.provenance import (
    ConsistencyGuarantee,
    PrecisionClass,
    ProvenanceEnvelope,
    TrustDimensions,
    TrustScore,
)

CONSISTENCY_SCORES: dict[ConsistencyGuarantee, float] = {
    ConsistencyGuarantee.STRONG: 1.0,
    ConsistencyGuarantee.READ_COMMITTED: 0.8,
    ConsistencyGuarantee.EVENTUAL: 0.5,
    ConsistencyGuarantee.BEST_EFFORT: 0.2,
}

PRECISION_SCORES: dict[PrecisionClass, float] = {
    PrecisionClass.EXACT: 1.0,
    PrecisionClass.EXACT_AGGREGATE: 0.95,
    PrecisionClass.LOGICALLY_ENTAILED: 0.9,
    PrecisionClass.ESTIMATED: 0.6,
    PrecisionClass.SIMILARITY_RANKED: 0.55,
    PrecisionClass.PREDICTED: 0.5,
    PrecisionClass.HEURISTIC: 0.3,
}


@dataclass
class TrustScorerConfig:
    weight_source_authority: float = 0.2
    weight_consistency: float = 0.3
    weight_freshness: float = 0.2
    weight_precision: float = 0.3
    source_authority_map: dict[str, float] = field(default_factory=dict)


class TrustScorer:
    def __init__(self, config: TrustScorerConfig | None = None) -> None:
        self.config = config or TrustScorerConfig()

    def score(self, envelope: ProvenanceEnvelope) -> TrustScore:
        source_authority = self.config.source_authority_map.get(envelope.source_system, 0.5)
        consistency_score = CONSISTENCY_SCORES[envelope.consistency]
        precision_score = PRECISION_SCORES[envelope.precision]
        freshness_score = self._compute_freshness(envelope)

        composite = (
            self.config.weight_source_authority * source_authority
            + self.config.weight_consistency * consistency_score
            + self.config.weight_freshness * freshness_score
            + self.config.weight_precision * precision_score
        )

        if composite >= 0.8:
            label = "high"
        elif composite >= 0.55:
            label = "medium"
        elif composite >= 0.3:
            label = "low"
        else:
            label = "uncertain"

        return TrustScore(
            composite=composite,
            dimensions=TrustDimensions(
                source_authority=source_authority,
                consistency_score=consistency_score,
                freshness_score=freshness_score,
                precision_score=precision_score,
            ),
            label=label,
        )

    def _compute_freshness(self, envelope: ProvenanceEnvelope) -> float:
        if envelope.staleness_window_sec is None:
            return 0.5  # unknown = neutral

        retrieved_at = datetime.fromisoformat(envelope.retrieved_at)
        now = datetime.now(timezone.utc)
        age_sec = (now - retrieved_at).total_seconds()
        window_sec = envelope.staleness_window_sec

        if window_sec <= 0:
            return 0.5

        ratio = age_sec / window_sec
        # 1.0 if just retrieved, decays to 0 at 2x staleness window
        return max(0.0, 1.0 - ratio / 2.0)
```

### Tests: `tests/unit/core/provenance/test_trust_scorer.py`

```python
"""Tests for TrustScorer."""

from datetime import datetime, timezone

import pytest

from provena.core.provenance.trust_scorer import TrustScorer, TrustScorerConfig
from provena.types.provenance import (
    ConsistencyGuarantee,
    PrecisionClass,
    ProvenanceEnvelope,
    RetrievalMethod,
)


def _make_envelope(**overrides) -> ProvenanceEnvelope:
    defaults = {
        "source_system": "test.db",
        "retrieval_method": RetrievalMethod.DIRECT_QUERY,
        "consistency": ConsistencyGuarantee.STRONG,
        "precision": PrecisionClass.EXACT,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "staleness_window_sec": 3600.0,
    }
    return ProvenanceEnvelope(**(defaults | overrides))


class TestTrustScorer:
    def test_high_trust_for_strong_exact_fresh(self) -> None:
        scorer = TrustScorer()
        score = scorer.score(_make_envelope())
        assert score.label == "high"
        assert score.composite >= 0.8

    def test_low_trust_for_weak_heuristic_stale(self) -> None:
        scorer = TrustScorer()
        score = scorer.score(_make_envelope(
            consistency=ConsistencyGuarantee.BEST_EFFORT,
            precision=PrecisionClass.HEURISTIC,
            retrieved_at="2020-01-01T00:00:00+00:00",  # very stale
            staleness_window_sec=60.0,
        ))
        assert score.label in ("low", "uncertain")
        assert score.composite < 0.4

    def test_unknown_source_gets_neutral_authority(self) -> None:
        scorer = TrustScorer()
        score = scorer.score(_make_envelope())
        assert score.dimensions.source_authority == 0.5

    def test_known_source_gets_configured_authority(self) -> None:
        config = TrustScorerConfig(source_authority_map={"test.db": 0.95})
        scorer = TrustScorer(config)
        score = scorer.score(_make_envelope())
        assert score.dimensions.source_authority == 0.95

    def test_null_staleness_gets_neutral_freshness(self) -> None:
        scorer = TrustScorer()
        score = scorer.score(_make_envelope(staleness_window_sec=None))
        assert score.dimensions.freshness_score == 0.5

    def test_freshly_retrieved_gets_high_freshness(self) -> None:
        scorer = TrustScorer()
        score = scorer.score(_make_envelope(staleness_window_sec=3600.0))
        assert score.dimensions.freshness_score > 0.9
```

### Additional tests to write:
- `tests/unit/core/provenance/test_envelope.py`: `create_envelope` produces valid envelopes, `create_default_envelope` uses conservative defaults, invalid inputs raise `ValidationError`

### Completion Check
- All provenance types pass `mypy`
- TrustScorer produces correct scores for all test cases
- `pytest tests/unit/core/provenance/` passes

---

## 5. Milestone 3: Context Compiler

### Goal
Build the Context Compiler that assembles retrieval results into typed context slots with provenance, detects conflicts, and produces structured context frames.

### File: `src/provena/types/context.py`

```python
"""Context frame types — the structured replacement for flat context windows."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from provena.types.provenance import ProvenanceEnvelope, TrustScore


class ContextSlotType(StrEnum):
    STRUCTURED = "STRUCTURED"
    RELATIONAL = "RELATIONAL"
    TEMPORAL = "TEMPORAL"
    UNSTRUCTURED = "UNSTRUCTURED"
    INFERRED = "INFERRED"


class ContextElement(BaseModel):
    """Atomic unit of the context frame — data + provenance."""

    id: str
    data: Any
    provenance: ProvenanceEnvelope
    trust: TrustScore
    source_intent_id: str
    entity_key: str | None = None


class ContextSlot(BaseModel):
    """Typed slot — elements share interpretation semantics."""

    type: ContextSlotType
    elements: list[ContextElement]
    interpretation_notes: str


class ConflictResolution(BaseModel):
    strategy: Literal[
        "prefer_freshest",
        "prefer_authoritative",
        "prefer_strongest_consistency",
        "defer_to_agent",
    ]
    winner: str | None = None  # element ID of winner, None for defer_to_agent
    reason: str


class ContextConflict(BaseModel):
    element_a: ContextElement
    element_b: ContextElement
    field: str
    value_a: Any
    value_b: Any
    resolution: ConflictResolution


class ContextFrameStats(BaseModel):
    total_elements: int
    avg_trust_score: float
    slot_counts: dict[str, int]


class ContextFrame(BaseModel):
    """The complete context frame passed to the agent."""

    slots: list[ContextSlot]
    conflicts: list[ContextConflict]
    stats: ContextFrameStats
    assembled_at: str
```

### File: `src/provena/core/context/context_compiler.py`

```python
"""
ContextCompiler assembles retrieval results into typed context frames.
This is the key innovation — replacing flat token soup with structured,
provenance-enriched context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from provena.core.context.conflict_detector import ConflictDetector
from provena.core.context.conflict_resolver import ConflictResolver
from provena.core.context.typed_slot import INTERPRETATION_NOTES
from provena.core.provenance.trust_scorer import TrustScorer
from provena.types.context import (
    ContextElement,
    ContextFrame,
    ContextFrameStats,
    ContextSlot,
    ContextSlotType,
)
from provena.types.provenance import ProvenanceEnvelope


class CompilerInput:
    """Input to add a single element to the compiler."""

    def __init__(
        self,
        slot_type: ContextSlotType,
        data: Any,
        provenance: ProvenanceEnvelope,
        source_intent_id: str,
        entity_key: str | None = None,
    ) -> None:
        self.slot_type = slot_type
        self.data = data
        self.provenance = provenance
        self.source_intent_id = source_intent_id
        self.entity_key = entity_key


class ContextCompiler:
    def __init__(self, trust_scorer: TrustScorer | None = None) -> None:
        self.trust_scorer = trust_scorer or TrustScorer()
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver()
        self._elements: list[tuple[ContextSlotType, ContextElement]] = []
        self._counter = 0

    def add_element(self, input: CompilerInput) -> ContextElement:
        """
        Add a data element. Call for each result from typed connectors.
        Returns the created ContextElement.
        """
        trust = self.trust_scorer.score(input.provenance)
        self._counter += 1
        element = ContextElement(
            id=f"elem-{self._counter}-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            data=input.data,
            provenance=input.provenance,
            trust=trust,
            source_intent_id=input.source_intent_id,
            entity_key=input.entity_key,
        )
        self._elements.append((input.slot_type, element))
        return element

    def compile(self) -> ContextFrame:
        """
        Compile all added elements into a ContextFrame.
        Groups into typed slots, detects conflicts, computes stats.
        """
        # 1. Group elements by slot type
        slot_groups: dict[ContextSlotType, list[ContextElement]] = {}
        for slot_type, element in self._elements:
            slot_groups.setdefault(slot_type, []).append(element)

        # 2. Build typed slots with interpretation notes
        slots = [
            ContextSlot(
                type=slot_type,
                elements=elements,
                interpretation_notes=INTERPRETATION_NOTES.get(slot_type, ""),
            )
            for slot_type, elements in slot_groups.items()
        ]

        # 3. Detect conflicts
        all_elements = [elem for _, elem in self._elements]
        conflicts = self.conflict_detector.detect(all_elements)

        # 4. Resolve conflicts
        resolved_conflicts = [
            self.conflict_resolver.resolve(c) for c in conflicts
        ]

        # 5. Compute stats
        trust_scores = [elem.trust.composite for _, elem in self._elements]
        avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

        stats = ContextFrameStats(
            total_elements=len(self._elements),
            avg_trust_score=round(avg_trust, 4),
            slot_counts={st.value: len(elems) for st, elems in slot_groups.items()},
        )

        return ContextFrame(
            slots=slots,
            conflicts=resolved_conflicts,
            stats=stats,
            assembled_at=datetime.now(timezone.utc).isoformat(),
        )

    def reset(self) -> None:
        self._elements.clear()
        self._counter = 0
```

### File: `src/provena/core/context/typed_slot.py`

```python
"""Interpretation notes per slot type."""

from provena.types.context import ContextSlotType

INTERPRETATION_NOTES: dict[ContextSlotType, str] = {
    ContextSlotType.STRUCTURED: (
        "Tabular data. Numbers are precise within query scope. "
        "NULLs mean absent, not unknown. Trust provenance precision class for confidence."
    ),
    ContextSlotType.RELATIONAL: (
        "Relationship data. Absence of an edge means not-found, not does-not-exist. "
        "Check traversal depth limits before assuming completeness."
    ),
    ContextSlotType.TEMPORAL: (
        "Time-series data. Always check window and granularity. "
        "Trends are computed within the stated window — do not extrapolate beyond it."
    ),
    ContextSlotType.UNSTRUCTURED: (
        "Natural language text. Apply standard NLP-level caution. "
        "Claims are assertions, not verified facts. Cross-reference with structured data."
    ),
    ContextSlotType.INFERRED: (
        "Derived by formal reasoning. Validity depends on ontology correctness. "
        "Uses open-world assumption: absence of entailment is not negation."
    ),
}
```

### File: `src/provena/core/context/conflict_detector.py`

Implement `ConflictDetector` with a `detect(elements: list[ContextElement])` method that:
- Groups elements by `entity_key` (skip elements with `entity_key=None`)
- For each group with 2+ elements from DIFFERENT source systems: compare `data` dicts
- If the same field has different values across elements → create an unresolved `ContextConflict`
- Only compare elements in STRUCTURED and TEMPORAL slots (text conflicts are too fuzzy)
- Return list of detected conflicts (without resolutions — those come from `ConflictResolver`)

### File: `src/provena/core/context/conflict_resolver.py`

Implement `ConflictResolver` with a `resolve(conflict)` method that applies these rules:
- If freshness difference > 10× staleness window → `prefer_freshest`
- If one source authority > 0.9 and other < 0.7 → `prefer_authoritative`
- If consistency gap ≥ 2 levels (e.g. strong vs eventual) → `prefer_strongest_consistency`
- Otherwise → `defer_to_agent`

The resolver should return a new `ContextConflict` with the `resolution` field filled in.

### Tests to write (`tests/unit/core/context/`):
- `test_context_compiler.py`: groups elements into correct slot types; computes stats correctly; handles empty input; produces valid `ContextFrame`
- `test_conflict_detector.py`: finds conflicts for same entity key with different values; ignores elements without entity keys; ignores UNSTRUCTURED slot elements; handles no conflicts gracefully
- `test_conflict_resolver.py` (create this file): picks freshest when age gap is large; defers to agent when signals are ambiguous; picks authoritative when authority gap is clear

### Completion Check
- ContextCompiler compiles a frame from mixed-type inputs
- Conflict detection and resolution produce correct results
- `pytest tests/unit/core/context/` passes

---

## 6. Milestone 4: Typed Connector Interface & Base Class

### Goal
Define the abstract connector contract that all typed connectors inherit from.

### File: `src/provena/types/connector.py`

```python
"""Connector result and health types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from provena.types.context import ContextSlotType
from provena.types.provenance import ProvenanceEnvelope


class ConnectorResultMeta(BaseModel):
    execution_ms: float
    record_count: int
    truncated: bool
    native_query: str | None = None  # actual query executed, for debugging


class ConnectorResult(BaseModel):
    """Result returned by a typed connector."""

    records: list[Any]
    provenance: ProvenanceEnvelope
    slot_type: ContextSlotType
    entity_keys: list[str] | None = None  # one per record
    meta: ConnectorResultMeta


class ConnectorHealth(BaseModel):
    connector_id: str
    status: str  # "healthy" | "degraded" | "unavailable"
    latency_ms: float
    last_checked: str
    message: str | None = None
```

### File: `src/provena/types/capability.py`

```python
"""Capability declarations for typed connectors."""

from __future__ import annotations

from pydantic import BaseModel


class ConnectorCapabilities(BaseModel):
    supports_aggregation: bool = False
    supports_windowing: bool = False
    supports_traversal: bool = False
    supports_similarity: bool = False
    supports_inference: bool = False
    supports_temporal_bucketing: bool = False
    supports_full_text_search: bool = False


class ConnectorPerformance(BaseModel):
    estimated_latency_ms: float
    max_result_cardinality: int
    supports_batch_lookup: bool = False


class ConnectorCapability(BaseModel):
    connector_id: str
    connector_type: str  # e.g. "olap", "oltp", "graph"
    supported_intent_types: list[str]
    capabilities: ConnectorCapabilities
    performance: ConnectorPerformance
    available_entities: list[str]
```

### File: `src/provena/connectors/base_connector.py`

```python
"""
Abstract base class for typed connectors.
Subclasses implement the four internal stages:
  1. interpret_intent  — validate and extract parameters
  2. synthesize_query  — build native query
  3. execute_query     — run the query (via MCP or direct)
  4. normalize_result  — transform to ConnectorResult
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from provena.types.capability import ConnectorCapability
from provena.types.connector import ConnectorHealth, ConnectorResult
from provena.types.intent import BaseIntent


class BaseConnector(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def connector_type(self) -> str: ...

    @abstractmethod
    def get_capabilities(self) -> ConnectorCapability: ...

    def can_handle(self, intent: BaseIntent) -> bool:
        """Check if this connector can handle the given intent."""
        caps = self.get_capabilities()
        return intent.type in caps.supported_intent_types

    async def execute(self, intent: BaseIntent) -> ConnectorResult:
        """
        Execute an intent end-to-end.
        Main entry point called by the Semantic Router.
        """
        start = time.monotonic()

        # Stage 1: Interpret
        params = self.interpret_intent(intent)

        # Stage 2: Synthesize
        query = self.synthesize_query(params)

        # Stage 3: Execute
        raw_result = await self.execute_query(query)

        # Stage 4: Normalize
        elapsed_ms = (time.monotonic() - start) * 1000
        return self.normalize_result(raw_result, intent, elapsed_ms)

    @abstractmethod
    async def check_health(self) -> ConnectorHealth: ...

    # ── Internal stages (implement in subclasses) ──

    @abstractmethod
    def interpret_intent(self, intent: BaseIntent) -> Any:
        """Stage 1: Validate intent and extract typed parameters."""
        ...

    @abstractmethod
    def synthesize_query(self, params: Any) -> Any:
        """Stage 2: Build native query from parameters."""
        ...

    @abstractmethod
    async def execute_query(self, query: Any) -> Any:
        """Stage 3: Execute the native query."""
        ...

    @abstractmethod
    def normalize_result(
        self,
        raw: Any,
        intent: BaseIntent,
        execution_ms: float,
    ) -> ConnectorResult:
        """Stage 4: Transform raw results to ConnectorResult."""
        ...
```

### Query Executor Protocol

Each connector takes a pluggable executor to decouple from real databases:

```python
# Put this in src/provena/connectors/executor.py

from __future__ import annotations

from typing import Any, Protocol


class QueryExecutor(Protocol):
    """Protocol for executing native queries against a data source."""

    async def execute(self, query: Any) -> dict[str, Any]:
        """Returns {"records": [...], "meta": {...}}"""
        ...


class MockQueryExecutor:
    """Mock executor for testing — returns pre-configured results."""

    def __init__(self, records: list[Any] | None = None, meta: dict[str, Any] | None = None) -> None:
        self.records = records or []
        self.meta = meta or {}
        self.last_query: Any = None

    async def execute(self, query: Any) -> dict[str, Any]:
        self.last_query = query
        return {"records": self.records, "meta": self.meta}
```

### Tests to write:
- `BaseConnector.can_handle` correctly checks against capability declarations
- `BaseConnector.execute` calls all four stages in order (test via a minimal concrete subclass)
- `BaseConnector.execute` measures execution time correctly
- `MockQueryExecutor` captures queries and returns configured results

### Completion Check
- All connector types pass `mypy`
- BaseConnector abstract contract is clean
- `pytest` passes

---

## 7. Milestone 5: Reference Connectors

### Goal
Build three reference connectors: OLAP, OLTP, and Document/Vector. Each demonstrates the full four-stage pattern with real query optimization.

### IMPORTANT: These connectors don't connect to real databases. They build optimized query strings/objects and return them via the pluggable `QueryExecutor`. This lets us test query optimization without database dependencies. The MCP layer (Milestone 8) provides actual execution.

### File: `src/provena/connectors/olap/olap_connector.py`

The OLAP connector handles `aggregate_analysis` and `temporal_trend` intents.

Key optimizations to implement:
1. **Push-down aggregation**: Build GROUP BY with server-side SUM/AVG/COUNT
2. **Partition pruning**: Include time-range predicates for partition elimination
3. **Predicate pushdown**: Convert FilterClauses to WHERE conditions
4. **Order pushdown**: Include ORDER BY and LIMIT server-side
5. **Rollup detection**: When granularity matches a known pre-computed rollup, use it

```python
from dataclasses import dataclass

@dataclass
class OLAPQuery:
    """Native query representation for OLAP systems."""
    sql: str
    params: list[Any]
    optimizations: list[str]   # which optimizations were applied
    estimated_rows_scanned: int
    uses_partition_pruning: bool
    uses_precomputed_rollup: bool
```

Capabilities to declare:
- `supported_intent_types`: `["aggregate_analysis", "temporal_trend"]`
- `supports_aggregation`: True
- `supports_windowing`: True
- `supports_temporal_bucketing`: True
- `estimated_latency_ms`: 500

Provenance to set:
- `precision`: `PrecisionClass.EXACT_AGGREGATE`
- `consistency`: `ConsistencyGuarantee.STRONG`
- `slot_type`: `ContextSlotType.STRUCTURED` for aggregations, `ContextSlotType.TEMPORAL` for trends

### File: `src/provena/connectors/oltp/oltp_connector.py`

Handles `point_lookup` and simple `aggregate_analysis` intents.

Key optimizations:
1. **Index-aware field selection**: Only SELECT fields the agent asked for
2. **Batch lookup**: Convert multiple point lookups into single IN query
3. **Parameterized queries**: Use `$1, $2` style params (not string interpolation)

Provenance: `precision=EXACT`, `consistency=READ_COMMITTED`, `slot_type=STRUCTURED`

### File: `src/provena/connectors/document/document_connector.py`

Handles `semantic_search` intents.

Key optimizations:
1. **Hybrid retrieval**: Combine keyword (BM25) and vector similarity via `hybrid_weight`
2. **Filter pushdown**: Convert metadata filters to native filter syntax
3. **Score-based truncation**: Only return above relevance threshold
4. **Reranking flag**: When `rerank=True`, include reranking stage

Provenance: `precision=SIMILARITY_RANKED`, `consistency=EVENTUAL`, `slot_type=UNSTRUCTURED`

### Tests to write (per connector in `tests/unit/connectors/`):
- Correctly translates intent to native query format
- Applies expected optimizations (check `query.optimizations` list)
- Produces correct provenance metadata
- Handles empty results gracefully
- Rejects intents it cannot handle (raises `InvalidIntentError`)
- Reports entity keys for conflict detection

### Completion Check
- All three connectors implement `BaseConnector` fully
- Query builders produce correct SQL/search queries
- `pytest tests/unit/connectors/` passes

---

## 8. Milestone 6: Capability Registry

### Goal
Build the registry that connectors register with and that the Semantic Router queries.

### File: `src/provena/connectors/capability_registry.py`

```python
"""
Registry of typed connectors and their capabilities.
The Semantic Router uses this to route intents to connectors.
"""

from __future__ import annotations

from dataclasses import dataclass

from provena.connectors.base_connector import BaseConnector
from provena.types.capability import ConnectorCapability
from provena.types.intent import BaseIntent


@dataclass
class ConnectorCandidate:
    connector: BaseConnector
    capability: ConnectorCapability
    suitability_score: float


class CapabilityRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, BaseConnector] = {}
        self._capabilities: dict[str, ConnectorCapability] = {}

    def register(self, connector: BaseConnector) -> None:
        """Register a connector. Reads its capabilities automatically."""
        caps = connector.get_capabilities()
        self._connectors[caps.connector_id] = connector
        self._capabilities[caps.connector_id] = caps

    def unregister(self, connector_id: str) -> None:
        self._connectors.pop(connector_id, None)
        self._capabilities.pop(connector_id, None)

    def find_candidates(self, intent: BaseIntent) -> list[ConnectorCandidate]:
        """
        Find connectors that can handle this intent, ranked by suitability.

        Ranking criteria:
        1. Direct intent type match (required — filter, not score)
        2. Entity availability (connector has access to the entity)
        3. Performance profile (lower latency preferred)
        4. Capability richness (more relevant capabilities preferred)
        """
        candidates: list[ConnectorCandidate] = []

        for conn_id, caps in self._capabilities.items():
            if intent.type not in caps.supported_intent_types:
                continue

            connector = self._connectors[conn_id]
            score = self._compute_suitability(intent, caps)
            candidates.append(ConnectorCandidate(
                connector=connector,
                capability=caps,
                suitability_score=score,
            ))

        candidates.sort(key=lambda c: c.suitability_score, reverse=True)
        return candidates

    def _compute_suitability(self, intent: BaseIntent, caps: ConnectorCapability) -> float:
        """
        Score how suitable a connector is for this intent.
        Implement scoring based on:
        - entity match: does the connector have the entity the intent targets?
        - latency: lower estimated latency = higher score
        - capability match: do the connector's capabilities align with intent needs?
        """
        # ... implement this
        pass

    def get_connector(self, connector_id: str) -> BaseConnector | None:
        return self._connectors.get(connector_id)

    def list_capabilities(self) -> list[ConnectorCapability]:
        return list(self._capabilities.values())
```

### Tests to write (`tests/unit/connectors/test_capability_registry.py`):
- Returns correct candidates for each intent type
- Ranks better-matched connectors higher
- Returns empty list when no connector matches
- Handles multiple connectors for same intent type
- Register and unregister work correctly

### Completion Check
- CapabilityRegistry routes all intent types correctly
- `pytest` passes

---

## 9. Milestone 7: Semantic Router & Query Planner

### Goal
Build the Semantic Router — the orchestration brain of Provena.

### File: `src/provena/types/router.py`

```python
"""Router and execution plan types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from provena.types.connector import ConnectorResult
from provena.types.intent import BaseIntent


class ExecutionStep(BaseModel):
    step_id: str
    intent: BaseIntent
    connector_id: str
    depends_on: list[str]  # step IDs that must complete first
    estimated_ms: float
    estimated_tokens: int


class ExecutionPlan(BaseModel):
    steps: list[ExecutionStep]
    estimated_total_ms: float  # accounting for parallelism
    estimated_total_tokens: int
    has_parallel_steps: bool
    fusion_strategy: str | None = None


class ExecutionError(BaseModel):
    step_id: str
    error_message: str
    error_code: str


class ExecutionResult(BaseModel):
    """
    Note: results is a dict of step_id → ConnectorResult.
    Pydantic v2 handles dict serialization.
    """
    results: dict[str, ConnectorResult]
    plan: ExecutionPlan
    actual_total_ms: float
    errors: list[ExecutionError]
```

### File: `src/provena/core/router/intent_decomposer.py`

```python
"""Decomposes composite intents into atomic sub-intents."""

from __future__ import annotations

from provena.types.intent import BaseIntent, CompositeIntent, Intent


class IntentDecomposer:
    def decompose(self, intent: BaseIntent) -> list[BaseIntent]:
        """
        If composite, recursively flatten into atomic intents.
        If atomic, return single-element list.
        """
        if not isinstance(intent, CompositeIntent):
            return [intent]

        flattened: list[BaseIntent] = []
        for sub in intent.sub_intents:
            flattened.extend(self.decompose(sub))
        return flattened

    def analyze_dependencies(
        self,
        sub_intents: list[BaseIntent],
        fusion_operator: str,
        fusion_key: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Determine execution dependencies between sub-intents.

        Returns: {intent_id: [list of intent_ids it depends on]}

        Dependencies exist when:
        - fusion_operator is "sequence" (strict ordering)
        - A sub-intent's filters reference another's join_key (scope narrowing)
        """
        # ... implement dependency analysis
        pass
```

### File: `src/provena/core/router/cost_estimator.py`

Implement `CostEstimator` that:
- Estimates latency from connector's declared `estimated_latency_ms` + 10ms overhead
- Estimates token count: `estimated_records × 50 tokens per record` (configurable)
- Accounts for parallelism: parallel steps contribute `max(latencies)`, not `sum`
- Tracks historical actuals to refine future estimates (simple running average)

### File: `src/provena/core/router/query_planner.py`

```python
"""
QueryPlanner generates execution plans for intents.
1. Decompose into sub-intents
2. Route each to best connector
3. Analyze dependencies
4. Build execution plan with parallel/sequential steps
5. Estimate total cost
"""

from __future__ import annotations

from provena.connectors.capability_registry import CapabilityRegistry
from provena.core.router.cost_estimator import CostEstimator
from provena.core.router.intent_decomposer import IntentDecomposer
from provena.types.errors import NoCapableConnectorError
from provena.types.intent import BaseIntent, CompositeIntent
from provena.types.router import ExecutionPlan, ExecutionStep


class QueryPlanner:
    def __init__(
        self,
        registry: CapabilityRegistry,
        decomposer: IntentDecomposer,
        cost_estimator: CostEstimator,
    ) -> None:
        self.registry = registry
        self.decomposer = decomposer
        self.cost_estimator = cost_estimator

    def plan(self, intent: BaseIntent) -> ExecutionPlan:
        # 1. Decompose
        sub_intents = self.decomposer.decompose(intent)

        # 2. Route each sub-intent to best connector
        steps: list[ExecutionStep] = []
        for i, sub in enumerate(sub_intents):
            candidates = self.registry.find_candidates(sub)
            if not candidates:
                raise NoCapableConnectorError(sub.type)

            best = candidates[0]
            step = ExecutionStep(
                step_id=f"step-{i}",
                intent=sub,
                connector_id=best.capability.connector_id,
                depends_on=[],  # filled in step 3
                estimated_ms=self.cost_estimator.estimate_latency(best.capability),
                estimated_tokens=self.cost_estimator.estimate_tokens(best.capability),
            )
            steps.append(step)

        # 3. Analyze dependencies (if original was composite)
        if isinstance(intent, CompositeIntent):
            deps = self.decomposer.analyze_dependencies(
                sub_intents, intent.fusion_operator.value, intent.fusion_key
            )
            for step in steps:
                step.depends_on = deps.get(step.intent.id, [])

        # 4. Compute plan-level stats
        has_parallel = any(len(s.depends_on) == 0 for s in steps[1:]) if len(steps) > 1 else False

        return ExecutionPlan(
            steps=steps,
            estimated_total_ms=self.cost_estimator.estimate_total_ms(steps),
            estimated_total_tokens=sum(s.estimated_tokens for s in steps),
            has_parallel_steps=has_parallel,
            fusion_strategy=intent.fusion_operator.value if isinstance(intent, CompositeIntent) else None,
        )
```

### File: `src/provena/core/router/semantic_router.py`

```python
"""
SemanticRouter — the main orchestrator.
Takes an intent, plans execution, runs the plan, compiles into ContextFrame.
"""

from __future__ import annotations

import asyncio
import time

from provena.connectors.capability_registry import CapabilityRegistry
from provena.core.context.context_compiler import CompilerInput, ContextCompiler
from provena.core.router.query_planner import QueryPlanner
from provena.types.context import ContextFrame
from provena.types.intent import BaseIntent
from provena.types.router import ExecutionError, ExecutionPlan, ExecutionResult


class SemanticRouter:
    def __init__(
        self,
        planner: QueryPlanner,
        compiler: ContextCompiler,
        registry: CapabilityRegistry,
    ) -> None:
        self.planner = planner
        self.compiler = compiler
        self.registry = registry

    async def route(self, intent: BaseIntent) -> ContextFrame:
        """
        Main entry point.
        Intent → Plan → Execute → Compile → ContextFrame
        """
        # 1. Plan
        plan = self.planner.plan(intent)

        # 2. Execute (respecting dependencies and parallelism)
        execution_result = await self._execute_plan(plan)

        # 3. Feed results into compiler
        self.compiler.reset()
        for step_id, result in execution_result.results.items():
            step = next(s for s in plan.steps if s.step_id == step_id)
            for i, record in enumerate(result.records):
                self.compiler.add_element(CompilerInput(
                    slot_type=result.slot_type,
                    data=record,
                    provenance=result.provenance,
                    source_intent_id=step.intent.id,
                    entity_key=result.entity_keys[i] if result.entity_keys else None,
                ))

        # 4. Compile and return
        return self.compiler.compile()

    async def _execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        Execute plan with dependency-aware parallelism.
        Uses topological sort to group steps into levels.
        Each level runs in parallel via asyncio.gather.
        """
        start = time.monotonic()
        results: dict[str, any] = {}
        errors: list[ExecutionError] = []
        completed_step_ids: set[str] = set()

        # Topological sort into levels
        levels = self._topological_levels(plan.steps)

        for level in levels:
            # Run all steps in this level concurrently
            tasks = []
            for step in level:
                connector = self.registry.get_connector(step.connector_id)
                if connector is None:
                    errors.append(ExecutionError(
                        step_id=step.step_id,
                        error_message=f"Connector not found: {step.connector_id}",
                        error_code="NO_CAPABLE_CONNECTOR",
                    ))
                    continue
                tasks.append(self._execute_step(step, connector))

            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(level, step_results):
                if isinstance(result, Exception):
                    errors.append(ExecutionError(
                        step_id=step.step_id,
                        error_message=str(result),
                        error_code="CONNECTOR_ERROR",
                    ))
                else:
                    results[step.step_id] = result
                    completed_step_ids.add(step.step_id)

        elapsed = (time.monotonic() - start) * 1000
        return ExecutionResult(
            results=results,
            plan=plan,
            actual_total_ms=elapsed,
            errors=errors,
        )

    async def _execute_step(self, step, connector):
        """Execute a single step."""
        return await connector.execute(step.intent)

    def _topological_levels(self, steps):
        """
        Group steps into execution levels.
        Level 0: steps with no dependencies.
        Level 1: steps depending only on level 0 steps.
        And so on.
        """
        # ... implement topological sort into levels
        pass
```

### Tests to write (`tests/unit/core/router/`):
- `test_intent_decomposer.py`: flattens nested composites; detects sequential dependencies; handles atomic intents
- `test_query_planner.py`: routes intents to correct connectors; parallelizes independent steps; sequences dependent steps; raises `NoCapableConnectorError` when needed
- `test_semantic_router.py`: produces valid `ContextFrame` from composite intent; handles connector errors (partial results); parallel execution actually runs concurrently (check timing)

### Completion Check
- Full route: intent → plan → execute → compile works end-to-end with mocks
- `pytest tests/unit/core/router/` passes

---

## 10. Milestone 8: MCP Integration Layer

### Goal
Build the adapter between Provena typed connectors and MCP servers.

### File: `src/provena/mcp/mcp_adapter.py`

```python
"""
Adapter between Provena typed connectors and MCP servers.
Typed connectors call this instead of hitting databases directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from provena.types.errors import MCPTransportError


@dataclass
class MCPServerConfig:
    server_id: str
    server_url: str
    declared_consistency: str | None = None
    declared_precision: str | None = None
    declared_staleness_window_sec: float | None = None


@dataclass
class MCPToolCall:
    tool_name: str
    parameters: dict[str, Any]


@dataclass
class MCPResponse:
    content: Any
    provena_metadata: dict[str, Any] | None = None  # Provena extension (may be absent)


class MCPTransport(Protocol):
    """Pluggable transport for MCP communication."""
    async def send(self, server: MCPServerConfig, call: MCPToolCall) -> MCPResponse: ...


class MockMCPTransport:
    """Mock transport for testing."""

    def __init__(self) -> None:
        self._responses: dict[str, MCPResponse] = {}

    def set_response(self, tool_name: str, response: MCPResponse) -> None:
        self._responses[tool_name] = response

    async def send(self, server: MCPServerConfig, call: MCPToolCall) -> MCPResponse:
        return self._responses.get(call.tool_name, MCPResponse(content=[]))


class MCPAdapter:
    def __init__(self, transport: MCPTransport) -> None:
        self._transport = transport
        self._servers: dict[str, MCPServerConfig] = {}

    def register_server(self, config: MCPServerConfig) -> None:
        self._servers[config.server_id] = config

    async def call(self, server_id: str, tool_call: MCPToolCall) -> MCPResponse:
        server = self._servers.get(server_id)
        if server is None:
            raise MCPTransportError(server_id, f"Server not registered: {server_id}")
        return await self._transport.send(server, tool_call)
```

### File: `src/provena/mcp/response_wrapper.py`

Implement `ResponseWrapper` that:
- Extracts Provena metadata from `MCPResponse.provena_metadata` if present
- Falls back to `MCPServerConfig.declared_*` properties if metadata absent
- Falls back to conservative defaults (`best_effort`, `estimated`, `None` staleness) as last resort
- Returns a `ProvenanceEnvelope`

### Tests (`tests/unit/mcp/`):
- `test_mcp_adapter.py`: routes calls to correct server; raises on unknown server
- `test_response_wrapper.py`: extracts Provena metadata when present; falls back to declared defaults; uses conservative defaults as last resort

### Completion Check
- MCPAdapter + ResponseWrapper produce correct ProvenanceEnvelopes
- `pytest tests/unit/mcp/` passes

---

## 11. Milestone 9: Epistemic Tracker

### Goal
Build the Epistemic Tracker — the agent's confidence-reasoning module.

### File: `src/provena/core/epistemic/epistemic_tracker.py`

```python
"""
EpistemicTracker maintains running confidence model over context.
Provides trust-weighted reasoning support and prompt injection.
"""

from __future__ import annotations

from provena.types.context import ContextElement, ContextFrame
from provena.types.provenance import TrustScore


class EpistemicTracker:
    def __init__(self) -> None:
        self._frames: list[ContextFrame] = []

    def ingest(self, frame: ContextFrame) -> None:
        """Ingest a new context frame."""
        self._frames.append(frame)

    def get_trust(self, element_id: str) -> TrustScore | None:
        """Get trust score for a specific element."""
        for frame in self._frames:
            for slot in frame.slots:
                for elem in slot.elements:
                    if elem.id == element_id:
                        return elem.trust
        return None

    def get_low_trust_elements(self, threshold: float = 0.4) -> list[ContextElement]:
        """Get all elements below a trust threshold."""
        results: list[ContextElement] = []
        for frame in self._frames:
            for slot in frame.slots:
                for elem in slot.elements:
                    if elem.trust.composite < threshold:
                        results.append(elem)
        return results

    def get_unresolved_conflicts(self):
        """Get all conflicts deferred to agent."""
        return [
            conflict
            for frame in self._frames
            for conflict in frame.conflicts
            if conflict.resolution.strategy == "defer_to_agent"
        ]

    def generate_epistemic_prompt(self) -> str:
        """
        Generate structured text that communicates epistemic context to the LLM.
        This is injected into the agent's prompt so it can reason about confidence.

        Output format:
        ## Data Confidence Summary
        - {N} data elements from {M} sources
        - Average trust: {score} ({label})
        - Low-trust elements: [list with source and reason]
        - Unresolved conflicts: [list with competing values]
        """
        if not self._frames:
            return "## Data Confidence Summary\nNo data ingested yet."

        all_elements: list[ContextElement] = []
        sources: set[str] = set()
        for frame in self._frames:
            for slot in frame.slots:
                for elem in slot.elements:
                    all_elements.append(elem)
                    sources.add(elem.provenance.source_system)

        avg_trust = (
            sum(e.trust.composite for e in all_elements) / len(all_elements)
            if all_elements
            else 0.0
        )

        low_trust = self.get_low_trust_elements()
        conflicts = self.get_unresolved_conflicts()

        lines = [
            "## Data Confidence Summary",
            f"- {len(all_elements)} data elements from {len(sources)} sources",
            f"- Average trust: {avg_trust:.2f}",
        ]

        if low_trust:
            lines.append(f"- {len(low_trust)} low-trust elements:")
            for elem in low_trust[:5]:  # cap at 5 to avoid prompt bloat
                lines.append(
                    f"  - {elem.id}: source={elem.provenance.source_system}, "
                    f"precision={elem.provenance.precision.value}, "
                    f"trust={elem.trust.composite:.2f}"
                )

        if conflicts:
            lines.append(f"- {len(conflicts)} unresolved conflicts:")
            for c in conflicts[:3]:  # cap at 3
                lines.append(
                    f"  - {c.field}: {c.value_a} ({c.element_a.provenance.source_system}) "
                    f"vs {c.value_b} ({c.element_b.provenance.source_system})"
                )

        return "\n".join(lines)

    def reset(self) -> None:
        self._frames.clear()
```

### Tests (`tests/unit/core/epistemic/test_epistemic_tracker.py`):
- Retrieves trust for specific elements
- `get_low_trust_elements` returns correct elements below threshold
- `get_unresolved_conflicts` returns only `defer_to_agent` conflicts
- `generate_epistemic_prompt` produces readable summary with correct counts
- Works across multiple ingested frames
- Returns empty summary when no frames ingested

### Completion Check
- EpistemicTracker passes all tests
- Prompt generation is accurate and readable

---

## 12. Milestone 10: Agent SDK & Intent Formulator

### Goal
Build the high-level SDK that agent frameworks use to interact with Provena.

### File: `src/provena/agent/intent_formulator.py`

```python
"""
IntentFormulator provides builder methods for constructing well-formed intents.
Validates everything via Pydantic before returning.
"""

from __future__ import annotations

import time
from typing import Any

from provena.types.intent import (
    AggregateAnalysisIntent,
    BaseIntent,
    CompositeIntent,
    EscapeHatchIntent,
    FilterClause,
    FusionOperator,
    GraphTraversalIntent,
    MeasureSpec,
    NodeSpec,
    OntologyQueryIntent,
    OrderSpec,
    PointLookupIntent,
    SemanticSearchIntent,
    TemporalTrendIntent,
    TimeWindow,
)


class IntentFormulator:
    def __init__(self) -> None:
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"intent-{self._counter}-{int(time.time() * 1000)}"

    def point_lookup(
        self,
        entity: str,
        identifier: dict[str, str | int],
        fields: list[str] | None = None,
    ) -> PointLookupIntent:
        return PointLookupIntent(
            id=self._next_id(), entity=entity, identifier=identifier, fields=fields,
        )

    def temporal_trend(
        self,
        entity: str,
        metric: str,
        window: dict[str, str],
        granularity: str | None = None,
        direction: str | None = None,
        join_key: str | None = None,
    ) -> TemporalTrendIntent:
        return TemporalTrendIntent(
            id=self._next_id(),
            entity=entity,
            metric=metric,
            window=TimeWindow(**window),
            granularity=granularity,
            direction=direction,
            join_key=join_key,
        )

    def aggregate_analysis(
        self,
        entity: str,
        measures: list[dict[str, Any]],
        dimensions: list[str],
        filters: list[dict[str, Any]] | None = None,
        order_by: list[dict[str, str]] | None = None,
        having: list[dict[str, Any]] | None = None,
    ) -> AggregateAnalysisIntent:
        return AggregateAnalysisIntent(
            id=self._next_id(),
            entity=entity,
            measures=[MeasureSpec(**m) for m in measures],
            dimensions=dimensions,
            filters=[FilterClause(**f) for f in filters] if filters else None,
            order_by=[OrderSpec(**o) for o in order_by] if order_by else None,
            having=[FilterClause(**h) for h in having] if having else None,
        )

    # ... implement builder methods for:
    # graph_traversal, semantic_search, ontology_query, escape_hatch

    def composite(
        self,
        sub_intents: list[BaseIntent],
        fusion_operator: str | FusionOperator,
        fusion_key: str | None = None,
    ) -> CompositeIntent:
        op = FusionOperator(fusion_operator) if isinstance(fusion_operator, str) else fusion_operator
        return CompositeIntent(
            id=self._next_id(),
            sub_intents=sub_intents,
            fusion_operator=op,
            fusion_key=fusion_key,
        )
```

### File: `src/provena/agent/agent_sdk.py`

```python
"""
High-level SDK — the main public API of Provena.
Agent frameworks instantiate this and call query().
"""

from __future__ import annotations

from provena.agent.intent_formulator import IntentFormulator
from provena.core.epistemic.epistemic_tracker import EpistemicTracker
from provena.core.router.semantic_router import SemanticRouter
from provena.types.context import ContextFrame
from provena.types.intent import BaseIntent


class Provena:
    """
    Main entry point for agent frameworks.

    Usage:
        provena = Provena(router)
        intent = provena.formulator.point_lookup("customer", {"id": "C-1042"})
        frame = await provena.query(intent)
        print(provena.get_epistemic_context())
    """

    def __init__(self, router: SemanticRouter) -> None:
        self.formulator = IntentFormulator()
        self.tracker = EpistemicTracker()
        self._router = router

    async def query(self, intent: BaseIntent) -> ContextFrame:
        """Send an intent and get back an enriched context frame."""
        frame = await self._router.route(intent)
        self.tracker.ingest(frame)
        return frame

    def get_epistemic_context(self) -> str:
        """Get epistemic summary. Inject into agent's system prompt."""
        return self.tracker.generate_epistemic_prompt()

    def reset(self) -> None:
        """Reset for new conversation/session."""
        self.tracker.reset()
```

### Tests (`tests/unit/agent/`):
- IntentFormulator produces valid intents for all builder methods
- IntentFormulator rejects invalid parameters (Pydantic raises `ValidationError`)
- `Provena.query` routes through router and ingests into tracker
- `Provena.get_epistemic_context` returns accurate summary after queries

### Completion Check
- Agent SDK provides clean API
- All builder methods produce valid intents
- `pytest tests/unit/agent/` passes

---

## 13. Milestone 11: Cross-Source Join Optimizer

### Goal
Implement intelligent cross-source join strategies.

### File: `src/provena/core/router/join_optimizer.py`

```python
"""
Cross-source join optimizer.
Determines how to efficiently combine results from different storage paradigms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Awaitable

from provena.types.connector import ConnectorResult
from provena.types.router import ExecutionStep


class JoinStrategy(StrEnum):
    HASH_MATERIALIZE = "hash_materialize"
    NESTED_LOOKUP = "nested_lookup"
    CONTEXT_WINDOW_JOIN = "context_window_join"
    PUSH_DOWN = "push_down"


@dataclass
class JoinPlan:
    strategy: JoinStrategy
    left_step_id: str
    right_step_id: str
    join_key: str
    build_side: str | None = None  # "left" or "right" for hash_materialize
    estimated_result_size: int = 0


class JoinOptimizer:
    def plan_join(
        self,
        left_step: ExecutionStep,
        right_step: ExecutionStep,
        join_key: str,
        left_cardinality: int,
        right_cardinality: int,
    ) -> JoinPlan:
        """
        Determine optimal join strategy.

        Decision logic:
        1. If both target same connector → push_down
        2. If one side < 100 records → hash_materialize (build on small side)
        3. If both sides < 50 records → context_window_join
        4. Otherwise → hash_materialize on smaller side
        """
        # ... implement this
        pass

    async def execute_hash_materialize(
        self,
        build_result: ConnectorResult,
        probe_step: ExecutionStep,
        join_key: str,
        execute_step: Callable[[ExecutionStep], Awaitable[ConnectorResult]],
    ) -> list[ConnectorResult]:
        """
        1. Extract join key values from build_result
        2. Add IN filter to probe_step's intent to narrow scope
        3. Execute narrowed probe_step
        4. Return both results
        """
        # ... implement this
        pass
```

### Tests (`tests/unit/core/router/test_join_optimizer.py`):
- Selects push_down when both steps target same connector
- Selects hash_materialize with correct build side
- Selects context_window_join for small cardinalities
- `execute_hash_materialize` correctly narrows probe-side scope

### Completion Check
- Join optimizer makes correct strategy decisions for all cases
- `pytest` passes

---

## 14. Milestone 12: End-to-End Integration & CLI

### Goal
Wire everything together, create examples, and validate the full stack.

### File: `src/provena/__init__.py` (public API)

```python
"""Provena — Epistemic provenance for AI agents."""

from provena.agent.agent_sdk import Provena
from provena.agent.intent_formulator import IntentFormulator
from provena.connectors.base_connector import BaseConnector
from provena.connectors.capability_registry import CapabilityRegistry
from provena.connectors.document.document_connector import DocumentConnector
from provena.connectors.olap.olap_connector import OLAPConnector
from provena.connectors.oltp.oltp_connector import OLTPConnector
from provena.core.context.context_compiler import ContextCompiler
from provena.core.epistemic.epistemic_tracker import EpistemicTracker
from provena.core.provenance.trust_scorer import TrustScorer
from provena.core.router.semantic_router import SemanticRouter
from provena.mcp.mcp_adapter import MCPAdapter

__all__ = [
    "Provena",
    "IntentFormulator",
    "SemanticRouter",
    "ContextCompiler",
    "TrustScorer",
    "EpistemicTracker",
    "CapabilityRegistry",
    "BaseConnector",
    "MCPAdapter",
    "OLAPConnector",
    "OLTPConnector",
    "DocumentConnector",
]
```

### File: `examples/cross_source_query.py`

```python
"""
Example: "Show me customers likely to churn who also have
unresolved support tickets and declining usage trends"

Spans OLAP (churn scores), OLTP (tickets), time-series (usage).
"""

import asyncio

from provena import (
    Provena,
    CapabilityRegistry,
    ContextCompiler,
    OLAPConnector,
    OLTPConnector,
    SemanticRouter,
    TrustScorer,
)
from provena.connectors.executor import MockQueryExecutor
from provena.core.provenance.trust_scorer import TrustScorerConfig
from provena.core.router.cost_estimator import CostEstimator
from provena.core.router.intent_decomposer import IntentDecomposer
from provena.core.router.query_planner import QueryPlanner


async def main() -> None:
    # 1. Set up connectors with mock executors
    olap_executor = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "churn_probability": 0.89, "region": "west"},
        {"customer_id": "C-2091", "churn_probability": 0.76, "region": "east"},
    ])
    oltp_executor = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "ticket_id": "T-501", "status": "unresolved"},
    ])

    registry = CapabilityRegistry()
    registry.register(OLAPConnector(executor=olap_executor))
    registry.register(OLTPConnector(executor=oltp_executor))

    # 2. Set up router
    trust_config = TrustScorerConfig(source_authority_map={
        "snowflake.analytics": 0.95,
        "postgres.production": 0.9,
    })
    compiler = ContextCompiler(TrustScorer(trust_config))
    planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
    router = SemanticRouter(planner, compiler, registry)

    # 3. Create Provena instance
    provena = Provena(router)

    # 4. Build composite intent
    intent = provena.formulator.composite(
        sub_intents=[
            provena.formulator.aggregate_analysis(
                entity="customer_churn_scores",
                measures=[{"field": "churn_probability", "aggregation": "max"}],
                dimensions=["customer_id", "region"],
                having=[{"field": "churn_probability", "operator": "gt", "value": 0.7}],
            ),
            provena.formulator.point_lookup("support_tickets", {"status": "unresolved"}),
        ],
        fusion_operator="intersect",
        fusion_key="customer_id",
    )

    # 5. Execute
    frame = await provena.query(intent)

    # 6. Inspect
    print("=== Context Frame Stats ===")
    print(f"  Elements: {frame.stats.total_elements}")
    print(f"  Avg trust: {frame.stats.avg_trust_score:.2f}")
    print(f"  Slots: {frame.stats.slot_counts}")
    print(f"  Conflicts: {len(frame.conflicts)}")
    print()
    print("=== Epistemic Context ===")
    print(provena.get_epistemic_context())


if __name__ == "__main__":
    asyncio.run(main())
```

### Integration Test: `tests/integration/test_end_to_end.py`

Write a test that:
1. Sets up all components with mock executors
2. Sends a composite intent spanning 2+ data paradigms
3. Asserts the execution plan has correct parallelism
4. Asserts the context frame has correct slot types
5. Asserts provenance is attached to ALL elements
6. Asserts the epistemic tracker reports accurate trust
7. Asserts conflicts are detected when mock sources disagree
8. Asserts partial results work when one connector raises an exception

### Completion Check
- Full end-to-end flow works
- Example script runs: `python examples/cross_source_query.py`
- `pytest tests/integration/` passes
- `pytest` (all tests) passes
- `mypy src/provena --strict` passes
- `ruff check src/` passes

---

## Appendix A: Full Type Reference

All types live in `src/provena/types/`. Dependency graph:

```
intent.py          (no internal deps)
    ↓
connector.py       (imports provenance, context)
capability.py      (no internal deps)
    ↓
provenance.py      (no internal deps)
    ↓
context.py         (imports provenance)
    ↓
router.py          (imports intent, connector)
    ↓
errors.py          (no internal deps)
```

**RULE: Type files never import from implementation files (`core/`, `connectors/`, etc.). Types only import other types.**

---

## Appendix B: Error Handling Strategy

Every public method that can fail should:
1. Raise typed `ProvenaError` subclasses (never bare `Exception`)
2. Include context in the error (`intent_id`, `connector_id`, etc.)
3. Be catchable by error code: `error.code == ProvenaErrorCode.CONNECTOR_TIMEOUT`

For `SemanticRouter`:
- If ONE connector in parallel execution fails → continue with partial results
- Attach the error to `ExecutionResult.errors`
- `ContextCompiler` still compiles available results
- `EpistemicTracker` notes which intents had errors

For `ContextCompiler`:
- If trust scoring fails → assign `TrustScore(composite=0.0, label="uncertain", ...)`
- Never raise during `compile()` — always produce a frame

---

## Appendix C: Testing Strategy

### Unit Tests (`tests/unit/`)
Every module gets tests. Mock all dependencies — unit tests never cross module boundaries.

### Integration Tests (`tests/integration/`)
Test cross-module flows with `MockQueryExecutor` and `MockMCPTransport`.

### Fixtures (`tests/conftest.py`)
Create shared fixtures:
```python
import pytest
from provena.connectors.executor import MockQueryExecutor
from provena.core.provenance.trust_scorer import TrustScorer

@pytest.fixture
def trust_scorer() -> TrustScorer:
    return TrustScorer()

@pytest.fixture
def mock_executor() -> MockQueryExecutor:
    return MockQueryExecutor(records=[{"id": 1, "value": "test"}])
```

### Naming Convention
- Test files: `test_{module_name}.py`
- Classes: `TestClassName`
- Methods: `test_behavior_description`
  - Good: `test_returns_high_trust_for_strong_consistency`
  - Bad: `test_compute_freshness_called`

### Coverage Target
- >85% line coverage on `core/` and `connectors/`
- 100% on `types/` (schemas) and error classes
- Integration tests cover all happy paths + key error paths

### Run all checks:
```bash
pytest --cov=provena --cov-report=term-missing
mypy src/provena --strict
ruff check src/
```

---

## Implementation Order Summary

```
Milestone 0:  Project Bootstrap                    → foundation
Milestone 1:  Types & Intent Schema                → Pydantic models
Milestone 2:  Provenance & Trust Scoring           → epistemic foundation
Milestone 3:  Context Compiler                     → context assembly
Milestone 4:  Connector Interface & Base Class     → ABC contract
Milestone 5:  Reference Connectors (3)             → storage intelligence
Milestone 6:  Capability Registry                  → connector discovery
Milestone 7:  Semantic Router & Query Planner      → orchestration brain
Milestone 8:  MCP Integration Layer                → transport bridge
Milestone 9:  Epistemic Tracker                    → confidence reasoning
Milestone 10: Agent SDK & Intent Formulator        → public API
Milestone 11: Cross-Source Join Optimizer           → advanced optimization
Milestone 12: End-to-End Integration               → validation & examples
```

**DO NOT skip milestones. Each builds on the previous. Run `pytest` and `mypy` after every milestone before proceeding.**
