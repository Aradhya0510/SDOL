# Databricks notebook source
# MAGIC %md
# MAGIC # Provena Concepts: Understanding the Building Blocks
# MAGIC
# MAGIC This notebook walks through every core concept in the **Semantic Data Orchestration
# MAGIC Layer (Provena)** using in-memory mock data from the fleet management domain.
# MAGIC Nothing is written to any table -- this is a purely educational notebook.
# MAGIC
# MAGIC | Section | Concept | Key Idea |
# MAGIC |---------|---------|----------|
# MAGIC | 1 | Typed Intents | Describe WHAT you need, never HOW to get it |
# MAGIC | 2 | Typed Connectors | Storage-aware adapters that declare capabilities |
# MAGIC | 3 | Provenance Tracking | Every datum carries a birth certificate |
# MAGIC | 4 | Trust Scoring | Deterministic confidence from objective metadata |
# MAGIC | 5 | Conflict Detection | Structural detection when sources disagree |
# MAGIC | 6 | Context Compilation | Typed slots replace flat context windows |
# MAGIC | 7 | Epistemic Context | What the LLM actually sees about data quality |
# MAGIC | 8 | Full Pipeline | End-to-end determinism in a probabilistic system |
# MAGIC
# MAGIC **Prerequisites:** A Databricks workspace with serverless compute (Python 3.10).
# MAGIC The Provena source tree must be synced to your workspace (see the `provena_project_root` widget).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 0: Setup

# COMMAND ----------

# MAGIC %pip install -U -qqqq pydantic>=2.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("provena_project_root", "/Workspace/Users/{user}/SDOL")

# COMMAND ----------

import sys, os

PROVENA_PROJECT_ROOT = dbutils.widgets.get("provena_project_root")

try:
    import provena
except ImportError:
    resolved = PROVENA_PROJECT_ROOT.replace(
        "{user}", spark.sql("SELECT current_user()").first()[0]
    )
    src_path = os.path.join(resolved, "src")
    if os.path.isdir(src_path):
        sys.path.insert(0, src_path)
        import provena
        print(f"Loaded Provena from {src_path}")
    else:
        raise ImportError(f"Provena not found at {resolved}")

print(f"Provena version loaded -- exports: {provena.__all__[:8]}...")

# COMMAND ----------

# Core imports used throughout the notebook
from provena import (
    IntentFormulator,
    GenericOLTPConnector,
    GenericOLAPConnector,
    GenericDocumentConnector,
    CapabilityRegistry,
    TrustScorer,
    ContextCompiler,
    EpistemicTracker,
    Provena as ProvenaClient,
)
from provena.connectors.executor import MockQueryExecutor
from provena.core.provenance.envelope import create_envelope
from provena.core.provenance.trust_scorer import TrustScorerConfig, CONSISTENCY_SCORES, PRECISION_SCORES
from provena.core.context.conflict_detector import ConflictDetector
from provena.core.context.conflict_resolver import ConflictResolver
from provena.core.context.context_compiler import CompilerInput
from provena.core.router.semantic_router import SemanticRouter
from provena.core.router.query_planner import QueryPlanner
from provena.core.router.intent_decomposer import IntentDecomposer
from provena.core.router.cost_estimator import CostEstimator
from provena.types.provenance import (
    RetrievalMethod,
    ConsistencyGuarantee,
    PrecisionClass,
    ProvenanceEnvelope,
)
from provena.types.context import ContextElement, ContextSlotType
import json, textwrap

print("All imports successful.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1: Typed Intents -- "What, Not How"
# MAGIC
# MAGIC Provena defines **8 intent types** that describe WHAT the agent needs from data,
# MAGIC never HOW to get it. Each intent is a Pydantic model with strict validation.
# MAGIC The same intent can be routed to completely different backends depending on what
# MAGIC connectors are registered.
# MAGIC
# MAGIC | Intent Type | Purpose |
# MAGIC |-------------|---------|
# MAGIC | `point_lookup` | Get the current state of one entity by ID |
# MAGIC | `aggregate_analysis` | Statistical summaries across dimensions |
# MAGIC | `temporal_trend` | Change patterns over a time window |
# MAGIC | `semantic_search` | Find data by meaning similarity |
# MAGIC | `composite` | Combine results from multiple sub-intents |
# MAGIC | `graph_traversal` | Walk entity relationships |
# MAGIC | `ontology_query` | Class-based inferences and entailments |
# MAGIC | `escape_hatch` | Bypass for queries outside the type system |

# COMMAND ----------

formulator = IntentFormulator()

# -- 1. Point Lookup: "Get me this specific machine" --
point = formulator.point_lookup(
    "fleet_machines",
    {"machine_id": "EXC-0342"},
)
print("=== Point Lookup Intent ===")
print(json.dumps(point.model_dump(), indent=2))

# COMMAND ----------

# -- 2. Aggregate Analysis: "Summarize temperature by region" --
agg = formulator.aggregate_analysis(
    "telemetry_daily",
    measures=[{"field": "avg_engine_temp", "aggregation": "avg"}],
    dimensions=["region"],
)
print("=== Aggregate Analysis Intent ===")
print(json.dumps(agg.model_dump(), indent=2))

# COMMAND ----------

# -- 3. Temporal Trend: "Show fuel efficiency trend over the last 90 days" --
trend = formulator.temporal_trend(
    "telemetry_readings",
    metric="fuel_efficiency_lpkm",
    window={"relative": "last_90d"},
    granularity="1M",
)
print("=== Temporal Trend Intent ===")
print(json.dumps(trend.model_dump(), indent=2))

# COMMAND ----------

# -- 4. Semantic Search: "Find similar maintenance issues" --
search = formulator.semantic_search(
    query="hydraulic failure patterns",
    collection="maintenance_logs",
)
print("=== Semantic Search Intent ===")
print(json.dumps(search.model_dump(), indent=2))

# COMMAND ----------

# -- 5. Composite: "Combine data from multiple sources" --
oltp_intent = formulator.point_lookup("fleet_machines", {"machine_id": "EXC-0342"})
olap_intent = formulator.aggregate_analysis(
    "telemetry_daily",
    measures=[{"field": "avg_engine_temp", "aggregation": "avg"}],
    dimensions=["machine_id"],
    filters=[{"field": "machine_id", "operator": "eq", "value": "EXC-0342"}],
)
composite = formulator.composite(
    sub_intents=[oltp_intent, olap_intent],
    fusion_operator="union",
    fusion_key="machine_id",
)
print("=== Composite Intent ===")
print(json.dumps(composite.model_dump(), indent=2, default=str))

# COMMAND ----------

# -- 6-8: Briefly mention the remaining intent types --
print("=== Additional Intent Types (available for other domains) ===\n")

graph = formulator.graph_traversal(
    start_node={"type": "machine", "identifier": {"machine_id": "EXC-0342"}},
    max_depth=3,
    edge_types=["supplies_parts_to", "maintained_by"],
    direction="outbound",
)
print(f"graph_traversal : {graph.type} -- walk entity relationships")
print(f"  start_node={graph.start_node.type}, max_depth={graph.max_depth}\n")

ontology = formulator.ontology_query(
    subject="ExcavatorClass",
    predicate="requires_maintenance_when",
    inference_depth=2,
)
print(f"ontology_query  : {ontology.type} -- class-based inference")
print(f"  subject={ontology.subject}, predicate={ontology.predicate}\n")

escape = formulator.escape_hatch(
    target_connector="legacy_erp",
    raw_parameters={"sql": "SELECT * FROM spare_parts WHERE sku = 'HYD-4420'"},
    description="Fetch spare part details from legacy ERP system",
)
print(f"escape_hatch    : {escape.type} -- bypass for edge cases")
print(f"  target_connector={escape.target_connector}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC The same intent can be routed to completely different backends -- the agent does
# MAGIC not need to know whether the data lives in a SQL warehouse, a key-value store,
# MAGIC or a vector index. A `point_lookup` for `fleet_machines` could hit Postgres today
# MAGIC and Databricks Lakebase tomorrow. The intent is the same; only the routing changes.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2: Typed Connectors -- "Storage-Aware Adapters"
# MAGIC
# MAGIC Provena connectors follow a three-tier architecture:
# MAGIC
# MAGIC 1. **BaseConnector** -- abstract foundation (4-stage pipeline: interpret, synthesize, execute, normalize)
# MAGIC 2. **Paradigm bases** -- `BaseOLTPConnector`, `BaseOLAPConnector`, `BaseDocumentConnector`
# MAGIC 3. **Provider extensions** -- `GenericOLTP...`, `DatabricksDBSQL...`, etc.
# MAGIC
# MAGIC Each connector *declares* what it can do via `get_capabilities()`. The
# MAGIC `CapabilityRegistry` uses those declarations to route intents automatically.

# COMMAND ----------

# Create mock executors that return realistic fleet data
oltp_executor = MockQueryExecutor(
    records=[{
        "machine_id": "EXC-0342",
        "model": "Model X",
        "firmware_version": "v2.1",
        "status": "offline",
        "region": "north_america",
        "gps_lat": 42.3601,
        "gps_lon": -71.0589,
        "last_heartbeat_at": "2026-04-02T14:32:00Z",
    }]
)

olap_executor = MockQueryExecutor(
    records=[
        {"region": "north_america", "avg_engine_temp": 92.4, "machine_count": 132},
        {"region": "europe", "avg_engine_temp": 88.1, "machine_count": 118},
        {"region": "asia_pacific", "avg_engine_temp": 95.7, "machine_count": 145},
        {"region": "middle_east", "avg_engine_temp": 101.2, "machine_count": 105},
    ]
)

doc_executor = MockQueryExecutor(
    records=[
        {
            "log_id": "LOG-002841",
            "machine_id": "EXC-0198",
            "fault_category": "hydraulic",
            "description": "Hydraulic cylinder response sluggish. Measured 1950 PSI "
                           "(expected 3000 PSI). Found internal leak in control valve.",
            "similarity_score": 0.92,
        },
        {
            "log_id": "LOG-004117",
            "machine_id": "EXC-0342",
            "fault_category": "hydraulic",
            "description": "Hydraulic fluid contamination detected during routine sample. "
                           "Particle count exceeded ISO 18/15 limits.",
            "similarity_score": 0.87,
        },
    ]
)

# Instantiate typed connectors
oltp_conn = GenericOLTPConnector(
    executor=oltp_executor,
    connector_id="fleet.oltp",
    source_system="databricks.lakebase",
    available_entities=["fleet_machines"],
)

olap_conn = GenericOLAPConnector(
    executor=olap_executor,
    connector_id="fleet.olap",
    source_system="databricks.dbsql",
    available_entities=["telemetry_readings", "telemetry_daily"],
)

doc_conn = GenericDocumentConnector(
    executor=doc_executor,
    connector_id="fleet.document",
    source_system="databricks.vector_search",
    available_entities=["maintenance_logs"],
)

print("Three connectors created: OLTP, OLAP, Document")

# COMMAND ----------

# Show each connector's capabilities
for conn in [oltp_conn, olap_conn, doc_conn]:
    caps = conn.get_capabilities()
    print(f"\n{'=' * 60}")
    print(f"Connector:     {caps.connector_id}")
    print(f"Type:          {caps.connector_type}")
    print(f"Intent types:  {caps.supported_intent_types}")
    print(f"Entities:      {caps.available_entities}")
    print(f"Latency:       {caps.performance.estimated_latency_ms}ms")
    print(f"Max results:   {caps.performance.max_result_cardinality:,}")
    print(f"Capabilities:  {caps.capabilities.model_dump()}")

# COMMAND ----------

# Register in a CapabilityRegistry and show intent routing
registry = CapabilityRegistry()
registry.register(oltp_conn)
registry.register(olap_conn)
registry.register(doc_conn)

# Route a point_lookup -- should go to OLTP
point_intent = formulator.point_lookup("fleet_machines", {"machine_id": "EXC-0342"})
candidates = registry.find_candidates(point_intent)
print(f"Intent: point_lookup('fleet_machines') --> Best connector: {candidates[0].capability.connector_id}")
print(f"  Suitability scores: {[(c.capability.connector_id, round(c.suitability_score, 3)) for c in candidates]}")

print()

# Route an aggregate -- should go to OLAP
agg_intent = formulator.aggregate_analysis(
    "telemetry_daily",
    measures=[{"field": "avg_engine_temp", "aggregation": "avg"}],
    dimensions=["region"],
)
candidates = registry.find_candidates(agg_intent)
print(f"Intent: aggregate_analysis('telemetry_daily') --> Best connector: {candidates[0].capability.connector_id}")
print(f"  Suitability scores: {[(c.capability.connector_id, round(c.suitability_score, 3)) for c in candidates]}")

print()

# Route a semantic search -- should go to Document
search_intent = formulator.semantic_search(
    query="hydraulic failure patterns",
    collection="maintenance_logs",
)
candidates = registry.find_candidates(search_intent)
print(f"Intent: semantic_search('maintenance_logs') --> Best connector: {candidates[0].capability.connector_id}")
print(f"  Suitability scores: {[(c.capability.connector_id, round(c.suitability_score, 3)) for c in candidates]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC Each connector declares what it CAN do. The registry matches intents to the best
# MAGIC connector automatically -- the agent never picks a backend. Suitability scoring
# MAGIC considers entity availability, latency profile, and capability alignment.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 3: Provenance Tracking -- "Where Did This Data Come From?"
# MAGIC
# MAGIC Every piece of data in Provena carries a **ProvenanceEnvelope** -- a structured
# MAGIC metadata record created at retrieval time (not generated by the LLM). It captures:
# MAGIC
# MAGIC - **source_system** -- which backend produced the data
# MAGIC - **retrieval_method** -- how it was obtained (direct query, cache hit, vector similarity, ...)
# MAGIC - **consistency** -- what guarantee the source provides (strong, eventual, best_effort, ...)
# MAGIC - **precision** -- how exact the data is (exact, exact_aggregate, similarity_ranked, ...)
# MAGIC - **staleness_window_sec** -- how stale the data might be
# MAGIC - **retrieved_at** -- timestamp of retrieval

# COMMAND ----------

# Create 3 envelopes representing different source characteristics

# 1. OLTP: real-time machine registry (strong consistency, exact, low staleness)
oltp_envelope = create_envelope(
    source_system="databricks.lakebase",
    retrieval_method=RetrievalMethod.DIRECT_QUERY,
    consistency=ConsistencyGuarantee.STRONG,
    precision=PrecisionClass.EXACT,
    staleness_window_sec=30.0,
    execution_ms=12.4,
)

# 2. OLAP: pre-aggregated daily telemetry (eventual consistency, 15-min batch lag)
olap_envelope = create_envelope(
    source_system="databricks.dbsql",
    retrieval_method=RetrievalMethod.COMPUTED_AGGREGATE,
    consistency=ConsistencyGuarantee.EVENTUAL,
    precision=PrecisionClass.EXACT_AGGREGATE,
    staleness_window_sec=900.0,
    execution_ms=245.8,
)

# 3. Vector Search: semantic search over maintenance logs
vector_envelope = create_envelope(
    source_system="databricks.vector_search",
    retrieval_method=RetrievalMethod.VECTOR_SIMILARITY,
    consistency=ConsistencyGuarantee.BEST_EFFORT,
    precision=PrecisionClass.SIMILARITY_RANKED,
    staleness_window_sec=180.0,
    execution_ms=89.3,
)

for label, env in [("OLTP", oltp_envelope), ("OLAP", olap_envelope), ("Vector Search", vector_envelope)]:
    print(f"\n{'=' * 60}")
    print(f"  {label} Provenance Envelope")
    print(f"{'=' * 60}")
    print(f"  source_system:        {env.source_system}")
    print(f"  retrieval_method:     {env.retrieval_method.value}")
    print(f"  consistency:          {env.consistency.value}")
    print(f"  precision:            {env.precision.value}")
    print(f"  staleness_window_sec: {env.staleness_window_sec}")
    print(f"  execution_ms:         {env.execution_ms}")
    print(f"  retrieved_at:         {env.retrieved_at}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC Every piece of data in Provena carries a "birth certificate" -- you can always trace
# MAGIC back to where it came from, when it was retrieved, and how reliable the source is.
# MAGIC This metadata is attached at retrieval time, not generated by the LLM.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 4: Trust Scoring -- "How Much Should I Trust This?"
# MAGIC
# MAGIC The `TrustScorer` computes a composite trust score from provenance metadata
# MAGIC across four dimensions:
# MAGIC
# MAGIC | Dimension | What it measures | Weight |
# MAGIC |-----------|-----------------|--------|
# MAGIC | `source_authority` | How authoritative is this source? (configurable per source) | 0.20 |
# MAGIC | `consistency_score` | What consistency guarantee? (STRONG=1.0, EVENTUAL=0.5, ...) | 0.30 |
# MAGIC | `freshness_score` | How recently retrieved vs. staleness window? | 0.20 |
# MAGIC | `precision_score` | How precise? (EXACT=1.0, SIMILARITY_RANKED=0.55, ...) | 0.30 |
# MAGIC
# MAGIC The composite score is a weighted sum: `0.2*authority + 0.3*consistency + 0.2*freshness + 0.3*precision`

# COMMAND ----------

# Configure custom authority weights for our fleet data sources
config = TrustScorerConfig(
    source_authority_map={
        "databricks.lakebase": 0.95,       # OLTP -- authoritative for real-time state
        "databricks.dbsql": 0.85,          # OLAP -- authoritative for analytics
        "databricks.vector_search": 0.70,  # Vector -- good for semantic, not authoritative
    }
)
scorer = TrustScorer(config=config)

# Score each envelope
envelopes = {
    "OLTP (Lakebase)": oltp_envelope,
    "OLAP (DBSQL)": olap_envelope,
    "Vector Search": vector_envelope,
}

print(f"{'Source':<22} {'Composite':>10} {'Authority':>10} {'Consistency':>12} {'Freshness':>10} {'Precision':>10} {'Label':>8}")
print("-" * 84)

scores = {}
for label, env in envelopes.items():
    trust = scorer.score(env)
    scores[label] = trust
    d = trust.dimensions
    print(
        f"{label:<22} "
        f"{trust.composite:>10.4f} "
        f"{d.source_authority:>10.4f} "
        f"{d.consistency_score:>12.4f} "
        f"{d.freshness_score:>10.4f} "
        f"{d.precision_score:>10.4f} "
        f"{trust.label:>8}"
    )

# COMMAND ----------

# Show how the composite score is computed for the OLTP source
oltp_trust = scores["OLTP (Lakebase)"]
d = oltp_trust.dimensions
print("=== Composite Score Breakdown: OLTP (Lakebase) ===\n")
print(f"  source_authority  = {d.source_authority:.4f}  x  weight 0.20  =  {0.20 * d.source_authority:.4f}")
print(f"  consistency_score = {d.consistency_score:.4f}  x  weight 0.30  =  {0.30 * d.consistency_score:.4f}")
print(f"  freshness_score   = {d.freshness_score:.4f}  x  weight 0.20  =  {0.20 * d.freshness_score:.4f}")
print(f"  precision_score   = {d.precision_score:.4f}  x  weight 0.30  =  {0.30 * d.precision_score:.4f}")
print(f"  {'':49s}--------")
computed = (
    0.20 * d.source_authority
    + 0.30 * d.consistency_score
    + 0.20 * d.freshness_score
    + 0.30 * d.precision_score
)
print(f"  composite = {computed:.4f}  (label: {oltp_trust.label})")

# COMMAND ----------

# Show the lookup tables that drive consistency and precision scores
print("=== Consistency Guarantee --> Score ===")
for k, v in CONSISTENCY_SCORES.items():
    print(f"  {k.value:<20s} -> {v:.2f}")

print()
print("=== Precision Class --> Score ===")
for k, v in PRECISION_SCORES.items():
    print(f"  {k.value:<20s} -> {v:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC Trust scores are COMPUTED from objective metadata, not estimated by the LLM.
# MAGIC When Provena says "trust = 0.85", that is a deterministic calculation from
# MAGIC source authority (0.95), consistency (STRONG -> 1.0), freshness (age/staleness),
# MAGIC and precision (EXACT -> 1.0). The weights are configurable but the computation
# MAGIC is always transparent and reproducible.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 5: Conflict Detection and Resolution -- "When Sources Disagree"
# MAGIC
# MAGIC In multi-source architectures, data conflicts are inevitable. The classic
# MAGIC fleet-management example: the OLTP registry says machine EXC-0342 is `offline`
# MAGIC (updated in real-time), but the OLAP daily aggregate still shows `online`
# MAGIC (15-minute batch lag).
# MAGIC
# MAGIC Provena detects these conflicts structurally by comparing `ContextElement` data
# MAGIC for the same entity key across different source systems.

# COMMAND ----------

# Build two ContextElements for the same machine from different sources

# OLTP says: status = "offline" (real-time, strong consistency)
oltp_element = ContextElement(
    id="elem-oltp-1",
    data={
        "machine_id": "EXC-0342",
        "status": "offline",
        "model": "Model X",
        "firmware_version": "v2.1",
        "last_heartbeat_at": "2026-04-02T14:32:00Z",
    },
    provenance=oltp_envelope,
    trust=scorer.score(oltp_envelope),
    source_intent_id="intent-oltp-1",
    entity_key="EXC-0342",
)

# OLAP says: status = "online" (15-min stale, eventual consistency)
# We use the field name "status" to trigger conflict detection on shared keys
olap_element = ContextElement(
    id="elem-olap-1",
    data={
        "machine_id": "EXC-0342",
        "status": "online",
        "avg_engine_temp": 105.3,
        "max_engine_temp": 125.0,
        "avg_fuel_efficiency": 16.2,
    },
    provenance=olap_envelope,
    trust=scorer.score(olap_envelope),
    source_intent_id="intent-olap-1",
    entity_key="EXC-0342",
)

print("OLTP element -- status:", oltp_element.data["status"])
print("OLAP element -- status:", olap_element.data["status"])
print("\nBoth elements share entity_key='EXC-0342' but come from different sources.")

# COMMAND ----------

# Detect the conflict
detector = ConflictDetector()
conflicts = detector.detect([oltp_element, olap_element])

print(f"Conflicts detected: {len(conflicts)}\n")
for c in conflicts:
    print(f"  Field:   {c.field}")
    print(f"  Value A: {c.value_a} (source: {c.element_a.provenance.source_system})")
    print(f"  Value B: {c.value_b} (source: {c.element_b.provenance.source_system})")
    print(f"  Status:  {c.resolution.strategy}")

# COMMAND ----------

# Resolve the conflict using provenance-based heuristics
resolver = ConflictResolver()
resolved = [resolver.resolve(c) for c in conflicts]

for r in resolved:
    print(f"=== Resolved Conflict: field='{r.field}' ===")
    print(f"  Strategy: {r.resolution.strategy}")
    print(f"  Winner:   {r.resolution.winner}")
    print(f"  Reason:   {r.resolution.reason}")
    print()
    # Show why: OLTP has STRONG consistency, OLAP has EVENTUAL -- gap of 2 levels
    print("  Why this strategy?")
    print(f"    Element A consistency: {r.element_a.provenance.consistency.value} (level 3)")
    print(f"    Element B consistency: {r.element_b.provenance.consistency.value} (level 1)")
    print(f"    Gap = 2 (threshold for prefer_strongest_consistency)")

# COMMAND ----------

# Demonstrate presence conflict detection (when one source has no data)
print("=== Presence Conflict Detection ===\n")

# Only OLTP returned data; OLAP returned nothing for this query
presence_conflicts = detector.detect_presence_conflicts(
    elements=[oltp_element],
    expected_sources=[
        {"source_system": "databricks.lakebase", "connector_id": "fleet.oltp"},
        {"source_system": "databricks.dbsql", "connector_id": "fleet.olap"},
    ],
)

for pc in presence_conflicts:
    print(f"  Missing source: {pc.missing_source_system} (connector: {pc.missing_connector_id})")
    print(f"  Resolution:     {pc.resolution.strategy}")
    print(f"  Reason:         {pc.resolution.reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC An LLM MIGHT notice that two data sources disagree. Provena ALWAYS detects it.
# MAGIC And it resolves it using deterministic heuristics based on provenance -- not vibes.
# MAGIC This is the difference between probabilistic behavior and structural guarantees.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 6: Context Compilation -- "Building the Full Picture"
# MAGIC
# MAGIC The `ContextCompiler` assembles individual data elements into a `ContextFrame` --
# MAGIC a structured replacement for the flat context window. The frame contains:
# MAGIC
# MAGIC - **Typed slots** -- elements grouped by semantic type (STRUCTURED, TEMPORAL, UNSTRUCTURED, ...)
# MAGIC - **Conflicts** -- detected and resolved automatically
# MAGIC - **Stats** -- element counts, average trust
# MAGIC - **Trust summary** -- overall confidence, lowest-trust source, advisory

# COMMAND ----------

# Build a ContextCompiler with our custom trust scorer
compiler = ContextCompiler(trust_scorer=scorer)

# Add the OLTP element (machine registry lookup)
compiler.add_element(CompilerInput(
    slot_type=ContextSlotType.STRUCTURED,
    data=oltp_element.data,
    provenance=oltp_envelope,
    source_intent_id="intent-oltp-1",
    entity_key="EXC-0342",
))

# Add the OLAP element (daily telemetry aggregate)
compiler.add_element(CompilerInput(
    slot_type=ContextSlotType.STRUCTURED,
    data=olap_element.data,
    provenance=olap_envelope,
    source_intent_id="intent-olap-1",
    entity_key="EXC-0342",
))

# Add a vector search result (maintenance log)
compiler.add_element(CompilerInput(
    slot_type=ContextSlotType.UNSTRUCTURED,
    data={
        "log_id": "LOG-002841",
        "machine_id": "EXC-0198",
        "fault_category": "hydraulic",
        "description": "Hydraulic cylinder response sluggish. Measured 1950 PSI.",
        "similarity_score": 0.92,
    },
    provenance=vector_envelope,
    source_intent_id="intent-search-1",
    entity_key=None,  # no entity key for unstructured search results
))

# Compile the frame
frame = compiler.compile()

# COMMAND ----------

# Display the frame structure
print("=== ContextFrame ===\n")
print(f"Assembled at: {frame.assembled_at}")
print(f"Total elements: {frame.stats.total_elements}")
print(f"Average trust:  {frame.stats.avg_trust_score:.4f}")
print(f"Slot counts:    {frame.stats.slot_counts}")

print(f"\n--- Slots ({len(frame.slots)}) ---")
for slot in frame.slots:
    print(f"\n  Slot type: {slot.type.value}")
    print(f"  Interpretation: {slot.interpretation_notes}")
    print(f"  Elements: {len(slot.elements)}")
    for elem in slot.elements:
        print(f"    [{elem.id}] trust={elem.trust.composite:.4f} source={elem.provenance.source_system}")
        if isinstance(elem.data, dict):
            for k, v in list(elem.data.items())[:4]:
                print(f"      {k}: {v}")

print(f"\n--- Conflicts ({len(frame.conflicts)}) ---")
for c in frame.conflicts:
    print(f"  field='{c.field}': {c.value_a} vs {c.value_b}")
    print(f"  resolution: {c.resolution.strategy} -> winner={c.resolution.winner}")
    print(f"  reason: {c.resolution.reason}")

if frame.trust_summary:
    print(f"\n--- Trust Summary ---")
    print(f"  Overall confidence:  {frame.trust_summary.overall_confidence}")
    print(f"  Lowest trust source: {frame.trust_summary.lowest_trust_source}")
    if frame.trust_summary.advisory:
        print(f"  Advisory:            {frame.trust_summary.advisory}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC The ContextFrame is the structured replacement for a flat context window. Instead
# MAGIC of dumping raw text, the agent receives typed slots, trust scores, conflict
# MAGIC resolutions, and an advisory -- all computed, not hallucinated.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 7: Epistemic Context -- "What the Agent Actually Sees"
# MAGIC
# MAGIC The `EpistemicTracker` takes a ContextFrame and generates a text fragment that
# MAGIC gets injected into the LLM's system message. This gives the agent concrete
# MAGIC numbers to reason over, rather than vague instructions like "be careful with
# MAGIC data quality."

# COMMAND ----------

# Generate the epistemic prompt from the compiled frame
tracker = EpistemicTracker()
tracker.ingest(frame)

epistemic_prompt = tracker.generate_epistemic_prompt()

print("=== Epistemic Prompt (injected into agent system message) ===\n")
print(epistemic_prompt)

# COMMAND ----------

# Compare to what a flat context window looks like -- just raw data, no metadata
print("=== For comparison: what a FLAT context window looks like ===\n")
flat_context = (
    "Machine EXC-0342: Model X, firmware v2.1, status offline, region north_america\n"
    "Telemetry for EXC-0342: avg_engine_temp 105.3, status online, avg_fuel_efficiency 16.2\n"
    "Maintenance log LOG-002841: Hydraulic cylinder response sluggish. Measured 1950 PSI."
)
print(flat_context)
print()
print("Notice what is MISSING from the flat version:")
print("  - No trust scores")
print("  - No source attribution")
print("  - No consistency guarantees")
print("  - No staleness information")
print("  - No conflict detection")
print("  - The status contradiction (offline vs online) is buried in raw text")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC This is the prompt fragment Provena injects into the agent's system message. It gives
# MAGIC the LLM concrete numbers to reason over -- not vague instructions to "be careful
# MAGIC with data quality." The agent can say "I trust the OLTP source (0.95 authority,
# MAGIC strong consistency) over the OLAP source (0.85 authority, eventual consistency)"
# MAGIC because Provena gave it those numbers.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 8: The Full Pipeline -- "Determinism in a Probabilistic System"
# MAGIC
# MAGIC Now we wire up the complete Provena pipeline end-to-end using mock executors:
# MAGIC
# MAGIC ```
# MAGIC Intent --> Registry (routing) --> Planner (decompose + assign) --> Router (execute)
# MAGIC   --> Connectors (synthesize + run) --> Compiler (slots + conflicts + trust)
# MAGIC   --> EpistemicTracker (prompt generation)
# MAGIC ```

# COMMAND ----------

import asyncio

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # nest_asyncio may not be installed; asyncio.run will work directly

# Re-create fresh connectors for the full pipeline demo
oltp_exec = MockQueryExecutor(records=[{
    "machine_id": "EXC-0342",
    "model": "Model X",
    "firmware_version": "v2.1",
    "status": "offline",
    "region": "north_america",
    "last_heartbeat_at": "2026-04-02T14:32:00Z",
}])

oltp_c = GenericOLTPConnector(
    executor=oltp_exec,
    connector_id="fleet.oltp",
    source_system="databricks.lakebase",
    available_entities=["fleet_machines"],
)

# Build the full pipeline
pipeline_registry = CapabilityRegistry()
pipeline_registry.register(oltp_c)

pipeline_scorer = TrustScorer(TrustScorerConfig(
    source_authority_map={"databricks.lakebase": 0.95}
))
pipeline_compiler = ContextCompiler(trust_scorer=pipeline_scorer)
decomposer = IntentDecomposer()
cost_estimator = CostEstimator()
planner = QueryPlanner(pipeline_registry, decomposer, cost_estimator)
router = SemanticRouter(planner, pipeline_compiler, pipeline_registry)
sdk = ProvenaClient(router)

print("Full Provena pipeline assembled:")
print("  Registry -> Planner -> Router -> Compiler -> EpistemicTracker")

# COMMAND ----------

# Execute a point_lookup intent end-to-end
intent = sdk.formulator.point_lookup("fleet_machines", {"machine_id": "EXC-0342"})

print("=== Stage 1: Intent ===")
print(f"  Type:   {intent.type}")
print(f"  Entity: {intent.entity}")
print(f"  ID:     {intent.identifier}")

# Show the execution plan
plan = planner.plan(intent)
print(f"\n=== Stage 2: Routing Decision ===")
for step in plan.steps:
    print(f"  Step {step.step_id}: connector={step.connector_id}, est_ms={step.estimated_ms:.0f}ms")

print(f"\n=== Stage 3: Query Synthesis ===")
synth_query = oltp_c.synthesize_query(intent)
print(f"  Native query: {synth_query}")

# Run the full pipeline
loop = asyncio.get_event_loop()
result_frame = loop.run_until_complete(sdk.query(intent))

print(f"\n=== Stage 4: Execution (mock) + Normalization ===")
print(f"  Elements in frame: {result_frame.stats.total_elements}")

print(f"\n=== Stage 5: Context Compilation ===")
for slot in result_frame.slots:
    for elem in slot.elements:
        print(f"  Slot: {slot.type.value}")
        print(f"  Data: {elem.data}")
        print(f"  Provenance: source={elem.provenance.source_system}, "
              f"consistency={elem.provenance.consistency.value}")

print(f"\n=== Stage 6: Trust Scoring ===")
for slot in result_frame.slots:
    for elem in slot.elements:
        d = elem.trust.dimensions
        print(f"  Composite: {elem.trust.composite:.4f} (label: {elem.trust.label})")
        print(f"    authority={d.source_authority:.2f}  consistency={d.consistency_score:.2f}  "
              f"freshness={d.freshness_score:.2f}  precision={d.precision_score:.2f}")

print(f"\n=== Stage 7: Epistemic Prompt ===")
epistemic = sdk.get_epistemic_context()
print(epistemic)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC The LLM is inherently probabilistic -- it might notice data quality issues, or it
# MAGIC might not. Provena wraps **deterministic guarantees** around the probabilistic core:
# MAGIC
# MAGIC - Provenance is ALWAYS tracked
# MAGIC - Conflicts are ALWAYS detected
# MAGIC - Trust is ALWAYS scored
# MAGIC - Resolution strategies are ALWAYS applied
# MAGIC
# MAGIC The LLM focuses on reasoning and communication. Provena handles the data integrity.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated the core building blocks of Provena:
# MAGIC
# MAGIC | Concept | What it does | Why it matters |
# MAGIC |---------|-------------|----------------|
# MAGIC | Typed Intents | Describe data needs declaratively | Backend-agnostic, validatable, decomposable |
# MAGIC | Typed Connectors | Adapt to storage paradigms | Capability-based routing, not hardcoded paths |
# MAGIC | Provenance | Track data lineage at retrieval time | Objective metadata, not LLM guesses |
# MAGIC | Trust Scoring | Compute confidence from provenance | Deterministic, weighted, transparent |
# MAGIC | Conflict Detection | Find disagreements across sources | Structural guarantee, never missed |
# MAGIC | Context Compilation | Build typed frames from raw results | Slots + conflicts + trust, not token soup |
# MAGIC | Epistemic Context | Generate LLM-ready confidence info | Concrete numbers for the agent to reason over |
# MAGIC | Full Pipeline | Wire everything end-to-end | Determinism wrapped around a probabilistic core |
# MAGIC
# MAGIC ### What comes next
# MAGIC
# MAGIC These concepts power the benchmark in the companion notebooks:
# MAGIC
# MAGIC - **02_fleet_setup** -- Creates real Databricks tables (500 machines, 360K telemetry
# MAGIC   readings, 90K daily aggregates, 5K maintenance logs) plus a Vector Search index.
# MAGIC   Seeds the OLTP/OLAP conflict for EXC-0342.
# MAGIC
# MAGIC - **03_fleet_benchmark** -- Runs both a vanilla MCP agent and a Provena-enhanced agent
# MAGIC   against the same questions, logging results to MLflow. Demonstrates the two failure
# MAGIC   modes that Provena structurally solves: token-busting cross-paradigm joins and
# MAGIC   epistemic conflicts.
# MAGIC
# MAGIC Run `02_fleet_setup` first to create the data, then `03_fleet_benchmark` to see
# MAGIC the evaluation results side-by-side.
