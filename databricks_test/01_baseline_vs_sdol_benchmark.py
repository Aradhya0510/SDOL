# Databricks notebook source
# MAGIC %md
# MAGIC # Provena in Action: Seeing the Concepts Work with Real Data
# MAGIC
# MAGIC This notebook is an **interactive walkthrough** of every core Provena concept,
# MAGIC executed against real Databricks tables created by notebook `02_fleet_setup`.
# MAGIC
# MAGIC Unlike the purely educational `00` notebook, this one uses real connectors
# MAGIC (Lakebase, DBSQL, Vector Search) against real data. Each section isolates
# MAGIC one concept so you can run the cells independently and inspect the output.
# MAGIC
# MAGIC | Section | Concept | What You Will See |
# MAGIC |---------|---------|-------------------|
# MAGIC | 0 | Setup | Wire the Provena pipeline with three Databricks connectors |
# MAGIC | 1 | Typed Intents | Declare what you need, not how to get it |
# MAGIC | 2 | Connector Pipeline | The 4-stage journey from intent to result |
# MAGIC | 3 | Trust Scoring | Computed confidence across three source types |
# MAGIC | 4 | Conflict Detection | When OLTP and OLAP disagree about the same machine |
# MAGIC | 5 | Epistemic Context | The data-quality briefing injected into the LLM |
# MAGIC | 6 | Semantic Search | Vector Search results with provenance metadata |
# MAGIC | 7 | Full Picture | Determinism wrapping probabilism — cost summary |
# MAGIC
# MAGIC **Prerequisite:** Run `02_fleet_setup` first to create the fleet tables and
# MAGIC Vector Search index.

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-vectorsearch nest_asyncio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 0: Setup and Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "users")
dbutils.widgets.text("schema", "aradhya_chouhan")
dbutils.widgets.text("vs_endpoint", "provena_fleet_vs")
dbutils.widgets.text("sdol_project_root", "/Workspace/Users/{user}/Provena")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint")
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.maintenance_logs_index"
SDOL_PROJECT_ROOT = dbutils.widgets.get("sdol_project_root")

print(f"Catalog:     {CATALOG}")
print(f"Schema:      {SCHEMA}")
print(f"VS Endpoint: {VS_ENDPOINT_NAME}")
print(f"VS Index:    {VS_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Provena from workspace path

# COMMAND ----------

import sys, os

try:
    import provena
except ImportError:
    resolved = SDOL_PROJECT_ROOT.replace("{user}", spark.sql("SELECT current_user()").first()[0])
    src_path = os.path.join(resolved, "src")
    if os.path.isdir(src_path):
        sys.path.insert(0, src_path)
        import provena
        print(f"Loaded Provena from {src_path}")
    else:
        raise ImportError(f"Provena not found at {resolved}")

print(f"Provena loaded — exports: {provena.__all__[:5]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wire up executors, connectors, and the full Provena pipeline

# COMMAND ----------

import asyncio, json, time, nest_asyncio
nest_asyncio.apply()

from provena import (
    Provena as ProvenaEngine,
    CapabilityRegistry,
    ContextCompiler,
    DatabricksDBSQLConnector,
    DatabricksLakebaseConnector,
    DatabricksVectorSearchConnector,
    SemanticRouter,
    TrustScorer,
)
from provena.core.provenance.trust_scorer import TrustScorerConfig
from provena.core.router.cost_estimator import CostEstimator
from provena.core.router.intent_decomposer import IntentDecomposer
from provena.core.router.query_planner import QueryPlanner
from provena.types.provenance import ConsistencyGuarantee


class SparkSQLExecutor:
    """Bridges Provena's QueryExecutor protocol with the notebook's SparkSession."""

    async def execute(self, query) -> dict:
        sql_str = query.sql
        for k, v in query.parameters.items():
            placeholder = f":{k}"
            if isinstance(v, str):
                sql_str = sql_str.replace(placeholder, f"'{v}'")
            else:
                sql_str = sql_str.replace(placeholder, str(v))
        df = spark.sql(sql_str)
        records = [row.asDict() for row in df.limit(500).collect()]
        return {"records": records, "meta": {"native_query": sql_str, "total_available": len(records)}}


class DatabricksVectorSearchExecutor:
    """Bridges Provena's QueryExecutor protocol with Databricks Vector Search."""

    def __init__(self, endpoint_name, index_name):
        from databricks.vector_search.client import VectorSearchClient
        self._index = VectorSearchClient().get_index(
            endpoint_name=endpoint_name, index_name=index_name,
        )

    async def execute(self, query) -> dict:
        kwargs = {
            "query_text": query.query_text,
            "columns": query.columns or [
                "log_id", "machine_id", "log_date",
                "fault_category", "severity", "description",
            ],
            "num_results": query.num_results,
        }
        if query.filters_json:
            kwargs["filters"] = query.filters_json
        results = self._index.similarity_search(**kwargs)
        columns = [c["name"] for c in results.get("manifest", {}).get("columns", [])]
        data_array = results.get("result", {}).get("data_array", [])
        records = [dict(zip(columns, row)) for row in data_array]
        return {"records": records, "meta": {"native_query": query.query_text, "total_available": len(records)}}


# --- Instantiate executors ---
sql_executor = SparkSQLExecutor()
vs_executor = DatabricksVectorSearchExecutor(VS_ENDPOINT_NAME, VS_INDEX_NAME)

ENTITY_KEYS = ("machine_id",)

# --- Register connectors ---
oltp_connector = DatabricksLakebaseConnector(
    executor=sql_executor,
    connector_id="fleet.oltp",
    source_system="databricks.lakebase.fleet",
    available_entities=["fleet_machines"],
    catalog=CATALOG,
    schema=SCHEMA,
    entity_key_fields=ENTITY_KEYS,
)

olap_connector = DatabricksDBSQLConnector(
    executor=sql_executor,
    connector_id="fleet.olap",
    source_system="databricks.sql_warehouse.telemetry",
    available_entities=["telemetry_readings", "telemetry_daily"],
    catalog=CATALOG,
    schema=SCHEMA,
    time_column_map={
        "telemetry_readings": "reading_time",
        "telemetry_daily": "report_date",
    },
    entity_key_fields=ENTITY_KEYS,
    consistency=ConsistencyGuarantee.EVENTUAL,
    staleness_sec=900.0,
)

doc_connector = DatabricksVectorSearchConnector(
    executor=vs_executor,
    connector_id="fleet.docs",
    source_system="databricks.vector_search.maintenance",
    available_entities=["maintenance_logs"],
    catalog=CATALOG,
    schema=SCHEMA,
    index_name=VS_INDEX_NAME,
)

# --- Build the Provena pipeline ---
registry = CapabilityRegistry()
registry.register(oltp_connector)
registry.register(olap_connector)
registry.register(doc_connector)

trust_cfg = TrustScorerConfig(source_authority_map={
    "databricks.lakebase.fleet": 0.95,
    "databricks.sql_warehouse.telemetry": 0.85,
    "databricks.vector_search.maintenance": 0.70,
})
compiler = ContextCompiler(TrustScorer(trust_cfg))
planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
router = SemanticRouter(planner, compiler, registry)
provena = ProvenaEngine(router)

print("Provena fleet pipeline ready")
print(f"  OLTP  entities: {oltp_connector._available_entities}")
print(f"  OLAP  entities: {olap_connector._available_entities} (consistency={olap_connector.default_consistency.value})")
print(f"  Doc   entities: {doc_connector._available_entities} (index={VS_INDEX_NAME})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper: pretty-print a ContextFrame

# COMMAND ----------

def inspect_frame(frame, title=""):
    """Pretty-print a ContextFrame showing data, provenance, and trust."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
    for slot in frame.slots:
        print(f"\nSlot Type: {slot.type}")
        print(f"Interpretation: {slot.interpretation_notes}")
        for i, elem in enumerate(slot.elements):
            print(f"\n  Element {i+1}:")
            print(f"    Data: {json.dumps(elem.data, default=str, indent=6) if isinstance(elem.data, dict) else elem.data}")
            print(f"    --- Provenance ---")
            print(f"    Source:       {elem.provenance.source_system}")
            print(f"    Method:       {elem.provenance.retrieval_method}")
            print(f"    Consistency:  {elem.provenance.consistency}")
            print(f"    Precision:    {elem.provenance.precision}")
            print(f"    Staleness:    {elem.provenance.staleness_window_sec}s")
            print(f"    --- Trust Score ---")
            print(f"    Composite:    {elem.trust.composite:.3f} ({elem.trust.label})")
            print(f"    Authority:    {elem.trust.dimensions.source_authority:.3f}")
            print(f"    Consistency:  {elem.trust.dimensions.consistency_score:.3f}")
            print(f"    Freshness:    {elem.trust.dimensions.freshness_score:.3f}")
            print(f"    Precision:    {elem.trust.dimensions.precision_score:.3f}")
    if frame.trust_summary:
        print(f"\n  Trust Summary:")
        print(f"    Overall: {frame.trust_summary.overall_confidence}")
        print(f"    Lowest:  {frame.trust_summary.lowest_trust_source}")
        print(f"    Advisory: {frame.trust_summary.advisory}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1: Typed Intents with Real Data
# MAGIC
# MAGIC ### Intent Formulation: Declaring What You Need
# MAGIC
# MAGIC With Provena, agents do not write SQL. They declare *typed intents* that describe
# MAGIC the information they need. The framework handles routing, SQL generation,
# MAGIC execution, and provenance tracking.
# MAGIC
# MAGIC Below we create and execute two different intent types against the real fleet data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Point Lookup — OLTP (fleet_machines)
# MAGIC
# MAGIC A `point_lookup` intent says: "I need the record for machine EXC-0342."
# MAGIC Provena routes this to the OLTP connector (Lakebase), which has strong
# MAGIC consistency guarantees and sub-second latency.

# COMMAND ----------

provena.reset()

# --- Step 1: Declare the intent ---
intent_oltp = provena.formulator.point_lookup("fleet_machines", {"machine_id": "EXC-0342"})

# Show the intent BEFORE execution — this is what the agent declared
print("INTENT (before execution):")
print(json.dumps(intent_oltp.model_dump(), indent=2, default=str))

# COMMAND ----------

# --- Step 2: Execute through the Provena pipeline ---
frame_oltp = asyncio.run(provena.query(intent_oltp))

# Show the ContextFrame AFTER execution — this is what Provena returned
inspect_frame(frame_oltp, "Point Lookup: EXC-0342 from OLTP (fleet_machines)")

# COMMAND ----------

# MAGIC %md
# MAGIC Notice three things in the output above:
# MAGIC 1. **Data** — the machine record, just like a SQL query would return
# MAGIC 2. **Provenance** — source system, retrieval method, consistency level, staleness window
# MAGIC 3. **Trust Score** — a computed composite score (0-1) with four dimensional breakdowns
# MAGIC
# MAGIC The agent did not write SQL. It declared an intent. Provena did the rest.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Aggregate Analysis — OLAP (telemetry_daily)
# MAGIC
# MAGIC An `aggregate_analysis` intent says: "I need average engine temperature by region."
# MAGIC Provena routes this to the OLAP connector (DBSQL) with push-down aggregation —
# MAGIC the `AVG()` and `GROUP BY` happen inside the SQL engine, not in Python.

# COMMAND ----------

intent_olap = provena.formulator.aggregate_analysis(
    entity="telemetry_daily",
    measures=[{"field": "avg_engine_temp", "aggregation": "avg"}],
    dimensions=["region"],
    order_by=[{"field": "avg_avg_engine_temp", "direction": "desc"}],
)

print("INTENT (before execution):")
print(json.dumps(intent_olap.model_dump(), indent=2, default=str))

# COMMAND ----------

frame_olap = asyncio.run(provena.query(intent_olap))
inspect_frame(frame_olap, "Aggregate Analysis: Avg Engine Temp by Region from OLAP")

# COMMAND ----------

# MAGIC %md
# MAGIC The same Provena pipeline routed this intent to the OLAP connector. The generated
# MAGIC SQL contains `AVG(avg_engine_temp)` and `GROUP BY region` — push-down aggregation
# MAGIC means only the small result set crosses the wire, not 90K raw rows.
# MAGIC
# MAGIC Check the provenance: the consistency is `EVENTUAL` and the staleness window is
# MAGIC `900s` (15 minutes) — reflecting the batch nature of the OLAP pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 2: The Connector Pipeline — What Happens Under the Hood
# MAGIC
# MAGIC ### Under the Hood: The 4-Stage Connector Pipeline
# MAGIC
# MAGIC Every Provena query goes through four deterministic stages:
# MAGIC
# MAGIC ```
# MAGIC  Intent  -->  interpret  -->  synthesize  -->  execute  -->  normalize
# MAGIC  (what)       (which?)        (how?)          (run!)       (enrich)
# MAGIC ```
# MAGIC
# MAGIC 1. **Interpret** — The registry finds connectors that can handle this intent type
# MAGIC 2. **Synthesize** — The chosen connector generates a parameterized query (SQL, VS query, etc.)
# MAGIC 3. **Execute** — The executor runs the query against the real backend
# MAGIC 4. **Normalize** — Raw results are wrapped in `ConnectorResult` with provenance attached
# MAGIC
# MAGIC Let us trace each stage for a point lookup.

# COMMAND ----------

provena.reset()

# Step 1: Create the intent
intent = provena.formulator.point_lookup("fleet_machines", {"machine_id": "EXC-0100"})
print("STEP 1 - Intent created:")
print(f"  Type:   {intent.type}")
print(f"  Entity: {intent.entity}")
print(f"  ID:     {intent.identifier}")
print(f"  Full:   {json.dumps(intent.model_dump(), indent=2, default=str)}")

# COMMAND ----------

# Step 2: Routing — find which connectors can handle this intent
candidates = registry.find_candidates(intent)
print("STEP 2 - Routing (connector candidates):")
for c in candidates:
    print(f"  Candidate: {c.connector.id}")
    print(f"    Source system:    {c.capability.source_system}")
    print(f"    Suitability:     {c.suitability_score:.2f}")
    print(f"    Entities:        {c.capability.entities}")
    print(f"    Supported types: {c.capability.supported_intent_types}")
    print()

if candidates:
    chosen = candidates[0]
    print(f"  >> Winner: {chosen.connector.id} (score={chosen.suitability_score:.2f})")

# COMMAND ----------

# Step 3 + 4: Execute through the full Provena pipeline and inspect the result
frame = asyncio.run(provena.query(intent))
inspect_frame(frame, "STEP 3+4 - Execution and Normalization: EXC-0100")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What each stage contributed:
# MAGIC
# MAGIC | Stage | What Happened |
# MAGIC |-------|---------------|
# MAGIC | **Interpret** | Registry matched `fleet.oltp` (Lakebase connector) as the best candidate |
# MAGIC | **Synthesize** | Connector generated `SELECT * FROM catalog.schema.fleet_machines WHERE machine_id = :machine_id` |
# MAGIC | **Execute** | SparkSQLExecutor ran the query against the lakehouse |
# MAGIC | **Normalize** | Raw row wrapped in `ConnectorResult` with provenance (source, consistency, staleness, precision) |
# MAGIC
# MAGIC The agent never touches SQL. The provenance is attached deterministically.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 3: Trust Scoring with Real Sources
# MAGIC
# MAGIC ### Trust Scoring: Computed Confidence, Not Guesswork
# MAGIC
# MAGIC Provena computes trust scores across four dimensions for every data element:
# MAGIC - **Source Authority** — configured weight per source system (e.g., OLTP = 0.95, VS = 0.70)
# MAGIC - **Consistency** — strong vs eventual consistency guarantees
# MAGIC - **Freshness** — how stale the data might be (staleness window)
# MAGIC - **Precision** — exact match vs approximate / similarity-ranked
# MAGIC
# MAGIC Below we execute three queries — one per source type — and compare the trust scores.

# COMMAND ----------

provena.reset()

# Query 1: OLTP point lookup (strong consistency, exact match)
intent_1 = provena.formulator.point_lookup("fleet_machines", {"machine_id": "EXC-0001"})
frame_1 = asyncio.run(provena.query(intent_1))
print("Query 1 complete: OLTP point lookup (fleet_machines)")

# Query 2: OLAP aggregation (eventual consistency, aggregated)
intent_2 = provena.formulator.aggregate_analysis(
    entity="telemetry_daily",
    measures=[{"field": "avg_engine_temp", "aggregation": "avg"}],
    dimensions=["region"],
)
frame_2 = asyncio.run(provena.query(intent_2))
print("Query 2 complete: OLAP aggregation (telemetry_daily)")

# Query 3: Vector Search semantic search (similarity-ranked, approximate)
intent_3 = provena.formulator.semantic_search(
    query="hydraulic failure high pressure",
    collection="maintenance_logs",
)
frame_3 = asyncio.run(provena.query(intent_3))
print("Query 3 complete: Vector Search semantic search (maintenance_logs)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trust Score Comparison Table

# COMMAND ----------

trust_rows = []

for label, frame, source_type in [
    ("OLTP Point Lookup", frame_1, "Lakebase (OLTP)"),
    ("OLAP Aggregation", frame_2, "DBSQL (OLAP)"),
    ("Vector Search", frame_3, "VS (Document)"),
]:
    for slot in frame.slots:
        for elem in slot.elements:
            trust_rows.append({
                "Query": label,
                "Source Type": source_type,
                "Source System": elem.provenance.source_system,
                "Consistency": str(elem.provenance.consistency),
                "Staleness (sec)": float(elem.provenance.staleness_window_sec),
                "Precision": str(elem.provenance.precision),
                "Trust: Authority": round(elem.trust.dimensions.source_authority, 3),
                "Trust: Consistency": round(elem.trust.dimensions.consistency_score, 3),
                "Trust: Freshness": round(elem.trust.dimensions.freshness_score, 3),
                "Trust: Precision": round(elem.trust.dimensions.precision_score, 3),
                "Trust: Composite": round(elem.trust.composite, 3),
                "Trust Label": elem.trust.label,
            })
            # Only take the first element per source to keep the table readable
            break

trust_df = spark.createDataFrame(trust_rows)
display(trust_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why Each Source Gets Its Score
# MAGIC
# MAGIC | Source | Authority | Consistency | Freshness | Precision | Explanation |
# MAGIC |--------|-----------|-------------|-----------|-----------|-------------|
# MAGIC | **OLTP (Lakebase)** | 0.95 | High (strong) | High (30s staleness) | High (exact match) | Real-time registry with strong guarantees |
# MAGIC | **OLAP (DBSQL)** | 0.85 | Lower (eventual) | Lower (900s staleness) | Medium (aggregated) | Batch-updated warehouse — reliable but not real-time |
# MAGIC | **Vector Search** | 0.70 | Lower (eventual) | Medium (180s staleness) | Lower (similarity) | Semantic results are approximate by nature |
# MAGIC
# MAGIC These scores are **computed deterministically** from the connector configuration.
# MAGIC They do not depend on the LLM's judgment. An agent receiving these scores can
# MAGIC make calibrated decisions about how much to trust each data source.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 4: Conflict Detection in Action
# MAGIC
# MAGIC ### Conflict Detection: When Sources Disagree
# MAGIC
# MAGIC This is the flagship demo. Machine `EXC-0342` is deliberately seeded with
# MAGIC a discrepancy:
# MAGIC - **OLTP** (`fleet_machines`): status = `offline` (real-time, strong consistency)
# MAGIC - **OLAP** (`telemetry_daily`): last_known_status = `online` (batch, 15-min stale)
# MAGIC
# MAGIC A vanilla MCP agent would get both values and either pick one arbitrarily
# MAGIC or hallucinate a reconciliation. Provena detects the conflict automatically
# MAGIC and resolves it using provenance-based heuristics.

# COMMAND ----------

provena.reset()

# Composite intent: query BOTH OLTP and OLAP for the same machine
oltp_intent = provena.formulator.point_lookup("fleet_machines", {"machine_id": "EXC-0342"})

today_str = str(spark.sql("SELECT current_date()").first()[0])

olap_intent = provena.formulator.aggregate_analysis(
    entity="telemetry_daily",
    measures=[{"field": "max_engine_temp", "aggregation": "max"}],
    dimensions=["machine_id", "last_known_status"],
    filters=[
        {"field": "machine_id", "operator": "eq", "value": "EXC-0342"},
        {"field": "report_date", "operator": "eq", "value": today_str},
    ],
)

composite = provena.formulator.composite(
    sub_intents=[oltp_intent, olap_intent],
    fusion_operator="union",
    fusion_key="machine_id",
)

print("Composite intent created:")
print(f"  Sub-intents: {len(composite.sub_intents)}")
print(f"  Fusion:      {composite.fusion_operator}")
print(f"  Fusion key:  {composite.fusion_key}")
print(f"  Today:       {today_str}")

# COMMAND ----------

# Execute the composite query
frame_conflict = asyncio.run(provena.query(composite))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect the raw data from both sources

# COMMAND ----------

inspect_frame(frame_conflict, "Composite Query: EXC-0342 from OLTP + OLAP")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conflicts Detected

# COMMAND ----------

if frame_conflict.conflicts:
    print(f"\nCONFLICTS DETECTED: {len(frame_conflict.conflicts)}")
    print("=" * 80)
    for i, c in enumerate(frame_conflict.conflicts):
        print(f"\n  Conflict {i+1}:")
        print(f"    Field:        {c.field}")
        print(f"    Value A:      {c.value_a} (from {c.element_a.provenance.source_system})")
        print(f"    Value B:      {c.value_b} (from {c.element_b.provenance.source_system})")
        print(f"    --- Resolution ---")
        print(f"    Strategy:     {c.resolution.strategy}")
        print(f"    Winner:       {c.resolution.winner}")
        print(f"    Reason:       {c.resolution.reason}")
else:
    print("No field-level conflicts detected.")
    print("(This can happen if the OLAP query returned no rows for today.)")
    print("Make sure 02_fleet_setup was run recently to seed the conflict row.")

# COMMAND ----------

# Check for presence conflicts (when an expected source returns no data)
if frame_conflict.presence_conflicts:
    print(f"\nPRESENCE CONFLICTS: {len(frame_conflict.presence_conflicts)}")
    print("=" * 80)
    for i, pc in enumerate(frame_conflict.presence_conflicts):
        print(f"\n  Presence Conflict {i+1}:")
        print(f"    Missing source:    {pc.missing_source_system}")
        print(f"    Missing connector: {pc.missing_connector_id}")
        print(f"    Resolution:        {pc.resolution.reason}")
else:
    print("\nNo presence conflicts — both sources returned data.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Insight
# MAGIC
# MAGIC OLTP says `offline` (strong consistency, 30s staleness). OLAP says `online`
# MAGIC (eventual consistency, 900s staleness). Provena detects this conflict automatically
# MAGIC and resolves it by preferring the source with stronger consistency guarantees.
# MAGIC
# MAGIC The resolution is **deterministic** — it will make the same decision every time,
# MAGIC regardless of which LLM is downstream. This is not a probabilistic judgment.
# MAGIC It is a computed decision based on the provenance metadata.
# MAGIC
# MAGIC A vanilla MCP agent receiving both values would have to rely on the LLM to
# MAGIC notice the discrepancy and reason about it. Sometimes it would. Sometimes
# MAGIC it would not. Provena makes this detection and resolution **structural**.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 5: Epistemic Context — The Agent's Data Quality Briefing
# MAGIC
# MAGIC ### Epistemic Context: The Agent's Data Quality Briefing
# MAGIC
# MAGIC After executing queries, Provena accumulates an epistemic context — a structured
# MAGIC summary of data quality, trust scores, conflicts, and caveats. This context
# MAGIC is designed to be injected into the LLM's system prompt so the agent can
# MAGIC reason about data quality and communicate uncertainty.

# COMMAND ----------

# The epistemic context was built up from all the queries above
epistemic_prompt = provena.get_epistemic_context()

print("FULL EPISTEMIC CONTEXT (injected into LLM system prompt):")
print("=" * 80)
print(epistemic_prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Side-by-Side: Vanilla MCP vs Provena-Enhanced Agent

# COMMAND ----------

# Simulate what a vanilla MCP agent sees vs what a Provena-enhanced agent sees

# --- LEFT: Vanilla MCP agent ---
vanilla_data = {
    "machine_id": "EXC-0342",
    "status": "offline",
    "model": "Model X",
    "firmware_version": "v2.1",
    "region": "north_america",
}

# --- RIGHT: Provena-enhanced agent ---
sdol_data_elements = []
for slot in frame_conflict.slots:
    for elem in slot.elements:
        sdol_data_elements.append({
            "data": elem.data,
            "provenance": {
                "source": elem.provenance.source_system,
                "consistency": str(elem.provenance.consistency),
                "staleness_sec": elem.provenance.staleness_window_sec,
                "precision": str(elem.provenance.precision),
            },
            "trust": {
                "composite": round(elem.trust.composite, 3),
                "label": elem.trust.label,
            },
        })

conflicts_summary = []
for c in frame_conflict.conflicts:
    conflicts_summary.append({
        "field": c.field,
        "source_a": c.element_a.provenance.source_system,
        "value_a": c.value_a,
        "source_b": c.element_b.provenance.source_system,
        "value_b": c.value_b,
        "winner": c.resolution.winner,
        "strategy": c.resolution.strategy,
    })

print("=" * 80)
print("  LEFT: What a vanilla MCP agent sees")
print("=" * 80)
print(json.dumps(vanilla_data, indent=2, default=str))
print()
print("  (No provenance. No trust score. No conflict detection.)")
print("  (The LLM must figure out data quality on its own.)")

print()
print("=" * 80)
print("  RIGHT: What a Provena-enhanced agent sees")
print("=" * 80)
print(json.dumps({
    "data_elements": sdol_data_elements,
    "conflicts": conflicts_summary,
    "epistemic_summary": "(see full epistemic prompt above)",
}, indent=2, default=str))
print()
print("  (Provenance attached. Trust scored. Conflicts detected and resolved.)")
print("  (The LLM receives a structured briefing on data quality.)")

# COMMAND ----------

# MAGIC %md
# MAGIC The difference is structural, not cosmetic. The vanilla agent gets raw JSON and
# MAGIC must rely on the LLM to notice quality issues. The Provena agent gets a
# MAGIC deterministic data-quality briefing that the LLM can reference in its response.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 6: Semantic Search with Provenance
# MAGIC
# MAGIC ### Semantic Search: Vector Search with Provenance
# MAGIC
# MAGIC Even similarity-ranked results from Vector Search carry full provenance
# MAGIC metadata. The trust score will be lower (approximate matching, eventual
# MAGIC consistency) but the metadata is still present and useful.

# COMMAND ----------

provena.reset()

intent_vs = provena.formulator.semantic_search(
    query="hydraulic failure high pressure",
    collection="maintenance_logs",
)

frame_vs = asyncio.run(provena.query(intent_vs))
inspect_frame(frame_vs, "Semantic Search: 'hydraulic failure high pressure'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Semantic Results with Trust Scores

# COMMAND ----------

vs_rows = []
for slot in frame_vs.slots:
    for i, elem in enumerate(slot.elements):
        record = {
            "rank": i + 1,
            "source": elem.provenance.source_system,
            "trust_composite": round(elem.trust.composite, 3),
            "trust_label": elem.trust.label,
            "consistency": str(elem.provenance.consistency),
            "precision": str(elem.provenance.precision),
        }
        if isinstance(elem.data, dict):
            record["machine_id"] = elem.data.get("machine_id", "")
            record["fault_category"] = elem.data.get("fault_category", "")
            record["severity"] = elem.data.get("severity", "")
            desc = elem.data.get("description", "")
            record["description_preview"] = desc[:100] + "..." if len(desc) > 100 else desc
        vs_rows.append(record)

if vs_rows:
    vs_df = spark.createDataFrame(vs_rows)
    display(vs_df)
else:
    print("No results returned from Vector Search.")

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that every semantic search result carries the same provenance envelope
# MAGIC as a SQL query result. The trust score reflects the approximate nature of
# MAGIC similarity search (lower precision score), but the agent still knows exactly
# MAGIC where the data came from, how fresh it is, and how confident it should be.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 7: The Full Picture — Determinism Wrapping Probabilism
# MAGIC
# MAGIC ### Putting It All Together
# MAGIC
# MAGIC The core value proposition of Provena is not making the agent smarter.
# MAGIC The LLM is already capable of noticing data quality issues — sometimes.
# MAGIC The problem is "sometimes."
# MAGIC
# MAGIC Provena adds **deterministic guarantees** around the probabilistic LLM:
# MAGIC
# MAGIC | Property | Without Provena | With Provena |
# MAGIC |----------|-------------|-----------|
# MAGIC | **Provenance** | Not tracked | ALWAYS tracked — every element has source, freshness, consistency |
# MAGIC | **Conflict detection** | Depends on LLM noticing | ALWAYS detected — structural comparison of overlapping fields |
# MAGIC | **Trust scoring** | Not computed | ALWAYS scored — 4-dimensional composite with configured authority weights |
# MAGIC | **Audit trail** | Not available | ALWAYS available — full epistemic context for every session |
# MAGIC | **SQL generation** | LLM writes SQL (error-prone) | Provena generates parameterized queries from typed intents |
# MAGIC | **Token efficiency** | Raw rows dumped into context | Pre-aggregated results with push-down computation |
# MAGIC
# MAGIC The value is not intelligence — it is **auditability and consistency**.
# MAGIC The LLM can be swapped out. The guarantees remain.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execution Cost Summary

# COMMAND ----------

cost_summary = provena.get_cost_summary()
print("Provena Session Cost Summary:")
print(json.dumps(cost_summary, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC The cost summary shows execution time by source across all queries in this
# MAGIC session. This metadata is useful for:
# MAGIC - Identifying slow connectors
# MAGIC - Budgeting compute costs across source types
# MAGIC - Optimizing which queries to cache vs re-execute

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Next Steps
# MAGIC
# MAGIC - **`02_fleet_setup`** — Creates the benchmark data tables and Vector Search index
# MAGIC   (run this first if you have not already)
# MAGIC - **`03_fleet_benchmark`** — Head-to-head agent evaluation: a vanilla MCP agent
# MAGIC   vs a Provena-enhanced agent answering the same questions, scored by LLM judges
# MAGIC
# MAGIC ---
# MAGIC *Provena — Epistemic provenance for AI agents*
