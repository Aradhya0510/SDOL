# Databricks notebook source
# MAGIC %md
# MAGIC # Fleet Management: Baseline MCP Agent vs SDOL-Enhanced Agent
# MAGIC
# MAGIC This notebook demonstrates two failure modes that **structurally cannot be solved**
# MAGIC by a vanilla MCP agent, and shows how SDOL resolves them:
# MAGIC
# MAGIC | Failure Mode | Question | Vanilla MCP Problem | SDOL Solution |
# MAGIC |-------------|----------|--------------------|-|
# MAGIC | **Token-busting cross-paradigm join** | Q1: Avg fuel efficiency of Model X v2.1 + failure themes | Scans 360K raw rows, generic log search | Push-down aggregation + targeted vector search |
# MAGIC | **Epistemic conflict** | Q2: Is EXC-0342 active? Status + peak temp | Gets contradictory data, picks one or hallucinates | Detects conflict via provenance, resolves by consistency |
# MAGIC
# MAGIC **Prerequisite:** Run `02_fleet_setup` first.

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-langchain databricks-agents langgraph langchain langchain-core nest_asyncio databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "sdol_benchmark"
SCHEMA = "fleet"

LLM_ENDPOINT = "databricks-claude-3-7-sonnet"

SDOL_PROJECT_ROOT = "/Workspace/Users/{user}/SDOL-python"

VS_ENDPOINT_NAME = "sdol_fleet_vs"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.maintenance_logs_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import SDOL

# COMMAND ----------

import sys, os

try:
    import sdol
except ImportError:
    resolved = SDOL_PROJECT_ROOT.replace("{user}", spark.sql("SELECT current_user()").first()[0])
    src_path = os.path.join(resolved, "src")
    if os.path.isdir(src_path):
        sys.path.insert(0, src_path)
        import sdol
        print(f"Loaded SDOL from {src_path}")
    else:
        raise ImportError(f"SDOL not found at {resolved}")

print(f"SDOL loaded — {sdol.__all__[:5]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize the SDOL Pipeline
# MAGIC
# MAGIC Three Databricks-native connectors spanning three paradigms:
# MAGIC - **OLTP** (Lakebase) → `fleet_machines` — strong consistency, 30s staleness
# MAGIC - **OLAP** (DBSQL) → `telemetry_readings`, `telemetry_daily` — **eventual** consistency, 900s staleness
# MAGIC - **Document** (Vector Search) → `maintenance_logs` — Databricks VS connector, eventual consistency, 180s staleness

# COMMAND ----------

import asyncio, json, time, nest_asyncio
nest_asyncio.apply()

from sdol import (
    SDOL as SDOLEngine,
    CapabilityRegistry,
    ContextCompiler,
    DatabricksDBSQLConnector,
    DatabricksLakebaseConnector,
    DatabricksVectorSearchConnector,
    SemanticRouter,
    TrustScorer,
)
from sdol.core.provenance.trust_scorer import TrustScorerConfig
from sdol.core.router.cost_estimator import CostEstimator
from sdol.core.router.intent_decomposer import IntentDecomposer
from sdol.core.router.query_planner import QueryPlanner
from sdol.types.provenance import ConsistencyGuarantee


class SparkSQLExecutor:
    """Bridges SDOL's QueryExecutor protocol with SparkSession. Errors propagate."""

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
    """Bridges SDOL's QueryExecutor protocol with Databricks Vector Search.

    Accepts DatabricksVSQuery objects produced by the
    DatabricksVectorSearchConnector's synthesize_query stage.
    """

    def __init__(self, endpoint_name: str, index_name: str):
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


sql_executor = SparkSQLExecutor()
vs_executor = DatabricksVectorSearchExecutor(VS_ENDPOINT_NAME, VS_INDEX_NAME)

ENTITY_KEYS = ("machine_id",)

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
sdol = SDOLEngine(router)

print("SDOL fleet pipeline ready")
print(f"  OLTP  entities: {oltp_connector._available_entities}")
print(f"  OLAP  entities: {olap_connector._available_entities} (consistency={olap_connector.default_consistency.value})")
print(f"  Doc   entities: {doc_connector._available_entities} (index={VS_INDEX_NAME})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Tools

# COMMAND ----------

from langchain_core.tools import tool

def _format_frame(frame):
    results = []
    for slot in frame.slots:
        for elem in slot.elements:
            results.append({
                "data": elem.data,
                "source": elem.provenance.source_system,
                "freshness_sec": elem.provenance.staleness_window_sec,
                "consistency": elem.provenance.consistency.value,
                "precision": elem.provenance.precision.value,
                "trust_score": round(elem.trust.composite, 3),
            })
    conflicts = []
    for c in frame.conflicts:
        conflicts.append({
            "field": c.field,
            "value_a": c.value_a,
            "source_a": c.element_a.provenance.source_system,
            "value_b": c.value_b,
            "source_b": c.element_b.provenance.source_system,
            "resolution_strategy": c.resolution.strategy,
            "resolution_winner": c.resolution.winner,
            "resolution_reason": c.resolution.reason,
        })
    return json.dumps({
        "results": results,
        "result_count": len(results),
        "conflicts": conflicts,
        "data_confidence": sdol.get_epistemic_context(),
    }, indent=2, default=str)

# ─────────────── Baseline Tools ───────────────

@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query against the Databricks lakehouse and return results as JSON.
    Args:
        query: A complete, valid Spark SQL query.
    """
    try:
        df = spark.sql(query)
        records = [row.asDict() for row in df.limit(100).collect()]
        return json.dumps(records, indent=2, default=str)
    except Exception as exc:
        return f"SQL error: {exc}"

@tool
def describe_tables() -> str:
    """List all fleet benchmark tables and their columns."""
    info = {}
    for tbl in ["fleet_machines", "telemetry_readings", "telemetry_daily", "maintenance_logs"]:
        cols = [
            row.col_name
            for row in spark.sql(f"DESCRIBE {CATALOG}.{SCHEMA}.{tbl}").collect()
            if not row.col_name.startswith("#")
        ]
        info[f"{CATALOG}.{SCHEMA}.{tbl}"] = cols
    return json.dumps(info, indent=2)

baseline_tools = [execute_sql, describe_tables]

# ─────────────── SDOL Tools ───────────────

@tool
def sdol_machine_lookup(machine_id: str, fields: str = "") -> str:
    """Look up current state of a machine from the real-time OLTP registry.
    Args:
        machine_id: e.g. 'EXC-0342'
        fields: Comma-separated columns (empty = all).
    """
    try:
        field_list = [f.strip() for f in fields.split(",") if f.strip()] or None
        intent = sdol.formulator.point_lookup("fleet_machines", {"machine_id": machine_id}, fields=field_list)
        frame = asyncio.run(sdol.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_fleet_aggregate(entity: str, measures: str, dimensions: str, filters: str = "", order_by: str = "") -> str:
    """Run an aggregate analysis on telemetry data via SDOL push-down.
    Args:
        entity: 'telemetry_readings' or 'telemetry_daily'
        measures: e.g. 'avg(fuel_efficiency_lpkm), max(engine_temp_c)'
        dimensions: e.g. 'machine_id' or 'machine_id,report_date'
        filters: e.g. 'machine_id:eq:EXC-0342; report_date:gte:2025-06-01'
        order_by: e.g. 'avg_fuel_efficiency_lpkm desc'
    """
    import re as _re
    measure_list = []
    for m in measures.split(","):
        m = m.strip()
        match = _re.match(r"(\w+)\((\w+)\)", m)
        if match:
            measure_list.append({"field": match.group(2), "aggregation": match.group(1)})
        else:
            measure_list.append({"field": m, "aggregation": "avg"})
    dims = [d.strip() for d in dimensions.split(",") if d.strip()]
    filter_list = _parse_filters(filters)
    ob = None
    if order_by and order_by.strip():
        parts = order_by.strip().rsplit(None, 1)
        ob_field = parts[0]
        ob_dir = parts[1] if len(parts) > 1 and parts[1] in ("asc", "desc") else "desc"
        ob = [{"field": ob_field, "direction": ob_dir}]
    try:
        intent = sdol.formulator.aggregate_analysis(
            entity=entity, measures=measure_list, dimensions=dims, filters=filter_list, order_by=ob,
        )
        frame = asyncio.run(sdol.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_telemetry_trend(entity: str, metric: str, window: str = "last_180d", granularity: str = "1M") -> str:
    """Analyze temporal trends in telemetry via SDOL DATE_TRUNC push-down.
    Args:
        entity: 'telemetry_readings' or 'telemetry_daily'
        metric: e.g. 'fuel_efficiency_lpkm', 'engine_temp_c'
        window: e.g. 'last_30d', 'last_180d'
        granularity: '1d', '1w', '1M'
    """
    try:
        intent = sdol.formulator.temporal_trend(
            entity=entity, metric=metric, window={"relative": window}, granularity=granularity,
        )
        frame = asyncio.run(sdol.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_search_logs(query: str, machine_ids: str = "", max_results: int = 10) -> str:
    """Semantic search over technician maintenance logs via Databricks Vector Search.
    Args:
        query: Natural-language search query, e.g. 'Model X overheating failure patterns'
        machine_ids: Optional comma-separated machine IDs to filter, e.g. 'EXC-0042,EXC-0100'
        max_results: Number of results (default 10).
    """
    try:
        filters = None
        if machine_ids and machine_ids.strip():
            ids = [m.strip() for m in machine_ids.split(",") if m.strip()]
            if ids:
                filters = [{"field": "machine_id", "operator": "in", "value": ids}]
        intent = sdol.formulator.semantic_search(
            query=query, collection="maintenance_logs",
            filters=filters, max_results=max_results,
        )
        frame = asyncio.run(sdol.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_cross_source_status(machine_id: str) -> str:
    """Query BOTH the real-time OLTP registry AND the OLAP telemetry for a machine.

    This is a composite query that goes through the full SDOL pipeline. If the
    two sources disagree (e.g. OLTP says 'offline' but OLAP says 'online'),
    SDOL automatically detects the conflict and resolves it using provenance-based
    heuristics (preferring the source with stronger consistency guarantees).

    Args:
        machine_id: e.g. 'EXC-0342'
    """
    try:
        oltp_intent = sdol.formulator.point_lookup("fleet_machines", {"machine_id": machine_id})
        olap_intent = sdol.formulator.aggregate_analysis(
            entity="telemetry_daily",
            measures=[{"field": "max_engine_temp", "aggregation": "max"}],
            dimensions=["machine_id", "last_known_status"],
            filters=[
                {"field": "machine_id", "operator": "eq", "value": machine_id},
                {"field": "report_date", "operator": "eq", "value": "today_placeholder"},
            ],
        )
        today_str = str(spark.sql("SELECT current_date()").first()[0])
        olap_intent.filters[-1].value = today_str

        composite = sdol.formulator.composite(
            sub_intents=[oltp_intent, olap_intent],
            fusion_operator="union",
            fusion_key="machine_id",
        )
        frame = asyncio.run(sdol.query(composite))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_sql(query: str) -> str:
    """Execute raw SQL with a note that SDOL provenance tracking is not applied.
    Use for complex JOINs that the typed SDOL tools cannot express.
    Args:
        query: A Spark SQL query using fully-qualified table names.
    """
    try:
        df = spark.sql(query)
        records = [row.asDict() for row in df.limit(100).collect()]
        return json.dumps({
            "results": records, "result_count": len(records),
            "note": "Raw SQL — no SDOL provenance for this query.",
            "data_confidence": sdol.get_epistemic_context(),
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_data_confidence() -> str:
    """Return overall data confidence summary for all data queried so far."""
    return sdol.get_epistemic_context()

def _parse_filters(raw):
    if not raw or not raw.strip():
        return None
    OP_ALIAS = {"=": "eq", "!=": "neq", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}
    result = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.count(":") >= 2:
            segs = part.split(":", 2)
            field, op, val = segs[0].strip(), segs[1].strip(), segs[2].strip()
        else:
            matched = False
            for sym in ("!=", ">=", "<=", ">", "<", "="):
                if sym in part:
                    field, val = [s.strip() for s in part.split(sym, 1)]
                    op = OP_ALIAS.get(sym, "eq")
                    matched = True
                    break
            if not matched:
                continue
        val = val.strip("'\"")
        try:
            val = float(val)
        except ValueError:
            pass
        result.append({"field": field, "operator": op, "value": val})
    return result or None

sdol_tools = [
    sdol_machine_lookup, sdol_fleet_aggregate, sdol_telemetry_trend,
    sdol_search_logs, sdol_cross_source_status, sdol_sql, sdol_data_confidence,
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build LangGraph Agents

# COMMAND ----------

from typing import Annotated, Any, Optional, Sequence, TypedDict
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]

llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

def build_agent(tools, system_prompt: str):
    bound = llm.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + list(state["messages"])
    )
    chain = preprocessor | bound

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "continue"
        return "end"

    def call_model(state: AgentState, config: RunnableConfig):
        return {"messages": [chain.invoke(state, config)]}

    g = StateGraph(AgentState)
    g.add_node("agent", RunnableLambda(call_model))
    g.add_node("tools", ToolNode(tools))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile()

# COMMAND ----------

# MAGIC %md
# MAGIC ### System Prompts

# COMMAND ----------

FQ = f"{CATALOG}.{SCHEMA}"

TABLE_SCHEMAS = f"""
Tables in `{FQ}`:

OLTP (real-time, strongly consistent):
  fleet_machines — machine_id, model, serial_number, firmware_version, status, region, gps_lat, gps_lon, last_heartbeat_at

OLAP (batch-updated every ~15 min, eventually consistent):
  telemetry_readings — machine_id, reading_time, engine_temp_c, rpm, fuel_efficiency_lpkm, vibration_mm_s, oil_pressure_psi, coolant_temp_c
  telemetry_daily    — machine_id, report_date, avg_engine_temp, max_engine_temp, avg_rpm, avg_fuel_efficiency, min_fuel_efficiency, reading_count, last_known_status

Document (Vector Search, semantic):
  maintenance_logs — log_id, machine_id, log_date, fault_category, severity, description, technician_name, resolution_notes
""".strip()

BASELINE_PROMPT = f"""You are a fleet reliability analyst assistant for an industrial machinery company.
You have access to a Databricks lakehouse with the following tables:

{TABLE_SCHEMAS}

Use `describe_tables` to inspect columns if unsure.
Use `execute_sql` to run Spark SQL queries. Always use fully-qualified names ({FQ}.<table>).
Present results clearly with actual values.
"""

SDOL_PROMPT = f"""You are a fleet reliability analyst assistant enhanced with SDOL (Semantic Data Orchestration Layer).

SDOL tracks data provenance (source, freshness, consistency), computes trust scores,
and automatically detects conflicts between data sources with different consistency guarantees.

{TABLE_SCHEMAS}

Available SDOL tools:
- `sdol_machine_lookup`       — real-time OLTP lookup (strong consistency, <30s stale)
- `sdol_fleet_aggregate`      — OLAP push-down aggregation (eventual consistency, ~15min stale)
    measures format: 'avg(fuel_efficiency_lpkm), max(engine_temp_c)'
    filters format: 'machine_id:eq:EXC-0042; report_date:gte:2025-06-01'
- `sdol_telemetry_trend`      — temporal bucketed analysis (DATE_TRUNC push-down)
- `sdol_search_logs`          — semantic search on maintenance logs via Vector Search
- `sdol_cross_source_status`  — **composite query** that hits BOTH OLTP and OLAP for the
    same machine, automatically detects data conflicts between sources, and resolves them
    using provenance-based heuristics. USE THIS when asked about a machine's current state.
- `sdol_sql`                  — raw SQL fallback for JOINs ({FQ}.<table>)
- `sdol_data_confidence`      — overall epistemic summary

CRITICAL GUIDELINES:
- When SDOL detects a conflict between sources, ALWAYS explain it transparently.
  State which source you trust more and why (consistency level, freshness).
- Always cite trust scores, source system, and freshness in your answers.
- For questions about a machine's CURRENT STATUS, use `sdol_cross_source_status`
  to query both OLTP and OLAP simultaneously — this triggers conflict detection.
"""

# COMMAND ----------

import mlflow
mlflow.langchain.autolog()

baseline_agent = build_agent(baseline_tools, BASELINE_PROMPT)
sdol_agent = build_agent(sdol_tools, SDOL_PROMPT)
print("Both agents compiled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke Test

# COMMAND ----------

def invoke_agent(agent, question: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)

q = "What is Machine EXC-0001's model and firmware version?"
print("── Baseline ──")
print(invoke_agent(baseline_agent, q)[:500])
print("\n── SDOL ──")
sdol.reset()
print(invoke_agent(sdol_agent, q)[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Dataset
# MAGIC
# MAGIC Two primary showcase questions plus supporting questions.

# COMMAND ----------

EVAL_QUESTIONS = [
    # ── SHOWCASE Q1: Cross-Paradigm Token Buster ──
    {
        "question": (
            "What is the average fuel efficiency of all Model X excavators running "
            "firmware v2.1 over the last 6 months, and what are the primary themes "
            "in the technician maintenance logs when these specific machines have failures?"
        ),
        "category": "cross_paradigm",
        "expected_response": (
            "Should return a concise average fuel efficiency number from OLAP "
            "push-down aggregation, plus summarized failure themes from semantic search "
            "on maintenance logs — NOT raw rows."
        ),
    },
    # ── SHOWCASE Q2: Epistemic Conflict ──
    {
        "question": (
            "Is Machine EXC-0342 currently active? I need its current operational "
            "status and its peak engine temperature for today."
        ),
        "category": "epistemic_conflict",
        "expected_response": (
            "Should detect that OLTP (real-time) says 'offline' while OLAP (15-min stale) "
            "says 'online'. Should resolve the conflict by trusting the OLTP source "
            "(stronger consistency) and explain the discrepancy transparently."
        ),
    },
    # ── Supporting: Point Lookup ──
    {
        "question": "What are the full details for machine EXC-0100? Include model, firmware, status, and region.",
        "category": "point_lookup",
        "expected_response": "Should return exact machine details from OLTP with provenance metadata.",
    },
    # ── Supporting: Aggregate ──
    {
        "question": "What is the average engine temperature by region from the telemetry_daily table?",
        "category": "aggregate",
        "expected_response": "Should return avg engine temp grouped by region via OLAP push-down.",
    },
    # ── Supporting: Semantic Search ──
    {
        "question": "Find maintenance logs related to hydraulic system failures. What are the common patterns?",
        "category": "semantic_search",
        "expected_response": "Should use vector search to find semantically relevant hydraulic failure logs.",
    },
    # ── Supporting: Confidence ──
    {
        "question": (
            "How reliable is the telemetry data for our fleet? What should I be "
            "cautious about when making decisions based on it?"
        ),
        "category": "confidence",
        "expected_response": (
            "SDOL should mention eventual consistency, 15-min staleness, and trust scores. "
            "Baseline will likely not mention data quality concerns."
        ),
    },
]

print(f"Evaluation set: {len(EVAL_QUESTIONS)} questions")
for eq in EVAL_QUESTIONS:
    print(f"  [{eq['category']:20s}] {eq['question'][:75]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Both Agents

# COMMAND ----------

import pandas as pd

rows = []
for i, eq in enumerate(EVAL_QUESTIONS):
    q = eq["question"]
    cat = eq["category"]
    print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {cat}: {q[:70]}...")

    sdol.reset()
    t0 = time.time()
    try:
        b_resp = invoke_agent(baseline_agent, q)
    except Exception as exc:
        b_resp = f"ERROR: {exc}"
    b_lat = round(time.time() - t0, 2)

    sdol.reset()
    t0 = time.time()
    try:
        s_resp = invoke_agent(sdol_agent, q)
    except Exception as exc:
        s_resp = f"ERROR: {exc}"
    s_lat = round(time.time() - t0, 2)

    rows.append({
        "question": q, "category": cat, "expected_response": eq["expected_response"],
        "baseline_response": b_resp, "baseline_latency_sec": b_lat,
        "sdol_response": s_resp, "sdol_latency_sec": s_lat,
    })
    print(f"   baseline={b_lat}s  sdol={s_lat}s")

results_df = pd.DataFrame(rows)
print(f"\nAll {len(results_df)} questions answered.")

# COMMAND ----------

display(spark.createDataFrame(results_df[["question", "category", "baseline_latency_sec", "sdol_latency_sec"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Judge Evaluation (Databricks-Managed)
# MAGIC
# MAGIC Custom scorers designed to expose the structural gaps in vanilla MCP:
# MAGIC - **epistemic_transparency**: Does the agent explain provenance, flag conflicts?
# MAGIC - **data_efficiency**: Does the agent avoid dumping raw rows?

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety

try:
    from mlflow.genai.scorers import Guidelines
    HAS_GUIDELINES = True
except ImportError:
    HAS_GUIDELINES = False

scorers = [RelevanceToQuery(), Safety()]

if HAS_GUIDELINES:
    scorers.append(Guidelines(
        name="epistemic_transparency",
        guidelines=(
            "The response should reference the data source system for each claim. "
            "The response should mention data freshness or staleness when relevant. "
            "If data from different sources conflicts, the response should explicitly "
            "flag the conflict, explain which source is more reliable, and state why. "
            "The response should include confidence or trust levels for key claims."
        ),
    ))
    scorers.append(Guidelines(
        name="data_efficiency",
        guidelines=(
            "The response should present concise, aggregated results rather than raw data rows. "
            "The response should not include hundreds of raw records dumped into the answer. "
            "The response should answer the question directly without unnecessary data. "
            "Numeric results should be clearly summarized (e.g. averages, totals) not listed row-by-row."
        ),
    ))

print(f"Scorers: {[type(s).__name__ for s in scorers]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Baseline

# COMMAND ----------

baseline_eval_data = [
    {"inputs": {"question": r["question"]}, "expected_response": r["expected_response"]}
    for _, r in results_df.iterrows()
]

with mlflow.start_run(run_name="fleet_baseline_mcp"):
    baseline_eval = mlflow.genai.evaluate(
        data=baseline_eval_data,
        predict_fn=lambda question: results_df.loc[
            results_df["question"] == question, "baseline_response"
        ].iloc[0],
        scorers=scorers,
    )
print("Baseline evaluation complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate SDOL

# COMMAND ----------

sdol_eval_data = [
    {"inputs": {"question": r["question"]}, "expected_response": r["expected_response"]}
    for _, r in results_df.iterrows()
]

with mlflow.start_run(run_name="fleet_sdol_enhanced"):
    sdol_eval = mlflow.genai.evaluate(
        data=sdol_eval_data,
        predict_fn=lambda question: results_df.loc[
            results_df["question"] == question, "sdol_response"
        ].iloc[0],
        scorers=scorers,
    )
print("SDOL evaluation complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Comparison

# COMMAND ----------

b_metrics = baseline_eval.metrics if hasattr(baseline_eval, "metrics") else {}
s_metrics = sdol_eval.metrics if hasattr(sdol_eval, "metrics") else {}
all_keys = sorted(set(list(b_metrics.keys()) + list(s_metrics.keys())))

comparison_rows = []
for k in all_keys:
    bv, sv = b_metrics.get(k), s_metrics.get(k)
    delta = round(sv - bv, 4) if isinstance(bv, (int, float)) and isinstance(sv, (int, float)) else None
    comparison_rows.append({"metric": k, "baseline": bv, "sdol_enhanced": sv, "delta": delta})

comparison_df = pd.DataFrame(comparison_rows)
display(spark.createDataFrame(comparison_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latency

# COMMAND ----------

latency_summary = pd.DataFrame({
    "Agent": ["Baseline", "SDOL-Enhanced"],
    "Mean (s)": [results_df["baseline_latency_sec"].mean(), results_df["sdol_latency_sec"].mean()],
    "Median (s)": [results_df["baseline_latency_sec"].median(), results_df["sdol_latency_sec"].median()],
})
display(spark.createDataFrame(latency_summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visual Comparison

# COMMAND ----------

import matplotlib.pyplot as plt

score_keys = [k for k in all_keys if isinstance(b_metrics.get(k), (int, float)) and isinstance(s_metrics.get(k), (int, float))]
if score_keys:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(score_keys))
    w = 0.35
    ax.bar([i - w/2 for i in x], [b_metrics[k] for k in score_keys], w, label="Baseline", color="#5B9BD5")
    ax.bar([i + w/2 for i in x], [s_metrics[k] for k in score_keys], w, label="SDOL", color="#70AD47")
    ax.set_xticks(list(x))
    ax.set_xticklabels(score_keys, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Fleet Benchmark: LLM Judge Scores")
    ax.legend()
    plt.tight_layout()
    display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Responses — Showcase Questions

# COMMAND ----------

for _, row in results_df.iterrows():
    print("=" * 100)
    print(f"CATEGORY: {row['category']}  |  QUESTION: {row['question']}")
    print("-" * 100)
    print(f"BASELINE ({row['baseline_latency_sec']}s):\n{row['baseline_response'][:600]}")
    print("-" * 50)
    print(f"SDOL ({row['sdol_latency_sec']}s):\n{row['sdol_response'][:600]}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Scenario | Vanilla MCP | SDOL-Enhanced |
# MAGIC |----------|-------------|---------------|
# MAGIC | **Q1: Cross-paradigm** | Scans raw telemetry rows, generic text search | Push-down AVG inside Databricks, targeted vector search by machine IDs |
# MAGIC | **Q2: Epistemic conflict** | Gets contradictory answers, picks one silently | Detects conflict, resolves via `prefer_strongest_consistency`, explains transparently |
# MAGIC | **Provenance** | None — all data treated as equally reliable | Every element tagged with source, freshness, consistency, trust score |
# MAGIC | **Token efficiency** | Dumps raw rows into context window | Returns tiny aggregated results + provenance metadata |
# MAGIC
# MAGIC Check the MLflow Experiment UI for full traces, judge rationales, and per-question drilldowns.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *Fleet Management Benchmark — SDOL (Semantic Data Orchestration Layer)*
