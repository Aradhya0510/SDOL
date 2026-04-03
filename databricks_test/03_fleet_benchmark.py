# Databricks notebook source
# MAGIC %md
# MAGIC # Fleet Management: Baseline MCP Agent vs Provena-Enhanced Agent
# MAGIC
# MAGIC This notebook demonstrates two failure modes that **structurally cannot be solved**
# MAGIC by a vanilla MCP agent, and shows how Provena resolves them:
# MAGIC
# MAGIC | Failure Mode | Question | Vanilla MCP Problem | Provena Solution |
# MAGIC |-------------|----------|--------------------|-|
# MAGIC | **Token-busting cross-paradigm join** | Q1: Avg fuel efficiency of Model X v2.1 + failure themes | Scans 360K raw rows, generic log search | Push-down aggregation + targeted vector search |
# MAGIC | **Epistemic conflict** | Q2: Is EXC-0342 active? Status + peak temp | Gets contradictory data, picks one or hallucinates | Detects conflict via provenance, resolves by consistency |
# MAGIC
# MAGIC **Prerequisite:** Run `02_fleet_setup` first.

# COMMAND ----------

# MAGIC %pip install -U -qqqq pydantic>=2.0 databricks-langchain databricks-agents langgraph langchain langchain-core nest_asyncio databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "users")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("llm_endpoint", "databricks-claude-sonnet-4-6")
dbutils.widgets.text("model_endpoints", "")
dbutils.widgets.text("provena_project_root", "/Workspace/Users/{user}/SDOL")
dbutils.widgets.text("vs_endpoint", "provena_fleet_vs")
dbutils.widgets.text("num_runs", "1")
dbutils.widgets.text("input_price_per_1k_tokens", "0.003")
dbutils.widgets.text("output_price_per_1k_tokens", "0.015")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
_model_endpoints_raw = dbutils.widgets.get("model_endpoints").strip()
MODEL_ENDPOINTS = [e.strip() for e in _model_endpoints_raw.split(",") if e.strip()] if _model_endpoints_raw else [LLM_ENDPOINT]

PROVENA_PROJECT_ROOT = dbutils.widgets.get("provena_project_root")

VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint")
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.maintenance_logs_index"

NUM_RUNS = int(dbutils.widgets.get("num_runs"))
INPUT_PRICE = float(dbutils.widgets.get("input_price_per_1k_tokens"))
OUTPUT_PRICE = float(dbutils.widgets.get("output_price_per_1k_tokens"))
CHARS_PER_TOKEN = 4  # standard approximation

print(f"Models:    {MODEL_ENDPOINTS}")
print(f"Num runs:  {NUM_RUNS}")
print(f"Pricing:   ${INPUT_PRICE}/1K input, ${OUTPUT_PRICE}/1K output")
est_minutes = len(MODEL_ENDPOINTS) * NUM_RUNS * 7 * 2 * 1.5  # ~1.5 min per question per agent
print(f"Estimated runtime: ~{est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Provena

# COMMAND ----------

import sys, os

try:
    import provena
except ImportError:
    resolved = PROVENA_PROJECT_ROOT.replace("{user}", spark.sql("SELECT current_user()").first()[0])
    src_path = os.path.join(resolved, "src")
    if os.path.isdir(src_path):
        sys.path.insert(0, src_path)
        import provena
        print(f"Loaded Provena from {src_path}")
    else:
        raise ImportError(f"Provena not found at {resolved}")

print(f"Provena loaded — {provena.__all__[:5]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize the Provena Pipeline
# MAGIC
# MAGIC Three Databricks-native connectors spanning three paradigms:
# MAGIC - **OLTP** (Lakebase) → `fleet_machines` — strong consistency, 30s staleness
# MAGIC - **OLAP** (DBSQL) → `telemetry_readings`, `telemetry_daily` — **eventual** consistency, 900s staleness
# MAGIC - **Document** (Vector Search) → `maintenance_logs` — Databricks VS connector, eventual consistency, 180s staleness

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
    """Bridges Provena's QueryExecutor protocol with SparkSession. Errors propagate."""

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
    """Bridges Provena's QueryExecutor protocol with Databricks Vector Search.

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
provena = ProvenaEngine(router)

print("Provena fleet pipeline ready")
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
    presence_conflicts = []
    for pc in frame.presence_conflicts:
        presence_conflicts.append({
            "missing_source": pc.missing_source_system,
            "missing_connector": pc.missing_connector_id,
            "resolution": pc.resolution.reason,
        })
    output = {
        "results": results,
        "result_count": len(results),
        "conflicts": conflicts,
        "data_confidence": provena.get_epistemic_context(),
    }
    if frame.trust_summary:
        output["trust_summary"] = {
            "overall_confidence": frame.trust_summary.overall_confidence,
            "lowest_trust_source": frame.trust_summary.lowest_trust_source,
            "advisory": frame.trust_summary.advisory,
        }
    if presence_conflicts:
        output["presence_conflicts"] = presence_conflicts
    return json.dumps(output, indent=2, default=str)

# ─────────────── Baseline Tools ───────────────
# Simulates a naive MCP data connector that exposes row-fetching tools
# rather than arbitrary SQL. This forces the baseline to consume raw rows
# into its context window — exposing the token-busting failure mode.

@tool
def get_table_sample(table: str, n_rows: int = 100, filter_expr: str = "") -> str:
    """Fetch a sample of rows from a fleet table. Returns raw rows as JSON.

    This is similar to how most MCP data connectors work — they expose
    data-fetching tools that return rows, not arbitrary SQL execution.

    Args:
        table: Table name (one of: fleet_machines, telemetry_readings,
               telemetry_daily, maintenance_logs).
        n_rows: Number of rows to return (max 500).
        filter_expr: Optional SQL WHERE clause (e.g. "machine_id = 'EXC-0342'").
    """
    try:
        n_rows = min(n_rows, 500)
        fq_table = f"{CATALOG}.{SCHEMA}.{table}"
        where = f" WHERE {filter_expr}" if filter_expr.strip() else ""
        query = f"SELECT * FROM {fq_table}{where} LIMIT {n_rows}"
        df = spark.sql(query)
        records = [row.asDict() for row in df.collect()]
        return json.dumps(records, indent=2, default=str)
    except Exception as exc:
        return f"Error: {exc}"

@tool
def search_logs_text(keyword: str, n_rows: int = 50) -> str:
    """Search maintenance logs by keyword match (LIKE). Returns raw rows.
    Args:
        keyword: Text to search for in the description field.
        n_rows: Number of rows to return (max 200).
    """
    try:
        n_rows = min(n_rows, 200)
        query = f"""
            SELECT * FROM {CATALOG}.{SCHEMA}.maintenance_logs
            WHERE lower(description) LIKE lower('%{keyword}%')
            LIMIT {n_rows}
        """
        df = spark.sql(query)
        records = [row.asDict() for row in df.collect()]
        return json.dumps(records, indent=2, default=str)
    except Exception as exc:
        return f"Error: {exc}"

baseline_tools = [get_table_sample, search_logs_text]

# ─────────────── Provena Tools ───────────────

@tool
def sdol_machine_lookup(machine_id: str, fields: str = "") -> str:
    """Look up current state of a machine from the real-time OLTP registry.
    Args:
        machine_id: e.g. 'EXC-0342'
        fields: Comma-separated columns (empty = all).
    """
    try:
        field_list = [f.strip() for f in fields.split(",") if f.strip()] or None
        intent = provena.formulator.point_lookup("fleet_machines", {"machine_id": machine_id}, fields=field_list)
        frame = asyncio.run(provena.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_fleet_aggregate(entity: str, measures: str, dimensions: str, filters: str = "", order_by: str = "") -> str:
    """Run an aggregate analysis on telemetry data via Provena push-down.
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
        intent = provena.formulator.aggregate_analysis(
            entity=entity, measures=measure_list, dimensions=dims, filters=filter_list, order_by=ob,
        )
        frame = asyncio.run(provena.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_telemetry_trend(entity: str, metric: str, window: str = "last_180d", granularity: str = "1M") -> str:
    """Analyze temporal trends in telemetry via Provena DATE_TRUNC push-down.
    Args:
        entity: 'telemetry_readings' or 'telemetry_daily'
        metric: e.g. 'fuel_efficiency_lpkm', 'engine_temp_c'
        window: e.g. 'last_30d', 'last_180d'
        granularity: '1d', '1w', '1M'
    """
    try:
        intent = provena.formulator.temporal_trend(
            entity=entity, metric=metric, window={"relative": window}, granularity=granularity,
        )
        frame = asyncio.run(provena.query(intent))
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
        intent = provena.formulator.semantic_search(
            query=query, collection="maintenance_logs",
            filters=filters, max_results=max_results,
        )
        frame = asyncio.run(provena.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_cross_source_status(machine_id: str) -> str:
    """Query BOTH the real-time OLTP registry AND the OLAP telemetry for a machine.

    This is a composite query that goes through the full Provena pipeline. If the
    two sources disagree (e.g. OLTP says 'offline' but OLAP says 'online'),
    Provena automatically detects the conflict and resolves it using provenance-based
    heuristics (preferring the source with stronger consistency guarantees).

    Args:
        machine_id: e.g. 'EXC-0342'
    """
    try:
        oltp_intent = provena.formulator.point_lookup("fleet_machines", {"machine_id": machine_id})
        olap_intent = provena.formulator.aggregate_analysis(
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

        composite = provena.formulator.composite(
            sub_intents=[oltp_intent, olap_intent],
            fusion_operator="union",
            fusion_key="machine_id",
        )
        frame = asyncio.run(provena.query(composite))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_sql(query: str) -> str:
    """Execute raw SQL with a note that Provena provenance tracking is not applied.
    Use for complex JOINs that the typed Provena tools cannot express.
    Args:
        query: A Spark SQL query using fully-qualified table names.
    """
    try:
        df = spark.sql(query)
        records = [row.asDict() for row in df.limit(100).collect()]
        return json.dumps({
            "results": records, "result_count": len(records),
            "note": "Raw SQL — no Provena provenance for this query.",
            "data_confidence": provena.get_epistemic_context(),
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

@tool
def sdol_data_confidence() -> str:
    """Return overall data confidence summary for all data queried so far."""
    return provena.get_epistemic_context()

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

def build_agent(tools, system_prompt: str, llm=None):
    if llm is None:
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
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

Use `get_table_sample` to fetch rows from tables. You can filter with a WHERE clause.
Use `search_logs_text` to search maintenance logs by keyword.
Present results clearly with actual values.
"""

SDOL_PROMPT = f"""You are a fleet reliability analyst assistant enhanced with Provena (Epistemic Provenance for AI Agents).

Provena tracks data provenance (source, freshness, consistency), computes trust scores,
and automatically detects conflicts between data sources with different consistency guarantees.

{TABLE_SCHEMAS}

Available Provena tools:
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
- When Provena detects a conflict between sources, ALWAYS explain it transparently.
  State which source you trust more and why (consistency level, freshness).
- Always cite trust scores, source system, and freshness in your answers.
- For questions about a machine's CURRENT STATUS, use `sdol_cross_source_status`
  to query both OLTP and OLAP simultaneously — this triggers conflict detection.
"""

# COMMAND ----------

import mlflow
mlflow.langchain.autolog()

def build_agents_for_endpoint(endpoint):
    """Build baseline and Provena agents for a given model endpoint."""
    model = ChatDatabricks(endpoint=endpoint)
    return build_agent(baseline_tools, BASELINE_PROMPT, model), build_agent(sdol_tools, SDOL_PROMPT, model)

baseline_agent, sdol_agent = build_agents_for_endpoint(MODEL_ENDPOINTS[0])
print(f"Both agents compiled (model: {MODEL_ENDPOINTS[0]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke Test

# COMMAND ----------

def invoke_agent(agent, question: str) -> tuple:
    """Invoke an agent and return (response, context_chars_consumed).

    context_chars_consumed is the total character count of all tool call
    results flowing through the agent's context window -- a proxy for
    intermediate token consumption.
    """
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    tool_result_chars = 0
    for msg in result["messages"]:
        if hasattr(msg, "content") and getattr(msg, "type", None) == "tool":
            tool_result_chars += len(str(msg.content))
    last = result["messages"][-1]
    response = last.content if hasattr(last, "content") else str(last)
    return response, tool_result_chars

def estimate_cost(context_chars, response_chars):
    """Estimate LLM API cost from character counts."""
    input_tokens = context_chars / CHARS_PER_TOKEN
    output_tokens = response_chars / CHARS_PER_TOKEN
    input_cost = (input_tokens / 1000) * INPUT_PRICE
    output_cost = (output_tokens / 1000) * OUTPUT_PRICE
    return round(input_cost + output_cost, 6)

q = "What is Machine EXC-0001's model and firmware version?"
print("-- Baseline --")
resp, tokens = invoke_agent(baseline_agent, q)
print(f"{resp[:500]}  [context_chars={tokens}]")
print("\n-- Provena --")
provena.reset()
resp, tokens = invoke_agent(sdol_agent, q)
print(f"{resp[:500]}  [context_chars={tokens}]")

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
            "Provena should mention eventual consistency, 15-min staleness, and trust scores. "
            "Baseline will likely not mention data quality concerns."
        ),
    },
    # ── SHOWCASE Q3: Trust Metadata — baseline structurally cannot answer ──
    {
        "question": (
            "Which data sources should I trust most for real-time fleet decisions, "
            "and which have known staleness risks? Give me concrete numbers."
        ),
        "category": "trust_meta",
        "expected_response": (
            "Provena should answer from its trust scorer config and epistemic tracker "
            "with concrete trust scores, consistency levels, and staleness windows. "
            "The baseline has no provenance metadata and can only speculate."
        ),
    },
]

print(f"Evaluation set: {len(EVAL_QUESTIONS)} questions")
for eq in EVAL_QUESTIONS:
    print(f"  [{eq['category']:20s}] {eq['question'][:75]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Both Agents
# MAGIC
# MAGIC Supports multi-run (`num_runs`) and multi-model (`model_endpoints`) sweeps.
# MAGIC With defaults (`num_runs=1`, single model), behavior is identical to previous versions.

# COMMAND ----------

import pandas as pd
import numpy as np

all_model_results = {}  # {endpoint: [list of run DataFrames]}

for endpoint in MODEL_ENDPOINTS:
    print(f"\n{'#'*60}\nMODEL: {endpoint}\n{'#'*60}")
    baseline_agent, sdol_agent = build_agents_for_endpoint(endpoint)

    model_runs = []
    for run_idx in range(NUM_RUNS):
        if NUM_RUNS > 1:
            print(f"\n--- Run {run_idx + 1}/{NUM_RUNS} ---")
        rows = []
        for i, eq in enumerate(EVAL_QUESTIONS):
            q = eq["question"]
            cat = eq["category"]
            print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {cat}: {q[:70]}...")

            provena.reset()
            t0 = time.time()
            try:
                b_resp, b_tokens = invoke_agent(baseline_agent, q)
            except Exception as exc:
                b_resp, b_tokens = f"ERROR: {exc}", 0
            b_lat = round(time.time() - t0, 2)

            provena.reset()
            t0 = time.time()
            try:
                s_resp, s_tokens = invoke_agent(sdol_agent, q)
            except Exception as exc:
                s_resp, s_tokens = f"ERROR: {exc}", 0
            s_lat = round(time.time() - t0, 2)

            rows.append({
                "question": q, "category": cat, "expected_response": eq["expected_response"],
                "baseline_response": b_resp, "baseline_latency_sec": b_lat,
                "baseline_context_chars": b_tokens,
                "baseline_cost_usd": estimate_cost(b_tokens, len(b_resp)),
                "sdol_response": s_resp, "sdol_latency_sec": s_lat,
                "sdol_context_chars": s_tokens,
                "sdol_cost_usd": estimate_cost(s_tokens, len(s_resp)),
            })
            print(f"   baseline={b_lat}s/{b_tokens}chars  provena={s_lat}s/{s_tokens}chars")

        run_df = pd.DataFrame(rows)
        run_df["run_index"] = run_idx
        run_df["model_endpoint"] = endpoint
        model_runs.append(run_df)
        print(f"\nRun {run_idx + 1} complete: {len(run_df)} questions answered.")

    all_model_results[endpoint] = model_runs

# For backwards compat and single-run display
results_df = all_model_results[MODEL_ENDPOINTS[0]][0]
print(f"\nBenchmark complete: {len(MODEL_ENDPOINTS)} model(s) x {NUM_RUNS} run(s)")

# COMMAND ----------

display(spark.createDataFrame(results_df[["question", "category", "baseline_latency_sec", "sdol_latency_sec", "baseline_context_chars", "sdol_context_chars", "baseline_cost_usd", "sdol_cost_usd"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Judge Evaluation (Databricks-Managed)
# MAGIC
# MAGIC Seven scorers measure different aspects of agent quality:
# MAGIC
# MAGIC | Scorer | What It Measures | Provena Advantage |
# MAGIC |--------|-----------------|----------------|
# MAGIC | **relevance_to_query** | Answer correctness | Neutral (LLMs are already good) |
# MAGIC | **safety** | Safety guardrails | Neutral |
# MAGIC | **epistemic_transparency** | Cites provenance, flags conflicts, reports trust | Strong |
# MAGIC | **data_efficiency** | Concise aggregated results vs raw rows | Moderate |
# MAGIC | **provenance_completeness** | Names specific sources, freshness, consistency levels | Strong |
# MAGIC | **conflict_detection_quality** | Detects, explains, and resolves cross-source conflicts | Strong |
# MAGIC | **cost_awareness** | Uses targeted queries and aggregations efficiently | Moderate |

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
    scorers.append(Guidelines(
        name="provenance_completeness",
        guidelines=(
            "The response should cite specific source systems (e.g. 'OLTP registry', "
            "'OLAP telemetry warehouse', 'vector search index') for each data claim. "
            "The response should mention data freshness (e.g. 'updated within 30 seconds', "
            "'batch-updated every 15 minutes') for at least one source. "
            "The response should mention consistency guarantees (e.g. 'strong consistency', "
            "'eventual consistency') when comparing data from different sources. "
            "Vague attributions like 'the database' or 'the system' are insufficient."
        ),
    ))
    scorers.append(Guidelines(
        name="conflict_detection_quality",
        guidelines=(
            "When the underlying data contains conflicts between sources (e.g. one source "
            "says a machine is online while another says offline), the response should: "
            "(1) explicitly identify the conflict and name both sources, "
            "(2) explain why the conflict exists (e.g. different update frequencies), and "
            "(3) state which source it trusts more and give a concrete reason. "
            "If no conflict exists in the data, the response should not fabricate one. "
            "Silently picking one value without acknowledging the disagreement is a failure."
        ),
    ))
    scorers.append(Guidelines(
        name="cost_awareness",
        guidelines=(
            "The response should demonstrate efficient data usage: presenting aggregated "
            "summaries (averages, counts, top-N) rather than listing raw individual records. "
            "The response should not dump large tables of row-level data into the answer. "
            "When multiple data sources are queried, the response should use targeted queries "
            "(filtering by relevant entity, using aggregations) rather than scanning entire tables. "
            "The response should show awareness of query cost implications when relevant."
        ),
    ))

print(f"Scorers ({len(scorers)}): {[type(s).__name__ + ('(' + s.name + ')' if hasattr(s, 'name') else '') for s in scorers]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate All Runs

# COMMAND ----------

all_eval_metrics = []  # list of {model, agent, run_index, metric, value}

with mlflow.start_run(run_name=f"benchmark_sweep_{int(time.time())}"):
    mlflow.set_tag("num_runs", NUM_RUNS)
    mlflow.set_tag("models", ",".join(MODEL_ENDPOINTS))

    for endpoint, model_runs in all_model_results.items():
        model_short = endpoint.split("/")[-1] if "/" in endpoint else endpoint

        for run_idx, run_df in enumerate(model_runs):
            for agent_name, resp_col, lat_col, chars_col, cost_col in [
                ("baseline", "baseline_response", "baseline_latency_sec", "baseline_context_chars", "baseline_cost_usd"),
                ("provena", "sdol_response", "sdol_latency_sec", "sdol_context_chars", "sdol_cost_usd"),
            ]:
                run_label = f"{model_short}_{agent_name}_run{run_idx}"
                with mlflow.start_run(run_name=run_label, nested=True):
                    mlflow.set_tag("model", endpoint)
                    mlflow.set_tag("agent", agent_name)
                    mlflow.set_tag("run_index", run_idx)

                    # Log latency, token, and cost metrics
                    for _, row in run_df.iterrows():
                        mlflow.log_metric(f"latency_{row['category']}", row[lat_col])
                        mlflow.log_metric(f"context_chars_{row['category']}", row[chars_col])
                    mlflow.log_metric("latency_mean", run_df[lat_col].mean())
                    mlflow.log_metric("context_chars_mean", run_df[chars_col].mean())
                    mlflow.log_metric("context_chars_total", run_df[chars_col].sum())
                    mlflow.log_metric("estimated_cost_usd_total", run_df[cost_col].sum())
                    mlflow.log_metric("estimated_cost_usd_mean", run_df[cost_col].mean())

                    # LLM judge evaluation
                    eval_data = [
                        {"inputs": {"question": r["question"]}, "expected_response": r["expected_response"]}
                        for _, r in run_df.iterrows()
                    ]
                    _resp_col = resp_col  # closure capture
                    _run_df = run_df
                    eval_result = mlflow.genai.evaluate(
                        data=eval_data,
                        predict_fn=lambda question, _rc=_resp_col, _rd=_run_df: _rd.loc[
                            _rd["question"] == question, _rc
                        ].iloc[0],
                        scorers=scorers,
                    )

                    # Collect metrics for aggregation
                    for mk, mv in (eval_result.metrics if hasattr(eval_result, "metrics") else {}).items():
                        all_eval_metrics.append({
                            "model": endpoint, "agent": agent_name,
                            "run_index": run_idx, "metric": mk, "value": mv,
                        })
                        mlflow.log_metric(mk, mv)

                print(f"  {run_label}: evaluated")

    # Summary child run with aggregated metrics
    if all_eval_metrics:
        with mlflow.start_run(run_name="summary", nested=True):
            metrics_agg_df = pd.DataFrame(all_eval_metrics)
            for (model, agent, metric), grp in metrics_agg_df.groupby(["model", "agent", "metric"]):
                model_short = model.split("/")[-1] if "/" in model else model
                prefix = f"{model_short}_{agent}_{metric}"
                mlflow.log_metric(f"{prefix}_mean", grp["value"].mean())
                if len(grp) > 1:
                    mlflow.log_metric(f"{prefix}_std", grp["value"].std())
                    mlflow.log_metric(f"{prefix}_min", grp["value"].min())
                    mlflow.log_metric(f"{prefix}_max", grp["value"].max())
            print("Summary metrics logged")

print(f"\nAll evaluations complete: {len(all_eval_metrics)} metric observations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Comparison

# COMMAND ----------

# Build comparison from aggregated metrics
metrics_agg_df = pd.DataFrame(all_eval_metrics)

if len(metrics_agg_df) > 0:
    pivot = metrics_agg_df.groupby(["model", "agent", "metric"])["value"].agg(["mean", "std"]).reset_index()
    # For display: show first model's results (or all if multi-model)
    for endpoint in MODEL_ENDPOINTS:
        model_data = pivot[pivot["model"] == endpoint]
        if len(model_data) == 0:
            continue
        model_short = endpoint.split("/")[-1] if "/" in endpoint else endpoint
        print(f"\nModel: {model_short}")
        b_data = model_data[model_data["agent"] == "baseline"].set_index("metric")
        s_data = model_data[model_data["agent"] == "provena"].set_index("metric")
        all_metrics = sorted(set(list(b_data.index) + list(s_data.index)))
        comp_rows = []
        for m in all_metrics:
            bv = b_data.loc[m, "mean"] if m in b_data.index else None
            sv = s_data.loc[m, "mean"] if m in s_data.index else None
            bs = b_data.loc[m, "std"] if m in b_data.index and NUM_RUNS > 1 else None
            ss = s_data.loc[m, "std"] if m in s_data.index and NUM_RUNS > 1 else None
            delta = round(sv - bv, 4) if isinstance(bv, (int, float)) and isinstance(sv, (int, float)) else None
            row = {"metric": m, "baseline_mean": round(bv, 4) if bv is not None else None,
                   "sdol_mean": round(sv, 4) if sv is not None else None, "delta": delta}
            if NUM_RUNS > 1:
                row["baseline_std"] = round(bs, 4) if bs is not None else None
                row["sdol_std"] = round(ss, 4) if ss is not None else None
            comp_rows.append(row)
        comparison_df = pd.DataFrame(comp_rows)
        display(spark.createDataFrame(comparison_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latency and Token Efficiency

# COMMAND ----------

latency_summary = pd.DataFrame({
    "Agent": ["Baseline", "Provena-Enhanced"],
    "Mean Latency (s)": [results_df["baseline_latency_sec"].mean(), results_df["sdol_latency_sec"].mean()],
    "Median Latency (s)": [results_df["baseline_latency_sec"].median(), results_df["sdol_latency_sec"].median()],
    "Mean Context Chars": [results_df["baseline_context_chars"].mean(), results_df["sdol_context_chars"].mean()],
    "Total Context Chars": [results_df["baseline_context_chars"].sum(), results_df["sdol_context_chars"].sum()],
    "Token Efficiency Ratio": [1.0, round(results_df["baseline_context_chars"].sum() / max(results_df["sdol_context_chars"].sum(), 1), 1)],
})
display(spark.createDataFrame(latency_summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estimated Cost Comparison
# MAGIC
# MAGIC Cost estimates based on context window token consumption (chars / 4 = tokens).
# MAGIC Pricing: $`input_price_per_1k_tokens` per 1K input tokens, $`output_price_per_1k_tokens` per 1K output tokens.

# COMMAND ----------

b_session_cost = results_df["baseline_cost_usd"].sum()
s_session_cost = results_df["sdol_cost_usd"].sum()

cost_comparison = pd.DataFrame({
    "Metric": [
        "Cost per session (7 questions)",
        "Cost per 100 sessions/day",
        "Cost per 1,000 sessions/day",
        "Cost per 10,000 sessions/day",
        "Annual cost at 1,000 sessions/day",
    ],
    "Baseline ($)": [
        round(b_session_cost, 4),
        round(b_session_cost * 100, 2),
        round(b_session_cost * 1000, 2),
        round(b_session_cost * 10000, 2),
        round(b_session_cost * 1000 * 365, 2),
    ],
    "Provena ($)": [
        round(s_session_cost, 4),
        round(s_session_cost * 100, 2),
        round(s_session_cost * 1000, 2),
        round(s_session_cost * 10000, 2),
        round(s_session_cost * 1000 * 365, 2),
    ],
    "Savings ($)": [
        round(b_session_cost - s_session_cost, 4),
        round((b_session_cost - s_session_cost) * 100, 2),
        round((b_session_cost - s_session_cost) * 1000, 2),
        round((b_session_cost - s_session_cost) * 10000, 2),
        round((b_session_cost - s_session_cost) * 1000 * 365, 2),
    ],
})
display(spark.createDataFrame(cost_comparison))
print(f"\nProvena saves ~${round(b_session_cost - s_session_cost, 4)} per session ({round((1 - s_session_cost/max(b_session_cost, 0.0001)) * 100, 1)}% reduction)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Provena Execution Cost Breakdown

# COMMAND ----------

cost_summary = provena.get_cost_summary()
print(f"Total Provena queries: {cost_summary['total_queries']}")
print(f"Total execution time: {cost_summary['total_execution_ms']:.0f}ms")
print("\nBy source:")
for source, stats in cost_summary.get("by_source", {}).items():
    print(f"  {source}: {stats['query_count']} queries, {stats['total_ms']:.0f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visual Comparison

# COMMAND ----------

import matplotlib.pyplot as plt

if len(metrics_agg_df) > 0:
    # Use first model's data for the chart
    chart_data = metrics_agg_df[metrics_agg_df["model"] == MODEL_ENDPOINTS[0]]
    b_means = chart_data[chart_data["agent"] == "baseline"].groupby("metric")["value"].agg(["mean", "std"])
    s_means = chart_data[chart_data["agent"] == "provena"].groupby("metric")["value"].agg(["mean", "std"])
    common_metrics = sorted(set(b_means.index) & set(s_means.index))

    if common_metrics:
        fig, ax = plt.subplots(figsize=(14, 5))
        x = range(len(common_metrics))
        w = 0.35
        b_vals = [b_means.loc[k, "mean"] for k in common_metrics]
        s_vals = [s_means.loc[k, "mean"] for k in common_metrics]
        b_err = [b_means.loc[k, "std"] if NUM_RUNS > 1 else 0 for k in common_metrics]
        s_err = [s_means.loc[k, "std"] if NUM_RUNS > 1 else 0 for k in common_metrics]

        ax.bar([i - w/2 for i in x], b_vals, w, yerr=b_err, label="Baseline", color="#5B9BD5", capsize=3)
        ax.bar([i + w/2 for i in x], s_vals, w, yerr=s_err, label="Provena", color="#70AD47", capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(common_metrics, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Score")
        title = "Fleet Benchmark: LLM Judge Scores"
        if NUM_RUNS > 1:
            title += f" (mean +/- std, n={NUM_RUNS})"
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross-Model Comparison

# COMMAND ----------

if len(MODEL_ENDPOINTS) > 1 and len(metrics_agg_df) > 0:
    cross_model_rows = []
    for (model, agent, metric), grp in metrics_agg_df.groupby(["model", "agent", "metric"]):
        model_short = model.split("/")[-1] if "/" in model else model
        cross_model_rows.append({
            "model": model_short, "agent": agent, "metric": metric,
            "mean": round(grp["value"].mean(), 4),
            "std": round(grp["value"].std(), 4) if len(grp) > 1 else 0.0,
        })
    cross_df = pd.DataFrame(cross_model_rows)
    display(spark.createDataFrame(cross_df))
else:
    print("Single model -- cross-model comparison not applicable.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Responses -- Showcase Questions

# COMMAND ----------

for _, row in results_df.iterrows():
    print("=" * 100)
    print(f"CATEGORY: {row['category']}  |  QUESTION: {row['question']}")
    print("-" * 100)
    print(f"BASELINE ({row['baseline_latency_sec']}s, ${row['baseline_cost_usd']}):\n{row['baseline_response'][:600]}")
    print("-" * 50)
    print(f"Provena ({row['sdol_latency_sec']}s, ${row['sdol_cost_usd']}):\n{row['sdol_response'][:600]}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Dimension | Baseline MCP | Provena-Enhanced | Measurement |
# MAGIC |-----------|-------------|---------------|-------------|
# MAGIC | **Answer correctness** | High | High | `relevance_to_query` (both ~1.0) |
# MAGIC | **Provenance** | None -- all data treated equally | Every element tagged with source, freshness, consistency, trust | `provenance_completeness` scorer |
# MAGIC | **Conflict handling** | Silently picks one value or hallucinates | Detects deterministically, resolves via provenance heuristics | `conflict_detection_quality` scorer |
# MAGIC | **Token efficiency** | Dumps raw rows into context window | Push-down aggregation, targeted search | `context_chars_total` (~25x reduction) |
# MAGIC | **Cost at scale** | Higher token consumption = higher cost | Lower token consumption = lower cost | `estimated_cost_usd_total` |
# MAGIC | **Latency** | Fewer tool calls, lower latency | More typed calls, higher latency | `latency_mean` (tradeoff) |
# MAGIC | **Auditability** | Cannot prove data origin | Machine-verifiable provenance chain | Structural guarantee |
# MAGIC
# MAGIC **Provena does not make agents smarter -- it makes them auditable, cost-efficient, and deterministically reliable.**
# MAGIC
# MAGIC Check the MLflow Experiment UI for full traces, judge rationales, and per-question drilldowns.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *Fleet Management Benchmark -- Provena (Epistemic Provenance for AI Agents)*
