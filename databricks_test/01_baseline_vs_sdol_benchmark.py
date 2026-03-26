# Databricks notebook source
# MAGIC %md
# MAGIC # SDOL Benchmark: Baseline MCP Agent vs SDOL-Enhanced Agent
# MAGIC
# MAGIC This notebook runs a **head-to-head evaluation** of two LangGraph agents that query the same
# MAGIC Databricks lakehouse data:
# MAGIC
# MAGIC | Agent | Tools | System Prompt | Key Capability |
# MAGIC |-------|-------|---------------|----------------|
# MAGIC | **Baseline** | Raw `execute_sql` | Generic data analyst | LLM writes SQL directly |
# MAGIC | **SDOL-Enhanced** | Typed SDOL tools | Epistemic-aware analyst | SDOL generates optimized SQL, tracks provenance & trust |
# MAGIC
# MAGIC Both agents use the same LLM endpoint and answer the same evaluation questions.
# MAGIC **Databricks-managed LLM judges** (via MLflow) score every response on relevance, safety,
# MAGIC data-confidence awareness, and completeness.
# MAGIC
# MAGIC ### What SDOL adds
# MAGIC 1. **Intent-based routing** — OLAP vs OLTP queries go to the right backend automatically
# MAGIC 2. **Optimized SQL generation** — parameterized queries, Photon hints, column pruning
# MAGIC 3. **Provenance tracking** — every result carries source, freshness, consistency metadata
# MAGIC 4. **Trust scoring** — quantified confidence (0–1) per data element
# MAGIC 5. **Epistemic context** — the agent can reason about and communicate data quality
# MAGIC
# MAGIC **Prerequisite:** Run `00_setup_benchmark_resources` first to create the data tables.

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-langchain databricks-agents langgraph langchain langchain-core nest_asyncio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "users"   # Must match 00_setup notebook
SCHEMA = "aradhya_chouhan"

LLM_ENDPOINT = "databricks-claude-3-7-sonnet"  # TODO: any Foundation Model endpoint

# Path to the SDOL source tree — update for your workspace.
# Option A: Workspace Repo / Git folder
# Option B: Uploaded wheel  →  comment out sys.path and %pip install the wheel instead
SDOL_SRC_PATH = "/Workspace/Users/{user}/SDOL-python/src"  # TODO: update

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install / import SDOL

# COMMAND ----------

import sys, os

# Try importing; fall back to sys.path injection from the configured source path
try:
    import sdol  # already installed (wheel / pip)
except ImportError:
    resolved = SDOL_SRC_PATH.replace("{user}", spark.sql("SELECT current_user()").first()[0])
    if os.path.isdir(resolved):
        sys.path.insert(0, resolved)
        import sdol
        print(f"Loaded SDOL from {resolved}")
    else:
        raise ImportError(
            f"SDOL not found. Either install the wheel or update SDOL_SRC_PATH "
            f"(tried {resolved})"
        )

print(f"SDOL package loaded — exports: {sdol.__all__[:5]} …")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize the SDOL pipeline
# MAGIC
# MAGIC Wire up a `SparkSQLExecutor` (implements SDOL's `QueryExecutor` protocol using the
# MAGIC notebook's Spark session), then register Databricks DBSQL and Lakebase connectors
# MAGIC with the full SDOL stack: registry → planner → router → SDK.

# COMMAND ----------

import asyncio, json, time, textwrap, nest_asyncio
nest_asyncio.apply()

from sdol import (
    SDOL as SDOLEngine,
    CapabilityRegistry,
    ContextCompiler,
    DatabricksDBSQLConnector,
    DatabricksLakebaseConnector,
    SemanticRouter,
    TrustScorer,
)
from sdol.core.provenance.trust_scorer import TrustScorerConfig
from sdol.core.router.cost_estimator import CostEstimator
from sdol.core.router.intent_decomposer import IntentDecomposer
from sdol.core.router.query_planner import QueryPlanner


class SparkSQLExecutor:
    """Bridges SDOL's QueryExecutor protocol with the notebook's SparkSession.

    Errors are **propagated** rather than swallowed so that SDOL tools can
    report the real failure to the LLM instead of returning empty results.
    """

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
        return {
            "records": records,
            "meta": {"native_query": sql_str, "total_available": len(records)},
        }


executor = SparkSQLExecutor()

dbsql_connector = DatabricksDBSQLConnector(
    executor=executor,
    connector_id="databricks.analytics",
    source_system="databricks.sql_warehouse",
    available_entities=["sales_transactions", "revenue_daily"],
    catalog=CATALOG,
    schema=SCHEMA,
    time_column_map={
        "sales_transactions": "order_date",
        "revenue_daily": "report_date",
    },
)

lakebase_connector = DatabricksLakebaseConnector(
    executor=executor,
    connector_id="databricks.serving",
    source_system="databricks.lakebase",
    available_entities=["customers", "products", "orders"],
    catalog=CATALOG,
    schema=SCHEMA,
)

registry = CapabilityRegistry()
registry.register(dbsql_connector)
registry.register(lakebase_connector)

trust_cfg = TrustScorerConfig(source_authority_map={
    "databricks.sql_warehouse": 0.95,
    "databricks.lakebase": 0.90,
})
compiler = ContextCompiler(TrustScorer(trust_cfg))
planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
router = SemanticRouter(planner, compiler, registry)
sdol = SDOLEngine(router)

print("SDOL pipeline ready")
print(f"  OLAP entities : {dbsql_connector._available_entities}")
print(f"  OLTP entities : {lakebase_connector._available_entities}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Agent Tools
# MAGIC
# MAGIC ### Baseline tools
# MAGIC The baseline agent gets raw SQL execution — it must write all SQL itself.
# MAGIC
# MAGIC ### SDOL tools
# MAGIC The SDOL agent gets intent-level tools that delegate SQL generation, execution,
# MAGIC and provenance tracking to the SDOL pipeline.

# COMMAND ----------

from langchain_core.tools import tool

def _parse_filters(raw: str) -> list[dict] | None:
    """Parse a flexible filter string into SDOL FilterClause dicts.

    Accepted formats (semicolon-separated):
      • field = value          (equality shorthand)
      • field:op:value         (explicit operator)
      • field op value         (space-separated, op in {=, !=, >, <, >=, <=})

    Returns None when *raw* is empty.
    """
    if not raw or not raw.strip():
        return None
    OP_ALIAS = {"=": "eq", "!=": "neq", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}
    result = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        # field:op:value
        if part.count(":") >= 2:
            segs = part.split(":", 2)
            field, op, val = segs[0].strip(), segs[1].strip(), segs[2].strip()
        # field = value / field != value …
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

def _format_frame(frame, include_confidence=True) -> str:
    """Extract results + optional epistemic context from a ContextFrame."""
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
    out = {"results": results, "result_count": len(results)}
    if include_confidence:
        out["data_confidence"] = sdol.get_epistemic_context()
    return json.dumps(out, indent=2, default=str)

# ─────────────────────── Baseline Tools ───────────────────────

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
    """List all benchmark tables and their columns."""
    info = {}
    for tbl in ["customers", "products", "orders", "sales_transactions", "revenue_daily"]:
        cols = [
            row.col_name
            for row in spark.sql(f"DESCRIBE {CATALOG}.{SCHEMA}.{tbl}").collect()
            if not row.col_name.startswith("#")
        ]
        info[f"{CATALOG}.{SCHEMA}.{tbl}"] = cols
    return json.dumps(info, indent=2)


baseline_tools = [execute_sql, describe_tables]

# ─────────────────────── SDOL Tools ───────────────────────

@tool
def sdol_point_lookup(entity: str, id_field: str, id_value: str, fields: str = "") -> str:
    """Look up a specific record by its identifier via SDOL optimized routing.

    Args:
        entity: Table name — 'customers', 'products', or 'orders'.
        id_field: Identifier column name, e.g. 'customer_id'.
        id_value: Identifier value, e.g. 'C-0042'.
        fields: Comma-separated columns to return (empty = all).
    """
    try:
        field_list = [f.strip() for f in fields.split(",") if f.strip()] or None
        intent = sdol.formulator.point_lookup(entity, {id_field: id_value}, fields=field_list)
        frame = asyncio.run(sdol.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc), "data_confidence": sdol.get_epistemic_context()})


@tool
def sdol_aggregate(
    entity: str,
    measures: str,
    dimensions: str,
    filters: str = "",
    order_by: str = "",
) -> str:
    """Run an aggregate analysis via SDOL (routes to the optimal OLAP backend).

    Args:
        entity: Table name — 'sales_transactions' or 'revenue_daily'.
        measures: One or more measures as 'agg(field)' separated by commas.
                  Examples: 'sum(total_amount)', 'count(order_id), avg(total_amount)'.
        dimensions: Comma-separated grouping columns, e.g. 'region' or 'region,channel'.
        filters: Optional filters. Accepts natural syntax separated by semicolons:
                 'status = completed', 'status:eq:completed', or 'total_amount > 100'.
        order_by: Optional ordering, e.g. 'sum_total_amount desc'.
    """
    import re as _re
    measure_list = []
    for m in measures.split(","):
        m = m.strip()
        match = _re.match(r"(\w+)\((\w+)\)", m)
        if match:
            measure_list.append({"field": match.group(2), "aggregation": match.group(1)})
        else:
            measure_list.append({"field": m, "aggregation": "sum"})

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
            entity=entity, measures=measure_list, dimensions=dims,
            filters=filter_list, order_by=ob,
        )
        frame = asyncio.run(sdol.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc), "data_confidence": sdol.get_epistemic_context()})


@tool
def sdol_temporal_trend(entity: str, metric: str, window: str = "last_90d", granularity: str = "1M") -> str:
    """Analyze temporal trends via SDOL (routes to OLAP with DATE_TRUNC).

    Args:
        entity: Table name, e.g. 'sales_transactions' or 'revenue_daily'.
        metric: Metric column to aggregate, e.g. 'total_amount' or 'total_revenue'.
        window: Relative time window — 'last_30d', 'last_90d', 'last_1y'.
        granularity: Time bucket size — '1d' (day), '1w' (week), '1M' (month).
    """
    try:
        intent = sdol.formulator.temporal_trend(
            entity=entity, metric=metric, window={"relative": window}, granularity=granularity,
        )
        frame = asyncio.run(sdol.query(intent))
        return _format_frame(frame)
    except Exception as exc:
        return json.dumps({"error": str(exc), "data_confidence": sdol.get_epistemic_context()})


@tool
def sdol_sql(query: str) -> str:
    """Execute a raw SQL query with SDOL provenance tracking.

    Use this for complex queries that need JOINs across tables, multiple
    sub-queries, or anything the other SDOL tools cannot express directly.

    Args:
        query: A complete Spark SQL query using fully-qualified table names.
    """
    try:
        df = spark.sql(query)
        records = [row.asDict() for row in df.limit(100).collect()]
        return json.dumps({
            "results": records,
            "result_count": len(records),
            "note": "Raw SQL — no SDOL provenance tracking for this query.",
            "data_confidence": sdol.get_epistemic_context(),
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def sdol_data_confidence() -> str:
    """Return a summary of data confidence, trust scores, and quality for all data queried so far."""
    return sdol.get_epistemic_context()


sdol_tools = [sdol_point_lookup, sdol_aggregate, sdol_temporal_trend, sdol_sql, sdol_data_confidence]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build LangGraph Agents
# MAGIC
# MAGIC Both agents share the same LLM and graph structure — only the **tools** and **system prompt** differ.

# COMMAND ----------

from typing import Annotated, Any, Optional, Sequence, TypedDict
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode


class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]


llm = ChatDatabricks(endpoint=LLM_ENDPOINT)


def build_agent(tools, system_prompt: str):
    """Return a compiled LangGraph tool-calling agent."""
    bound = llm.bind_tools(tools)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "continue"
        return "end"

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + list(state["messages"])
    )
    chain = preprocessor | bound

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

TABLE_SCHEMAS = f"""
Available tables in `{CATALOG}.{SCHEMA}`:

• customers (OLTP)  — customer_id, name, email, tier, region, signup_date, last_login, lifetime_value, is_active
• products  (OLTP)  — product_id, name, category, price, stock_quantity, supplier, is_available
• orders    (OLTP)  — order_id, customer_id, product_id, quantity, unit_price, total_amount, status, created_at, updated_at
• sales_transactions (OLAP) — transaction_id, customer_id, product_id, quantity, unit_price, total_amount, region, channel, order_date, status
• revenue_daily      (OLAP) — report_date, region, total_revenue, order_count, avg_order_value
""".strip()

BASELINE_SYSTEM_PROMPT = f"""You are a data analyst assistant with access to a Databricks lakehouse.

{TABLE_SCHEMAS}

Guidelines:
- Use `describe_tables` to inspect column names if unsure.
- Use `execute_sql` to run Spark SQL queries. Always use fully-qualified table names ({CATALOG}.{SCHEMA}.<table>).
- Present results clearly with actual values. Summarize large result sets.
- If you are unsure about data quality, say so honestly.
"""

SDOL_SYSTEM_PROMPT = f"""You are a data analyst assistant enhanced with SDOL (Semantic Data Orchestration Layer).

SDOL automatically routes queries to the optimal backend, generates optimized SQL,
tracks data provenance (source, freshness, consistency), and computes trust scores.

{TABLE_SCHEMAS}

Available SDOL tools (choose the best match for each question):
- `sdol_point_lookup`    — look up a single record by ID  (routes → OLTP / Lakebase)
- `sdol_aggregate`       — aggregations, counts, top-N    (routes → OLAP / DBSQL)
    measures format: 'sum(total_amount)' or 'count(order_id), avg(total_amount)'
    filters format: 'status = completed; region = west'  (semicolon-separated)
- `sdol_temporal_trend`  — time-series bucketed analysis   (routes → OLAP / DBSQL)
- `sdol_sql`             — raw SQL for JOINs or complex queries that cross tables
- `sdol_data_confidence` — overall trust & quality summary

Guidelines:
- For questions that require JOINing two tables (e.g. customer details + their orders),
  use `sdol_sql` with a proper SQL JOIN using fully-qualified table names ({CATALOG}.{SCHEMA}.<table>).
- Always include data confidence context in your answer — mention source, trust score, and freshness.
- When trust scores are below 0.7, explicitly flag reduced confidence.
- When presenting numbers, cite the source system (e.g. DBSQL / Lakebase).
"""

# COMMAND ----------

import mlflow
mlflow.langchain.autolog()

baseline_agent = build_agent(baseline_tools, BASELINE_SYSTEM_PROMPT)
sdol_agent     = build_agent(sdol_tools, SDOL_SYSTEM_PROMPT)

print("Both agents compiled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke Test

# COMMAND ----------

def invoke_agent(agent, question: str) -> str:
    """Run an agent and return the final text response."""
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)

# Quick sanity check
q = "How many customers are in the enterprise tier?"
print("── Baseline ──")
print(invoke_agent(baseline_agent, q)[:600])
print("\n── SDOL ──")
sdol.reset()
print(invoke_agent(sdol_agent, q)[:600])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Dataset
# MAGIC
# MAGIC 10 questions spanning OLTP lookups, OLAP aggregations, cross-paradigm queries, and
# MAGIC confidence / uncertainty reasoning — the categories where SDOL should shine.

# COMMAND ----------

EVAL_QUESTIONS = [
    # ── OLTP: Point Lookups ──
    {
        "question": "What are the full details for customer C-0042? Include name, tier, region, and lifetime value.",
        "category": "point_lookup",
        "expected_response": "Should return the exact row for customer C-0042 from the customers table with name, tier, region, and lifetime_value.",
    },
    {
        "question": "Look up product P-105. What is its category, price, and current stock quantity?",
        "category": "point_lookup",
        "expected_response": "Should return the exact row for product P-105 with category, price, and stock_quantity.",
    },
    # ── OLAP: Aggregations ──
    {
        "question": "What is the total revenue by region for completed transactions? Sort from highest to lowest.",
        "category": "aggregate",
        "expected_response": "Should return SUM(total_amount) grouped by region for status='completed', ordered descending.",
    },
    {
        "question": "Which sales channel (online, in_store, mobile) has the highest average order value for completed sales?",
        "category": "aggregate",
        "expected_response": "Should return AVG(total_amount) grouped by channel for completed transactions, identifying the top channel.",
    },
    {
        "question": "How many completed orders are there per customer tier? Also show the average order amount per tier.",
        "category": "aggregate",
        "expected_response": "Should join or query to get COUNT and AVG(total_amount) grouped by customer tier for completed orders.",
    },
    # ── OLAP: Temporal ──
    {
        "question": "Show the monthly total revenue trend from the revenue_daily table. Which month had the highest revenue?",
        "category": "temporal",
        "expected_response": "Should aggregate revenue_daily by month, showing the trend and identifying the peak month.",
    },
    # ── Cross-Paradigm ──
    {
        "question": "Who are the top 5 customers by total spending in sales_transactions (completed only)? For each, show customer name and tier.",
        "category": "cross_paradigm",
        "expected_response": "Should aggregate sales_transactions by customer_id, then look up customer details for the top 5.",
    },
    {
        "question": "For each region, what is the total revenue from enterprise-tier customers only?",
        "category": "cross_paradigm",
        "expected_response": "Should filter customers by tier='enterprise', then aggregate their transactions by region.",
    },
    # ── Data Confidence / Uncertainty ──
    {
        "question": "How confident should we be in the total revenue numbers for the south region? Are there any data quality concerns?",
        "category": "confidence",
        "expected_response": "Should query the data AND communicate confidence level, data freshness, and any limitations.",
    },
    {
        "question": "Give me a summary of overall data quality across all the tables you have access to. What should I be cautious about?",
        "category": "confidence",
        "expected_response": "Should introspect data sources and provide a nuanced view of data reliability and coverage.",
    },
]

print(f"Evaluation set: {len(EVAL_QUESTIONS)} questions")
for eq in EVAL_QUESTIONS:
    print(f"  [{eq['category']:15s}] {eq['question'][:80]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Both Agents
# MAGIC
# MAGIC Execute every question on both agents, recording responses and latency.

# COMMAND ----------

import pandas as pd

rows = []
for i, eq in enumerate(EVAL_QUESTIONS):
    q = eq["question"]
    cat = eq["category"]
    print(f"[{i+1}/{len(EVAL_QUESTIONS)}] {cat}: {q[:70]}…")

    # Baseline
    sdol.reset()
    t0 = time.time()
    try:
        b_resp = invoke_agent(baseline_agent, q)
    except Exception as exc:
        b_resp = f"ERROR: {exc}"
    b_lat = round(time.time() - t0, 2)

    # SDOL
    sdol.reset()
    t0 = time.time()
    try:
        s_resp = invoke_agent(sdol_agent, q)
    except Exception as exc:
        s_resp = f"ERROR: {exc}"
    s_lat = round(time.time() - t0, 2)

    rows.append({
        "question": q,
        "category": cat,
        "expected_response": eq["expected_response"],
        "baseline_response": b_resp,
        "baseline_latency_sec": b_lat,
        "sdol_response": s_resp,
        "sdol_latency_sec": s_lat,
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
# MAGIC We use MLflow's **predefined scorers** (backed by Databricks Foundation Model judges) plus
# MAGIC a **custom Guidelines scorer** for data-confidence awareness — the metric where SDOL should
# MAGIC have the clearest advantage.

# COMMAND ----------

from mlflow.genai.scorers import RelevanceToQuery, Safety

try:
    from mlflow.genai.scorers import Guidelines
    HAS_GUIDELINES = True
except ImportError:
    HAS_GUIDELINES = False
    print("⚠ Guidelines scorer not available in this MLflow version — using built-in scorers only.")

scorers = [RelevanceToQuery(), Safety()]

if HAS_GUIDELINES:
    scorers.append(Guidelines(
        name="data_confidence_awareness",
        guidelines=(
            "The response should reference or acknowledge where the data comes from. "
            "The response should mention data freshness, recency, or staleness when relevant. "
            "The response should communicate confidence or uncertainty about quantitative claims. "
            "The response should note any caveats, limitations, or potential data quality issues."
        ),
    ))
    scorers.append(Guidelines(
        name="response_completeness",
        guidelines=(
            "The response should directly and fully answer the question asked. "
            "The response should include all specifically requested data points or metrics. "
            "The response should present data in a clear, structured format (tables, lists, etc.). "
            "The response should not omit important details that were explicitly asked for."
        ),
    ))

print(f"Scorers: {[s.__class__.__name__ for s in scorers]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Baseline Agent

# COMMAND ----------

baseline_eval_data = [
    {
        "inputs": {"question": r["question"]},
        "expected_response": r["expected_response"],
    }
    for _, r in results_df.iterrows()
]

with mlflow.start_run(run_name="baseline_mcp_agent"):
    baseline_eval = mlflow.genai.evaluate(
        data=baseline_eval_data,
        predict_fn=lambda question: results_df.loc[
            results_df["question"] == question, "baseline_response"
        ].iloc[0],
        scorers=scorers,
    )

print("Baseline evaluation complete")
baseline_metrics_df = pd.DataFrame([
    {"metric": k, "value": v} for k, v in baseline_eval.metrics.items()
])
display(spark.createDataFrame(baseline_metrics_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate SDOL-Enhanced Agent

# COMMAND ----------

sdol_eval_data = [
    {
        "inputs": {"question": r["question"]},
        "expected_response": r["expected_response"],
    }
    for _, r in results_df.iterrows()
]

with mlflow.start_run(run_name="sdol_enhanced_agent"):
    sdol_eval = mlflow.genai.evaluate(
        data=sdol_eval_data,
        predict_fn=lambda question: results_df.loc[
            results_df["question"] == question, "sdol_response"
        ].iloc[0],
        scorers=scorers,
    )

print("SDOL evaluation complete")
sdol_metrics_df = pd.DataFrame([
    {"metric": k, "value": v} for k, v in sdol_eval.metrics.items()
])
display(spark.createDataFrame(sdol_metrics_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Comparison

# COMMAND ----------

b_metrics = baseline_eval.metrics if hasattr(baseline_eval, "metrics") else {}
s_metrics = sdol_eval.metrics if hasattr(sdol_eval, "metrics") else {}

all_keys = sorted(set(list(b_metrics.keys()) + list(s_metrics.keys())))

comparison_rows = []
for k in all_keys:
    bv = b_metrics.get(k)
    sv = s_metrics.get(k)
    delta = None
    if isinstance(bv, (int, float)) and isinstance(sv, (int, float)):
        delta = round(sv - bv, 4)
    comparison_rows.append({"metric": k, "baseline": bv, "sdol_enhanced": sv, "delta": delta})

comparison_df = pd.DataFrame(comparison_rows)
display(spark.createDataFrame(comparison_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latency Comparison

# COMMAND ----------

latency_summary = pd.DataFrame({
    "Agent": ["Baseline", "SDOL-Enhanced"],
    "Mean Latency (s)": [
        results_df["baseline_latency_sec"].mean(),
        results_df["sdol_latency_sec"].mean(),
    ],
    "Median Latency (s)": [
        results_df["baseline_latency_sec"].median(),
        results_df["sdol_latency_sec"].median(),
    ],
    "Max Latency (s)": [
        results_df["baseline_latency_sec"].max(),
        results_df["sdol_latency_sec"].max(),
    ],
})
display(spark.createDataFrame(latency_summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Per-Question Score Comparison
# MAGIC
# MAGIC Merge the per-row evaluation tables from both runs for a side-by-side view.

# COMMAND ----------

try:
    b_table = baseline_eval.tables["eval_results"].rename(
        columns=lambda c: f"baseline_{c}" if c != "question" else c
    )
    s_table = sdol_eval.tables["eval_results"].rename(
        columns=lambda c: f"sdol_{c}" if c != "question" else c
    )
    merged = b_table.merge(s_table, left_index=True, right_index=True, suffixes=("_b", "_s"))
    display(spark.createDataFrame(merged))
except Exception:
    print("Per-row tables not available — see aggregate metrics above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visual Comparison

# COMMAND ----------

import matplotlib.pyplot as plt

score_metrics = [k for k in all_keys if isinstance(b_metrics.get(k), (int, float)) and isinstance(s_metrics.get(k), (int, float))]

if score_metrics:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(score_metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], [b_metrics[k] for k in score_metrics], width, label="Baseline", color="#5B9BD5")
    ax.bar([i + width/2 for i in x], [s_metrics[k] for k in score_metrics], width, label="SDOL Enhanced", color="#70AD47")
    ax.set_xticks(list(x))
    ax.set_xticklabels(score_metrics, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("LLM Judge Scores: Baseline vs SDOL-Enhanced Agent")
    ax.legend()
    ax.set_ylim(0, max(max(b_metrics.get(k, 0) for k in score_metrics), max(s_metrics.get(k, 0) for k in score_metrics)) * 1.15)
    plt.tight_layout()
    display(fig)
else:
    print("No comparable numeric metrics found for charting.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Response Comparison

# COMMAND ----------

for _, row in results_df.iterrows():
    print("=" * 100)
    print(f"CATEGORY: {row['category']}  |  QUESTION: {row['question']}")
    print("-" * 100)
    print(f"BASELINE ({row['baseline_latency_sec']}s):\n{row['baseline_response'][:500]}")
    print("-" * 50)
    print(f"SDOL ({row['sdol_latency_sec']}s):\n{row['sdol_response'][:500]}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary & Interpretation
# MAGIC
# MAGIC | Dimension | Expected SDOL Advantage | Why |
# MAGIC |-----------|------------------------|-----|
# MAGIC | **Relevance** | Moderate | SDOL routes to the optimal backend; generates correct SQL |
# MAGIC | **Data Confidence** | Strong | Provenance + trust scores → agent communicates uncertainty |
# MAGIC | **Completeness** | Moderate | Intent-based tools ensure all requested fields are returned |
# MAGIC | **Safety** | Neutral | Both agents use the same LLM |
# MAGIC | **Latency** | Varies | SDOL adds orchestration overhead but generates simpler SQL |
# MAGIC
# MAGIC Check the MLflow Experiment UI for full traces, judge rationales, and per-question drilldowns.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *Benchmark generated by SDOL — Semantic Data Orchestration Layer*
