"""Databricks SQL query builder with Photon and Delta Lake optimizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sdol.connectors.sql_utils import OPERATOR_MAP, qualify_table
from sdol.types.intent import (
    AggregateAnalysisIntent,
    FilterClause,
    MeasureSpec,
    TemporalTrendIntent,
)


@dataclass
class DBSQLQuery:
    """Native query representation for Databricks SQL (DBSQL) endpoints."""

    sql: str
    parameters: dict[str, Any]
    optimizations: list[str]
    estimated_rows_scanned: int
    uses_photon: bool
    uses_delta_data_skipping: bool
    uses_liquid_clustering: bool
    catalog: str | None
    schema: str | None


import re

_RELATIVE_RE = re.compile(r"^last_(\d+)(h|d|w|M|Q|y)$")
_UNIT_TO_SQL: dict[str, str] = {
    "h": "HOUR",
    "d": "DAY",
    "w": "WEEK",
    "M": "MONTH",
    "Q": "QUARTER",  # mapped to MONTH * 3 below
    "y": "YEAR",
}


def parse_relative_window(raw: str) -> str:
    """Convert 'last_90d' → 'INTERVAL 90 DAY', etc.

    Falls back to ``INTERVAL '<raw>'`` so absolute strings like '30 DAY'
    pass through unchanged.
    """
    m = _RELATIVE_RE.match(raw)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "Q":
            return f"INTERVAL {n * 3} MONTH"
        return f"INTERVAL {n} {_UNIT_TO_SQL[unit]}"
    return f"INTERVAL {raw}"


KNOWN_ROLLUP_GRANULARITIES = {"day", "week", "month", "quarter", "year"}

GRANULARITY_TO_TRUNC: dict[str, str] = {
    "1h": "HOUR",
    "1d": "DAY",
    "1w": "WEEK",
    "1M": "MONTH",
    "1Q": "QUARTER",
    "1y": "YEAR",
    "day": "DAY",
    "week": "WEEK",
    "month": "MONTH",
    "quarter": "QUARTER",
    "year": "YEAR",
}


def _build_where(
    filters: list[FilterClause] | None,
    params: dict[str, Any],
    counter: list[int],
) -> str:
    if not filters:
        return ""
    clauses: list[str] = []
    for f in filters:
        op = OPERATOR_MAP.get(f.operator, "=")
        if f.operator == "in":
            placeholders = ", ".join(
                [f":p{counter[0] + i}" for i in range(len(f.value))]
            )
            for i, v in enumerate(f.value):
                params[f"p{counter[0] + i}"] = v
            counter[0] += len(f.value)
            clauses.append(f"{f.field} IN ({placeholders})")
        elif f.operator == "exists":
            clauses.append(f"{f.field} IS NOT NULL")
        elif f.operator == "contains":
            pname = f"p{counter[0]}"
            params[pname] = f"%{f.value}%"
            counter[0] += 1
            clauses.append(f"{f.field} LIKE :{pname}")
        else:
            pname = f"p{counter[0]}"
            params[pname] = f.value
            counter[0] += 1
            clauses.append(f"{f.field} {op} :{pname}")
    return " WHERE " + " AND ".join(clauses)


def _agg_sql(m: MeasureSpec) -> str:
    alias = m.alias or f"{m.aggregation}_{m.field}"
    if m.aggregation in ("p50", "p95", "p99"):
        pct = {"p50": 0.5, "p95": 0.95, "p99": 0.99}[m.aggregation]
        return f"PERCENTILE_APPROX({m.field}, {pct}) AS {alias}"
    if m.aggregation == "count_distinct":
        return f"COUNT(DISTINCT {m.field}) AS {alias}"
    return f"{m.aggregation.upper()}({m.field}) AS {alias}"


def build_dbsql_aggregate_query(
    intent: AggregateAnalysisIntent,
    *,
    catalog: str | None = None,
    schema: str | None = None,
) -> DBSQLQuery:
    """Build an aggregate analysis query targeting Databricks SQL."""
    params: dict[str, Any] = {}
    counter = [0]
    optimizations: list[str] = ["photon_acceleration", "predicate_pushdown"]

    select_parts = [_agg_sql(m) for m in intent.measures]
    select_parts = list(intent.dimensions) + select_parts
    select_clause = ", ".join(select_parts)

    table = qualify_table(intent.entity, catalog, schema)

    where_clause = _build_where(intent.filters, params, counter)
    if where_clause:
        optimizations.append("delta_data_skipping")

    group_by = ", ".join(intent.dimensions)

    having_clause = ""
    if intent.having:
        having_parts: list[str] = []
        for h in intent.having:
            op = OPERATOR_MAP.get(h.operator, "=")
            pname = f"p{counter[0]}"
            params[pname] = h.value
            counter[0] += 1
            having_parts.append(f"{h.field} {op} :{pname}")
        having_clause = " HAVING " + " AND ".join(having_parts)

    order_clause = ""
    if intent.order_by:
        order_parts = [f"{o.field} {o.direction.upper()}" for o in intent.order_by]
        order_clause = " ORDER BY " + ", ".join(order_parts)
        optimizations.append("order_pushdown")

    limit_clause = ""
    if intent.max_results:
        limit_clause = f" LIMIT {intent.max_results}"

    sql = (
        f"SELECT {select_clause} FROM {table}"
        f"{where_clause} GROUP BY {group_by}{having_clause}{order_clause}{limit_clause}"
    )

    optimizations.append("pushdown_aggregation")

    return DBSQLQuery(
        sql=sql,
        parameters=params,
        optimizations=optimizations,
        estimated_rows_scanned=100_000,
        uses_photon=True,
        uses_delta_data_skipping=bool(where_clause),
        uses_liquid_clustering=False,
        catalog=catalog,
        schema=schema,
    )


def build_dbsql_temporal_query(
    intent: TemporalTrendIntent,
    *,
    catalog: str | None = None,
    schema: str | None = None,
    time_column: str = "timestamp",
) -> DBSQLQuery:
    """Build a temporal trend query targeting Databricks SQL with DATE_TRUNC.

    Args:
        time_column: The column that holds the time dimension for this entity
                     (e.g. ``order_date``, ``report_date``).  Defaults to
                     ``timestamp`` for backward compatibility.
    """
    params: dict[str, Any] = {}
    counter = [0]
    optimizations: list[str] = ["photon_acceleration", "predicate_pushdown"]
    uses_delta_data_skipping = False
    uses_liquid_clustering = False

    time_bucket = intent.granularity or "1d"
    trunc_unit = GRANULARITY_TO_TRUNC.get(time_bucket, "DAY")

    if trunc_unit.lower() in KNOWN_ROLLUP_GRANULARITIES:
        table = qualify_table(intent.entity, catalog, schema)
        optimizations.append("delta_caching")
    else:
        table = qualify_table(intent.entity, catalog, schema)

    tc = time_column
    bucket_expr = f"DATE_TRUNC('{trunc_unit}', {tc})"
    select_clause = f"{bucket_expr} AS bucket, AVG({intent.metric}) AS {intent.metric}"

    where_parts: list[str] = []
    if intent.window.start:
        pname = f"p{counter[0]}"
        params[pname] = intent.window.start
        counter[0] += 1
        where_parts.append(f"{tc} >= :{pname}")
        uses_delta_data_skipping = True
    if intent.window.end:
        pname = f"p{counter[0]}"
        params[pname] = intent.window.end
        counter[0] += 1
        where_parts.append(f"{tc} <= :{pname}")
        uses_delta_data_skipping = True
    if intent.window.relative:
        interval_expr = parse_relative_window(intent.window.relative)
        where_parts.append(f"{tc} >= CURRENT_TIMESTAMP() - {interval_expr}")
        uses_delta_data_skipping = True

    if uses_delta_data_skipping:
        optimizations.append("delta_data_skipping")

    filter_where = _build_where(intent.filters, params, counter)
    extra_where = " AND ".join(where_parts)
    if filter_where and extra_where:
        where_clause = filter_where + " AND " + extra_where
    elif extra_where:
        where_clause = " WHERE " + extra_where
    else:
        where_clause = filter_where

    sql = (
        f"SELECT {select_clause} FROM {table}{where_clause}"
        f" GROUP BY bucket ORDER BY bucket"
    )

    if intent.max_results:
        sql += f" LIMIT {intent.max_results}"

    return DBSQLQuery(
        sql=sql,
        parameters=params,
        optimizations=optimizations,
        estimated_rows_scanned=50_000,
        uses_photon=True,
        uses_delta_data_skipping=uses_delta_data_skipping,
        uses_liquid_clustering=uses_liquid_clustering,
        catalog=catalog,
        schema=schema,
    )
