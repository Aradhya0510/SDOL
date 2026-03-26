"""SQL/analytical query builder for generic OLAP systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sdol.connectors.sql_utils import OPERATOR_MAP
from sdol.types.intent import (
    AggregateAnalysisIntent,
    FilterClause,
    MeasureSpec,
    TemporalTrendIntent,
)


@dataclass
class OLAPQuery:
    """Native query representation for OLAP systems."""

    sql: str
    params: list[Any]
    optimizations: list[str]
    estimated_rows_scanned: int
    uses_partition_pruning: bool
    uses_precomputed_rollup: bool


KNOWN_ROLLUPS = {"1d", "1w", "1M"}


def _build_where(
    filters: list[FilterClause] | None,
    params: list[Any],
) -> str:
    if not filters:
        return ""
    clauses: list[str] = []
    for f in filters:
        op = OPERATOR_MAP.get(f.operator, "=")
        if f.operator == "in":
            placeholders = ", ".join(["$" + str(len(params) + 1 + i) for i in range(len(f.value))])
            clauses.append(f"{f.field} IN ({placeholders})")
            params.extend(f.value)
        elif f.operator == "exists":
            clauses.append(f"{f.field} IS NOT NULL")
        elif f.operator == "contains":
            params.append(f"%{f.value}%")
            clauses.append(f"{f.field} LIKE ${len(params)}")
        else:
            params.append(f.value)
            clauses.append(f"{f.field} {op} ${len(params)}")
    return " WHERE " + " AND ".join(clauses)


def _agg_sql(m: MeasureSpec) -> str:
    alias = m.alias or f"{m.aggregation}_{m.field}"
    if m.aggregation in ("p50", "p95", "p99"):
        pct = {"p50": 0.5, "p95": 0.95, "p99": 0.99}[m.aggregation]
        return f"PERCENTILE_CONT({pct}) WITHIN GROUP (ORDER BY {m.field}) AS {alias}"
    return f"{m.aggregation.upper()}({m.field}) AS {alias}"


def build_aggregate_query(intent: AggregateAnalysisIntent) -> OLAPQuery:
    """Build an aggregate analysis SQL query with push-down optimizations."""
    params: list[Any] = []
    optimizations: list[str] = ["predicate_pushdown"]

    select_parts = [_agg_sql(m) for m in intent.measures]
    select_parts = intent.dimensions + select_parts
    select_clause = ", ".join(select_parts)

    where_clause = _build_where(intent.filters, params)
    if where_clause:
        optimizations.append("filter_pushdown")

    group_by = ", ".join(intent.dimensions)

    having_clause = ""
    if intent.having:
        having_parts: list[str] = []
        for h in intent.having:
            op = OPERATOR_MAP.get(h.operator, "=")
            params.append(h.value)
            agg_field = h.field
            having_parts.append(f"{agg_field} {op} ${len(params)}")
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
        f"SELECT {select_clause} FROM {intent.entity}"
        f"{where_clause} GROUP BY {group_by}{having_clause}{order_clause}{limit_clause}"
    )

    optimizations.append("pushdown_aggregation")

    return OLAPQuery(
        sql=sql,
        params=params,
        optimizations=optimizations,
        estimated_rows_scanned=100000,
        uses_partition_pruning=False,
        uses_precomputed_rollup=False,
    )


def build_temporal_query(
    intent: TemporalTrendIntent,
    *,
    time_column: str = "timestamp",
) -> OLAPQuery:
    """Build a temporal trend SQL query with partition pruning.

    Args:
        time_column: Column holding the time dimension (e.g. ``created_at``).
    """
    params: list[Any] = []
    optimizations: list[str] = ["predicate_pushdown"]
    uses_partition_pruning = False
    uses_precomputed_rollup = False

    tc = time_column
    time_bucket = intent.granularity or "1d"
    if time_bucket in KNOWN_ROLLUPS:
        uses_precomputed_rollup = True
        optimizations.append("rollup_detection")
        table = f"{intent.entity}_rollup_{time_bucket}"
    else:
        table = intent.entity

    select_clause = f"time_bucket('{time_bucket}', {tc}) AS bucket, AVG({intent.metric}) AS {intent.metric}"

    where_parts: list[str] = []
    if intent.window.start:
        params.append(intent.window.start)
        where_parts.append(f"{tc} >= ${len(params)}")
        uses_partition_pruning = True
    if intent.window.end:
        params.append(intent.window.end)
        where_parts.append(f"{tc} <= ${len(params)}")
        uses_partition_pruning = True
    if intent.window.relative:
        where_parts.append(f"{tc} >= NOW() - INTERVAL '{intent.window.relative}'")
        uses_partition_pruning = True

    if uses_partition_pruning:
        optimizations.append("partition_pruning")

    where_clause = _build_where(intent.filters, params)
    extra_where = " AND ".join(where_parts)
    if where_clause and extra_where:
        where_clause = where_clause + " AND " + extra_where
    elif extra_where:
        where_clause = " WHERE " + extra_where

    sql = (
        f"SELECT {select_clause} FROM {table}{where_clause}"
        f" GROUP BY bucket ORDER BY bucket"
    )

    if intent.max_results:
        sql += f" LIMIT {intent.max_results}"

    return OLAPQuery(
        sql=sql,
        params=params,
        optimizations=optimizations,
        estimated_rows_scanned=50000,
        uses_partition_pruning=uses_partition_pruning,
        uses_precomputed_rollup=uses_precomputed_rollup,
    )
