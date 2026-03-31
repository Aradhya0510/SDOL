"""Lakebase query builder for Databricks OLTP workloads.

Lakebase provides low-latency row-level serving on top of Delta Lake tables,
optimized for point lookups and simple aggregations with automatic indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sdol.connectors.sql_utils import OPERATOR_MAP, qualify_table
from sdol.types.intent import (
    AggregateAnalysisIntent,
    FilterClause,
    PointLookupIntent,
)


@dataclass
class LakebaseQuery:
    """Native query representation for Databricks Lakebase."""

    sql: str
    parameters: dict[str, Any]
    optimizations: list[str]
    is_batch: bool
    uses_row_index: bool
    catalog: str | None
    schema: str | None


def build_lakebase_point_lookup(
    intent: PointLookupIntent,
    *,
    catalog: str | None = None,
    schema: str | None = None,
) -> LakebaseQuery:
    """Build a point lookup query optimized for Lakebase row-level access."""
    params: dict[str, Any] = {}
    counter = 0
    optimizations: list[str] = ["lakebase_row_index", "parameterized_query"]

    if intent.fields:
        select_clause = ", ".join(intent.fields)
        optimizations.append("column_pruning")
    else:
        select_clause = "*"

    where_parts: list[str] = []
    for key, value in intent.identifier.items():
        pname = f"p{counter}"
        params[pname] = value
        counter += 1
        where_parts.append(f"{key} = :{pname}")

    where_clause = " AND ".join(where_parts)
    table = qualify_table(intent.entity, catalog, schema)
    sql = f"SELECT {select_clause} FROM {table} WHERE {where_clause}"

    if intent.max_results:
        sql += f" LIMIT {intent.max_results}"

    return LakebaseQuery(
        sql=sql,
        parameters=params,
        optimizations=optimizations,
        is_batch=False,
        uses_row_index=True,
        catalog=catalog,
        schema=schema,
    )


def build_lakebase_batch_lookup(
    entity: str,
    id_field: str,
    ids: list[str | int],
    fields: list[str] | None = None,
    *,
    catalog: str | None = None,
    schema: str | None = None,
) -> LakebaseQuery:
    """Convert multiple point lookups into a single IN query via Lakebase."""
    params: dict[str, Any] = {}
    optimizations = ["lakebase_row_index", "parameterized_query", "batch_lookup"]

    select_clause = ", ".join(fields) if fields else "*"
    placeholders: list[str] = []
    for i, id_val in enumerate(ids):
        pname = f"p{i}"
        params[pname] = id_val
        placeholders.append(f":{pname}")

    table = qualify_table(entity, catalog, schema)
    sql = f"SELECT {select_clause} FROM {table} WHERE {id_field} IN ({', '.join(placeholders)})"

    return LakebaseQuery(
        sql=sql,
        parameters=params,
        optimizations=optimizations,
        is_batch=True,
        uses_row_index=True,
        catalog=catalog,
        schema=schema,
    )


def build_lakebase_simple_aggregate(
    intent: AggregateAnalysisIntent,
    *,
    catalog: str | None = None,
    schema: str | None = None,
) -> LakebaseQuery:
    """Build a simple aggregate query for Lakebase-backed tables."""
    params: dict[str, Any] = {}
    counter = 0
    optimizations: list[str] = ["parameterized_query"]

    agg_parts: list[str] = []
    for m in intent.measures:
        alias = m.alias or f"{m.aggregation}_{m.field}"
        agg_parts.append(f"{m.aggregation.upper()}({m.field}) AS {alias}")

    select_clause = ", ".join(list(intent.dimensions) + agg_parts)

    where_clause = ""
    if intent.filters:
        where_parts: list[str] = []
        for f in intent.filters:
            op = OPERATOR_MAP.get(f.operator, "=")
            pname = f"p{counter}"
            params[pname] = f.value
            counter += 1
            where_parts.append(f"{f.field} {op} :{pname}")
        where_clause = " WHERE " + " AND ".join(where_parts)
        optimizations.append("filter_pushdown")

    group_by = ", ".join(intent.dimensions)
    table = qualify_table(intent.entity, catalog, schema)

    sql = f"SELECT {select_clause} FROM {table}{where_clause} GROUP BY {group_by}"

    if intent.max_results:
        sql += f" LIMIT {intent.max_results}"

    return LakebaseQuery(
        sql=sql,
        parameters=params,
        optimizations=optimizations,
        is_batch=False,
        uses_row_index=False,
        catalog=catalog,
        schema=schema,
    )
