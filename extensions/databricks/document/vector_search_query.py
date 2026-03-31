"""Databricks Vector Search query builder.

Translates SemanticSearchIntent into the native request shape expected by
the Databricks Vector Search ``query_index`` API, supporting both
approximate nearest-neighbor (ANN) and hybrid keyword+vector retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sdol.types.intent import FilterClause, SemanticSearchIntent


@dataclass
class DatabricksVSQuery:
    """Native query representation for Databricks Vector Search."""

    index_name: str
    query_text: str
    columns: list[str] | None
    filters: dict[str, Any]
    filters_json: str | None
    num_results: int
    query_type: str
    score_threshold: float | None
    optimizations: list[str] = field(default_factory=list)


def _build_filter_string(filters: list[FilterClause]) -> str:
    """Convert SDOL FilterClauses to a Databricks VS SQL-style filter string.

    Databricks VS accepts filter expressions as SQL WHERE-clause fragments,
    e.g. ``"category = 'electronics' AND price < 100"``.
    """
    parts: list[str] = []
    for f in filters:
        if f.operator == "eq":
            parts.append(f"{f.field} = '{f.value}'" if isinstance(f.value, str)
                         else f"{f.field} = {f.value}")
        elif f.operator == "neq":
            parts.append(f"{f.field} != '{f.value}'" if isinstance(f.value, str)
                         else f"{f.field} != {f.value}")
        elif f.operator == "gt":
            parts.append(f"{f.field} > {f.value}")
        elif f.operator == "gte":
            parts.append(f"{f.field} >= {f.value}")
        elif f.operator == "lt":
            parts.append(f"{f.field} < {f.value}")
        elif f.operator == "lte":
            parts.append(f"{f.field} <= {f.value}")
        elif f.operator == "in":
            vals = ", ".join(
                f"'{v}'" if isinstance(v, str) else str(v) for v in f.value
            )
            parts.append(f"{f.field} IN ({vals})")
        elif f.operator == "contains":
            parts.append(f"{f.field} LIKE '%{f.value}%'")
        elif f.operator == "exists":
            parts.append(f"{f.field} IS NOT NULL")
        else:
            parts.append(f"{f.field} = '{f.value}'" if isinstance(f.value, str)
                         else f"{f.field} = {f.value}")
    return " AND ".join(parts)


def _qualify_index(
    collection: str,
    catalog: str | None,
    schema: str | None,
) -> str:
    """Build fully qualified index name ``catalog.schema.index``."""
    if catalog and schema:
        return f"{catalog}.{schema}.{collection}"
    if schema:
        return f"{schema}.{collection}"
    return collection


def build_vs_similarity_query(
    intent: SemanticSearchIntent,
    *,
    catalog: str | None = None,
    schema: str | None = None,
    index_name: str | None = None,
    score_threshold: float | None = 0.3,
) -> DatabricksVSQuery:
    """Build a Databricks Vector Search query from a SemanticSearchIntent.

    Args:
        intent: The semantic search intent to translate.
        catalog: Unity Catalog catalog name (optional).
        schema: Unity Catalog schema name (optional).
        index_name: Explicit index name override (takes precedence over
            ``intent.collection``).
        score_threshold: Minimum similarity score for results.  Defaults to
            0.3 to filter low-quality matches.
    """
    optimizations: list[str] = []

    resolved_index = index_name or _qualify_index(intent.collection, catalog, schema)

    hybrid_weight = intent.hybrid_weight if intent.hybrid_weight is not None else 0.7
    if hybrid_weight < 1.0 and hybrid_weight > 0.0:
        query_type = "HYBRID"
        optimizations.append("hybrid_retrieval")
    else:
        query_type = "ANN"
        optimizations.append("ann_search")

    filters: dict[str, Any] = {}
    filters_json: str | None = None
    if intent.filters:
        for f in intent.filters:
            filters[f.field] = {"operator": f.operator, "value": f.value}
        filters_json = _build_filter_string(intent.filters)
        optimizations.append("metadata_filter_pushdown")

    num_results = intent.max_results or 20

    if intent.rerank:
        optimizations.append("reranking")

    optimizations.append("delta_sync_auto_update")

    return DatabricksVSQuery(
        index_name=resolved_index,
        query_text=intent.query,
        columns=None,
        filters=filters,
        filters_json=filters_json,
        num_results=num_results,
        query_type=query_type,
        score_threshold=score_threshold,
        optimizations=optimizations,
    )
