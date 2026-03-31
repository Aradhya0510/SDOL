"""Databricks Lakebase OLTP connector — low-latency point lookups and simple aggregates on Delta Lake tables."""

from __future__ import annotations

from typing import Any

from sdol.connectors.executor import QueryExecutor
from sdol.connectors.oltp.base import BaseOLTPConnector
from sdol.types.capability import ConnectorPerformance
from sdol.types.errors import InvalidIntentError
from sdol.types.intent import AggregateAnalysisIntent, PointLookupIntent
from sdol.types.provenance import ConsistencyGuarantee

from .lakebase_query import (
    LakebaseQuery,
    build_lakebase_point_lookup,
    build_lakebase_simple_aggregate,
)


class DatabricksLakebaseConnector(BaseOLTPConnector):
    """OLTP connector backed by Databricks Lakebase.

    Lakebase provides sub-10ms row-level serving on Delta Lake tables with
    automatic indexing, Unity Catalog governance, and seamless lakehouse
    integration — no data duplication or ETL pipelines needed.
    """

    def __init__(
        self,
        executor: QueryExecutor,
        connector_id: str = "databricks.lakebase",
        source_system: str = "databricks.lakebase",
        available_entities: list[str] | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        entity_key_fields: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            executor=executor,
            connector_id=connector_id,
            source_system=source_system,
            available_entities=available_entities,
            entity_key_fields=entity_key_fields,
        )
        self._catalog = catalog
        self._schema = schema

    @property
    def default_staleness_sec(self) -> float:
        return 30.0

    @property
    def default_consistency(self) -> ConsistencyGuarantee:
        return ConsistencyGuarantee.READ_COMMITTED

    def get_performance(self) -> ConnectorPerformance:
        return ConnectorPerformance(
            estimated_latency_ms=10,
            max_result_cardinality=10_000,
            supports_batch_lookup=True,
        )

    def synthesize_query(self, params: Any) -> LakebaseQuery:
        if isinstance(params, PointLookupIntent):
            return build_lakebase_point_lookup(
                params, catalog=self._catalog, schema=self._schema
            )
        if isinstance(params, AggregateAnalysisIntent):
            return build_lakebase_simple_aggregate(
                params, catalog=self._catalog, schema=self._schema
            )
        raise InvalidIntentError(
            "Unexpected intent type in synthesize_query",
            [{"type": type(params).__name__}],
        )
