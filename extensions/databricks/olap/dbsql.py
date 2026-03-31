"""Databricks SQL (DBSQL) OLAP connector — aggregate analysis and temporal trends via Photon-accelerated SQL warehouses."""

from __future__ import annotations

from typing import Any

from sdol.connectors.executor import QueryExecutor
from sdol.connectors.olap.base import BaseOLAPConnector
from sdol.types.capability import ConnectorPerformance
from sdol.types.errors import InvalidIntentError
from sdol.types.intent import AggregateAnalysisIntent, TemporalTrendIntent
from sdol.types.provenance import ConsistencyGuarantee

from .dbsql_query import (
    DBSQLQuery,
    build_dbsql_aggregate_query,
    build_dbsql_temporal_query,
)


class DatabricksDBSQLConnector(BaseOLAPConnector):
    """OLAP connector backed by a Databricks SQL warehouse.

    Leverages Photon acceleration, Delta Lake data skipping, and Unity Catalog
    three-level namespacing for high-throughput analytical queries.
    """

    def __init__(
        self,
        executor: QueryExecutor,
        connector_id: str = "databricks.dbsql",
        source_system: str = "databricks.sql_warehouse",
        available_entities: list[str] | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        time_column_map: dict[str, str] | None = None,
        entity_key_fields: tuple[str, ...] | None = None,
        consistency: ConsistencyGuarantee | None = None,
        staleness_sec: float | None = None,
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
        self._time_column_map = time_column_map or {}
        self._consistency_override = consistency
        self._staleness_override = staleness_sec

    @property
    def default_staleness_sec(self) -> float:
        if self._staleness_override is not None:
            return self._staleness_override
        return 600.0

    @property
    def default_consistency(self) -> ConsistencyGuarantee:
        if self._consistency_override is not None:
            return self._consistency_override
        return ConsistencyGuarantee.STRONG

    def get_performance(self) -> ConnectorPerformance:
        return ConnectorPerformance(
            estimated_latency_ms=300,
            max_result_cardinality=10_000_000,
        )

    def synthesize_query(self, params: Any) -> DBSQLQuery:
        if isinstance(params, AggregateAnalysisIntent):
            return build_dbsql_aggregate_query(
                params, catalog=self._catalog, schema=self._schema
            )
        if isinstance(params, TemporalTrendIntent):
            tc = self._time_column_map.get(params.entity, "timestamp")
            return build_dbsql_temporal_query(
                params,
                catalog=self._catalog,
                schema=self._schema,
                time_column=tc,
            )
        raise InvalidIntentError(
            "Unexpected intent type in synthesize_query",
            [{"type": type(params).__name__}],
        )
