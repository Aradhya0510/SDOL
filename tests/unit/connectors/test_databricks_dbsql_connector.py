"""Tests for Databricks DBSQL (OLAP) connector."""

import pytest

from sdol.connectors.olap.databricks_dbsql import DatabricksDBSQLConnector
from sdol.connectors.olap.databricks_dbsql_query import (
    build_dbsql_aggregate_query,
    build_dbsql_temporal_query,
    parse_relative_window,
)
from sdol.connectors.executor import MockQueryExecutor
from sdol.types.context import ContextSlotType
from sdol.types.errors import InvalidIntentError
from sdol.types.intent import (
    AggregateAnalysisIntent,
    FilterClause,
    MeasureSpec,
    OrderSpec,
    PointLookupIntent,
    TemporalTrendIntent,
    TimeWindow,
)


class TestDatabricksDBSQLConnector:
    def test_handles_aggregate_analysis(self) -> None:
        connector = DatabricksDBSQLConnector(executor=MockQueryExecutor())
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["region"],
        )
        assert connector.can_handle(intent)

    def test_handles_temporal_trend(self) -> None:
        connector = DatabricksDBSQLConnector(executor=MockQueryExecutor())
        intent = TemporalTrendIntent(
            id="i-1",
            entity="usage",
            metric="api_calls",
            window=TimeWindow(relative="last_90d"),
        )
        assert connector.can_handle(intent)

    def test_rejects_point_lookup(self) -> None:
        connector = DatabricksDBSQLConnector(executor=MockQueryExecutor())
        intent = PointLookupIntent(
            id="i-1", entity="customer", identifier={"id": "C-1"}
        )
        assert not connector.can_handle(intent)

    def test_connector_type_is_olap(self) -> None:
        connector = DatabricksDBSQLConnector(executor=MockQueryExecutor())
        assert connector.connector_type == "olap"

    def test_capabilities_report_photon_scale(self) -> None:
        connector = DatabricksDBSQLConnector(executor=MockQueryExecutor())
        caps = connector.get_capabilities()
        assert caps.performance.max_result_cardinality == 10_000_000
        assert caps.capabilities.supports_aggregation
        assert caps.capabilities.supports_windowing
        assert caps.capabilities.supports_temporal_bucketing

    @pytest.mark.asyncio
    async def test_execute_aggregate(self) -> None:
        executor = MockQueryExecutor(
            records=[{"region": "west", "sum_revenue": 5000}]
        )
        connector = DatabricksDBSQLConnector(executor=executor)
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["region"],
        )
        result = await connector.execute(intent)
        assert result.slot_type == ContextSlotType.STRUCTURED
        assert len(result.records) == 1
        assert result.provenance.precision.value == "exact_aggregate"
        assert result.provenance.source_system == "databricks.sql_warehouse"

    @pytest.mark.asyncio
    async def test_execute_temporal(self) -> None:
        executor = MockQueryExecutor(
            records=[{"bucket": "2024-01-01", "api_calls": 42}]
        )
        connector = DatabricksDBSQLConnector(executor=executor)
        intent = TemporalTrendIntent(
            id="i-1",
            entity="usage",
            metric="api_calls",
            window=TimeWindow(relative="last_90d"),
            granularity="1d",
        )
        result = await connector.execute(intent)
        assert result.slot_type == ContextSlotType.TEMPORAL
        assert len(result.records) == 1

    @pytest.mark.asyncio
    async def test_rejects_invalid_intent_type(self) -> None:
        connector = DatabricksDBSQLConnector(executor=MockQueryExecutor())
        intent = PointLookupIntent(
            id="i-1", entity="customer", identifier={"id": "C-1"}
        )
        with pytest.raises(InvalidIntentError):
            await connector.execute(intent)

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        executor = MockQueryExecutor(records=[])
        connector = DatabricksDBSQLConnector(executor=executor)
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["region"],
        )
        result = await connector.execute(intent)
        assert result.records == []
        assert result.meta.record_count == 0

    @pytest.mark.asyncio
    async def test_entity_keys_detected(self) -> None:
        executor = MockQueryExecutor(
            records=[{"customer_id": "C-1", "revenue": 100}]
        )
        connector = DatabricksDBSQLConnector(executor=executor)
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["customer_id"],
        )
        result = await connector.execute(intent)
        assert result.entity_keys == ["C-1"]

    @pytest.mark.asyncio
    async def test_staleness_window_shorter_than_generic_olap(self) -> None:
        executor = MockQueryExecutor(records=[{"region": "west", "cnt": 10}])
        connector = DatabricksDBSQLConnector(executor=executor)
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="id", aggregation="count")],
            dimensions=["region"],
        )
        result = await connector.execute(intent)
        assert result.provenance.staleness_window_sec == 600.0


class TestDBSQLQueryBuilder:
    def test_aggregate_uses_photon(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["region"],
        )
        query = build_dbsql_aggregate_query(intent)
        assert "photon_acceleration" in query.optimizations
        assert "pushdown_aggregation" in query.optimizations
        assert query.uses_photon is True
        assert "GROUP BY" in query.sql

    def test_aggregate_uses_named_params(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["region"],
            filters=[FilterClause(field="status", operator="eq", value="active")],
        )
        query = build_dbsql_aggregate_query(intent)
        assert ":p0" in query.sql
        assert query.parameters["p0"] == "active"
        assert "delta_data_skipping" in query.optimizations

    def test_aggregate_with_unity_catalog(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["region"],
        )
        query = build_dbsql_aggregate_query(
            intent, catalog="main", schema="analytics"
        )
        assert "main.analytics.orders" in query.sql
        assert query.catalog == "main"
        assert query.schema == "analytics"

    def test_aggregate_percentile_approx(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="latency", aggregation="p95")],
            dimensions=["region"],
        )
        query = build_dbsql_aggregate_query(intent)
        assert "PERCENTILE_APPROX(latency, 0.95)" in query.sql

    def test_aggregate_count_distinct(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="customer_id", aggregation="count_distinct")],
            dimensions=["region"],
        )
        query = build_dbsql_aggregate_query(intent)
        assert "COUNT(DISTINCT customer_id)" in query.sql

    def test_aggregate_with_having_and_order(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["region"],
            having=[FilterClause(field="SUM(revenue)", operator="gt", value=1000)],
            order_by=[OrderSpec(field="SUM(revenue)", direction="desc")],
            max_results=10,
        )
        query = build_dbsql_aggregate_query(intent)
        assert "HAVING" in query.sql
        assert "ORDER BY" in query.sql
        assert "LIMIT 10" in query.sql
        assert "order_pushdown" in query.optimizations

    def test_temporal_uses_date_trunc(self) -> None:
        intent = TemporalTrendIntent(
            id="i-1",
            entity="usage",
            metric="api_calls",
            window=TimeWindow(relative="last_90d"),
            granularity="1d",
        )
        query = build_dbsql_temporal_query(intent)
        assert "DATE_TRUNC('DAY', timestamp)" in query.sql
        assert query.uses_photon is True

    def test_temporal_with_absolute_window(self) -> None:
        intent = TemporalTrendIntent(
            id="i-1",
            entity="usage",
            metric="api_calls",
            window=TimeWindow(start="2024-01-01", end="2024-03-31"),
        )
        query = build_dbsql_temporal_query(intent)
        assert query.uses_delta_data_skipping is True
        assert "delta_data_skipping" in query.optimizations
        assert query.parameters["p0"] == "2024-01-01"
        assert query.parameters["p1"] == "2024-03-31"

    def test_temporal_with_unity_catalog(self) -> None:
        intent = TemporalTrendIntent(
            id="i-1",
            entity="metrics",
            metric="cpu",
            window=TimeWindow(relative="last_7d"),
        )
        query = build_dbsql_temporal_query(
            intent, catalog="prod", schema="telemetry"
        )
        assert "prod.telemetry.metrics" in query.sql

    def test_temporal_with_filters(self) -> None:
        intent = TemporalTrendIntent(
            id="i-1",
            entity="usage",
            metric="api_calls",
            window=TimeWindow(relative="last_30d"),
            filters=[FilterClause(field="service", operator="eq", value="api-gw")],
        )
        query = build_dbsql_temporal_query(intent)
        assert ":p0" in query.sql
        assert query.parameters["p0"] == "api-gw"

    def test_in_filter_generates_multiple_params(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="revenue", aggregation="sum")],
            dimensions=["region"],
            filters=[FilterClause(field="region", operator="in", value=["west", "east"])],
        )
        query = build_dbsql_aggregate_query(intent)
        assert ":p0" in query.sql
        assert ":p1" in query.sql
        assert query.parameters["p0"] == "west"
        assert query.parameters["p1"] == "east"

    # ── time_column support ──

    def test_temporal_custom_time_column(self) -> None:
        intent = TemporalTrendIntent(
            id="i-1",
            entity="sales",
            metric="total_amount",
            window=TimeWindow(relative="last_30d"),
            granularity="1d",
        )
        query = build_dbsql_temporal_query(intent, time_column="order_date")
        assert "DATE_TRUNC('DAY', order_date)" in query.sql
        assert "order_date >= CURRENT_TIMESTAMP()" in query.sql
        assert "timestamp" not in query.sql

    def test_temporal_absolute_window_uses_custom_column(self) -> None:
        intent = TemporalTrendIntent(
            id="i-1",
            entity="revenue",
            metric="total_revenue",
            window=TimeWindow(start="2025-01-01", end="2025-12-31"),
        )
        query = build_dbsql_temporal_query(intent, time_column="report_date")
        assert "report_date >= :p0" in query.sql
        assert "report_date <= :p1" in query.sql


class TestParseRelativeWindow:
    def test_days(self) -> None:
        assert parse_relative_window("last_90d") == "INTERVAL 90 DAY"

    def test_hours(self) -> None:
        assert parse_relative_window("last_24h") == "INTERVAL 24 HOUR"

    def test_weeks(self) -> None:
        assert parse_relative_window("last_4w") == "INTERVAL 4 WEEK"

    def test_months(self) -> None:
        assert parse_relative_window("last_6M") == "INTERVAL 6 MONTH"

    def test_years(self) -> None:
        assert parse_relative_window("last_1y") == "INTERVAL 1 YEAR"

    def test_quarters_convert_to_months(self) -> None:
        assert parse_relative_window("last_2Q") == "INTERVAL 6 MONTH"

    def test_passthrough_for_unknown_format(self) -> None:
        assert parse_relative_window("30 DAY") == "INTERVAL 30 DAY"


class TestTimeColumnMap:
    @pytest.mark.asyncio
    async def test_connector_routes_time_column_per_entity(self) -> None:
        executor = MockQueryExecutor(
            records=[{"bucket": "2025-07-01", "total_amount": 1000}]
        )
        connector = DatabricksDBSQLConnector(
            executor=executor,
            time_column_map={"sales": "order_date", "metrics": "event_ts"},
        )
        intent = TemporalTrendIntent(
            id="i-1",
            entity="sales",
            metric="total_amount",
            window=TimeWindow(relative="last_30d"),
            granularity="1M",
        )
        result = await connector.execute(intent)
        native_sql = executor.last_query.sql
        assert "order_date" in native_sql
        assert "timestamp" not in native_sql
        assert result.slot_type == ContextSlotType.TEMPORAL
