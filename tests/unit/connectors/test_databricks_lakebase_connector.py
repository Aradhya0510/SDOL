"""Tests for Databricks Lakebase (OLTP) connector."""

import pytest

from sdol.extensions.databricks.oltp.lakebase import DatabricksLakebaseConnector
from sdol.extensions.databricks.oltp.lakebase_query import (
    build_lakebase_batch_lookup,
    build_lakebase_point_lookup,
    build_lakebase_simple_aggregate,
)
from sdol.connectors.executor import MockQueryExecutor
from sdol.types.context import ContextSlotType
from sdol.types.errors import InvalidIntentError
from sdol.types.intent import (
    AggregateAnalysisIntent,
    FilterClause,
    MeasureSpec,
    PointLookupIntent,
    SemanticSearchIntent,
)


class TestDatabricksLakebaseConnector:
    def test_handles_point_lookup(self) -> None:
        connector = DatabricksLakebaseConnector(executor=MockQueryExecutor())
        intent = PointLookupIntent(
            id="i-1", entity="customer", identifier={"id": "C-1"}
        )
        assert connector.can_handle(intent)

    def test_handles_aggregate_analysis(self) -> None:
        connector = DatabricksLakebaseConnector(executor=MockQueryExecutor())
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="total", aggregation="count")],
            dimensions=["status"],
        )
        assert connector.can_handle(intent)

    def test_rejects_semantic_search(self) -> None:
        connector = DatabricksLakebaseConnector(executor=MockQueryExecutor())
        intent = SemanticSearchIntent(
            id="i-1", query="find docs", collection="kb"
        )
        assert not connector.can_handle(intent)

    def test_connector_type_is_oltp(self) -> None:
        connector = DatabricksLakebaseConnector(executor=MockQueryExecutor())
        assert connector.connector_type == "oltp"

    def test_capabilities_report_low_latency(self) -> None:
        connector = DatabricksLakebaseConnector(executor=MockQueryExecutor())
        caps = connector.get_capabilities()
        assert caps.performance.estimated_latency_ms == 10
        assert caps.performance.supports_batch_lookup is True

    @pytest.mark.asyncio
    async def test_execute_point_lookup(self) -> None:
        executor = MockQueryExecutor(
            records=[{"id": "C-1", "name": "Alice"}]
        )
        connector = DatabricksLakebaseConnector(executor=executor)
        intent = PointLookupIntent(
            id="i-1",
            entity="customer",
            identifier={"id": "C-1"},
            fields=["name"],
        )
        result = await connector.execute(intent)
        assert result.slot_type == ContextSlotType.STRUCTURED
        assert result.provenance.precision.value == "exact"
        assert result.provenance.source_system == "databricks.lakebase"
        assert len(result.records) == 1

    @pytest.mark.asyncio
    async def test_execute_aggregate(self) -> None:
        executor = MockQueryExecutor(
            records=[{"status": "active", "count_total": 42}]
        )
        connector = DatabricksLakebaseConnector(executor=executor)
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="total", aggregation="count")],
            dimensions=["status"],
        )
        result = await connector.execute(intent)
        assert result.provenance.precision.value == "exact_aggregate"
        assert result.provenance.retrieval_method.value == "computed_aggregate"

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        executor = MockQueryExecutor(records=[])
        connector = DatabricksLakebaseConnector(executor=executor)
        intent = PointLookupIntent(
            id="i-1", entity="customer", identifier={"id": "MISSING"}
        )
        result = await connector.execute(intent)
        assert result.records == []

    @pytest.mark.asyncio
    async def test_rejects_invalid_intent_type(self) -> None:
        connector = DatabricksLakebaseConnector(executor=MockQueryExecutor())
        intent = SemanticSearchIntent(id="i-1", query="test", collection="kb")
        with pytest.raises(InvalidIntentError):
            await connector.execute(intent)

    @pytest.mark.asyncio
    async def test_entity_keys_detected(self) -> None:
        executor = MockQueryExecutor(
            records=[{"customer_id": "C-1", "name": "Alice"}]
        )
        connector = DatabricksLakebaseConnector(executor=executor)
        intent = PointLookupIntent(
            id="i-1",
            entity="customer",
            identifier={"customer_id": "C-1"},
        )
        result = await connector.execute(intent)
        assert result.entity_keys == ["C-1"]

    @pytest.mark.asyncio
    async def test_staleness_window_very_short(self) -> None:
        executor = MockQueryExecutor(records=[{"id": "C-1"}])
        connector = DatabricksLakebaseConnector(executor=executor)
        intent = PointLookupIntent(
            id="i-1", entity="customer", identifier={"id": "C-1"}
        )
        result = await connector.execute(intent)
        assert result.provenance.staleness_window_sec == 30.0


class TestLakebaseQueryBuilder:
    def test_point_lookup_uses_row_index(self) -> None:
        intent = PointLookupIntent(
            id="i-1",
            entity="customer",
            identifier={"id": "C-1"},
            fields=["name", "email"],
        )
        query = build_lakebase_point_lookup(intent)
        assert "lakebase_row_index" in query.optimizations
        assert "column_pruning" in query.optimizations
        assert query.uses_row_index is True
        assert query.is_batch is False
        assert "name, email" in query.sql

    def test_point_lookup_uses_named_params(self) -> None:
        intent = PointLookupIntent(
            id="i-1",
            entity="customer",
            identifier={"id": "C-1"},
        )
        query = build_lakebase_point_lookup(intent)
        assert ":p0" in query.sql
        assert query.parameters["p0"] == "C-1"

    def test_point_lookup_with_unity_catalog(self) -> None:
        intent = PointLookupIntent(
            id="i-1",
            entity="customer",
            identifier={"id": "C-1"},
        )
        query = build_lakebase_point_lookup(
            intent, catalog="main", schema="production"
        )
        assert "main.production.customer" in query.sql
        assert query.catalog == "main"
        assert query.schema == "production"

    def test_batch_lookup(self) -> None:
        query = build_lakebase_batch_lookup(
            "customer", "id", ["C-1", "C-2", "C-3"], fields=["name"]
        )
        assert "batch_lookup" in query.optimizations
        assert query.is_batch is True
        assert query.parameters["p0"] == "C-1"
        assert query.parameters["p1"] == "C-2"
        assert query.parameters["p2"] == "C-3"
        assert "name" in query.sql

    def test_batch_lookup_with_unity_catalog(self) -> None:
        query = build_lakebase_batch_lookup(
            "customer", "id", ["C-1"],
            catalog="main", schema="prod",
        )
        assert "main.prod.customer" in query.sql

    def test_simple_aggregate(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="total", aggregation="count")],
            dimensions=["status"],
        )
        query = build_lakebase_simple_aggregate(intent)
        assert "parameterized_query" in query.optimizations
        assert "GROUP BY" in query.sql
        assert query.uses_row_index is False

    def test_simple_aggregate_with_filters(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="total", aggregation="sum")],
            dimensions=["region"],
            filters=[FilterClause(field="status", operator="eq", value="active")],
        )
        query = build_lakebase_simple_aggregate(intent)
        assert ":p0" in query.sql
        assert query.parameters["p0"] == "active"
        assert "filter_pushdown" in query.optimizations

    def test_simple_aggregate_with_unity_catalog(self) -> None:
        intent = AggregateAnalysisIntent(
            id="i-1",
            entity="orders",
            measures=[MeasureSpec(field="total", aggregation="count")],
            dimensions=["status"],
        )
        query = build_lakebase_simple_aggregate(
            intent, catalog="main", schema="sales"
        )
        assert "main.sales.orders" in query.sql

    def test_point_lookup_with_limit(self) -> None:
        intent = PointLookupIntent(
            id="i-1",
            entity="customer",
            identifier={"id": "C-1"},
            max_results=5,
        )
        query = build_lakebase_point_lookup(intent)
        assert "LIMIT 5" in query.sql


class TestLakebaseEntityKeyFields:
    @pytest.mark.asyncio
    async def test_custom_entity_key_fields(self) -> None:
        executor = MockQueryExecutor(
            records=[{"machine_id": "EXC-0342", "status": "online"}]
        )
        connector = DatabricksLakebaseConnector(
            executor=executor,
            entity_key_fields=("machine_id",),
        )
        intent = PointLookupIntent(
            id="i-1",
            entity="fleet_machines",
            identifier={"machine_id": "EXC-0342"},
        )
        result = await connector.execute(intent)
        assert result.entity_keys == ["EXC-0342"]

    @pytest.mark.asyncio
    async def test_default_key_fields_backward_compat(self) -> None:
        executor = MockQueryExecutor(
            records=[{"customer_id": "C-1", "name": "Alice"}]
        )
        connector = DatabricksLakebaseConnector(executor=executor)
        intent = PointLookupIntent(
            id="i-1",
            entity="customer",
            identifier={"customer_id": "C-1"},
        )
        result = await connector.execute(intent)
        assert result.entity_keys == ["C-1"]
