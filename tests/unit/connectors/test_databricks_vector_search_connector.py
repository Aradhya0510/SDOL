"""Tests for Databricks Vector Search (document) connector."""

import pytest

from sdol.extensions.databricks.document.vector_search import (
    DatabricksVectorSearchConnector,
)
from sdol.extensions.databricks.document.vector_search_query import (
    DatabricksVSQuery,
    build_vs_similarity_query,
)
from sdol.connectors.executor import MockQueryExecutor
from sdol.types.context import ContextSlotType
from sdol.types.errors import InvalidIntentError
from sdol.types.intent import (
    FilterClause,
    PointLookupIntent,
    SemanticSearchIntent,
)
from sdol.types.provenance import ConsistencyGuarantee


class TestDatabricksVectorSearchConnector:
    def test_handles_semantic_search(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        intent = SemanticSearchIntent(
            id="i-1", query="machine failure symptoms", collection="kb_articles"
        )
        assert connector.can_handle(intent)

    def test_rejects_point_lookup(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        intent = PointLookupIntent(
            id="i-1", entity="customer", identifier={"id": "C-1"}
        )
        assert not connector.can_handle(intent)

    def test_connector_type_is_document(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        assert connector.connector_type == "document"

    def test_default_source_system(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        assert connector.source_system == "databricks.vector_search"

    def test_capabilities_report_similarity(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        caps = connector.get_capabilities()
        assert caps.capabilities.supports_similarity is True
        assert caps.capabilities.supports_full_text_search is True
        assert caps.performance.estimated_latency_ms == 150
        assert caps.performance.max_result_cardinality == 1000

    @pytest.mark.asyncio
    async def test_execute_semantic_search(self) -> None:
        executor = MockQueryExecutor(
            records=[
                {"id": "doc-1", "text": "hydraulic failure guide", "score": 0.92},
                {"id": "doc-2", "text": "motor diagnostics", "score": 0.85},
            ]
        )
        connector = DatabricksVectorSearchConnector(executor=executor)
        intent = SemanticSearchIntent(
            id="i-1",
            query="machine failure symptoms",
            collection="kb_articles",
        )
        result = await connector.execute(intent)
        assert result.slot_type == ContextSlotType.UNSTRUCTURED
        assert len(result.records) == 2
        assert result.provenance.retrieval_method.value == "vector_similarity"
        assert result.provenance.precision.value == "similarity_ranked"
        assert result.provenance.source_system == "databricks.vector_search"

    @pytest.mark.asyncio
    async def test_execute_with_filters(self) -> None:
        executor = MockQueryExecutor(
            records=[{"id": "doc-1", "text": "filtered result", "score": 0.88}]
        )
        connector = DatabricksVectorSearchConnector(executor=executor)
        intent = SemanticSearchIntent(
            id="i-1",
            query="machine failure",
            collection="kb_articles",
            filters=[FilterClause(field="category", operator="eq", value="maintenance")],
        )
        result = await connector.execute(intent)
        assert len(result.records) == 1
        query = executor.last_query
        assert isinstance(query, DatabricksVSQuery)
        assert "metadata_filter_pushdown" in query.optimizations

    @pytest.mark.asyncio
    async def test_rejects_invalid_intent_type(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        intent = PointLookupIntent(
            id="i-1", entity="customer", identifier={"id": "C-1"}
        )
        with pytest.raises(InvalidIntentError):
            await connector.execute(intent)

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        executor = MockQueryExecutor(records=[])
        connector = DatabricksVectorSearchConnector(executor=executor)
        intent = SemanticSearchIntent(
            id="i-1", query="nonexistent topic", collection="kb_articles"
        )
        result = await connector.execute(intent)
        assert result.records == []
        assert result.meta.record_count == 0

    @pytest.mark.asyncio
    async def test_staleness_window(self) -> None:
        executor = MockQueryExecutor(records=[{"id": "doc-1"}])
        connector = DatabricksVectorSearchConnector(executor=executor)
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb"
        )
        result = await connector.execute(intent)
        assert result.provenance.staleness_window_sec == 180.0

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        health = await connector.check_health()
        assert health.status == "healthy"
        assert health.connector_id == "databricks.vector_search"


class TestDatabricksVSQueryBuilder:
    def test_basic_ann_query(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1",
            query="machine failure",
            collection="kb_articles",
            hybrid_weight=1.0,
        )
        query = build_vs_similarity_query(intent)
        assert query.query_type == "ANN"
        assert query.query_text == "machine failure"
        assert query.index_name == "kb_articles"
        assert "ann_search" in query.optimizations

    def test_hybrid_query(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1",
            query="machine failure",
            collection="kb_articles",
            hybrid_weight=0.7,
        )
        query = build_vs_similarity_query(intent)
        assert query.query_type == "HYBRID"
        assert "hybrid_retrieval" in query.optimizations

    def test_default_hybrid_weight_is_hybrid(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb"
        )
        query = build_vs_similarity_query(intent)
        assert query.query_type == "HYBRID"

    def test_unity_catalog_qualification(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb_articles"
        )
        query = build_vs_similarity_query(
            intent, catalog="main", schema="ml"
        )
        assert query.index_name == "main.ml.kb_articles"

    def test_explicit_index_name_overrides_collection(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="ignored"
        )
        query = build_vs_similarity_query(
            intent, index_name="prod.search.my_index"
        )
        assert query.index_name == "prod.search.my_index"

    def test_filter_pushdown(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1",
            query="test",
            collection="kb",
            filters=[
                FilterClause(field="category", operator="eq", value="maintenance"),
                FilterClause(field="priority", operator="gt", value=3),
            ],
        )
        query = build_vs_similarity_query(intent)
        assert "metadata_filter_pushdown" in query.optimizations
        assert query.filters_json is not None
        assert "category = 'maintenance'" in query.filters_json
        assert "priority > 3" in query.filters_json

    def test_in_filter(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1",
            query="test",
            collection="kb",
            filters=[
                FilterClause(field="region", operator="in", value=["us", "eu"]),
            ],
        )
        query = build_vs_similarity_query(intent)
        assert "region IN ('us', 'eu')" in query.filters_json

    def test_num_results_from_intent(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb", max_results=50
        )
        query = build_vs_similarity_query(intent)
        assert query.num_results == 50

    def test_default_num_results(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb"
        )
        query = build_vs_similarity_query(intent)
        assert query.num_results == 20

    def test_score_threshold(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb"
        )
        query = build_vs_similarity_query(intent, score_threshold=0.5)
        assert query.score_threshold == 0.5

    def test_reranking_optimization(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb", rerank=True
        )
        query = build_vs_similarity_query(intent)
        assert "reranking" in query.optimizations

    def test_delta_sync_optimization_always_present(self) -> None:
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb"
        )
        query = build_vs_similarity_query(intent)
        assert "delta_sync_auto_update" in query.optimizations


class TestConsistencyOverrides:
    def test_default_consistency_is_eventual(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        assert connector.default_consistency == ConsistencyGuarantee.EVENTUAL

    def test_override_consistency(self) -> None:
        connector = DatabricksVectorSearchConnector(
            executor=MockQueryExecutor(),
            consistency=ConsistencyGuarantee.STRONG,
        )
        assert connector.default_consistency == ConsistencyGuarantee.STRONG

    def test_default_staleness(self) -> None:
        connector = DatabricksVectorSearchConnector(executor=MockQueryExecutor())
        assert connector.default_staleness_sec == 180.0

    def test_override_staleness(self) -> None:
        connector = DatabricksVectorSearchConnector(
            executor=MockQueryExecutor(),
            staleness_sec=60.0,
        )
        assert connector.default_staleness_sec == 60.0

    @pytest.mark.asyncio
    async def test_provenance_reflects_overrides(self) -> None:
        executor = MockQueryExecutor(records=[{"id": "doc-1"}])
        connector = DatabricksVectorSearchConnector(
            executor=executor,
            consistency=ConsistencyGuarantee.STRONG,
            staleness_sec=60.0,
        )
        intent = SemanticSearchIntent(
            id="i-1", query="test", collection="kb"
        )
        result = await connector.execute(intent)
        assert result.provenance.consistency == ConsistencyGuarantee.STRONG
        assert result.provenance.staleness_window_sec == 60.0
