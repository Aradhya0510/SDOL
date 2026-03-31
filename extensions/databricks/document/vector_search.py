"""Databricks Vector Search connector — semantic/similarity search via managed vector indices.

Supports both Delta Sync indices (automatic embedding + sync from a source
Delta table) and Direct Vector Access indices (user-managed embeddings),
with optional hybrid keyword+vector retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sdol.connectors.document.base import BaseDocumentConnector
from sdol.connectors.executor import QueryExecutor
from sdol.types.capability import (
    ConnectorCapabilities,
    ConnectorCapability,
    ConnectorPerformance,
)
from sdol.types.connector import ConnectorHealth, ConnectorResult, ConnectorResultMeta
from sdol.types.context import ContextSlotType
from sdol.types.errors import InvalidIntentError
from sdol.types.intent import SemanticSearchIntent
from sdol.types.provenance import (
    ConsistencyGuarantee,
    PrecisionClass,
    ProvenanceEnvelope,
    RetrievalMethod,
)

from .vector_search_query import DatabricksVSQuery, build_vs_similarity_query


class DatabricksVectorSearchConnector(BaseDocumentConnector):
    """Document connector backed by Databricks Vector Search.

    Leverages managed vector indices on Unity Catalog with automatic
    Delta Sync for embedding generation and index maintenance.
    """

    def __init__(
        self,
        executor: QueryExecutor,
        connector_id: str = "databricks.vector_search",
        source_system: str = "databricks.vector_search",
        available_entities: list[str] | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        index_name: str | None = None,
        embedding_model_endpoint: str | None = None,
        score_threshold: float | None = 0.3,
        consistency: ConsistencyGuarantee | None = None,
        staleness_sec: float | None = None,
    ) -> None:
        super().__init__(
            executor=executor,
            connector_id=connector_id,
            source_system=source_system,
            available_entities=available_entities,
        )
        self._catalog = catalog
        self._schema = schema
        self._index_name = index_name
        self._embedding_model_endpoint = embedding_model_endpoint
        self._score_threshold = score_threshold
        self._consistency_override = consistency
        self._staleness_override = staleness_sec

    @property
    def default_staleness_sec(self) -> float:
        if self._staleness_override is not None:
            return self._staleness_override
        return 180.0

    @property
    def default_consistency(self) -> ConsistencyGuarantee:
        if self._consistency_override is not None:
            return self._consistency_override
        return ConsistencyGuarantee.EVENTUAL

    def get_performance(self) -> ConnectorPerformance:
        return ConnectorPerformance(
            estimated_latency_ms=150,
            max_result_cardinality=1000,
        )

    def get_capabilities(self) -> ConnectorCapability:
        return ConnectorCapability(
            connector_id=self._id,
            connector_type="document",
            supported_intent_types=["semantic_search"],
            capabilities=ConnectorCapabilities(
                supports_similarity=True,
                supports_full_text_search=True,
            ),
            performance=self.get_performance(),
            available_entities=self._available_entities,
        )

    def synthesize_query(self, params: Any) -> DatabricksVSQuery:
        if isinstance(params, SemanticSearchIntent):
            return build_vs_similarity_query(
                params,
                catalog=self._catalog,
                schema=self._schema,
                index_name=self._index_name,
                score_threshold=self._score_threshold,
            )
        raise InvalidIntentError(
            "Unexpected intent type in synthesize_query",
            [{"type": type(params).__name__}],
        )

    def normalize_result(
        self,
        raw: Any,
        intent: Any,
        execution_ms: float,
    ) -> ConnectorResult:
        records = raw.get("records", [])
        max_results = getattr(intent, "max_results", None) or 100

        return ConnectorResult(
            records=records,
            provenance=ProvenanceEnvelope(
                source_system=self._source_system,
                retrieval_method=RetrievalMethod.VECTOR_SIMILARITY,
                consistency=self.default_consistency,
                precision=PrecisionClass.SIMILARITY_RANKED,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                staleness_window_sec=self.default_staleness_sec,
                execution_ms=execution_ms,
                result_truncated=len(records) >= max_results,
                total_available=raw.get("meta", {}).get("total_available"),
            ),
            slot_type=ContextSlotType.UNSTRUCTURED,
            entity_keys=None,
            meta=ConnectorResultMeta(
                execution_ms=execution_ms,
                record_count=len(records),
                truncated=len(records) >= max_results,
                native_query=raw.get("meta", {}).get("native_query"),
            ),
        )

    async def check_health(self) -> ConnectorHealth:
        return ConnectorHealth(
            connector_id=self._id,
            status="healthy",
            latency_ms=0.0,
            last_checked=datetime.now(timezone.utc).isoformat(),
        )
