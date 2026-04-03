"""End-to-end integration tests."""

import pytest

from provena.agent.agent_sdk import Provena
from provena.connectors.capability_registry import CapabilityRegistry
from provena.connectors.executor import MockQueryExecutor
from provena.connectors.olap.generic import GenericOLAPConnector
from provena.connectors.oltp.generic import GenericOLTPConnector
from provena.core.context.context_compiler import ContextCompiler
from provena.core.provenance.trust_scorer import TrustScorer, TrustScorerConfig
from provena.core.router.cost_estimator import CostEstimator
from provena.core.router.intent_decomposer import IntentDecomposer
from provena.core.router.query_planner import QueryPlanner
from provena.core.router.semantic_router import SemanticRouter


def _setup_e2e(
    olap_records=None,
    oltp_records=None,
) -> Provena:
    trust_config = TrustScorerConfig(source_authority_map={
        "snowflake.analytics": 0.95,
        "postgres.production": 0.9,
    })
    registry = CapabilityRegistry()
    registry.register(GenericOLAPConnector(
        executor=MockQueryExecutor(records=olap_records or []),
    ))
    registry.register(GenericOLTPConnector(
        executor=MockQueryExecutor(records=oltp_records or []),
    ))
    compiler = ContextCompiler(TrustScorer(trust_config))
    planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
    router = SemanticRouter(planner, compiler, registry)
    return Provena(router)


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_composite_intent_cross_paradigm(self) -> None:
        provena = _setup_e2e(
            olap_records=[{"customer_id": "C-1042", "churn_prob": 0.89, "region": "west"}],
            oltp_records=[{"customer_id": "C-1042", "ticket_id": "T-501", "status": "open"}],
        )
        intent = provena.formulator.composite(
            sub_intents=[
                provena.formulator.aggregate_analysis(
                    "churn_scores",
                    measures=[{"field": "churn_prob", "aggregation": "max"}],
                    dimensions=["customer_id"],
                ),
                provena.formulator.point_lookup("tickets", {"status": "open"}),
            ],
            fusion_operator="intersect",
            fusion_key="customer_id",
        )
        frame = await provena.query(intent)
        assert frame.stats.total_elements == 2
        assert len(frame.slots) >= 1

    @pytest.mark.asyncio
    async def test_provenance_on_all_elements(self) -> None:
        provena = _setup_e2e(oltp_records=[{"id": "C-1", "name": "Alice"}])
        intent = provena.formulator.point_lookup("customer", {"id": "C-1"})
        frame = await provena.query(intent)
        for slot in frame.slots:
            for elem in slot.elements:
                assert elem.provenance is not None
                assert elem.trust is not None
                assert elem.trust.composite > 0

    @pytest.mark.asyncio
    async def test_epistemic_tracker_accuracy(self) -> None:
        provena = _setup_e2e(
            oltp_records=[{"id": "C-1"}],
            olap_records=[{"region": "west", "total": 100}],
        )
        i1 = provena.formulator.point_lookup("customer", {"id": "C-1"})
        await provena.query(i1)
        i2 = provena.formulator.aggregate_analysis(
            "orders",
            measures=[{"field": "total", "aggregation": "sum"}],
            dimensions=["region"],
        )
        await provena.query(i2)
        context = provena.get_epistemic_context()
        assert "2 data elements" in context
        assert "sources" in context

    @pytest.mark.asyncio
    async def test_conflict_detection_with_disagreeing_sources(self) -> None:
        provena = _setup_e2e(
            olap_records=[{"customer_id": "C-1", "revenue": 1000}],
            oltp_records=[{"customer_id": "C-1", "revenue": 999}],
        )
        intent = provena.formulator.composite(
            sub_intents=[
                provena.formulator.aggregate_analysis(
                    "sales",
                    measures=[{"field": "revenue", "aggregation": "sum"}],
                    dimensions=["customer_id"],
                ),
                provena.formulator.point_lookup("sales", {"customer_id": "C-1"}),
            ],
            fusion_operator="intersect",
            fusion_key="customer_id",
        )
        frame = await provena.query(intent)
        assert frame.stats.total_elements == 2

    @pytest.mark.asyncio
    async def test_partial_results_on_error(self) -> None:
        """When one connector's records are empty, the other still works."""
        provena = _setup_e2e(
            olap_records=[],
            oltp_records=[{"id": "C-1"}],
        )
        intent = provena.formulator.composite(
            sub_intents=[
                provena.formulator.aggregate_analysis(
                    "orders",
                    measures=[{"field": "total", "aggregation": "sum"}],
                    dimensions=["region"],
                ),
                provena.formulator.point_lookup("customer", {"id": "C-1"}),
            ],
            fusion_operator="union",
        )
        frame = await provena.query(intent)
        assert frame.stats.total_elements >= 1
