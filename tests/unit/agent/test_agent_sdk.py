"""Tests for Agent SDK."""

import pytest

from provena.agent.agent_sdk import Provena
from provena.connectors.capability_registry import CapabilityRegistry
from provena.connectors.executor import MockQueryExecutor
from provena.connectors.oltp.generic import GenericOLTPConnector
from provena.core.context.context_compiler import ContextCompiler
from provena.core.router.cost_estimator import CostEstimator
from provena.core.router.intent_decomposer import IntentDecomposer
from provena.core.router.query_planner import QueryPlanner
from provena.core.router.semantic_router import SemanticRouter


def _setup_provena(records=None) -> Provena:
    registry = CapabilityRegistry()
    registry.register(GenericOLTPConnector(
        executor=MockQueryExecutor(records=records or [{"id": "C-1", "name": "Alice"}])
    ))
    compiler = ContextCompiler()
    planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
    router = SemanticRouter(planner, compiler, registry)
    return Provena(router)


class TestProvena:
    @pytest.mark.asyncio
    async def test_query_returns_context_frame(self) -> None:
        provena = _setup_provena()
        intent = provena.formulator.point_lookup("customer", {"id": "C-1"})
        frame = await provena.query(intent)
        assert frame.stats.total_elements == 1

    @pytest.mark.asyncio
    async def test_epistemic_context_after_query(self) -> None:
        provena = _setup_provena()
        intent = provena.formulator.point_lookup("customer", {"id": "C-1"})
        await provena.query(intent)
        context = provena.get_epistemic_context()
        assert "1 data elements" in context
        assert "1 sources" in context

    @pytest.mark.asyncio
    async def test_reset_clears_tracker(self) -> None:
        provena = _setup_provena()
        intent = provena.formulator.point_lookup("customer", {"id": "C-1"})
        await provena.query(intent)
        provena.reset()
        context = provena.get_epistemic_context()
        assert "No data ingested" in context

    def test_formulator_available(self) -> None:
        provena = _setup_provena()
        assert provena.formulator is not None
