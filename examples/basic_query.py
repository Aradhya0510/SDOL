"""Simple single-source query example."""

import asyncio

from provena import Provena, CapabilityRegistry, ContextCompiler, GenericOLTPConnector, SemanticRouter, TrustScorer
from provena.connectors.executor import MockQueryExecutor
from provena.core.router.cost_estimator import CostEstimator
from provena.core.router.intent_decomposer import IntentDecomposer
from provena.core.router.query_planner import QueryPlanner


async def main() -> None:
    executor = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "name": "Alice", "email": "alice@example.com"},
    ])

    registry = CapabilityRegistry()
    registry.register(GenericOLTPConnector(executor=executor))

    compiler = ContextCompiler(TrustScorer())
    planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
    router = SemanticRouter(planner, compiler, registry)
    provena = Provena(router)

    intent = provena.formulator.point_lookup(
        "customer", {"customer_id": "C-1042"}, fields=["name", "email"]
    )

    frame = await provena.query(intent)

    print("=== Context Frame ===")
    print(f"  Elements: {frame.stats.total_elements}")
    print(f"  Avg trust: {frame.stats.avg_trust_score:.2f}")
    for slot in frame.slots:
        print(f"  Slot [{slot.type}]: {len(slot.elements)} elements")
        for elem in slot.elements:
            print(f"    - {elem.data} (trust: {elem.trust.composite:.2f})")
    print()
    print("=== Epistemic Context ===")
    print(provena.get_epistemic_context())


if __name__ == "__main__":
    asyncio.run(main())
