"""
Example: "Show me customers likely to churn who also have
unresolved support tickets and declining usage trends"

Spans OLAP (churn scores), OLTP (tickets), time-series (usage).
"""

import asyncio

from provena import (
    Provena,
    CapabilityRegistry,
    ContextCompiler,
    GenericOLAPConnector,
    GenericOLTPConnector,
    SemanticRouter,
    TrustScorer,
)
from provena.connectors.executor import MockQueryExecutor
from provena.core.provenance.trust_scorer import TrustScorerConfig
from provena.core.router.cost_estimator import CostEstimator
from provena.core.router.intent_decomposer import IntentDecomposer
from provena.core.router.query_planner import QueryPlanner


async def main() -> None:
    olap_executor = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "churn_probability": 0.89, "region": "west"},
        {"customer_id": "C-2091", "churn_probability": 0.76, "region": "east"},
    ])
    oltp_executor = MockQueryExecutor(records=[
        {"customer_id": "C-1042", "ticket_id": "T-501", "status": "unresolved"},
    ])

    registry = CapabilityRegistry()
    registry.register(GenericOLAPConnector(executor=olap_executor))
    registry.register(GenericOLTPConnector(executor=oltp_executor))

    trust_config = TrustScorerConfig(source_authority_map={
        "snowflake.analytics": 0.95,
        "postgres.production": 0.9,
    })
    compiler = ContextCompiler(TrustScorer(trust_config))
    planner = QueryPlanner(registry, IntentDecomposer(), CostEstimator())
    router = SemanticRouter(planner, compiler, registry)

    provena = Provena(router)

    intent = provena.formulator.composite(
        sub_intents=[
            provena.formulator.aggregate_analysis(
                entity="customer_churn_scores",
                measures=[{"field": "churn_probability", "aggregation": "max"}],
                dimensions=["customer_id", "region"],
                having=[{"field": "churn_probability", "operator": "gt", "value": 0.7}],
            ),
            provena.formulator.point_lookup("support_tickets", {"status": "unresolved"}),
        ],
        fusion_operator="intersect",
        fusion_key="customer_id",
    )

    frame = await provena.query(intent)

    print("=== Context Frame Stats ===")
    print(f"  Elements: {frame.stats.total_elements}")
    print(f"  Avg trust: {frame.stats.avg_trust_score:.2f}")
    print(f"  Slots: {frame.stats.slot_counts}")
    print(f"  Conflicts: {len(frame.conflicts)}")
    print()
    print("=== Epistemic Context ===")
    print(provena.get_epistemic_context())


if __name__ == "__main__":
    asyncio.run(main())
