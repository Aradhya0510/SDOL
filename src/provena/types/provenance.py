"""Provenance and trust types — the epistemic foundation of Provena."""

from __future__ import annotations

from provena.types._compat import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ConsistencyGuarantee(StrEnum):
    STRONG = "strong"
    READ_COMMITTED = "read_committed"
    EVENTUAL = "eventual"
    BEST_EFFORT = "best_effort"


class PrecisionClass(StrEnum):
    EXACT = "exact"
    EXACT_AGGREGATE = "exact_aggregate"
    ESTIMATED = "estimated"
    PREDICTED = "predicted"
    HEURISTIC = "heuristic"
    SIMILARITY_RANKED = "similarity_ranked"
    LOGICALLY_ENTAILED = "logically_entailed"


class RetrievalMethod(StrEnum):
    DIRECT_QUERY = "direct_query"
    CACHE_HIT = "cache_hit"
    COMPUTED_AGGREGATE = "computed_aggregate"
    ML_PREDICTION = "ml_prediction"
    VECTOR_SIMILARITY = "vector_similarity"
    GRAPH_TRAVERSAL = "graph_traversal"
    INFERENCE_ENGINE = "inference_engine"
    MCP_PASSTHROUGH = "mcp_passthrough"


class ProvenanceEnvelope(BaseModel):
    """Metadata attached to every data element entering the context."""

    source_system: str = Field(min_length=1)
    retrieval_method: RetrievalMethod
    consistency: ConsistencyGuarantee
    precision: PrecisionClass
    retrieved_at: str
    staleness_window_sec: float | None = None
    execution_ms: float | None = None
    result_truncated: bool | None = None
    total_available: int | None = None


class TrustDimensions(BaseModel):
    source_authority: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    precision_score: float = Field(ge=0.0, le=1.0)


class TrustScore(BaseModel):
    composite: float = Field(ge=0.0, le=1.0)
    dimensions: TrustDimensions
    label: Literal["high", "medium", "low", "uncertain"]
