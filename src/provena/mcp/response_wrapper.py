"""Wraps MCP responses with Provena provenance metadata."""

from __future__ import annotations

from datetime import datetime, timezone

from provena.mcp.mcp_adapter import MCPResponse, MCPServerConfig
from provena.types.provenance import (
    ConsistencyGuarantee,
    PrecisionClass,
    ProvenanceEnvelope,
    RetrievalMethod,
)


class ResponseWrapper:
    """Extracts or synthesizes ProvenanceEnvelope from MCP responses."""

    def wrap(
        self,
        response: MCPResponse,
        server_config: MCPServerConfig,
    ) -> ProvenanceEnvelope:
        """
        Build a ProvenanceEnvelope from the MCP response.
        Priority: provena_metadata > server declared defaults > conservative defaults.
        """
        if response.provena_metadata:
            return self._from_provena_metadata(response.provena_metadata, server_config)
        if self._has_declared_defaults(server_config):
            return self._from_server_defaults(server_config)
        return self._conservative_defaults(server_config)

    def _from_provena_metadata(
        self,
        metadata: dict[str, object],
        config: MCPServerConfig,
    ) -> ProvenanceEnvelope:
        return ProvenanceEnvelope(
            source_system=config.server_id,
            retrieval_method=RetrievalMethod(
                str(metadata.get("retrieval_method", "mcp_passthrough"))
            ),
            consistency=ConsistencyGuarantee(
                str(metadata.get("consistency", "best_effort"))
            ),
            precision=PrecisionClass(str(metadata.get("precision", "estimated"))),
            retrieved_at=str(
                metadata.get("retrieved_at", datetime.now(timezone.utc).isoformat())
            ),
            staleness_window_sec=_to_float(metadata.get("staleness_window_sec")),
            execution_ms=_to_float(metadata.get("execution_ms")),
        )

    def _has_declared_defaults(self, config: MCPServerConfig) -> bool:
        return any([
            config.declared_consistency,
            config.declared_precision,
            config.declared_staleness_window_sec is not None,
        ])

    def _from_server_defaults(self, config: MCPServerConfig) -> ProvenanceEnvelope:
        return ProvenanceEnvelope(
            source_system=config.server_id,
            retrieval_method=RetrievalMethod.MCP_PASSTHROUGH,
            consistency=ConsistencyGuarantee(config.declared_consistency or "best_effort"),
            precision=PrecisionClass(config.declared_precision or "estimated"),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            staleness_window_sec=config.declared_staleness_window_sec,
        )

    def _conservative_defaults(self, config: MCPServerConfig) -> ProvenanceEnvelope:
        return ProvenanceEnvelope(
            source_system=config.server_id,
            retrieval_method=RetrievalMethod.MCP_PASSTHROUGH,
            consistency=ConsistencyGuarantee.BEST_EFFORT,
            precision=PrecisionClass.ESTIMATED,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            staleness_window_sec=None,
        )


def _to_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
