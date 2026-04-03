"""Tests for ResponseWrapper."""

from provena.mcp.mcp_adapter import MCPResponse, MCPServerConfig
from provena.mcp.response_wrapper import ResponseWrapper
from provena.types.provenance import ConsistencyGuarantee, PrecisionClass, RetrievalMethod


class TestResponseWrapper:
    def test_extracts_provena_metadata(self) -> None:
        wrapper = ResponseWrapper()
        response = MCPResponse(
            content=[],
            provena_metadata={
                "retrieval_method": "direct_query",
                "consistency": "strong",
                "precision": "exact",
                "staleness_window_sec": 3600.0,
            },
        )
        config = MCPServerConfig(server_id="s1", server_url="http://localhost")
        envelope = wrapper.wrap(response, config)
        assert envelope.retrieval_method == RetrievalMethod.DIRECT_QUERY
        assert envelope.consistency == ConsistencyGuarantee.STRONG
        assert envelope.precision == PrecisionClass.EXACT

    def test_falls_back_to_server_defaults(self) -> None:
        wrapper = ResponseWrapper()
        response = MCPResponse(content=[])
        config = MCPServerConfig(
            server_id="s1",
            server_url="http://localhost",
            declared_consistency="read_committed",
            declared_precision="exact_aggregate",
            declared_staleness_window_sec=600.0,
        )
        envelope = wrapper.wrap(response, config)
        assert envelope.consistency == ConsistencyGuarantee.READ_COMMITTED
        assert envelope.precision == PrecisionClass.EXACT_AGGREGATE
        assert envelope.staleness_window_sec == 600.0

    def test_uses_conservative_defaults(self) -> None:
        wrapper = ResponseWrapper()
        response = MCPResponse(content=[])
        config = MCPServerConfig(server_id="s1", server_url="http://localhost")
        envelope = wrapper.wrap(response, config)
        assert envelope.consistency == ConsistencyGuarantee.BEST_EFFORT
        assert envelope.precision == PrecisionClass.ESTIMATED
        assert envelope.staleness_window_sec is None
        assert envelope.retrieval_method == RetrievalMethod.MCP_PASSTHROUGH
