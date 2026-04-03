"""Example: Integration with MCP server (using mock transport)."""

import asyncio

from provena.mcp.mcp_adapter import MCPAdapter, MCPResponse, MCPServerConfig, MCPToolCall, MockMCPTransport
from provena.mcp.response_wrapper import ResponseWrapper


async def main() -> None:
    transport = MockMCPTransport()
    transport.set_response("query_customers", MCPResponse(
        content=[{"customer_id": "C-1042", "name": "Alice"}],
        provena_metadata={
            "retrieval_method": "direct_query",
            "consistency": "read_committed",
            "precision": "exact",
            "staleness_window_sec": 60.0,
        },
    ))

    adapter = MCPAdapter(transport)
    adapter.register_server(MCPServerConfig(
        server_id="production-db",
        server_url="http://localhost:3000",
        declared_consistency="read_committed",
    ))

    response = await adapter.call(
        "production-db",
        MCPToolCall(tool_name="query_customers", parameters={"id": "C-1042"}),
    )

    wrapper = ResponseWrapper()
    config = MCPServerConfig(server_id="production-db", server_url="http://localhost:3000")
    envelope = wrapper.wrap(response, config)

    print("=== MCP Response ===")
    print(f"  Content: {response.content}")
    print(f"  Provena Metadata: {response.provena_metadata}")
    print()
    print("=== Provenance Envelope ===")
    print(f"  Source: {envelope.source_system}")
    print(f"  Consistency: {envelope.consistency}")
    print(f"  Precision: {envelope.precision}")
    print(f"  Retrieval: {envelope.retrieval_method}")
    print(f"  Staleness: {envelope.staleness_window_sec}s")


if __name__ == "__main__":
    asyncio.run(main())
