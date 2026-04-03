"""
Adapter between Provena typed connectors and MCP servers.
Typed connectors call this instead of hitting databases directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from provena.types.errors import MCPTransportError


@dataclass
class MCPServerConfig:
    server_id: str
    server_url: str
    declared_consistency: str | None = None
    declared_precision: str | None = None
    declared_staleness_window_sec: float | None = None


@dataclass
class MCPToolCall:
    tool_name: str
    parameters: dict[str, Any]


@dataclass
class MCPResponse:
    content: Any
    provena_metadata: dict[str, Any] | None = None


class MCPTransport(Protocol):
    """Pluggable transport for MCP communication."""

    async def send(self, server: MCPServerConfig, call: MCPToolCall) -> MCPResponse: ...


class MockMCPTransport:
    """Mock transport for testing."""

    def __init__(self) -> None:
        self._responses: dict[str, MCPResponse] = {}

    def set_response(self, tool_name: str, response: MCPResponse) -> None:
        self._responses[tool_name] = response

    async def send(self, server: MCPServerConfig, call: MCPToolCall) -> MCPResponse:
        return self._responses.get(call.tool_name, MCPResponse(content=[]))


class MCPAdapter:
    def __init__(self, transport: MCPTransport) -> None:
        self._transport = transport
        self._servers: dict[str, MCPServerConfig] = {}

    def register_server(self, config: MCPServerConfig) -> None:
        self._servers[config.server_id] = config

    async def call(self, server_id: str, tool_call: MCPToolCall) -> MCPResponse:
        server = self._servers.get(server_id)
        if server is None:
            raise MCPTransportError(server_id, f"Server not registered: {server_id}")
        return await self._transport.send(server, tool_call)
