"""
Error hierarchy for Provena.
Every public method that can fail throws a typed ProvenaError subclass.
Errors always carry context for debugging.
"""

from __future__ import annotations

from provena.types._compat import StrEnum
from typing import Any


class ProvenaErrorCode(StrEnum):
    INVALID_INTENT = "INVALID_INTENT"
    NO_CAPABLE_CONNECTOR = "NO_CAPABLE_CONNECTOR"
    CONNECTOR_TIMEOUT = "CONNECTOR_TIMEOUT"
    CONNECTOR_ERROR = "CONNECTOR_ERROR"
    DECOMPOSITION_FAILED = "DECOMPOSITION_FAILED"
    CONFLICT_UNRESOLVABLE = "CONFLICT_UNRESOLVABLE"
    MCP_TRANSPORT_ERROR = "MCP_TRANSPORT_ERROR"
    TRUST_BELOW_THRESHOLD = "TRUST_BELOW_THRESHOLD"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class ProvenaError(Exception):
    """Base error class for all Provena errors."""

    def __init__(
        self,
        message: str,
        code: ProvenaErrorCode,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}


class InvalidIntentError(ProvenaError):
    def __init__(self, message: str, validation_errors: list[Any]) -> None:
        super().__init__(
            message,
            ProvenaErrorCode.INVALID_INTENT,
            {"validation_errors": validation_errors},
        )
        self.validation_errors = validation_errors


class NoCapableConnectorError(ProvenaError):
    def __init__(self, intent_type: str) -> None:
        super().__init__(
            f"No registered connector can handle intent type: {intent_type}",
            ProvenaErrorCode.NO_CAPABLE_CONNECTOR,
            {"intent_type": intent_type},
        )


class ConnectorTimeoutError(ProvenaError):
    def __init__(self, connector_id: str, budget_ms: int, actual_ms: float) -> None:
        super().__init__(
            f"Connector {connector_id} timed out: {actual_ms:.0f}ms > {budget_ms}ms budget",
            ProvenaErrorCode.CONNECTOR_TIMEOUT,
            {"connector_id": connector_id, "budget_ms": budget_ms, "actual_ms": actual_ms},
        )


class MCPTransportError(ProvenaError):
    def __init__(self, server_id: str, detail: str) -> None:
        super().__init__(
            f"MCP transport error for server {server_id}: {detail}",
            ProvenaErrorCode.MCP_TRANSPORT_ERROR,
            {"server_id": server_id, "detail": detail},
        )
