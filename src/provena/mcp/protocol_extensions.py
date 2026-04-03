"""Provena metadata envelope for MCP protocol extensions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ProvenaMetadataEnvelope(BaseModel):
    """Extended metadata that Provena-aware MCP servers can include."""

    retrieval_method: str | None = None
    consistency: str | None = None
    precision: str | None = None
    staleness_window_sec: float | None = None
    execution_ms: float | None = None
    retrieved_at: str | None = None
    extra: dict[str, Any] | None = None
