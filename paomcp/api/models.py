"""Shared MCP tool response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolError(BaseModel):
    """Structured information about a failed tool operation."""

    code: str
    message: str
    type: Optional[str] = None
    details: Optional[Any] = None


class ToolMeta(BaseModel):
    """Metadata recorded for every tool invocation."""

    duration_ms: int = 0
    operation_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    idempotency_replayed: bool = False


class ToolResponse(BaseModel):
    """Versioned response envelope returned by every MCP tool."""

    api_version: str = "1.0"
    success: bool
    tool: str
    session_id: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[ToolError] = None
    warnings: List[str] = Field(default_factory=list)
    meta: ToolMeta = Field(default_factory=ToolMeta)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable response without truncating payload data."""
        return self.model_dump(mode="json")
