from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field, StringConstraints

GdbCommand = Annotated[str, StringConstraints(max_length=4096)]


class AttachRequest(BaseModel):
    """Request body for /attach."""

    pre: Optional[List[GdbCommand]] = Field(default=None, max_length=32)
    pid: int = Field(gt=1)
    after: Optional[List[GdbCommand]] = Field(default=None, max_length=32)
    where: Optional[str] = Field(default=None, max_length=1024)
    session_id: str = Field(max_length=64)


class AttachResponse(BaseModel):
    """Response body for /attach."""

    successful: bool
    attach: Optional[Dict[str, Any]] = None
    result: Dict[str, Any]
