"""Shared API response schemas."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class ErrorDetail(BaseModel):
    """Structured error payload for non-2xx responses."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error envelope for .NET integration."""

    success: bool = False
    timestamp: str = Field(default_factory=utc_now_iso)
    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check payload."""

    success: bool = True
    status: str = "healthy"
    service: str
    version: str
    timestamp: str = Field(default_factory=utc_now_iso)
    models: Dict[str, Any] = Field(default_factory=dict)


class RootResponse(BaseModel):
    """Root endpoint payload."""

    success: bool = True
    message: str
    service: str
    version: str
    docs_url: str = "/docs"
    health_url: str = "/health"
