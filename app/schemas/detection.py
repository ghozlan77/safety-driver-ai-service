"""Detection request/response schemas for the AI inference API."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.response import utc_now_iso


class DetectionType(str, Enum):
    """Supported detection categories."""

    FACE = "face"
    DROWSINESS = "drowsiness"
    YAWNING = "yawning"
    PHONE_PRESENCE = "phone_presence"
    PHONE_USAGE = "phone_usage"
    DISTRACTION = "distraction"
    ABNORMAL_MOVEMENT = "abnormal_movement"


class Severity(str, Enum):
    """Relative severity for detections that are positive."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class DetectionResult(BaseModel):
    """Single detection result entry.

    confidence:
      - float in [0, 1] when computed from model/metrics
      - null when confidence is unavailable (never fabricated)
    """

    type: DetectionType
    detected: bool
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Actual confidence from processing, or null if unavailable",
    )
    severity: Severity = Severity.NONE
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeFrameResponse(BaseModel):
    """Unified response for frame analysis."""

    success: bool = True
    timestamp: str = Field(default_factory=utc_now_iso)
    session_id: Optional[str] = None
    frame_index: Optional[int] = None
    detections: List[DetectionResult] = Field(default_factory=list)
    processing_time_ms: float
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues (e.g. missing optional model weights)",
    )


class AnalyzeVideoResponse(BaseModel):
    """Response for short video analysis (sampled frames)."""

    success: bool = True
    timestamp: str = Field(default_factory=utc_now_iso)
    session_id: Optional[str] = None
    frames_analyzed: int
    detections_summary: List[DetectionResult] = Field(default_factory=list)
    per_frame: List[AnalyzeFrameResponse] = Field(default_factory=list)
    processing_time_ms: float
    warnings: List[str] = Field(default_factory=list)
