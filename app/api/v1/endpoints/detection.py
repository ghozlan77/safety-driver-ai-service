"""Frame / video analysis endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.detection import (
    AnalyzeFrameResponse,
    AnalyzeVideoResponse,
    DetectionResult,
    DetectionType,
)
from app.schemas.response import ErrorDetail, ErrorResponse
from app.services.orchestrator import DetectionOrchestrator, orchestrator
from app.utils.image_processing import (
    ImageValidationError,
    decode_image_bytes,
    validate_content_type,
    validate_file_size,
)
from app.utils.video_processing import VideoProcessingError, sample_frames_from_path

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["detection"])


def get_orchestrator() -> DetectionOrchestrator:
    return orchestrator


def _error(status_code: int, code: str, message: str, details: Optional[dict] = None) -> HTTPException:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))
    return HTTPException(status_code=status_code, detail=payload.model_dump())


@router.post(
    "/analyze-frame",
    response_model=AnalyzeFrameResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_frame(
    file: UploadFile = File(..., description="In-cabin camera frame (JPEG/PNG/WebP/BMP)"),
    session_id: Optional[str] = Form(
        default=None,
        description="Optional session id for temporal tracking across frames",
    ),
    fps: Optional[float] = Form(
        default=None,
        description="Optional source FPS hint for temporal duration gates",
    ),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    settings: Settings = Depends(get_settings),
    orch: DetectionOrchestrator = Depends(get_orchestrator),
) -> AnalyzeFrameResponse:
    """Analyze a single in-cabin image frame.

    Pass the same `session_id` (form field or `X-Session-Id` header) on
    consecutive frames to enable drowsiness/yawning/distraction temporal logic.
    """
    effective_session = session_id or x_session_id
    content_type = file.content_type

    try:
        validate_content_type(content_type, settings.allowed_content_types)
        data = await file.read()
        validate_file_size(len(data), settings.max_image_size_bytes)
        image = decode_image_bytes(data)
    except ImageValidationError as exc:
        code = "PAYLOAD_TOO_LARGE" if "max size" in str(exc).lower() else "INVALID_IMAGE"
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if code == "PAYLOAD_TOO_LARGE"
            else status.HTTP_400_BAD_REQUEST
        )
        raise _error(status_code, code, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected image validation failure")
        raise _error(
            status.HTTP_400_BAD_REQUEST,
            "INVALID_IMAGE",
            f"Failed to read uploaded image: {exc}",
        ) from exc

    try:
        return orch.analyze_frame(image, session_id=effective_session, fps=fps)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Frame analysis failed")
        raise _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "PROCESSING_ERROR",
            f"Frame analysis failed: {exc}",
        ) from exc


@router.post(
    "/analyze-video",
    response_model=AnalyzeVideoResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_video(
    file: UploadFile = File(..., description="Short in-cabin video clip"),
    session_id: Optional[str] = Form(default=None),
    max_frames: int = Form(default=20, ge=1, le=60),
    settings: Settings = Depends(get_settings),
    orch: DetectionOrchestrator = Depends(get_orchestrator),
) -> AnalyzeVideoResponse:
    """Sample frames from a short video and run sequential analysis.

    Practical for offline clips; real-time streaming should use analyze-frame.
    """
    data = await file.read()
    if len(data) > settings.max_image_size_bytes * 10:
        raise _error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "PAYLOAD_TOO_LARGE",
            "Video exceeds allowed upload size",
        )
    if not data:
        raise _error(status.HTTP_400_BAD_REQUEST, "INVALID_VIDEO", "Empty video upload")

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    import time

    started = time.perf_counter()
    per_frame: List[AnalyzeFrameResponse] = []
    warnings: List[str] = []
    effective_session = session_id

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(data)

        try:
            frames, source_fps = sample_frames_from_path(
                tmp_path,
                max_frames=max_frames,
                target_fps=settings.default_fps,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        for frame in frames:
            result = orch.analyze_frame(
                frame,
                session_id=effective_session,
                fps=source_fps,
            )
            effective_session = result.session_id
            per_frame.append(result)
            warnings.extend(result.warnings)

        summary = _summarize_detections(per_frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return AnalyzeVideoResponse(
            success=True,
            session_id=effective_session,
            frames_analyzed=len(per_frame),
            detections_summary=summary,
            per_frame=per_frame,
            processing_time_ms=round(elapsed_ms, 2),
            warnings=sorted(set(warnings)),
        )
    except VideoProcessingError as exc:
        raise _error(status.HTTP_400_BAD_REQUEST, "INVALID_VIDEO", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Video analysis failed")
        raise _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "PROCESSING_ERROR",
            f"Video analysis failed: {exc}",
        ) from exc


def _summarize_detections(frames: List[AnalyzeFrameResponse]) -> List[DetectionResult]:
    """Aggregate per-type: detected if any frame detected; max confidence seen."""
    by_type: Dict[DetectionType, DetectionResult] = {}
    for frame in frames:
        for det in frame.detections:
            existing = by_type.get(det.type)
            if existing is None:
                by_type[det.type] = det.model_copy(deep=True)
                continue
            if det.detected:
                existing.detected = True
            if det.confidence is not None:
                if existing.confidence is None or det.confidence > existing.confidence:
                    existing.confidence = det.confidence
            # Keep highest severity by enum order
            severity_rank = {
                "none": 0,
                "low": 1,
                "medium": 2,
                "high": 3,
                "unknown": -1,
            }
            if severity_rank.get(det.severity.value, 0) > severity_rank.get(
                existing.severity.value, 0
            ):
                existing.severity = det.severity
            existing.metadata = {
                **existing.metadata,
                "aggregated_from_frames": True,
                "last_metadata": det.metadata,
            }
    return list(by_type.values())
