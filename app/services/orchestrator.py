"""Coordinates all in-cabin detection services for a single frame."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.model_loader import ModelLoader, model_loader
from app.schemas.detection import AnalyzeFrameResponse, DetectionResult
from app.services.abnormal_movement import AbnormalMovementService
from app.services.drowsiness import DrowsinessService
from app.services.face_detection import FaceDetectionService
from app.services.head_pose import HeadPoseService
from app.services.phone_detection import PhoneDetectionService
from app.services.temporal import SessionState, SessionStore, session_store
from app.services.yawning import YawningService
from app.utils.image_processing import resize_if_needed

logger = get_logger(__name__)


class DetectionOrchestrator:
    """Run independent detectors and unify their outputs."""

    def __init__(
        self,
        loader: Optional[ModelLoader] = None,
        settings: Optional[Settings] = None,
        store: Optional[SessionStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loader = loader or model_loader
        self.store = store or session_store

        registry = self.loader.registry
        self.face_service = FaceDetectionService(registry.face_landmarker)
        self.drowsiness_service = DrowsinessService(self.settings)
        self.yawning_service = YawningService(self.settings)
        self.phone_service = PhoneDetectionService(registry.yolo, self.settings)
        self.head_pose_service = HeadPoseService(self.settings)
        self.abnormal_movement_service = AbnormalMovementService(self.settings)

    def refresh_models(self) -> None:
        """Re-bind model handles after loader.load()."""
        self.face_service.set_model(self.loader.registry.face_landmarker)
        self.phone_service.set_model(self.loader.registry.yolo)

    def analyze_frame(
        self,
        image_bgr: np.ndarray,
        session_id: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> AnalyzeFrameResponse:
        started = time.perf_counter()
        warnings: List[str] = []
        effective_fps = fps or self.settings.default_fps

        session: SessionState = self.store.get_or_create(session_id)
        image, _scale = resize_if_needed(image_bgr)

        # Face + landmarks
        face_output = self.face_service.detect(image)
        if face_output.warning:
            warnings.append(face_output.warning)

        detections: List[DetectionResult] = [
            self.face_service.to_detection_result(face_output)
        ]

        primary = face_output.primary

        detections.append(
            self.drowsiness_service.analyze(primary, session.drowsiness, effective_fps)
        )
        detections.append(
            self.yawning_service.analyze(primary, session.yawning, effective_fps)
        )
        detections.append(
            self.head_pose_service.analyze(
                primary, image.shape, session.distraction, effective_fps
            )
        )
        detections.append(
            self.abnormal_movement_service.analyze(
                primary, session, session.abnormal_movement, effective_fps
            )
        )

        presence, usage, phone_warning = self.phone_service.analyze(
            image, primary, session.phone_usage
        )
        detections.append(presence)
        detections.append(usage)
        if phone_warning:
            warnings.append(phone_warning)

        self.store.touch(session)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return AnalyzeFrameResponse(
            success=True,
            session_id=session.session_id,
            frame_index=session.frame_index,
            detections=detections,
            processing_time_ms=round(elapsed_ms, 2),
            warnings=warnings,
        )


# Shared orchestrator instance (models rebound on startup)
orchestrator = DetectionOrchestrator()
