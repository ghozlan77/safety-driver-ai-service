"""Drowsiness detection via Eye Aspect Ratio (EAR) with temporal filtering."""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.config import Settings, get_settings
from app.schemas.detection import DetectionResult, DetectionType, Severity
from app.services.face_detection import LEFT_EYE_IDX, RIGHT_EYE_IDX, FaceLandmarks
from app.services.temporal import CounterState
from app.utils.geometry import ear_from_landmarks


class DrowsinessService:
    """Detect prolonged eye closure; never alerts on a single closed-eye frame."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def compute_ear(self, face: FaceLandmarks) -> float:
        left = ear_from_landmarks(face.points(LEFT_EYE_IDX))
        right = ear_from_landmarks(face.points(RIGHT_EYE_IDX))
        return float((left + right) / 2.0)

    def analyze(
        self,
        face: Optional[FaceLandmarks],
        state: CounterState,
        fps: Optional[float] = None,
    ) -> DetectionResult:
        if face is None:
            state.update(False)
            return DetectionResult(
                type=DetectionType.DROWSINESS,
                detected=False,
                confidence=None,
                severity=Severity.NONE,
                metadata={"reason": "no_face", "eye_status": "unknown"},
            )

        ear = self.compute_ear(face)
        eyes_closed = ear < self.settings.ear_threshold
        state.last_value = ear
        state.update(eyes_closed)

        # Also enforce frame-count derived from FPS when duration is used
        min_frames = self.settings.drowsiness_consecutive_frames
        if fps and fps > 0:
            frames_for_duration = int(
                np.ceil(self.settings.drowsiness_min_duration_seconds * fps)
            )
            min_frames = max(min_frames, frames_for_duration)

        detected = state.is_triggered(
            min_frames=min_frames,
            min_duration_seconds=self.settings.drowsiness_min_duration_seconds,
        )

        # Confidence from how far EAR is below threshold (clamped), only when closed
        if eyes_closed:
            span = max(self.settings.ear_threshold, 1e-6)
            confidence = float(np.clip((self.settings.ear_threshold - ear) / span, 0.0, 1.0))
        else:
            confidence = float(np.clip(ear / max(self.settings.ear_threshold, 1e-6), 0.0, 1.0))
            # open-eye confidence is "awake confidence"; for alert we report closure confidence
            if not detected:
                pass

        severity = Severity.NONE
        if detected:
            duration = state.duration_seconds()
            if duration >= self.settings.drowsiness_min_duration_seconds * 2:
                severity = Severity.HIGH
            else:
                severity = Severity.MEDIUM

        return DetectionResult(
            type=DetectionType.DROWSINESS,
            detected=detected,
            confidence=confidence if detected or eyes_closed else confidence,
            severity=severity,
            metadata={
                "ear": round(ear, 4),
                "ear_threshold": self.settings.ear_threshold,
                "eye_status": "closed" if eyes_closed else "open",
                "consecutive_closed_frames": state.consecutive,
                "duration_seconds": round(state.duration_seconds(), 3),
                "min_frames_required": min_frames,
                "temporal_gate": "requires_consecutive_closed_frames",
            },
        )
