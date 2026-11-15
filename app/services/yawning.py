"""Yawning detection via Mouth Aspect Ratio (MAR) with temporal filtering."""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.config import Settings, get_settings
from app.schemas.detection import DetectionResult, DetectionType, Severity
from app.services.face_detection import MOUTH_IDX, FaceLandmarks
from app.services.temporal import CounterState
from app.utils.geometry import mar_from_landmarks


class YawningService:
    """Detect prolonged mouth opening patterns consistent with yawning."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def compute_mar(self, face: FaceLandmarks) -> float:
        # MediaPipe indices: left, right, upper, lower, inner-left, inner-right
        pts = face.points(MOUTH_IDX)
        # Use 6-point MAR: left, right, top1, top2, bottom1, bottom2
        ordered = [pts[0], pts[1], pts[2], pts[4], pts[3], pts[5]]
        return mar_from_landmarks(ordered)

    def analyze(
        self,
        face: Optional[FaceLandmarks],
        state: CounterState,
        fps: Optional[float] = None,
    ) -> DetectionResult:
        if face is None:
            state.update(False)
            return DetectionResult(
                type=DetectionType.YAWNING,
                detected=False,
                confidence=None,
                severity=Severity.NONE,
                metadata={"reason": "no_face", "mouth_status": "unknown"},
            )

        mar = self.compute_mar(face)
        mouth_open = mar >= self.settings.mar_threshold
        state.last_value = mar
        state.update(mouth_open)

        min_frames = self.settings.yawning_consecutive_frames
        if fps and fps > 0:
            frames_for_duration = int(
                np.ceil(self.settings.yawning_min_duration_seconds * fps)
            )
            min_frames = max(min_frames, frames_for_duration)

        detected = state.is_triggered(
            min_frames=min_frames,
            min_duration_seconds=self.settings.yawning_min_duration_seconds,
        )

        if mouth_open:
            # How far above threshold, normalized soft score
            confidence = float(
                np.clip(
                    (mar - self.settings.mar_threshold) / max(self.settings.mar_threshold, 1e-6)
                    + 0.5,
                    0.0,
                    1.0,
                )
            )
        else:
            confidence = float(
                np.clip(mar / max(self.settings.mar_threshold, 1e-6), 0.0, 1.0)
            )

        severity = Severity.NONE
        if detected:
            duration = state.duration_seconds()
            severity = (
                Severity.HIGH
                if duration >= self.settings.yawning_min_duration_seconds * 2
                else Severity.MEDIUM
            )

        return DetectionResult(
            type=DetectionType.YAWNING,
            detected=detected,
            confidence=confidence,
            severity=severity,
            metadata={
                "mar": round(mar, 4),
                "mar_threshold": self.settings.mar_threshold,
                "mouth_status": "open" if mouth_open else "closed",
                "consecutive_open_frames": state.consecutive,
                "duration_seconds": round(state.duration_seconds(), 3),
                "min_frames_required": min_frames,
                "temporal_gate": "requires_consecutive_open_frames",
            },
        )
