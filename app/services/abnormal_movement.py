"""Abnormal in-cabin movement heuristics (non-medical).

This module measures sudden/large landmark displacement between frames.
It does NOT diagnose medical emergencies or physiological conditions.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.core.config import Settings, get_settings
from app.schemas.detection import DetectionResult, DetectionType, Severity
from app.services.face_detection import FaceLandmarks
from app.services.temporal import CounterState, SessionState


class AbnormalMovementService:
    """Detect unusually large frame-to-frame facial/upper-body landmark motion.

    Intended as a modular hook for future passenger/driver activity analysis.
    Results are behavioral heuristics only.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def analyze(
        self,
        face: Optional[FaceLandmarks],
        session: SessionState,
        state: CounterState,
        fps: Optional[float] = None,
    ) -> DetectionResult:
        disclaimer = (
            "Heuristic motion signal only — not a medical diagnosis or emergency detector"
        )

        if face is None:
            state.update(False)
            session.previous_landmarks = None
            return DetectionResult(
                type=DetectionType.ABNORMAL_MOVEMENT,
                detected=False,
                confidence=None,
                severity=Severity.NONE,
                metadata={"reason": "no_face", "disclaimer": disclaimer},
            )

        current = face.landmarks_norm[:, :2]
        previous: Optional[np.ndarray] = session.previous_landmarks
        session.previous_landmarks = current.copy()

        if previous is None or previous.shape != current.shape:
            state.update(False)
            return DetectionResult(
                type=DetectionType.ABNORMAL_MOVEMENT,
                detected=False,
                confidence=None,
                severity=Severity.NONE,
                metadata={
                    "reason": "insufficient_history",
                    "disclaimer": disclaimer,
                    "note": "Requires at least two sequential frames in the same session",
                },
            )

        # Mean landmark displacement in normalized image coordinates
        delta = np.linalg.norm(current - previous, axis=1)
        mean_velocity = float(np.mean(delta))
        # Scale by fps when available so threshold is roughly per-second motion
        if fps and fps > 0:
            mean_velocity_per_sec = mean_velocity * fps
        else:
            mean_velocity_per_sec = mean_velocity * self.settings.default_fps

        # Compare per-frame velocity against configured threshold
        sudden = mean_velocity >= self.settings.movement_velocity_threshold
        state.last_value = mean_velocity
        state.update(sudden)

        detected = state.is_triggered(
            min_frames=self.settings.abnormal_movement_consecutive_frames,
            min_duration_seconds=0.0,
        )

        confidence = float(
            np.clip(
                mean_velocity / max(self.settings.movement_velocity_threshold, 1e-6),
                0.0,
                1.0,
            )
        )

        severity = Severity.NONE
        if detected:
            severity = Severity.MEDIUM
        elif sudden:
            severity = Severity.LOW

        metadata: dict[str, Any] = {
            "mean_landmark_delta": round(mean_velocity, 5),
            "mean_delta_scaled_per_sec": round(mean_velocity_per_sec, 5),
            "threshold": self.settings.movement_velocity_threshold,
            "sudden_this_frame": sudden,
            "consecutive_frames": state.consecutive,
            "min_frames_required": self.settings.abnormal_movement_consecutive_frames,
            "disclaimer": disclaimer,
        }

        return DetectionResult(
            type=DetectionType.ABNORMAL_MOVEMENT,
            detected=detected,
            confidence=confidence,
            severity=severity,
            metadata=metadata,
        )
