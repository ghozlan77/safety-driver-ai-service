"""Head pose estimation and distraction detection (look-away over time)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.schemas.detection import DetectionResult, DetectionType, Severity
from app.services.face_detection import HEAD_POSE_IDX, FaceLandmarks
from app.services.temporal import CounterState


# Canonical 3D face model points (approximate, in mm) matching HEAD_POSE_IDX order:
# nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


class HeadPoseService:
    """Estimate yaw/pitch/roll and flag prolonged look-away as distraction."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def estimate(
        self,
        face: FaceLandmarks,
        image_shape: Tuple[int, ...],
    ) -> Optional[Dict[str, float]]:
        # Prefer MediaPipe facial transformation matrix when available
        if face.transformation_matrix is not None:
            try:
                return self._angles_from_matrix(face.transformation_matrix)
            except Exception:  # noqa: BLE001
                pass

        height, width = image_shape[:2]
        image_points = np.array(face.points(HEAD_POSE_IDX), dtype=np.float64)

        focal = float(width)
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal, 0, center[0]],
                [0, focal, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, _tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return self._angles_from_matrix(rotation_matrix)

    @staticmethod
    def _angles_from_matrix(matrix: np.ndarray) -> Dict[str, float]:
        """Extract yaw/pitch/roll degrees from a 3x3 or 4x4 transform."""
        rotation_matrix = np.asarray(matrix, dtype=np.float64)
        if rotation_matrix.shape == (4, 4):
            rotation_matrix = rotation_matrix[:3, :3]
        sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        singular = sy < 1e-6
        if not singular:
            pitch = float(np.degrees(np.arctan2(-rotation_matrix[2, 1], rotation_matrix[2, 2])))
            yaw = float(np.degrees(np.arctan2(rotation_matrix[2, 0], sy)))
            roll = float(np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])))
        else:
            pitch = float(np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])))
            yaw = float(np.degrees(np.arctan2(rotation_matrix[2, 0], sy)))
            roll = 0.0
        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    def analyze(
        self,
        face: Optional[FaceLandmarks],
        image_shape: Tuple[int, ...],
        state: CounterState,
        fps: Optional[float] = None,
    ) -> DetectionResult:
        if face is None:
            state.update(False)
            return DetectionResult(
                type=DetectionType.DISTRACTION,
                detected=False,
                confidence=None,
                severity=Severity.NONE,
                metadata={"reason": "no_face"},
            )

        pose = self.estimate(face, image_shape)
        if pose is None:
            state.update(False)
            return DetectionResult(
                type=DetectionType.DISTRACTION,
                detected=False,
                confidence=None,
                severity=Severity.UNKNOWN,
                metadata={"reason": "pose_estimation_failed"},
            )

        looking_away = (
            abs(pose["yaw"]) >= self.settings.head_yaw_threshold_degrees
            or abs(pose["pitch"]) >= self.settings.head_pitch_threshold_degrees
        )
        state.last_value = max(abs(pose["yaw"]), abs(pose["pitch"]))
        state.update(looking_away)

        min_frames = self.settings.distraction_consecutive_frames
        if fps and fps > 0:
            frames_for_duration = int(
                np.ceil(self.settings.distraction_min_duration_seconds * fps)
            )
            min_frames = max(min_frames, frames_for_duration)

        detected = state.is_triggered(
            min_frames=min_frames,
            min_duration_seconds=self.settings.distraction_min_duration_seconds,
        )

        # Confidence from angular excess beyond thresholds
        yaw_excess = max(
            0.0,
            abs(pose["yaw"]) - self.settings.head_yaw_threshold_degrees,
        ) / max(self.settings.head_yaw_threshold_degrees, 1e-6)
        pitch_excess = max(
            0.0,
            abs(pose["pitch"]) - self.settings.head_pitch_threshold_degrees,
        ) / max(self.settings.head_pitch_threshold_degrees, 1e-6)
        confidence = float(np.clip(max(yaw_excess, pitch_excess, 0.15 if looking_away else 0.0), 0.0, 1.0))

        severity = Severity.NONE
        if detected:
            severity = (
                Severity.HIGH
                if state.duration_seconds()
                >= self.settings.distraction_min_duration_seconds * 2
                else Severity.MEDIUM
            )
        elif looking_away:
            severity = Severity.LOW

        return DetectionResult(
            type=DetectionType.DISTRACTION,
            detected=detected,
            confidence=confidence if looking_away or detected else float(
                np.clip(
                    1.0
                    - max(
                        abs(pose["yaw"]) / max(self.settings.head_yaw_threshold_degrees, 1e-6),
                        abs(pose["pitch"]) / max(self.settings.head_pitch_threshold_degrees, 1e-6),
                    ),
                    0.0,
                    1.0,
                )
            ),
            severity=severity,
            metadata={
                "yaw_degrees": round(pose["yaw"], 2),
                "pitch_degrees": round(pose["pitch"], 2),
                "roll_degrees": round(pose["roll"], 2),
                "looking_away_this_frame": looking_away,
                "yaw_threshold_degrees": self.settings.head_yaw_threshold_degrees,
                "pitch_threshold_degrees": self.settings.head_pitch_threshold_degrees,
                "consecutive_away_frames": state.consecutive,
                "duration_seconds": round(state.duration_seconds(), 3),
                "min_frames_required": min_frames,
                "temporal_gate": "requires_consecutive_look_away_frames",
                "scope": "in_cabin_driver_attention",
            },
        )
