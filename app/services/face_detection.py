"""Face detection and facial landmark extraction via MediaPipe Face Landmarker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.logging import get_logger
from app.schemas.detection import DetectionResult, DetectionType, Severity
from app.utils.image_processing import to_rgb

logger = get_logger(__name__)

# Landmark index sets (MediaPipe Face Mesh / Face Landmarker topology)
LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)
MOUTH_IDX = (61, 291, 13, 14, 78, 308)
HEAD_POSE_IDX = (1, 152, 33, 263, 61, 291)


@dataclass
class FaceLandmarks:
    """Normalized + pixel landmark container for one face."""

    landmarks_norm: np.ndarray  # (N, 3) x,y,z normalized
    landmarks_px: np.ndarray  # (N, 2) pixel coordinates
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2
    face_index: int = 0
    transformation_matrix: Optional[np.ndarray] = None

    def points(self, indices: Tuple[int, ...]) -> List[np.ndarray]:
        return [self.landmarks_px[i] for i in indices]


@dataclass
class FaceDetectionOutput:
    """Raw face pipeline output shared by downstream detectors."""

    faces: List[FaceLandmarks] = field(default_factory=list)
    available: bool = True
    warning: Optional[str] = None

    @property
    def primary(self) -> Optional[FaceLandmarks]:
        """Primary face (index 0) — treated as driver candidate."""
        return self.faces[0] if self.faces else None


class FaceDetectionService:
    """Detect faces and extract mesh landmarks for in-cabin monitoring."""

    def __init__(self, face_landmarker: Any = None) -> None:
        self.face_landmarker = face_landmarker

    def set_model(self, face_landmarker: Any) -> None:
        self.face_landmarker = face_landmarker

    def detect(self, image_bgr: np.ndarray) -> FaceDetectionOutput:
        if self.face_landmarker is None:
            return FaceDetectionOutput(
                available=False,
                warning="MediaPipe Face Landmarker is not loaded",
            )

        try:
            import mediapipe as mp
        except ImportError as exc:
            return FaceDetectionOutput(
                available=False,
                warning=f"mediapipe import failed: {exc}",
            )

        height, width = image_bgr.shape[:2]
        rgb = to_rgb(image_bgr)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        try:
            result = self.face_landmarker.detect(mp_image)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Face Landmarker inference failed")
            return FaceDetectionOutput(
                available=False,
                warning=f"Face Landmarker inference failed: {exc}",
            )

        faces: List[FaceLandmarks] = []
        face_landmarks_list = getattr(result, "face_landmarks", None) or []
        matrices = getattr(result, "facial_transformation_matrixes", None) or []

        for idx, face_lms in enumerate(face_landmarks_list):
            coords_norm = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_lms],
                dtype=np.float64,
            )
            coords_px = np.column_stack(
                [
                    coords_norm[:, 0] * width,
                    coords_norm[:, 1] * height,
                ]
            )
            x_min = float(coords_px[:, 0].min())
            y_min = float(coords_px[:, 1].min())
            x_max = float(coords_px[:, 0].max())
            y_max = float(coords_px[:, 1].max())
            matrix = None
            if idx < len(matrices):
                matrix = np.asarray(matrices[idx], dtype=np.float64)
            faces.append(
                FaceLandmarks(
                    landmarks_norm=coords_norm,
                    landmarks_px=coords_px,
                    bbox=(x_min, y_min, x_max, y_max),
                    face_index=idx,
                    transformation_matrix=matrix,
                )
            )

        return FaceDetectionOutput(faces=faces)

    def to_detection_result(self, output: FaceDetectionOutput) -> DetectionResult:
        if not output.available:
            return DetectionResult(
                type=DetectionType.FACE,
                detected=False,
                confidence=None,
                severity=Severity.UNKNOWN,
                metadata={
                    "available": False,
                    "warning": output.warning,
                    "face_count": 0,
                },
            )

        face_count = len(output.faces)
        metadata: Dict[str, Any] = {
            "available": True,
            "face_count": face_count,
            "faces": [
                {
                    "face_index": face.face_index,
                    "bbox": {
                        "x1": face.bbox[0],
                        "y1": face.bbox[1],
                        "x2": face.bbox[2],
                        "y2": face.bbox[3],
                    },
                    "landmark_count": int(face.landmarks_px.shape[0]),
                    "role_hint": (
                        "driver_candidate"
                        if face.face_index == 0
                        else "passenger_candidate"
                    ),
                }
                for face in output.faces
            ],
        }
        return DetectionResult(
            type=DetectionType.FACE,
            detected=face_count > 0,
            confidence=1.0 if face_count > 0 else 0.0,
            severity=Severity.NONE,
            metadata=metadata,
        )
