"""In-cabin phone presence and phone-usage heuristics via YOLO."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.detection import DetectionResult, DetectionType, Severity
from app.services.face_detection import FaceLandmarks
from app.services.temporal import CounterState
from app.utils.geometry import bbox_center, bbox_iou, point_to_bbox_distance

logger = get_logger(__name__)

# COCO class id for cell phone in Ultralytics YOLO pretrained models
COCO_CELL_PHONE_CLASS_ID = 67


class PhoneDetectionService:
    """Detect phones and infer likely usage near the driver face region.

    Presence and usage are separate detections:
    - phone_presence: YOLO saw a phone above confidence threshold
    - phone_usage: phone is spatially near the driver face for consecutive frames
    """

    def __init__(
        self,
        yolo_model: Any = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.yolo = yolo_model
        self.settings = settings or get_settings()

    def set_model(self, yolo_model: Any) -> None:
        self.yolo = yolo_model

    def detect_phones(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Run YOLO and return phone boxes. Does not invent detections."""
        if self.yolo is None:
            return [], "YOLO model is not loaded; phone detection unavailable"

        conf = self.settings.phone_detection_confidence
        try:
            results = self.yolo.predict(
                source=image_bgr,
                conf=conf,
                verbose=False,
                classes=[COCO_CELL_PHONE_CLASS_ID],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("YOLO inference failed")
            return [], f"YOLO inference failed: {exc}"

        phones: List[Dict[str, Any]] = []
        if not results:
            return phones, None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return phones, None

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for box, score in zip(xyxy, confs):
            x1, y1, x2, y2 = [float(v) for v in box]
            phones.append(
                {
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": float(score),
                    "class_id": COCO_CELL_PHONE_CLASS_ID,
                    "label": "cell phone",
                }
            )
        return phones, None

    def _usage_candidate(
        self,
        phones: List[Dict[str, Any]],
        face: Optional[FaceLandmarks],
        image_shape: Tuple[int, ...],
    ) -> Tuple[bool, Optional[float], Dict[str, Any]]:
        """Decide if any phone is near enough to the driver face to suggest usage."""
        if not phones or face is None:
            return False, None, {"reason": "no_phone_or_no_face"}

        height, width = image_shape[:2]
        diag = float(np.hypot(width, height))
        face_box = face.bbox
        face_ctr = bbox_center(face_box)

        best_score: Optional[float] = None
        best_meta: Dict[str, Any] = {}
        usage = False

        for phone in phones:
            pb = phone["bbox"]
            phone_box = (pb["x1"], pb["y1"], pb["x2"], pb["y2"])
            iou = bbox_iou(face_box, phone_box)
            dist = point_to_bbox_distance(face_ctr, phone_box)
            dist_ratio = dist / max(diag, 1e-6)
            near = (
                iou >= self.settings.phone_usage_iou_threshold
                or dist_ratio <= self.settings.phone_usage_max_distance_ratio
            )
            # Combine detection confidence with proximity (actual computed score)
            proximity_score = float(
                np.clip(1.0 - dist_ratio / max(self.settings.phone_usage_max_distance_ratio, 1e-6), 0.0, 1.0)
            )
            combined = float(0.6 * phone["confidence"] + 0.4 * proximity_score)
            if best_score is None or combined > best_score:
                best_score = combined
                best_meta = {
                    "phone_confidence": phone["confidence"],
                    "iou_with_face": round(iou, 4),
                    "distance_ratio": round(dist_ratio, 4),
                    "proximity_score": round(proximity_score, 4),
                    "near_face": near,
                }
            if near:
                usage = True

        return usage, best_score, best_meta

    def analyze(
        self,
        image_bgr: np.ndarray,
        face: Optional[FaceLandmarks],
        state: CounterState,
    ) -> Tuple[DetectionResult, DetectionResult, Optional[str]]:
        phones, warning = self.detect_phones(image_bgr)
        presence = DetectionResult(
            type=DetectionType.PHONE_PRESENCE,
            detected=len(phones) > 0,
            confidence=max((p["confidence"] for p in phones), default=None),
            severity=Severity.LOW if phones else Severity.NONE,
            metadata={
                "phone_count": len(phones),
                "phones": phones,
                "note": "Presence alone is not classified as phone usage",
            },
        )

        usage_now, usage_conf, usage_meta = self._usage_candidate(
            phones, face, image_bgr.shape
        )
        state.update(usage_now)
        detected_usage = state.is_triggered(
            min_frames=self.settings.phone_usage_consecutive_frames,
            min_duration_seconds=0.0,
        )

        severity = Severity.NONE
        if detected_usage:
            severity = Severity.HIGH
        elif usage_now:
            severity = Severity.MEDIUM

        usage = DetectionResult(
            type=DetectionType.PHONE_USAGE,
            detected=detected_usage,
            confidence=usage_conf if usage_now or detected_usage else None,
            severity=severity,
            metadata={
                **usage_meta,
                "usage_candidate_this_frame": usage_now,
                "consecutive_usage_frames": state.consecutive,
                "min_frames_required": self.settings.phone_usage_consecutive_frames,
                "temporal_gate": "requires_consecutive_near_face_frames",
                "note": "Usage requires phone near driver face over multiple frames",
            },
        )
        return presence, usage, warning
