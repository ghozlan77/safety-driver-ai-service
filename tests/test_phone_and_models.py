"""Phone detection logic and missing-model behavior tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.config import Settings
from app.models.model_loader import ModelLoader
from app.schemas.detection import DetectionType
from app.services.face_detection import FaceDetectionService, FaceLandmarks
from app.services.phone_detection import PhoneDetectionService
from app.services.temporal import CounterState


@pytest.fixture
def mock_yolo_with_phone():
    model = MagicMock()
    boxes = MagicMock()
    boxes.__len__.return_value = 1
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        [[500.0, 350.0, 620.0, 460.0]], dtype=np.float32
    )
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.91], dtype=np.float32)
    result = MagicMock()
    result.boxes = boxes
    model.predict.return_value = [result]
    return model


def test_phone_presence_without_usage(mock_yolo_with_phone):
    settings = Settings(
        phone_detection_confidence=0.3,
        phone_usage_max_distance_ratio=0.05,
        phone_usage_iou_threshold=0.9,
        phone_usage_consecutive_frames=1,
    )
    service = PhoneDetectionService(mock_yolo_with_phone, settings)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    face = FaceLandmarks(
        landmarks_norm=np.zeros((10, 3)),
        landmarks_px=np.zeros((10, 2)),
        bbox=(10, 10, 80, 80),
    )
    state = CounterState()
    presence, usage, warning = service.analyze(image, face, state)
    assert warning is None
    assert presence.type == DetectionType.PHONE_PRESENCE
    assert presence.detected is True
    assert usage.detected is False
    assert usage.metadata.get("near_face") is False


def test_missing_yolo_returns_warning_not_fake_detection():
    service = PhoneDetectionService(yolo_model=None, settings=Settings())
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    state = CounterState()
    presence, usage, warning = service.analyze(image, None, state)
    assert warning is not None
    assert presence.detected is False
    assert presence.confidence is None
    assert usage.detected is False


def test_missing_face_landmarker_reports_unavailable():
    service = FaceDetectionService(face_landmarker=None)
    output = service.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert output.available is False
    result = service.to_detection_result(output)
    assert result.detected is False
    assert result.metadata["available"] is False


def test_model_loader_handles_missing_weights(tmp_path, monkeypatch):
    """Missing model setup should not crash loader; statuses marked not loaded."""
    settings = Settings(
        yolo_model_path=str(tmp_path / "missing-model.pt"),
        face_landmarker_model_path=str(tmp_path / "missing.task"),
    )
    loader = ModelLoader(settings=settings)

    class BoomYOLO:
        def __init__(self, *_a, **_k):
            raise FileNotFoundError("weights missing")

    fake_ultra = MagicMock()
    fake_ultra.YOLO = BoomYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    # Prevent network download of face model
    monkeypatch.setattr(
        loader,
        "_ensure_face_model",
        lambda path: (_ for _ in ()).throw(FileNotFoundError("no face model")),
    )

    registry = loader.load()
    assert registry.statuses["yolo_phone"].loaded is False
    assert registry.statuses["mediapipe_face_landmarker"].loaded is False
