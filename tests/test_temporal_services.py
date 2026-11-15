"""Unit tests for temporal detection logic (mocked landmarks, no real models)."""

from __future__ import annotations

import numpy as np

from app.core.config import Settings
from app.services.drowsiness import DrowsinessService
from app.services.face_detection import FaceLandmarks
from app.services.temporal import CounterState
from app.services.yawning import YawningService
from app.utils.geometry import ear_from_landmarks, mar_from_landmarks


def _face_with_points(points: dict[int, tuple[float, float]]) -> FaceLandmarks:
    """Build a FaceLandmarks object with selective landmark overrides."""
    coords = np.zeros((478, 2), dtype=np.float64)
    for idx, (x, y) in points.items():
        coords[idx] = (x, y)
    return FaceLandmarks(
        landmarks_norm=np.zeros((478, 3), dtype=np.float64),
        landmarks_px=coords,
        bbox=(0, 0, 100, 100),
        face_index=0,
    )


def test_ear_computation_open_vs_closed():
    # Horizontal wide, vertical large -> open
    open_eye = [
        np.array([0.0, 0.0]),
        np.array([1.0, 2.0]),
        np.array([2.0, 2.0]),
        np.array([3.0, 0.0]),
        np.array([2.0, -2.0]),
        np.array([1.0, -2.0]),
    ]
    closed_eye = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.1]),
        np.array([2.0, 0.1]),
        np.array([3.0, 0.0]),
        np.array([2.0, -0.1]),
        np.array([1.0, -0.1]),
    ]
    assert ear_from_landmarks(open_eye) > ear_from_landmarks(closed_eye)


def test_mar_computation():
    open_mouth = [
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        np.array([5.0, 4.0]),
        np.array([5.0, -4.0]),
    ]
    closed_mouth = [
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        np.array([5.0, 0.2]),
        np.array([5.0, -0.2]),
    ]
    assert mar_from_landmarks(open_mouth) > mar_from_landmarks(closed_mouth)


def test_drowsiness_requires_consecutive_frames():
    settings = Settings(
        ear_threshold=0.5,
        drowsiness_consecutive_frames=3,
        drowsiness_min_duration_seconds=0.0,
    )
    service = DrowsinessService(settings)
    state = CounterState()

    # Build closed-eye-like landmarks for both eyes (very small vertical)
    from app.services.face_detection import LEFT_EYE_IDX, RIGHT_EYE_IDX

    points = {}
    for indices in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
        # p1..p6 around a nearly flat eye
        base_x = 10 if indices == LEFT_EYE_IDX else 40
        pts = [
            (base_x, 20),
            (base_x + 1, 20.05),
            (base_x + 2, 20.05),
            (base_x + 3, 20),
            (base_x + 2, 19.95),
            (base_x + 1, 19.95),
        ]
        for idx, pt in zip(indices, pts):
            points[idx] = pt
    face = _face_with_points(points)

    r1 = service.analyze(face, state, fps=10)
    assert r1.detected is False
    assert r1.metadata["eye_status"] == "closed"

    r2 = service.analyze(face, state, fps=10)
    assert r2.detected is False

    r3 = service.analyze(face, state, fps=10)
    assert r3.detected is True
    assert r3.metadata["consecutive_closed_frames"] >= 3


def test_yawning_single_frame_does_not_trigger():
    settings = Settings(
        mar_threshold=0.2,
        yawning_consecutive_frames=5,
        yawning_min_duration_seconds=0.0,
    )
    service = YawningService(settings)
    state = CounterState()
    from app.services.face_detection import MOUTH_IDX

    points = {
        MOUTH_IDX[0]: (0, 0),
        MOUTH_IDX[1]: (20, 0),
        MOUTH_IDX[2]: (10, 8),
        MOUTH_IDX[3]: (10, -8),
        MOUTH_IDX[4]: (5, 7),
        MOUTH_IDX[5]: (15, -7),
    }
    face = _face_with_points(points)
    result = service.analyze(face, state, fps=10)
    assert result.metadata["mouth_status"] == "open"
    assert result.detected is False


def test_counter_state_resets_on_negative():
    state = CounterState()
    state.update(True)
    state.update(True)
    assert state.consecutive == 2
    state.update(False)
    assert state.consecutive == 0
    assert state.active_since is None
