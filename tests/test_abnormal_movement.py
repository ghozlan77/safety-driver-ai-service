"""Abnormal movement service unit tests."""

import numpy as np

from app.core.config import Settings
from app.services.abnormal_movement import AbnormalMovementService
from app.services.face_detection import FaceLandmarks
from app.services.temporal import SessionState


def _face(offset: float = 0.0) -> FaceLandmarks:
    coords = np.full((20, 2), offset, dtype=np.float64)
    norm = np.zeros((20, 3), dtype=np.float64)
    norm[:, 0:2] = coords
    return FaceLandmarks(
        landmarks_norm=norm,
        landmarks_px=coords * 100,
        bbox=(0, 0, 50, 50),
    )


def test_abnormal_movement_needs_history():
    service = AbnormalMovementService(
        Settings(movement_velocity_threshold=0.01, abnormal_movement_consecutive_frames=2)
    )
    session = SessionState(session_id="t1")
    result = service.analyze(_face(0.0), session, session.abnormal_movement, fps=10)
    assert result.detected is False
    assert result.metadata["reason"] == "insufficient_history"


def test_abnormal_movement_triggers_after_consecutive_large_deltas():
    service = AbnormalMovementService(
        Settings(movement_velocity_threshold=0.05, abnormal_movement_consecutive_frames=2)
    )
    session = SessionState(session_id="t2")
    service.analyze(_face(0.0), session, session.abnormal_movement, fps=10)
    r2 = service.analyze(_face(0.2), session, session.abnormal_movement, fps=10)
    assert r2.metadata["sudden_this_frame"] is True
    assert r2.detected is False  # only 1 consecutive after history established
    r3 = service.analyze(_face(0.45), session, session.abnormal_movement, fps=10)
    assert r3.detected is True
    assert "not a medical diagnosis" in r3.metadata["disclaimer"].lower()
