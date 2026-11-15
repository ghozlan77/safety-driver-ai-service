"""Shared pytest fixtures.

Unit tests mock AI models. Integration-style API tests use TestClient
with a lifespan that skips heavy model downloads when possible.
"""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.models.model_loader import ModelRegistry, ModelStatus, model_loader
from app.services.orchestrator import orchestrator
from app.services.temporal import SessionStore, session_store


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def blank_image_bgr() -> np.ndarray:
    """Synthetic BGR image with no face content."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def jpeg_bytes(blank_image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", blank_image_bgr)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def png_bytes(blank_image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", blank_image_bgr)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def mock_face_landmarker() -> MagicMock:
    """Face Landmarker mock that returns no faces by default."""
    landmarker = MagicMock()
    result = MagicMock()
    result.face_landmarks = []
    result.facial_transformation_matrixes = []
    landmarker.detect.return_value = result
    landmarker.close = MagicMock()
    return landmarker


@pytest.fixture
def mock_yolo_empty() -> MagicMock:
    model = MagicMock()
    empty = MagicMock()
    empty.boxes = None
    model.predict.return_value = [empty]
    return model


@pytest.fixture
def configured_loader(
    mock_face_landmarker: MagicMock,
    mock_yolo_empty: MagicMock,
) -> Generator[None, None, None]:
    """Inject mocked models into the global loader/orchestrator."""
    original_registry = model_loader.registry
    original_loaded = model_loader._loaded

    registry = ModelRegistry(
        face_landmarker=mock_face_landmarker,
        yolo=mock_yolo_empty,
        statuses={
            "mediapipe_face_landmarker": ModelStatus(
                name="mediapipe_face_landmarker",
                loaded=True,
                detail="mocked",
            ),
            "yolo_phone": ModelStatus(
                name="yolo_phone",
                loaded=True,
                detail="mocked",
                optional=True,
            ),
        },
    )
    model_loader.registry = registry
    model_loader._loaded = True
    orchestrator.loader = model_loader
    orchestrator.refresh_models()
    session_store._sessions.clear()

    yield

    model_loader.registry = original_registry
    model_loader._loaded = original_loaded
    orchestrator.refresh_models()
    session_store._sessions.clear()


@pytest.fixture
def client(configured_loader: None) -> Generator[TestClient, None, None]:
    """FastAPI test client with mocked models (no real weight download)."""

    def _fake_load():
        return model_loader.registry

    original_load = model_loader.load
    original_shutdown = model_loader.shutdown
    model_loader.load = _fake_load  # type: ignore[method-assign]
    model_loader.shutdown = lambda: None  # type: ignore[method-assign]

    with TestClient(app) as test_client:
        yield test_client

    model_loader.load = original_load  # type: ignore[method-assign]
    model_loader.shutdown = original_shutdown  # type: ignore[method-assign]


@pytest.fixture
def fresh_session_store() -> SessionStore:
    return SessionStore()
