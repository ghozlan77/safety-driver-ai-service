"""Image validation and invalid upload tests."""

import numpy as np
import pytest

from app.utils.image_processing import (
    ImageValidationError,
    decode_image_bytes,
    validate_content_type,
    validate_file_size,
)


def test_validate_content_type_accepts_jpeg():
    assert validate_content_type("image/jpeg", ["image/jpeg", "image/png"]) == "image/jpeg"


def test_validate_content_type_rejects_pdf():
    with pytest.raises(ImageValidationError):
        validate_content_type("application/pdf", ["image/jpeg"])


def test_validate_file_size_rejects_empty():
    with pytest.raises(ImageValidationError):
        validate_file_size(0, 1000)


def test_validate_file_size_rejects_oversized():
    with pytest.raises(ImageValidationError):
        validate_file_size(2000, 1000)


def test_decode_invalid_bytes():
    with pytest.raises(ImageValidationError):
        decode_image_bytes(b"not-an-image")


def test_analyze_frame_rejects_non_image(client):
    response = client.post(
        "/api/v1/analyze-frame",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_IMAGE"


def test_analyze_frame_rejects_corrupt_image(client):
    response = client.post(
        "/api/v1/analyze-frame",
        files={"file": ("bad.jpg", b"not-really-jpeg", "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_analyze_frame_accepts_valid_jpeg(client, jpeg_bytes):
    response = client.post(
        "/api/v1/analyze-frame",
        files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "detections" in body
    assert isinstance(body["processing_time_ms"], (int, float))


def test_decode_roundtrip_shape(jpeg_bytes):
    image = decode_image_bytes(jpeg_bytes)
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3
