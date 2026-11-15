"""API response schema contract tests."""

from app.schemas.detection import (
    AnalyzeFrameResponse,
    DetectionResult,
    DetectionType,
    Severity,
)


def test_analyze_frame_response_schema(client, jpeg_bytes):
    response = client.post(
        "/api/v1/analyze-frame",
        files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    parsed = AnalyzeFrameResponse.model_validate(response.json())
    assert parsed.success is True
    assert parsed.session_id is not None
    types = {d.type for d in parsed.detections}
    assert DetectionType.FACE in types
    assert DetectionType.DROWSINESS in types
    assert DetectionType.YAWNING in types
    assert DetectionType.PHONE_PRESENCE in types
    assert DetectionType.PHONE_USAGE in types
    assert DetectionType.DISTRACTION in types
    assert DetectionType.ABNORMAL_MOVEMENT in types


def test_detection_result_allows_null_confidence():
    result = DetectionResult(
        type=DetectionType.DROWSINESS,
        detected=False,
        confidence=None,
        severity=Severity.NONE,
        metadata={"reason": "no_face"},
    )
    assert result.confidence is None


def test_detection_result_rejects_fake_out_of_range_confidence():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DetectionResult(
            type=DetectionType.FACE,
            detected=True,
            confidence=1.5,
        )
