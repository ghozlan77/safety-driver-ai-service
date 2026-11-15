"""Image decoding and validation helpers."""

from typing import Iterable, Set, Tuple

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# MIME types commonly sent for uploaded images
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "image/webp": {".webp"},
    "image/bmp": {".bmp"},
}


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


def validate_content_type(content_type: str | None, allowed: Iterable[str]) -> str:
    """Validate and normalize content type."""
    if not content_type:
        raise ImageValidationError("Missing Content-Type for uploaded image")
    normalized = content_type.split(";")[0].strip().lower()
    allowed_set = {item.lower() for item in allowed}
    if normalized not in allowed_set:
        raise ImageValidationError(
            f"Unsupported content type '{normalized}'. Allowed: {sorted(allowed_set)}"
        )
    return normalized


def validate_file_size(size_bytes: int, max_bytes: int) -> None:
    """Reject oversized uploads."""
    if size_bytes <= 0:
        raise ImageValidationError("Empty image upload")
    if size_bytes > max_bytes:
        raise ImageValidationError(
            f"Image exceeds max size of {max_bytes} bytes (got {size_bytes})"
        )


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR OpenCV image.

    Raises:
        ImageValidationError: if bytes cannot be decoded as an image.
    """
    if not data:
        raise ImageValidationError("Empty image payload")

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("Unable to decode image bytes with OpenCV")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError("Decoded image must be a 3-channel color image")
    return image


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR OpenCV image to RGB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_if_needed(
    image: np.ndarray,
    max_side: int = 1280,
) -> Tuple[np.ndarray, float]:
    """Downscale large images for faster inference while preserving aspect ratio.

    Returns:
        (possibly resized image, scale factor applied to original -> resized)
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2D/3D points."""
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))


def aspect_ratio(vertical: float, horizontal: float, eps: float = 1e-6) -> float:
    """Safe aspect ratio helper used by EAR/MAR calculations."""
    return float(vertical / (horizontal + eps))


def allowed_extensions_for(content_types: Iterable[str]) -> Set[str]:
    """Map allowed MIME types to file extensions (informational)."""
    result: Set[str] = set()
    for mime in content_types:
        result.update(CONTENT_TYPE_EXTENSIONS.get(mime.lower(), set()))
    return result
