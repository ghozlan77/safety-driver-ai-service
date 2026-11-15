"""Video sampling helpers for optional analyze-video endpoint."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class VideoProcessingError(ValueError):
    """Raised when a video cannot be opened or sampled."""


def sample_frames_from_bytes(
    data: bytes,
    max_frames: int = 30,
    target_fps: Optional[float] = None,
    temp_path: Optional[Path] = None,
) -> Tuple[List[np.ndarray], float]:
    """Decode a short video from bytes and sample frames.

    OpenCV needs a filesystem path for most containers, so bytes are written
    to a temporary file when temp_path is provided by the caller.

    Returns:
        (list of BGR frames, estimated source fps)
    """
    if temp_path is None:
        raise VideoProcessingError(
            "temp_path is required to sample video frames with OpenCV on this platform"
        )
    temp_path.write_bytes(data)
    return sample_frames_from_path(temp_path, max_frames=max_frames, target_fps=target_fps)


def sample_frames_from_path(
    path: Path,
    max_frames: int = 30,
    target_fps: Optional[float] = None,
) -> Tuple[List[np.ndarray], float]:
    """Sample up to max_frames evenly from a video file."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoProcessingError(f"Unable to open video: {path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 10.0

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            # Fallback: read sequentially until exhausted / max_frames
            frames = list(_read_sequential(capture, max_frames))
            if not frames:
                raise VideoProcessingError("Video contains no readable frames")
            return frames, source_fps

        step = 1
        if target_fps and target_fps > 0 and target_fps < source_fps:
            step = max(1, int(round(source_fps / target_fps)))

        indices = list(range(0, total_frames, step))
        if len(indices) > max_frames:
            # Evenly subsample to max_frames
            selector = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = [indices[i] for i in selector]

        frames: List[np.ndarray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)

        if not frames:
            raise VideoProcessingError("Failed to sample any frames from video")
        return frames, source_fps
    finally:
        capture.release()


def _read_sequential(
    capture: cv2.VideoCapture,
    max_frames: int,
) -> Generator[np.ndarray, None, None]:
    count = 0
    while count < max_frames:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        yield frame
        count += 1
