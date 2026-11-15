"""Geometry helpers for facial metrics and bounding-box logic."""

from typing import Sequence, Tuple

import numpy as np


def ear_from_landmarks(
    eye_points: Sequence[np.ndarray],
) -> float:
    """Compute Eye Aspect Ratio from 6 eye landmarks (MediaPipe-style).

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    if len(eye_points) != 6:
        raise ValueError("EAR requires exactly 6 eye landmarks")
    pts = [np.asarray(p, dtype=np.float64) for p in eye_points]
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-6:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def mar_from_landmarks(
    mouth_points: Sequence[np.ndarray],
) -> float:
    """Compute Mouth Aspect Ratio from selected mouth landmarks.

    Expects [left, right, top, bottom] or 6-point mouth set:
    [left, right, top1, top2, bottom1, bottom2]
    """
    pts = [np.asarray(p, dtype=np.float64) for p in mouth_points]
    if len(pts) == 4:
        horizontal = np.linalg.norm(pts[0] - pts[1])
        vertical = np.linalg.norm(pts[2] - pts[3])
    elif len(pts) == 6:
        horizontal = np.linalg.norm(pts[0] - pts[1])
        vertical = (
            np.linalg.norm(pts[2] - pts[4]) + np.linalg.norm(pts[3] - pts[5])
        ) / 2.0
    else:
        raise ValueError("MAR requires 4 or 6 mouth landmarks")
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU for boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def bbox_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_to_bbox_distance(
    point: Tuple[float, float],
    box: Tuple[float, float, float, float],
) -> float:
    """Distance from a point to the closest point on a box (0 if inside)."""
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))
