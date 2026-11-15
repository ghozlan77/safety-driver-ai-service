"""Centralized application configuration via environment variables."""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Safety Driver AI Service"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Image upload
    max_image_size_bytes: int = 5 * 1024 * 1024
    allowed_image_content_types: str = "image/jpeg,image/png,image/webp,image/bmp"

    # Models
    yolo_model_path: str = "models/yolov8n.pt"
    face_landmarker_model_path: str = "models/face_landmarker.task"

    # Drowsiness
    ear_threshold: float = 0.21
    drowsiness_consecutive_frames: int = 15
    drowsiness_min_duration_seconds: float = 1.5

    # Yawning
    mar_threshold: float = 0.55
    yawning_consecutive_frames: int = 12
    yawning_min_duration_seconds: float = 1.2

    # Phone
    phone_detection_confidence: float = 0.45
    phone_usage_iou_threshold: float = 0.05
    phone_usage_max_distance_ratio: float = 0.45
    phone_usage_consecutive_frames: int = 8

    # Head pose / distraction
    head_yaw_threshold_degrees: float = 25.0
    head_pitch_threshold_degrees: float = 20.0
    distraction_consecutive_frames: int = 12
    distraction_min_duration_seconds: float = 1.5

    # Abnormal movement
    movement_velocity_threshold: float = 0.08
    abnormal_movement_consecutive_frames: int = 10

    # Temporal tracking
    default_fps: float = 10.0
    session_ttl_seconds: int = 300

    @property
    def allowed_content_types(self) -> List[str]:
        """Parse allowed MIME types into a list."""
        return [
            item.strip().lower()
            for item in self.allowed_image_content_types.split(",")
            if item.strip()
        ]

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (FastAPI dependency-friendly)."""
    return Settings()
