"""Centralized AI model loading (singleton-style via FastAPI lifespan)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlretrieve

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class ModelStatus:
    """Status of an individual model resource."""

    name: str
    loaded: bool
    path: Optional[str] = None
    detail: str = ""
    optional: bool = False


@dataclass
class ModelRegistry:
    """Holds loaded model handles and status metadata."""

    face_landmarker: Any = None
    yolo: Any = None
    statuses: Dict[str, ModelStatus] = field(default_factory=dict)

    def status_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            key: {
                "loaded": status.loaded,
                "path": status.path,
                "detail": status.detail,
                "optional": status.optional,
            }
            for key, status in self.statuses.items()
        }


class ModelLoader:
    """Load MediaPipe Face Landmarker and YOLO once at startup."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.registry = ModelRegistry()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> ModelRegistry:
        """Load all configured models. Safe to call once at app startup."""
        if self._loaded:
            return self.registry

        self._load_face_landmarker()
        self._load_yolo()
        self._loaded = True
        return self.registry

    def _ensure_face_model(self, model_path: Path) -> Path:
        if model_path.exists() and model_path.stat().st_size > 0:
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Face Landmarker model to %s", model_path)
        urlretrieve(FACE_LANDMARKER_URL, model_path)
        return model_path

    def _load_face_landmarker(self) -> None:
        name = "mediapipe_face_landmarker"
        model_path = Path(self.settings.face_landmarker_model_path)
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision

            model_path = self._ensure_face_model(model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path.resolve())),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
            self.registry.face_landmarker = vision.FaceLandmarker.create_from_options(
                options
            )
            self.registry.statuses[name] = ModelStatus(
                name=name,
                loaded=True,
                path=str(model_path),
                detail="MediaPipe Face Landmarker ready (landmarks, EAR/MAR, head pose)",
            )
            logger.info("Loaded MediaPipe Face Landmarker from %s", model_path)
        except Exception as exc:  # noqa: BLE001 - startup must remain resilient
            logger.exception("Failed to load MediaPipe Face Landmarker")
            self.registry.face_landmarker = None
            self.registry.statuses[name] = ModelStatus(
                name=name,
                loaded=False,
                path=str(model_path),
                detail=(
                    f"Face Landmarker unavailable: {exc}. "
                    f"Place '{model_path.name}' under models/ or allow network download from "
                    f"{FACE_LANDMARKER_URL}"
                ),
            )

    def _load_yolo(self) -> None:
        name = "yolo_phone"
        model_path = Path(self.settings.yolo_model_path)
        try:
            from ultralytics import YOLO

            if not model_path.exists():
                logger.warning(
                    "YOLO weights not found at %s — attempting Ultralytics download by name",
                    model_path,
                )
                model = YOLO(model_path.name)
                try:
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    ckpt = getattr(model, "ckpt_path", None)
                    if isinstance(ckpt, (str, Path)) and Path(ckpt).exists():
                        import shutil

                        shutil.copy2(ckpt, model_path)
                except Exception:  # noqa: BLE001
                    logger.warning("Could not copy YOLO weights to %s", model_path)
            else:
                model = YOLO(str(model_path))

            self.registry.yolo = model
            self.registry.statuses[name] = ModelStatus(
                name=name,
                loaded=True,
                path=str(model_path),
                optional=True,
                detail="YOLO loaded for in-cabin phone detection",
            )
            logger.info("Loaded YOLO model from %s", model_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load YOLO model")
            self.registry.yolo = None
            self.registry.statuses[name] = ModelStatus(
                name=name,
                loaded=False,
                path=str(model_path),
                optional=True,
                detail=(
                    f"YOLO unavailable: {exc}. "
                    f"Place weights at '{model_path}' or ensure network access "
                    "for Ultralytics to download yolov8n.pt."
                ),
            )

    def shutdown(self) -> None:
        """Release resources."""
        landmarker = self.registry.face_landmarker
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:  # noqa: BLE001
                logger.debug("Face landmarker close() raised", exc_info=True)
        self.registry.face_landmarker = None
        self.registry.yolo = None
        self._loaded = False


# Module-level singleton set by FastAPI lifespan
model_loader = ModelLoader()
