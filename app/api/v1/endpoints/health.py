"""Health and root information endpoints."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.models.model_loader import ModelLoader, model_loader
from app.schemas.response import HealthResponse, RootResponse

router = APIRouter(tags=["health"])


@router.get("/", response_model=RootResponse)
def root(settings: Settings = Depends(get_settings)) -> RootResponse:
    return RootResponse(
        message="Safety Driver AI Service is running",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    loader: ModelLoader = Depends(lambda: model_loader),
) -> HealthResponse:
    models = loader.registry.status_dict()
    # Degraded if required face landmarker is missing
    face = models.get("mediapipe_face_landmarker", {})
    status = "healthy" if face.get("loaded") else "degraded"
    return HealthResponse(
        status=status,
        service=settings.app_name,
        version=settings.app_version,
        models=models,
    )
