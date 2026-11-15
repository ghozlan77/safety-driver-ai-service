"""FastAPI application entrypoint for Safety Driver AI Service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.models.model_loader import model_loader
from app.schemas.response import ErrorDetail, ErrorResponse, utc_now_iso
from app.services.orchestrator import orchestrator

settings = get_settings()
setup_logging(settings)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load AI models once at startup; release on shutdown."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    model_loader.load()
    orchestrator.refresh_models()
    logger.info("Model status: %s", model_loader.registry.status_dict())
    yield
    logger.info("Shutting down AI service")
    model_loader.shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "In-cabin driver monitoring AI service for the Safety Driver graduation project. "
        "Provides face, drowsiness, yawning, phone usage, distraction, and abnormal movement "
        "detections via REST for integration with the .NET backend."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    payload = ErrorResponse(
        timestamp=utc_now_iso(),
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()},
        ),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Return ErrorResponse envelope directly (no nested FastAPI `detail` wrapper)."""
    detail = exc.detail
    if isinstance(detail, dict) and "success" in detail and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    payload = ErrorResponse(
        timestamp=utc_now_iso(),
        error=ErrorDetail(
            code="HTTP_ERROR",
            message=str(detail),
        ),
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    payload = ErrorResponse(
        timestamp=utc_now_iso(),
        error=ErrorDetail(
            code="INTERNAL_ERROR",
            message="An unexpected server error occurred",
            details={"type": type(exc).__name__} if settings.debug else None,
        ),
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


def run() -> None:
    """CLI helper: `python -m app.main`."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
