"""
Health check endpoint for the BYOAI ML Service.

Provides a quick status check including whether the ML model
is loaded and ready to serve predictions.
"""

from fastapi import APIRouter, Request

from app.config import settings
from app.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.

    Returns the current service status, model loading state,
    and service metadata.
    """
    model_loaded = getattr(request.app.state, "model_loaded", False)

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        service=settings.APP_NAME,
        model_loaded=model_loaded,
        model_name=settings.MODEL_NAME,
        version="1.0.0",
    )
