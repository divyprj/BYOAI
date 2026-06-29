"""Health check endpoints.

Provides liveness and readiness probes for container orchestrators
like Kubernetes and Docker health checks.
"""

import logging
import time

from fastapi import APIRouter, Request

from app.config import settings
from app.models import HealthResponse
from app.services.ml_client import MLServiceClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])

_APP_VERSION = "1.0.0"


@router.get(
    "",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns basic health status and uptime. Used as a liveness probe.",
)
async def health_check(request: Request) -> HealthResponse:
    """Return the current health status of the gateway.

    This is a lightweight liveness check - it confirms the service
    is running and responsive without checking downstream dependencies.
    """
    uptime = time.time() - request.app.state.start_time
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=_APP_VERSION,
        uptime_seconds=round(uptime, 2),
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness check",
    description="Checks gateway health AND ML service connectivity.",
    responses={503: {"description": "ML service unreachable"}},
)
async def readiness_check(request: Request) -> HealthResponse:
    """Check readiness by verifying downstream ML service connectivity.

    Returns 'healthy' if the ML service is reachable, 'degraded' otherwise.
    Always returns HTTP 200 - orchestrators should inspect the status field.
    """
    uptime = time.time() - request.app.state.start_time
    ml_client = MLServiceClient(request.app.state.http_client)
    ml_healthy = await ml_client.health_check()

    status = "healthy" if ml_healthy else "degraded"
    if not ml_healthy:
        logger.warning("Readiness check: ML service is unreachable")

    return HealthResponse(
        status=status,
        service=settings.APP_NAME,
        version=_APP_VERSION,
        uptime_seconds=round(uptime, 2),
    )
