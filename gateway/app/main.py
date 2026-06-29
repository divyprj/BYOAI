"""BYOAI API Gateway — FastAPI application entrypoint.

Sets up the FastAPI application with:
- Lifespan-managed httpx.AsyncClient for downstream calls
- CORS middleware (permissive for demo use)
- Request ID middleware for distributed tracing
- Structured JSON logging
- Health and conversation routers
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import settings
from app.middleware.logging import RequestLoggingMiddleware, setup_logging
from app.routes import conversation, health

# Initialize structured logging before anything else
setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle resources.

    On startup:
        - Creates a shared httpx.AsyncClient for downstream service calls
        - Records the application start time

    On shutdown:
        - Closes the httpx.AsyncClient to release connections
    """
    logger.info("Starting %s...", settings.APP_NAME)

    # Shared HTTP client for all outbound requests
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    app.state.start_time = time.time()

    logger.info("%s started successfully", settings.APP_NAME)
    yield

    # Shutdown
    logger.info("Shutting down %s...", settings.APP_NAME)
    await app.state.http_client.aclose()
    logger.info("%s shut down cleanly", settings.APP_NAME)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "API Gateway for the BYOAI conversational automation system. "
        "Receives user chat messages, forwards them to the ML model service, "
        "and manages conversation history."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS — permissive for demo; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique X-Request-ID to every request.

    If the incoming request already carries an X-Request-ID header
    (e.g., from a reverse proxy), it is preserved. Otherwise a new
    UUID4 is generated.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Store on request state for downstream access
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(conversation.router)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Service info",
    description="Returns basic information about the gateway service.",
    tags=["Root"],
)
async def root() -> dict[str, str]:
    """Return basic service information."""
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
