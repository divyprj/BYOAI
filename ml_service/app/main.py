"""
Main FastAPI application for the BYOAI ML Service.

Manages the application lifecycle including model loading on startup,
graceful cleanup on shutdown, and route registration.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routes import health, predict
from app.services.intent_classifier import IntentClassifier

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    On startup: loads the HuggingFace zero-shot classification pipeline.
    On shutdown: cleans up resources and releases memory.
    """
    # --- Startup ---
    logger.info("Starting %s...", settings.APP_NAME)
    logger.info("Loading model: %s on device: %s", settings.MODEL_NAME, settings.DEVICE)

    start_time = time.time()
    try:
        classifier = IntentClassifier(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        )
        app.state.classifier = classifier
        app.state.model_loaded = True

        load_time = (time.time() - start_time) * 1000
        logger.info("Model loaded successfully in %.2f ms", load_time)
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        app.state.classifier = None
        app.state.model_loaded = False

    yield

    # --- Shutdown ---
    logger.info("Shutting down %s...", settings.APP_NAME)
    if hasattr(app.state, "classifier") and app.state.classifier is not None:
        del app.state.classifier
        app.state.model_loaded = False
        logger.info("Model resources released.")
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Zero-shot intent classification and contextual response generation "
        "service for the BYOAI conversational automation system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", tags=["root"])
async def root() -> dict:
    """Root endpoint returning service information."""
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
