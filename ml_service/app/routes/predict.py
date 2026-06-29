"""
Prediction endpoint for the BYOAI ML Service.

Accepts user text, classifies intent using the zero-shot model,
generates a contextual response, and returns structured results
with timing information.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.models import IntentScore, PredictRequest, PredictResponse
from app.services.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["prediction"])

response_generator = ResponseGenerator()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """
    Predict the intent of the given text and generate a contextual response.

    Args:
        request: The incoming FastAPI request (used to access app state).
        body: The prediction request containing the input text.

    Returns:
        PredictResponse with intent, confidence, generated response,
        all intent scores, and processing time.

    Raises:
        HTTPException 503: If the ML model is not loaded.
    """
    # Check if model is loaded
    model_loaded = getattr(request.app.state, "model_loaded", False)
    classifier = getattr(request.app.state, "classifier", None)

    if not model_loaded or classifier is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded. Service is not ready for predictions.",
        )

    # Classify intent
    start_time = time.time()
    try:
        result = classifier.classify(body.text)
    except Exception as e:
        logger.error("Classification failed for text '%s': %s", body.text[:50], str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}",
        )

    # Generate contextual response
    intent = result["intent"]
    confidence = result["confidence"]
    response_text = response_generator.generate(
        intent=intent,
        confidence=confidence,
        original_text=body.text,
    )

    processing_time_ms = (time.time() - start_time) * 1000

    # Build all_intents list
    all_intents = [
        IntentScore(intent=item["intent"], score=item["score"])
        for item in result["all_intents"]
    ]

    logger.info(
        "Prediction: text='%s...' -> intent=%s (%.3f) in %.2f ms",
        body.text[:30],
        intent,
        confidence,
        processing_time_ms,
    )

    return PredictResponse(
        intent=intent,
        confidence=confidence,
        response=response_text,
        all_intents=all_intents,
        processing_time_ms=round(processing_time_ms, 2),
        model_name=settings.MODEL_NAME,
    )
