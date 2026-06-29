"""
Pydantic models for request/response schemas in the BYOAI ML Service.

Defines strict validation for prediction requests and structured responses
including intent scores, confidence metrics, and health status.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for intent prediction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The input text to classify intent for.",
        examples=["Hello, I need help with my booking"],
    )


class IntentScore(BaseModel):
    """A single intent with its classification score."""

    intent: str = Field(..., description="The intent label.")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this intent.")


class PredictResponse(BaseModel):
    """Response schema for intent prediction."""

    intent: str = Field(..., description="The top predicted intent.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the top intent.")
    response: str = Field(..., description="Generated contextual response for the user.")
    all_intents: list[IntentScore] = Field(
        ..., description="All candidate intents with their scores."
    )
    processing_time_ms: float = Field(..., description="Time taken for prediction in milliseconds.")
    model_name: str = Field(..., description="Name of the ML model used for classification.")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service health status.")
    service: str = Field(..., description="Service name.")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready.")
    model_name: str = Field(..., description="Name of the ML model configured.")
    version: str = Field(default="1.0.0", description="Service version.")
