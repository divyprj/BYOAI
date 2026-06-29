"""Pydantic models for request/response validation and serialization."""

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from a user.

    Attributes:
        message: The user's chat message text.
        session_id: Optional session identifier; auto-generated if omitted.
    """

    message: str = Field(..., min_length=1, max_length=1000, description="User chat message")
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Session identifier (auto-generated UUID4 if not provided)",
    )


class ChatResponse(BaseModel):
    """Response returned after processing a chat message.

    Attributes:
        session_id: The session this response belongs to.
        message: The original user message echoed back.
        intent: Detected intent from the ML model.
        confidence: Model confidence score for the detected intent.
        response: Generated response text.
        timestamp: ISO-8601 timestamp of when the response was created.
    """

    session_id: str
    message: str
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationEntry(BaseModel):
    """A single entry in a conversation history.

    Attributes:
        role: Whether this entry is from the user or the assistant.
        message: The text content of the entry.
        timestamp: When this entry was recorded.
        intent: Detected intent (only for assistant entries).
        confidence: Model confidence (only for assistant entries).
    """

    role: Literal["user", "assistant"]
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    intent: Optional[str] = None
    confidence: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Current service status (e.g., 'healthy', 'degraded').
        service: Name of the service.
        version: Service version string.
        uptime_seconds: Seconds since service startup.
    """

    status: str
    service: str
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response body.

    Attributes:
        detail: Human-readable error description.
        error_code: Optional machine-readable error code.
    """

    detail: str
    error_code: Optional[str] = None
