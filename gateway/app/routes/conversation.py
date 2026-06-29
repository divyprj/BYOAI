"""Conversation and chat endpoints.

Handles user chat messages by forwarding them to the ML service,
storing conversation history, and returning structured responses.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.middleware.rate_limiter import rate_limiter
from app.models import ChatRequest, ChatResponse, ConversationEntry, ErrorResponse
from app.services.conversation import ConversationManager
from app.services.ml_client import MLServiceClient, MLServiceError, MLServiceTimeoutError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Conversation"])

# Shared conversation manager (in-memory, single-instance)
conversation_manager = ConversationManager()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="Processes a user message through the ML service and returns an intent-classified response.",
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        502: {"model": ErrorResponse, "description": "ML service error"},
        504: {"model": ErrorResponse, "description": "ML service timeout"},
    },
    dependencies=[Depends(rate_limiter)],
)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """Process a user chat message.

    1. Records the user message in conversation history.
    2. Forwards the message to the ML service for intent prediction.
    3. Records the assistant response in conversation history.
    4. Returns the full ChatResponse.

    Args:
        request: The FastAPI request (for accessing app.state).
        chat_request: The validated chat request body.

    Returns:
        ChatResponse with intent, confidence, and generated response.
    """
    session_id = chat_request.session_id
    logger.info("Chat request: session=%s message=%r", session_id, chat_request.message[:100])

    # Store user message
    await conversation_manager.add_entry(
        session_id=session_id,
        role="user",
        message=chat_request.message,
    )

    # Call ML service
    ml_client = MLServiceClient(request.app.state.http_client)
    try:
        prediction = await ml_client.predict(chat_request.message)
    except MLServiceTimeoutError as exc:
        logger.error("ML service timeout for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="ML service timed out. Please try again later.",
        ) from exc
    except MLServiceError as exc:
        logger.error("ML service error for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ML service is currently unavailable.",
        ) from exc

    intent = prediction.get("intent", "unknown")
    confidence = prediction.get("confidence", 0.0)
    response_text = prediction.get("response", "I'm not sure how to respond to that.")

    # Store assistant response
    await conversation_manager.add_entry(
        session_id=session_id,
        role="assistant",
        message=response_text,
        intent=intent,
        confidence=confidence,
    )

    chat_response = ChatResponse(
        session_id=session_id,
        message=chat_request.message,
        intent=intent,
        confidence=confidence,
        response=response_text,
    )

    logger.info(
        "Chat response: session=%s intent=%s confidence=%.3f",
        session_id,
        intent,
        confidence,
    )
    return chat_response


@router.get(
    "/history/{session_id}",
    response_model=list[ConversationEntry],
    summary="Get conversation history",
    description="Retrieves the full conversation history for a session.",
    responses={404: {"model": ErrorResponse, "description": "Session not found"}},
    dependencies=[Depends(rate_limiter)],
)
async def get_history(session_id: str) -> list[ConversationEntry]:
    """Retrieve conversation history for a session.

    Args:
        session_id: The session identifier.

    Returns:
        List of ConversationEntry objects in chronological order.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    history = await conversation_manager.get_history(session_id)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conversation history found for session '{session_id}'.",
        )
    return history


@router.delete(
    "/history/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear conversation history",
    description="Deletes all conversation history for a session.",
    responses={404: {"model": ErrorResponse, "description": "Session not found"}},
    dependencies=[Depends(rate_limiter)],
)
async def delete_history(session_id: str) -> dict[str, str]:
    """Clear conversation history for a session.

    Args:
        session_id: The session identifier to clear.

    Returns:
        Confirmation message.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    cleared = await conversation_manager.clear_history(session_id)
    if not cleared:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conversation history found for session '{session_id}'.",
        )
    logger.info("Cleared conversation history for session %s", session_id)
    return {"message": f"Conversation history for session '{session_id}' cleared."}
