"""ML Service client with retry logic and error handling.

Communicates with the downstream ML model service to get
intent predictions and generated responses for user messages.
"""

import asyncio
import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_RETRY_COUNT = 3
_RETRY_BASE_DELAY = 0.5  # seconds
_REQUEST_TIMEOUT = 10.0  # seconds


class MLServiceError(Exception):
    """Raised when the ML service returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MLServiceTimeoutError(MLServiceError):
    """Raised when the ML service request times out."""


class MLServiceClient:
    """Client for the ML model prediction service.

    Uses the shared httpx.AsyncClient from app.state for connection
    pooling and lifecycle management.

    Attributes:
        client: The httpx.AsyncClient instance.
        base_url: Base URL of the ML service.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client
        self.base_url = settings.ML_SERVICE_URL

    async def predict(self, text: str) -> dict[str, Any]:
        """Send text to the ML service for intent prediction.

        Implements retry logic with exponential backoff (3 attempts).

        Args:
            text: The user message to classify.

        Returns:
            Dict containing 'intent', 'confidence', and 'response' keys.

        Raises:
            MLServiceTimeoutError: If all retries time out.
            MLServiceError: If the ML service returns an error.
        """
        url = f"{self.base_url}/api/v1/predict"
        last_exception: Exception | None = None

        for attempt in range(1, _RETRY_COUNT + 1):
            try:
                logger.info(
                    "ML service request attempt %d/%d to %s",
                    attempt,
                    _RETRY_COUNT,
                    url,
                )
                response = await self.client.post(
                    url,
                    json={"text": text},
                    timeout=_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                logger.info(
                    "ML service responded: intent=%s confidence=%.3f",
                    data.get("intent"),
                    data.get("confidence", 0.0),
                )
                return data

            except httpx.TimeoutException as exc:
                last_exception = exc
                logger.warning(
                    "ML service timeout on attempt %d/%d: %s",
                    attempt,
                    _RETRY_COUNT,
                    exc,
                )

            except httpx.HTTPStatusError as exc:
                last_exception = exc
                status = exc.response.status_code
                # Don't retry on client errors (4xx) except 429
                if 400 <= status < 500 and status != 429:
                    raise MLServiceError(
                        f"ML service client error: {status}",
                        status_code=status,
                    ) from exc
                logger.warning(
                    "ML service error on attempt %d/%d: %s",
                    attempt,
                    _RETRY_COUNT,
                    exc,
                )

            except httpx.RequestError as exc:
                last_exception = exc
                logger.warning(
                    "ML service connection error on attempt %d/%d: %s",
                    attempt,
                    _RETRY_COUNT,
                    exc,
                )

            # Exponential backoff before next retry
            if attempt < _RETRY_COUNT:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info("Retrying in %.1fs...", delay)
                await asyncio.sleep(delay)

        # All retries exhausted
        if isinstance(last_exception, httpx.TimeoutException):
            raise MLServiceTimeoutError(
                f"ML service timed out after {_RETRY_COUNT} attempts"
            ) from last_exception

        raise MLServiceError(
            f"ML service unavailable after {_RETRY_COUNT} attempts: {last_exception}"
        ) from last_exception

    async def health_check(self) -> bool:
        """Check if the ML service is healthy.

        Returns:
            True if the ML service responds to /health, False otherwise.
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False
