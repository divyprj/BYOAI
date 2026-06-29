"""Token bucket rate limiter for FastAPI.

Implements per-client-IP rate limiting using the token bucket algorithm.
Designed to be used as a FastAPI dependency.
"""

import time
import logging
from collections import defaultdict
from typing import NamedTuple

from fastapi import HTTPException, Request, status

from app.config import settings

logger = logging.getLogger(__name__)


class _Bucket(NamedTuple):
    """Internal token bucket state."""
    tokens: float
    last_refill: float


class RateLimiter:
    """Token bucket rate limiter.

    Each client IP gets a bucket that refills at a steady rate.
    When the bucket is empty, requests are rejected with HTTP 429.

    Attributes:
        max_requests: Maximum tokens (requests) per window.
        window_seconds: Time window for the rate limit in seconds.
    """

    def __init__(
        self,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        self.max_requests = max_requests or settings.RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or settings.RATE_LIMIT_WINDOW
        self._buckets: dict[str, _Bucket] = defaultdict(
            lambda: _Bucket(tokens=self.max_requests, last_refill=time.monotonic())
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract the client IP from the request.

        Checks X-Forwarded-For for reverse proxy setups, falls back
        to the direct client host.

        Args:
            request: The incoming FastAPI request.

        Returns:
            Client IP address string.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _refill(self, bucket: _Bucket) -> _Bucket:
        """Refill tokens based on elapsed time since last refill.

        Args:
            bucket: Current bucket state.

        Returns:
            Updated bucket with refilled tokens.
        """
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        refill_rate = self.max_requests / self.window_seconds
        new_tokens = min(self.max_requests, bucket.tokens + elapsed * refill_rate)
        return _Bucket(tokens=new_tokens, last_refill=now)

    async def __call__(self, request: Request) -> None:
        """FastAPI dependency that enforces the rate limit.

        Args:
            request: The incoming request.

        Raises:
            HTTPException: 429 Too Many Requests when rate limit exceeded.
        """
        client_ip = self._get_client_ip(request)
        bucket = self._refill(self._buckets[client_ip])

        if bucket.tokens < 1.0:
            # Calculate how long the client should wait
            refill_rate = self.max_requests / self.window_seconds
            retry_after = int((1.0 - bucket.tokens) / refill_rate) + 1

            logger.warning(
                "Rate limit exceeded for %s (tokens: %.2f)",
                client_ip,
                bucket.tokens,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(retry_after)},
            )

        # Consume one token
        self._buckets[client_ip] = _Bucket(
            tokens=bucket.tokens - 1.0,
            last_refill=bucket.last_refill,
        )


# Shared rate limiter instance used as a FastAPI dependency
rate_limiter = RateLimiter()
