"""Comprehensive tests for the BYOAI API Gateway.

Uses pytest-asyncio with httpx AsyncClient and ASGITransport
to test all gateway endpoints without starting a real server.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from app.main import app
from app.routes.conversation import conversation_manager


@pytest_asyncio.fixture
async def client():
    """Create an async test client for the FastAPI app.

    Manually initializes app.state since ASGITransport does not
    trigger FastAPI lifespan events.
    """
    # Set up app state that lifespan would normally create
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    app.state.start_time = time.time()

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    # Teardown
    await app.state.http_client.aclose()


@pytest_asyncio.fixture(autouse=True)
async def clear_conversation_store():
    """Clear the conversation store between tests."""
    yield
    async with conversation_manager._lock:
        conversation_manager._store.clear()


# ---------------------------------------------------------------------------
# 1. Health endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(client: httpx.AsyncClient):
    """Test that the health endpoint returns 200 with expected fields."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "BYOAI Gateway"
    assert data["version"] == "1.0.0"
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_health_ready_ml_down(client: httpx.AsyncClient):
    """Test readiness check reports 'degraded' when ML service is down."""
    response = await client.get("/health/ready")
    assert response.status_code == 200

    data = response.json()
    # ML service is not running in tests, so status should be degraded
    assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# 2. Root endpoint test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_endpoint(client: httpx.AsyncClient):
    """Test that the root endpoint returns service info."""
    response = await client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "BYOAI Gateway"
    assert data["version"] == "1.0.0"
    assert "docs" in data


# ---------------------------------------------------------------------------
# 3. Chat endpoint tests (with mocked ML service)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_endpoint_success(client: httpx.AsyncClient):
    """Test chat endpoint with a mocked ML service response."""
    mock_prediction = {
        "intent": "greeting",
        "confidence": 0.95,
        "response": "Hello! How can I help you?",
    }

    with patch(
        "app.services.ml_client.MLServiceClient.predict",
        new_callable=AsyncMock,
        return_value=mock_prediction,
    ):
        response = await client.post(
            "/api/v1/chat",
            json={"message": "Hello there!", "session_id": "test-session-001"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-001"
    assert data["message"] == "Hello there!"
    assert data["intent"] == "greeting"
    assert data["confidence"] == 0.95
    assert data["response"] == "Hello! How can I help you?"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_chat_auto_generates_session_id(client: httpx.AsyncClient):
    """Test that session_id is auto-generated when not provided."""
    mock_prediction = {
        "intent": "greeting",
        "confidence": 0.9,
        "response": "Hi!",
    }

    with patch(
        "app.services.ml_client.MLServiceClient.predict",
        new_callable=AsyncMock,
        return_value=mock_prediction,
    ):
        response = await client.post(
            "/api/v1/chat",
            json={"message": "Hey"},
        )

    assert response.status_code == 200
    data = response.json()
    # session_id should be a valid UUID4 string
    assert len(data["session_id"]) == 36  # UUID4 format: 8-4-4-4-12


@pytest.mark.asyncio
async def test_chat_empty_message_rejected(client: httpx.AsyncClient):
    """Test that an empty message is rejected with 422."""
    response = await client.post(
        "/api/v1/chat",
        json={"message": ""},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 4. Conversation history tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_history(client: httpx.AsyncClient):
    """Test that chat messages are stored and retrievable via history."""
    mock_prediction = {
        "intent": "weather",
        "confidence": 0.88,
        "response": "It looks sunny today!",
    }

    with patch(
        "app.services.ml_client.MLServiceClient.predict",
        new_callable=AsyncMock,
        return_value=mock_prediction,
    ):
        await client.post(
            "/api/v1/chat",
            json={"message": "What's the weather?", "session_id": "history-test"},
        )

    response = await client.get("/api/v1/history/history-test")
    assert response.status_code == 200

    history = response.json()
    assert len(history) == 2  # user + assistant

    assert history[0]["role"] == "user"
    assert history[0]["message"] == "What's the weather?"

    assert history[1]["role"] == "assistant"
    assert history[1]["message"] == "It looks sunny today!"
    assert history[1]["intent"] == "weather"
    assert history[1]["confidence"] == 0.88


@pytest.mark.asyncio
async def test_history_not_found(client: httpx.AsyncClient):
    """Test that requesting history for unknown session returns 404."""
    response = await client.get("/api/v1/history/nonexistent-session")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_history(client: httpx.AsyncClient):
    """Test that deleting conversation history works correctly."""
    mock_prediction = {
        "intent": "test",
        "confidence": 0.5,
        "response": "OK",
    }

    with patch(
        "app.services.ml_client.MLServiceClient.predict",
        new_callable=AsyncMock,
        return_value=mock_prediction,
    ):
        await client.post(
            "/api/v1/chat",
            json={"message": "Test message", "session_id": "delete-test"},
        )

    # Delete the history
    response = await client.delete("/api/v1/history/delete-test")
    assert response.status_code == 200

    # Verify it's gone
    response = await client.get("/api/v1/history/delete-test")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_session(client: httpx.AsyncClient):
    """Test that deleting a non-existent session returns 404."""
    response = await client.delete("/api/v1/history/does-not-exist")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 5. Rate limiting tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiting(client: httpx.AsyncClient):
    """Test that rate limiting returns 429 after exceeding the limit.

    Overrides the rate limiter to a very low limit (2 requests/60s)
    to verify the mechanism without sending 60+ requests.
    """
    from app.middleware.rate_limiter import RateLimiter, rate_limiter

    # Save original settings and set a very low limit
    original_max = rate_limiter.max_requests
    original_window = rate_limiter.window_seconds

    rate_limiter.max_requests = 2
    rate_limiter.window_seconds = 60
    rate_limiter._buckets.clear()

    mock_prediction = {
        "intent": "test",
        "confidence": 0.5,
        "response": "OK",
    }

    try:
        with patch(
            "app.services.ml_client.MLServiceClient.predict",
            new_callable=AsyncMock,
            return_value=mock_prediction,
        ):
            # First 2 requests should succeed
            for i in range(2):
                response = await client.post(
                    "/api/v1/chat",
                    json={"message": f"msg {i}", "session_id": f"rate-test-{i}"},
                )
                assert response.status_code == 200, f"Request {i} should succeed"

            # Third request should be rate limited
            response = await client.post(
                "/api/v1/chat",
                json={"message": "should fail", "session_id": "rate-test-blocked"},
            )
            assert response.status_code == 429
            assert "Retry-After" in response.headers
    finally:
        # Restore original settings
        rate_limiter.max_requests = original_max
        rate_limiter.window_seconds = original_window
        rate_limiter._buckets.clear()


# ---------------------------------------------------------------------------
# 6. Request ID middleware test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_id_header(client: httpx.AsyncClient):
    """Test that every response includes an X-Request-ID header."""
    response = await client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 36  # UUID4 format


@pytest.mark.asyncio
async def test_request_id_passthrough(client: httpx.AsyncClient):
    """Test that an existing X-Request-ID is preserved in the response."""
    custom_id = "custom-request-id-12345"
    response = await client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers["X-Request-ID"] == custom_id
