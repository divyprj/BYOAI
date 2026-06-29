"""
Tests for the BYOAI ML Service.

Uses pytest with httpx AsyncClient and ASGITransport to test endpoints.
The HuggingFace pipeline is mocked to avoid loading the real model in tests.
"""

from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.response_generator import ResponseGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_pipeline_result():
    """Create a mock result that mimics HuggingFace zero-shot pipeline output."""
    return {
        "labels": [
            "greeting", "help", "question", "feedback", "farewell",
            "booking", "complaint", "cancel", "status_check", "out_of_scope",
        ],
        "scores": [
            0.85, 0.05, 0.03, 0.02, 0.01,
            0.01, 0.01, 0.01, 0.005, 0.005,
        ],
    }


@pytest_asyncio.fixture
async def client():
    """Create a test client with a mocked ML pipeline."""
    with patch("app.main.IntentClassifier") as MockClassifier:
        # Configure the mock classifier instance
        mock_instance = MagicMock()
        mock_instance.classify.return_value = {
            "intent": "greeting",
            "confidence": 0.85,
            "all_intents": [
                {"intent": "greeting", "score": 0.85},
                {"intent": "help", "score": 0.05},
                {"intent": "question", "score": 0.03},
                {"intent": "feedback", "score": 0.02},
                {"intent": "farewell", "score": 0.01},
                {"intent": "booking", "score": 0.01},
                {"intent": "complaint", "score": 0.01},
                {"intent": "cancel", "score": 0.01},
                {"intent": "status_check", "score": 0.005},
                {"intent": "out_of_scope", "score": 0.005},
            ],
        }
        MockClassifier.return_value = mock_instance

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# Tests: Root Endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test the root endpoint returns service metadata."""
    response = await client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "BYOAI ML Service"
    assert data["version"] == "1.0.0"
    assert "docs" in data
    assert "health" in data


# ---------------------------------------------------------------------------
# Tests: Health Endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Test the health endpoint returns model status."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["service"] == "BYOAI ML Service"
    assert "model_loaded" in data
    assert "model_name" in data
    assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Tests: Predict Endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_endpoint_success(client: AsyncClient):
    """Test successful intent prediction with valid input."""
    response = await client.post(
        "/api/v1/predict",
        json={"text": "Hello, I need some help"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["intent"] == "greeting"
    assert data["confidence"] == 0.85
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0
    assert isinstance(data["all_intents"], list)
    assert len(data["all_intents"]) > 0
    assert data["processing_time_ms"] >= 0
    assert data["model_name"] == "facebook/bart-large-mnli"


@pytest.mark.asyncio
async def test_predict_endpoint_empty_text(client: AsyncClient):
    """Test that empty text is rejected with 422."""
    response = await client.post(
        "/api/v1/predict",
        json={"text": ""},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_too_long_text(client: AsyncClient):
    """Test that text exceeding max_length is rejected with 422."""
    long_text = "a" * 1001
    response = await client.post(
        "/api/v1/predict",
        json={"text": long_text},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_missing_text(client: AsyncClient):
    """Test that missing text field is rejected with 422."""
    response = await client.post(
        "/api/v1/predict",
        json={},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_max_length_text(client: AsyncClient):
    """Test that text at exactly max_length (1000) is accepted."""
    text = "a" * 1000
    response = await client.post(
        "/api/v1/predict",
        json={"text": text},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests: Response Generator
# ---------------------------------------------------------------------------


class TestResponseGenerator:
    """Tests for the ResponseGenerator service."""

    def test_generate_known_intent(self):
        """Test that known intents produce valid responses."""
        generator = ResponseGenerator()
        for intent in ["greeting", "farewell", "complaint", "booking", "help"]:
            response = generator.generate(
                intent=intent, confidence=0.9, original_text="test"
            )
            assert isinstance(response, str)
            assert len(response) > 0

    def test_generate_out_of_scope(self):
        """Test that out_of_scope intent returns a fallback response."""
        generator = ResponseGenerator()
        response = generator.generate(
            intent="out_of_scope", confidence=0.1, original_text="xyz"
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_unknown_intent_fallback(self):
        """Test that an unknown intent falls back gracefully."""
        generator = ResponseGenerator()
        response = generator.generate(
            intent="nonexistent_intent", confidence=0.5, original_text="test"
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_all_intents_have_templates(self):
        """Test that all candidate labels have response templates."""
        from app.services.intent_classifier import CANDIDATE_LABELS
        from app.services.response_generator import RESPONSE_TEMPLATES

        for label in CANDIDATE_LABELS:
            assert label in RESPONSE_TEMPLATES, f"Missing template for intent: {label}"
            assert len(RESPONSE_TEMPLATES[label]) >= 3, (
                f"Intent '{label}' has fewer than 3 templates"
            )

    def test_generate_variety(self):
        """Test that the generator produces varied responses across calls."""
        generator = ResponseGenerator()
        responses = set()
        for _ in range(20):
            response = generator.generate(
                intent="greeting", confidence=0.9, original_text="hello"
            )
            responses.add(response)
        # With 5 templates and 20 draws, we should see at least 2 unique responses
        assert len(responses) >= 2, "Response generator lacks variety"
