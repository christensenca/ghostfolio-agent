"""Tests for FastAPI endpoints — written BEFORE implementation (TDD)."""

from unittest.mock import patch


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_returns_response(client):
    with patch("ghostfolio_agent.server.run_agent") as mock_agent:
        mock_agent.return_value = {
            "message": "Your portfolio has 5 holdings.",
            "conversation_id": "abc-123",
            "tool_calls": [],
            "confidence": 0.95,
        }

        response = client.post(
            "/chat",
            json={
                "message": "Show my portfolio",
                "jwt": "fake-jwt-token",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "conversation_id" in data
    assert "confidence" in data


def test_chat_with_conversation_id(client):
    with patch("ghostfolio_agent.server.run_agent") as mock_agent:
        mock_agent.return_value = {
            "message": "Here is a follow-up.",
            "conversation_id": "existing-id",
            "tool_calls": [],
            "confidence": 0.9,
        }

        response = client.post(
            "/chat",
            json={
                "message": "Tell me more",
                "conversation_id": "existing-id",
                "jwt": "fake-jwt-token",
            },
        )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "existing-id"


def test_chat_missing_message_returns_422(client):
    response = client.post(
        "/chat",
        json={"jwt": "fake-jwt-token"},
    )
    assert response.status_code == 422


def test_chat_missing_jwt_returns_401(client):
    """JWT is optional in request but requires settings fallback; returns 401 if neither exists."""
    response = client.post(
        "/chat",
        json={"message": "hello"},
    )
    assert response.status_code == 401


def test_chat_empty_message_returns_422(client):
    response = client.post(
        "/chat",
        json={"message": "", "jwt": "fake-jwt-token"},
    )
    assert response.status_code == 422
