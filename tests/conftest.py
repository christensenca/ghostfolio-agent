import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

import ghostfolio_agent.db as db_module
from ghostfolio_agent.config import settings
from ghostfolio_agent.server import app


@pytest.fixture(autouse=True)
def _use_in_memory_db(monkeypatch):
    """Use an in-memory SQLite database for all tests."""
    monkeypatch.setattr(settings, "db_path", ":memory:")
    # Reset the singleton connection so each test gets a fresh DB
    db_module._connection = None


@pytest.fixture
def client():
    """Synchronous test client for FastAPI."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture
async def async_client():
    """Async test client for FastAPI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
