"""Tests for ghostfolio_agent.auth module."""
import pytest
import httpx
import respx

from ghostfolio_agent.auth import exchange_token, AuthenticationError


@pytest.mark.asyncio
async def test_exchange_token_success():
    with respx.mock:
        respx.post("http://localhost:3333/api/v1/auth/anonymous").mock(
            return_value=httpx.Response(201, json={"authToken": "jwt-token-123"})
        )
        jwt = await exchange_token("http://localhost:3333", "my-security-token")
        assert jwt == "jwt-token-123"


@pytest.mark.asyncio
async def test_exchange_token_bad_status():
    with respx.mock:
        respx.post("http://localhost:3333/api/v1/auth/anonymous").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )
        with pytest.raises(AuthenticationError, match="Auth failed"):
            await exchange_token("http://localhost:3333", "bad-token")


@pytest.mark.asyncio
async def test_exchange_token_missing_auth_token():
    with respx.mock:
        respx.post("http://localhost:3333/api/v1/auth/anonymous").mock(
            return_value=httpx.Response(201, json={"something": "else"})
        )
        with pytest.raises(AuthenticationError, match="missing authToken"):
            await exchange_token("http://localhost:3333", "my-token")


@pytest.mark.asyncio
async def test_exchange_token_strips_trailing_slash():
    with respx.mock:
        respx.post("http://localhost:3333/api/v1/auth/anonymous").mock(
            return_value=httpx.Response(201, json={"authToken": "jwt-123"})
        )
        jwt = await exchange_token("http://localhost:3333/", "my-token")
        assert jwt == "jwt-123"
