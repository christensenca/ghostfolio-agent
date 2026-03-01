"""Authentication: exchange Ghostfolio security token for JWT."""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when JWT exchange fails."""


async def exchange_token(api_url: str, access_token: str) -> str:
    """Exchange a Ghostfolio security token for a JWT.

    Args:
        api_url: Ghostfolio API base URL (e.g. http://localhost:3333)
        access_token: Security token from Ghostfolio Settings page

    Returns:
        JWT string for authenticated API calls

    Raises:
        AuthenticationError: If the exchange fails
    """
    url = "{base}/api/v1/auth/anonymous".format(base=api_url.rstrip("/"))
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json={"accessToken": access_token})

    if resp.status_code != 201:
        raise AuthenticationError(
            "Auth failed (HTTP {code}): {body}".format(
                code=resp.status_code, body=resp.text[:200]
            )
        )

    jwt = resp.json().get("authToken")
    if not jwt:
        raise AuthenticationError(
            "Auth response missing authToken: {body}".format(body=resp.text[:200])
        )

    logger.info("Successfully exchanged security token for JWT")
    return jwt
