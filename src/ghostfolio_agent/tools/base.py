"""Base tool interface for all Ghostfolio agent tools."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import httpx

from ghostfolio_agent.config import settings

logger = logging.getLogger(__name__)

# Persistent HTTP client with connection pooling for Ghostfolio API calls.
# Reused across all tool instances to avoid per-request connection overhead.
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=10.0)
    return _http_client


class GhostfolioTool(ABC):
    """Base class for tools that call the Ghostfolio API."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used for LLM tool selection."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description shown to the LLM for tool selection."""

    @abstractmethod
    async def execute(self, jwt: str, **kwargs) -> Dict[str, Any]:
        """Execute the tool and return results."""

    async def _api_get(self, path: str, jwt: str) -> httpx.Response:
        """Make an authenticated GET request to Ghostfolio API."""
        url = "{base}{path}".format(base=settings.ghostfolio_api_url, path=path)
        client = _get_http_client()
        try:
            response = await client.get(
                url,
                headers={"Authorization": "Bearer {jwt}".format(jwt=jwt)},
            )
            if response.status_code >= 400:
                logger.error(
                    "API call to %s failed with status %d: %s",
                    path,
                    response.status_code,
                    response.text[:500],
                )
            return response
        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ghostfolio API at %s — is the server running?", url
            )
            raise
        except httpx.TimeoutException:
            logger.error("Request to %s timed out after 10s", url)
            raise
