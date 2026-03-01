"""Ghostfolio AI Agent - AI-powered financial assistant for Ghostfolio portfolios."""

__version__ = "0.1.0"

from ghostfolio_agent.auth import AuthenticationError, exchange_token
from ghostfolio_agent.config import Settings

__all__ = [
    "__version__",
    "AuthenticationError",
    "exchange_token",
    "Settings",
]
