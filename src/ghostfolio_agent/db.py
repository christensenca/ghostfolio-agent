"""SQLite database lifecycle and schema initialization."""
from __future__ import annotations

import logging

import aiosqlite

from ghostfolio_agent.config import settings

logger = logging.getLogger(__name__)

_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    """Return a singleton async SQLite connection."""
    global _connection
    if _connection is None:
        _connection = await aiosqlite.connect(settings.db_path)
        _connection.row_factory = aiosqlite.Row
        await _connection.execute("PRAGMA journal_mode=WAL")
        await _connection.execute("PRAGMA foreign_keys=ON")
    return _connection


async def init_db():
    """Create tables if they don't exist."""
    db = await get_db()
    await db.executescript(_SCHEMA_SQL)
    await db.commit()
    logger.info("SQLite database initialized at %s", settings.db_path)


async def close_db():
    """Close the database connection."""
    global _connection
    if _connection:
        await _connection.close()
        _connection = None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
"""
