"""Persistent conversation store backed by SQLite."""
from __future__ import annotations

import time
import uuid
from typing import List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ghostfolio_agent.config import settings
from ghostfolio_agent.db import get_db


class ConversationStore:
    """SQLite-backed conversation history with TTL-based cleanup."""

    async def get_or_create(
        self, conversation_id: str | None
    ) -> Tuple[str, List[BaseMessage]]:
        await self._cleanup()
        db = await get_db()

        if conversation_id:
            cursor = await db.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            )
            if await cursor.fetchone():
                return conversation_id, await self._load_messages(conversation_id)

        new_id = str(uuid.uuid4())
        now = time.time()
        await db.execute(
            "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
            (new_id, now, now),
        )
        await db.commit()
        return new_id, []

    async def add_message(self, conversation_id: str, role: str, content: str):
        db = await get_db()
        now = time.time()
        await db.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await db.commit()

    async def _load_messages(self, conversation_id: str) -> List[BaseMessage]:
        db = await get_db()
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        messages: List[BaseMessage] = []
        for row in rows:
            if row[0] == "user":
                messages.append(HumanMessage(content=row[1]))
            else:
                messages.append(AIMessage(content=row[1]))
        return messages

    async def _cleanup(self):
        db = await get_db()
        cutoff = time.time() - settings.conversation_ttl_seconds
        await db.execute(
            "DELETE FROM conversations WHERE created_at < ?", (cutoff,)
        )
        await db.commit()


conversation_store = ConversationStore()
