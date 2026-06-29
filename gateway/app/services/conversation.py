"""In-memory conversation history manager.

Provides thread-safe storage and retrieval of conversation entries
per session, using asyncio locks to prevent race conditions.
"""

import asyncio
import logging
from typing import Optional

from app.models import ConversationEntry

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages per-session conversation history in memory.

    Uses asyncio.Lock for thread-safe access to the shared
    conversation store. Suitable for single-instance deployments;
    for multi-instance, replace with Redis or a database backend.

    Attributes:
        _store: Mapping of session_id to list of ConversationEntry.
        _lock: Asyncio lock for thread-safe mutations.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[ConversationEntry]] = {}
        self._lock = asyncio.Lock()

    async def add_entry(
        self,
        session_id: str,
        role: str,
        message: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> ConversationEntry:
        """Add a conversation entry to a session.

        Args:
            session_id: The session to add the entry to.
            role: Either 'user' or 'assistant'.
            message: The message content.
            intent: Detected intent (assistant entries only).
            confidence: Model confidence (assistant entries only).

        Returns:
            The created ConversationEntry.
        """
        entry = ConversationEntry(
            role=role,
            message=message,
            intent=intent,
            confidence=confidence,
        )
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append(entry)

        logger.debug(
            "Added %s entry to session %s (total: %d)",
            role,
            session_id,
            len(self._store[session_id]),
        )
        return entry

    async def get_history(self, session_id: str) -> list[ConversationEntry]:
        """Retrieve the full conversation history for a session.

        Args:
            session_id: The session to retrieve history for.

        Returns:
            List of ConversationEntry objects, or empty list if session not found.
        """
        async with self._lock:
            entries = self._store.get(session_id, [])
            # Return a copy to prevent external mutation
            return list(entries)

    async def clear_history(self, session_id: str) -> bool:
        """Clear all conversation history for a session.

        Args:
            session_id: The session to clear.

        Returns:
            True if the session existed and was cleared, False if not found.
        """
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.info("Cleared history for session %s", session_id)
                return True
            logger.warning("Attempted to clear non-existent session %s", session_id)
            return False

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists in the store.

        Args:
            session_id: The session to check.

        Returns:
            True if the session exists, False otherwise.
        """
        async with self._lock:
            return session_id in self._store
