"""Session-based state management for Interactive Mode.

This module provides thread-safe, session-isolated storage for LLMTracer instances.
It eliminates state confusion bugs caused by Gradio's state management of complex objects.
"""

from __future__ import annotations
from typing import Optional, Dict
import threading
import uuid
import time
from datetime import datetime, timedelta
from miru_tracer.core.tracer import LLMTracer
from miru_tracer.core.logging_config import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Thread-safe session manager for LLMTracer instances.

    Each session is identified by a unique session_id and has its own:
    - LLMTracer instance
    - Thread lock for synchronization
    - Last access timestamp for cleanup

    This prevents race conditions and state corruption from rapid user interactions.
    """

    def __init__(self, cleanup_timeout_minutes: int = 30):
        """
        Initialize the session manager.

        Args:
            cleanup_timeout_minutes: Sessions inactive for this duration will be cleaned up
        """
        self._sessions: Dict[str, Dict] = {}
        self._global_lock = threading.Lock()
        self._cleanup_timeout = timedelta(minutes=cleanup_timeout_minutes)

    def create_session(self, model, tokenizer, device) -> str:
        """
        Create a new session with a fresh tracer instance.

        Args:
            model: The LLM model
            tokenizer: The tokenizer
            device: Device (cuda/cpu)

        Returns:
            session_id: Unique identifier for this session
        """
        session_id = str(uuid.uuid4())

        with self._global_lock:
            # Enforce single session: clear any existing sessions first
            # We do this manually here to avoid re-acquiring the lock if we called clear_all_sessions
            existing_ids = list(self._sessions.keys())
            for old_id in existing_ids:
                # We can't call self.delete_session because it acquires the lock
                # So we duplicate the cleanup logic here for the single-session enforcement
                old_session = self._sessions[old_id]
                old_tracer = old_session["tracer"]
                old_tracer.past_key_values = None
                old_tracer.input_ids = None
                del self._sessions[old_id]
                logger.info(f"Auto-cleared old session: {old_id}")

            tracer = LLMTracer(model, tokenizer, device)
            self._sessions[session_id] = {
                "tracer": tracer,
                "lock": threading.Lock(),
                "last_access": datetime.now(),
                "created": datetime.now(),
            }

        logger.info(f"Session created: {session_id} (device={device})")
        logger.debug(f"Active sessions: {len(self._sessions)}")

        return session_id

    def get_tracer(self, session_id: str) -> Optional[LLMTracer]:
        """
        Get the tracer for a session.

        Args:
            session_id: Session identifier

        Returns:
            LLMTracer instance or None if session doesn't exist
        """
        with self._global_lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(f"Session not found: {session_id}")
                return None

            # Update last access time
            session["last_access"] = datetime.now()
            logger.debug(f"Session accessed: {session_id}")
            return session["tracer"]

    def get_lock(self, session_id: str) -> Optional[threading.Lock]:
        """
        Get the lock for a session.

        Use this to synchronize access to the tracer within a session.

        Args:
            session_id: Session identifier

        Returns:
            Thread lock or None if session doesn't exist
        """
        with self._global_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session["lock"]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and clean up resources.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if it didn't exist
        """
        with self._global_lock:
            if session_id in self._sessions:
                # Clean up tracer resources
                session = self._sessions[session_id]
                tracer = session["tracer"]
                # Clear PyTorch tensors
                tracer.past_key_values = None
                tracer.input_ids = None

                # Calculate session lifetime
                lifetime = datetime.now() - session["created"]
                lifetime_seconds = lifetime.total_seconds()

                del self._sessions[session_id]

                logger.info(
                    f"Session deleted: {session_id} (lifetime={lifetime_seconds:.1f}s)"
                )
                logger.debug(f"Active sessions: {len(self._sessions)}")
                return True

            logger.warning(f"Cannot delete session (not found): {session_id}")
            return False

    def cleanup_old_sessions(self) -> int:
        """
        Remove sessions that haven't been accessed recently.

        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        to_delete = []

        with self._global_lock:
            for session_id, session in self._sessions.items():
                if now - session["last_access"] > self._cleanup_timeout:
                    to_delete.append(session_id)

        # Delete outside the lock to avoid holding it too long
        for session_id in to_delete:
            self.delete_session(session_id)

        if len(to_delete) > 0:
            logger.info(
                f"Cleaned up {len(to_delete)} old sessions (timeout={self._cleanup_timeout.total_seconds()/60:.0f}min)"
            )

        return len(to_delete)

    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        with self._global_lock:
            return len(self._sessions)

    def clear_all_sessions(self) -> int:
        """
        Clear all sessions (useful before unloading model).

        Returns:
            Number of sessions cleared
        """
        with self._global_lock:
            session_ids = list(self._sessions.keys())

        # Delete outside the lock
        for session_id in session_ids:
            self.delete_session(session_id)

        count = len(session_ids)
        if count > 0:
            logger.info(f"Cleared all {count} session(s)")

        return count

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session info or None if session doesn't exist
        """
        with self._global_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            tracer = session["tracer"]
            return {
                "session_id": session_id,
                "created": session["created"].isoformat(),
                "last_access": session["last_access"].isoformat(),
                "steps": len(tracer.history),
                "mode": tracer.mode,
                "prompt": (
                    tracer.prompt[:100] + "..."
                    if len(tracer.prompt) > 100
                    else tracer.prompt
                ),
            }


# Global singleton instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get the global session manager instance (singleton pattern).

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
