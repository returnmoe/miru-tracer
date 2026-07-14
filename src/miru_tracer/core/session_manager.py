"""Session-based state management for Interactive Mode.

This module provides thread-safe, session-isolated storage for LLMTracer
instances. Gradio state only carries the session id (a string); the live
tracer object stays server-side, which avoids Gradio's serialization issues
with complex objects.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from miru_tracer.core.logging_config import get_logger
from miru_tracer.core.tracer import LLMTracer

logger = get_logger(__name__)


@dataclass
class Session:
    """A live tracer plus its synchronization lock and bookkeeping."""

    session_id: str
    tracer: LLMTracer
    kind: str = "interactive"
    model_generation: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    created: datetime = field(default_factory=datetime.now)
    last_access: datetime = field(default_factory=datetime.now)


class SessionManager:
    """
    Thread-safe session manager for LLMTracer instances.

    Each session is identified by a unique session_id and has its own
    LLMTracer, thread lock, and last-access timestamp. Expired sessions are
    reaped opportunistically whenever a new session is created.
    """

    def __init__(self, cleanup_timeout_minutes: int = 30):
        """
        Args:
            cleanup_timeout_minutes: Sessions inactive for this duration are
                removed on the next create_session call.
        """
        self._sessions: dict[str, Session] = {}
        self._global_lock = threading.Lock()
        self._cleanup_timeout = timedelta(minutes=cleanup_timeout_minutes)

    def create_session(
        self,
        model,
        tokenizer,
        device,
        *,
        kind: str = "interactive",
        model_generation: int = 0,
    ) -> str:
        """
        Create a new session with a fresh tracer instance.

        The UI is single-user, so a new session supersedes the previous session
        of the same kind. Interactive and Logging sessions may coexist.

        Returns:
            session_id: Unique identifier for the new session.
        """
        session_id = str(uuid.uuid4())
        tracer = LLMTracer(model, tokenizer, device)

        with self._global_lock:
            for old in list(self._sessions.values()):
                if old.kind == kind:
                    self._dispose_locked(old, reason="superseded")
            self._sessions[session_id] = Session(
                session_id=session_id,
                tracer=tracer,
                kind=kind,
                model_generation=model_generation,
            )

        logger.info(f"Session created: {session_id} (device={device})")
        return session_id

    def get_session(
        self, session_id: str, *, expected_generation: int | None = None
    ) -> Session | None:
        """
        Get a session (tracer + lock) in a single lock acquisition.

        Callers should hold ``session.lock`` while operating on the tracer.

        Returns:
            The Session, or None if it doesn't exist (expired or never created).
        """
        with self._global_lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(f"Session not found: {session_id}")
                return None
            if (
                expected_generation is not None
                and session.model_generation != expected_generation
            ):
                logger.warning(
                    f"Session generation mismatch: {session.model_generation} != "
                    f"{expected_generation}"
                )
                return None
            session.last_access = datetime.now()
            return session

    def get_tracer(self, session_id: str) -> LLMTracer | None:
        """Get the tracer for a session (prefer get_session for lock access)."""
        session = self.get_session(session_id)
        return session.tracer if session else None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and release its tensors.

        Returns:
            True if the session existed.
        """
        with self._global_lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(f"Cannot delete session (not found): {session_id}")
                return False
            self._dispose_locked(session, reason="deleted")
            return True

    def cleanup_old_sessions(self) -> int:
        """Remove sessions that haven't been accessed within the timeout."""
        now = datetime.now()
        with self._global_lock:
            expired = [
                s
                for s in self._sessions.values()
                if now - s.last_access > self._cleanup_timeout
            ]
            for session in expired:
                self._dispose_locked(session, reason="expired")

        if expired:
            logger.info(
                f"Cleaned up {len(expired)} old sessions "
                f"(timeout={self._cleanup_timeout.total_seconds() / 60:.0f}min)"
            )
        return len(expired)

    def clear_all_sessions(self) -> int:
        """Clear all sessions (useful before unloading the model)."""
        with self._global_lock:
            sessions = list(self._sessions.values())
            # Signal generators before waiting on their per-session locks.
            # request_stop only flips a threading.Event and is safe without
            # taking the session lock held by an active generation loop.
            for session in sessions:
                session.tracer.request_stop()
            for session in sessions:
                self._dispose_locked(session, reason="cleared")
        if sessions:
            logger.info(f"Cleared all {len(sessions)} session(s)")
        return len(sessions)

    def get_session_count(self) -> int:
        with self._global_lock:
            return len(self._sessions)

    def get_session_info(self, session_id: str) -> dict | None:
        """Get bookkeeping info about a session, or None if it doesn't exist."""
        with self._global_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            tracer = session.tracer
            return {
                "session_id": session_id,
                "kind": session.kind,
                "model_generation": session.model_generation,
                "created": session.created.isoformat(),
                "last_access": session.last_access.isoformat(),
                "steps": len(tracer.history),
                "mode": tracer.mode,
                "prompt": (
                    tracer.prompt[:100] + "..."
                    if len(tracer.prompt) > 100
                    else tracer.prompt
                ),
            }

    def _dispose_locked(self, session: Session, reason: str) -> None:
        """Drop a session and its tensors. Caller must hold the global lock.

        Takes the session lock so an in-flight tracer operation finishes
        before its tensors are released. Lock ordering is always global ->
        session (handlers never call manager methods while holding a session
        lock), so this cannot deadlock.
        """
        with session.lock:
            session.tracer.close()
        del self._sessions[session.session_id]
        lifetime = (datetime.now() - session.created).total_seconds()
        logger.info(
            f"Session {reason}: {session.session_id} (lifetime={lifetime:.1f}s)"
        )


# Global singleton instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance (singleton pattern)."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
