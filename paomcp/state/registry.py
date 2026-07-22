"""Multi-session debug registry for Pao Tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from paomcp.tools.backends import GdbController, PwndbgTools
from paomcp.state.session import SessionState
from paomcp.utils.paths import RuntimePaths, validate_session_id

logger = logging.getLogger(__name__)
DEFAULT_MAX_DEBUG_SESSIONS = int(os.environ.get("PAO_MAX_DEBUG_SESSIONS", "8"))


class SessionLimitError(RuntimeError):
    """Raised when no additional debugger session can be allocated."""


class SessionClosedError(RuntimeError):
    """Raised when work targets a closing or closed debugger session."""


class SessionNotFoundError(LookupError):
    """Raised when a required debugger session does not exist."""


@dataclass
class DebugSession:
    """Container for a full debugger session and associated resources."""

    session_id: str
    runtime_dir: str
    gdb: GdbController
    state: SessionState
    tools: PwndbgTools
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lock: threading.RLock = field(default_factory=threading.RLock)
    driver_pid: Optional[int] = None
    closing: bool = False
    closed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lightweight metadata for API responses."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "runtime_dir": self.runtime_dir,
            "binary_path": self.state.binary_path,
            "binary_loaded": self.state.binary_loaded,
            "inferior_pid": self.state.pid,
            "driver_pid": self.driver_pid,
            "state": self.state.state,
            "gdb_state": self.gdb.get_state(),
            "closing": self.closing,
            "closed": self.closed,
        }


class DebugSessionRegistry:
    """Tracks independent debugger sessions keyed by session id."""

    def __init__(self, runtime_paths: RuntimePaths, max_sessions: Optional[int] = None):
        self.runtime_paths = runtime_paths
        self.max_sessions = (
            DEFAULT_MAX_DEBUG_SESSIONS if max_sessions is None else max_sessions
        )
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.sessions: Dict[str, DebugSession] = {}
        self.default_session_id: Optional[str] = None
        self._lock = threading.RLock()

    def _new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _runtime_dir_for(self, session_id: str) -> str:
        path = os.path.join(self.runtime_paths.sessions_dir, session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def create_session(self, session_id: Optional[str] = None) -> DebugSession:
        """Create (or return existing) debugger session by id."""
        with self._lock:
            chosen_id = (
                validate_session_id(session_id)
                if session_id
                else self._new_session_id()
            )
            if chosen_id in self.sessions:
                return self.sessions[chosen_id]
            if len(self.sessions) >= self.max_sessions:
                raise SessionLimitError(
                    "Debug session limit reached "
                    f"({len(self.sessions)}/{self.max_sessions})"
                )

            runtime_dir = self._runtime_dir_for(chosen_id)
            gdb = GdbController()
            try:
                gdb.initialize()
                state = SessionState(session_id=chosen_id)
                tools = PwndbgTools(gdb, state)
                session = DebugSession(
                    session_id=chosen_id,
                    runtime_dir=runtime_dir,
                    gdb=gdb,
                    state=state,
                    tools=tools,
                )
            except Exception:
                try:
                    gdb.close()
                except Exception:
                    logger.exception(
                        "Failed to close partially initialized session '%s'",
                        chosen_id,
                    )
                raise
            self.sessions[chosen_id] = session
            if self.default_session_id is None:
                self.default_session_id = chosen_id
            logger.info("Created debug session '%s'", chosen_id)
            return session

    def get_session(self, session_id: str) -> Optional[DebugSession]:
        lookup_id = validate_session_id(session_id)
        with self._lock:
            return self.sessions.get(lookup_id)

    def ensure_session(self, session_id: Optional[str] = None) -> DebugSession:
        """Return an existing session, creating if necessary."""
        with self._lock:
            if session_id:
                validated_id = validate_session_id(session_id)
                existing = self.sessions.get(validated_id)
                if existing:
                    return existing
                return self.create_session(validated_id)

            if self.default_session_id and self.default_session_id in self.sessions:
                return self.sessions[self.default_session_id]
            return self.create_session("default")

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [session.to_dict() for session in self.sessions.values()]

    def status(self) -> Dict[str, Any]:
        """Return debugger-session capacity without exposing live objects."""
        with self._lock:
            active = len(self.sessions)
            return {
                "active": active,
                "available": max(0, self.max_sessions - active),
                "max_sessions": self.max_sessions,
                "default_session_id": self.default_session_id,
            }

    def close_session(
        self,
        session_id: str,
        on_closing: Optional[Callable[[DebugSession], Any]] = None,
    ) -> Dict[str, Any]:
        """Close a session and release associated resources."""
        lookup_id = validate_session_id(session_id)
        with self._lock:
            session = self.sessions.pop(lookup_id, None)
            if not session:
                return {
                    "success": False,
                    "error": f"Session '{lookup_id}' not found",
                    "type": "SessionNotFoundError",
                    "code": "NOT_FOUND",
                }

            # Mark closing before waiting for the per-session lock. Any action that
            # already resolved this session but is queued behind an active action
            # will then fail closed instead of running after close was requested.
            session.closing = True
            cleanup_result = None
            with session.lock:
                if on_closing is not None:
                    try:
                        cleanup_result = on_closing(session)
                    except Exception as exc:
                        logger.exception(
                            "Session cleanup callback failed for '%s'", lookup_id
                        )
                        cleanup_result = {
                            "success": False,
                            "type": type(exc).__name__,
                            "error": str(exc),
                        }
                try:
                    session.gdb.close()
                except Exception:
                    logger.exception("Failed to close gdb for session '%s'", lookup_id)
                finally:
                    session.closed = True
                    session.closing = False

            if self.default_session_id == lookup_id:
                self.default_session_id = next(iter(self.sessions), None)

            logger.info("Closed debug session '%s'", lookup_id)
            result = {"success": True, "session_id": lookup_id}
            if cleanup_result is not None:
                result["cleanup"] = cleanup_result
            return result

    def close_all(self) -> None:
        """Close all tracked sessions."""
        with self._lock:
            for session_id in list(self.sessions.keys()):
                self.close_session(session_id)
