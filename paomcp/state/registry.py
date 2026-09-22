"""Multi-session debug registry for Pao Tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from paomcp.tools.backends import (
    BackendSpec,
    GdbController,
    LldbController,
    PwndbgTools,
    resolve_backend,
)
from paomcp.state.session import SessionState
from paomcp.utils.paths import RuntimePaths, validate_session_id

logger = logging.getLogger(__name__)
DEFAULT_MAX_DEBUG_SESSIONS = int(os.environ.get("PAO_MAX_DEBUG_SESSIONS", "8"))
DEFAULT_SESSION_CLOSE_LOCK_TIMEOUT_SEC = 2.0
# A cleanup callback that reported live helper processes is retried by later
# reaps; the cap keeps an unfixable leak from reserving a session slot forever.
MAX_CLEANUP_ATTEMPTS = 3


class SessionLimitError(RuntimeError):
    """Raised when no additional debugger session can be allocated."""


class SessionClosedError(RuntimeError):
    """Raised when work targets a closing or closed debugger session."""


class SessionNotFoundError(LookupError):
    """Raised when a required debugger session does not exist."""


def _build_controller_for_spec(spec: BackendSpec) -> Any:
    """Construct the controller for a resolved backend spec.

    Controller classes are resolved as module-global names at call time (not
    imported at module load) so the existing tests that monkeypatch
    ``registry_module.GdbController`` / ``registry_module.LldbController``
    keep working: this factory is the one construction path they intercept.

    The executable is only passed when it differs from the controller's
    default, preserving the historical zero-argument construction shape
    (``GdbController()``) that monkeypatched fakes depend on.
    """
    if spec.protocol == "lldb-cli":
        if spec.executable and spec.executable != "pwndbg-lldb":
            return LldbController(executable=spec.executable)
        return LldbController()
    if spec.executable and spec.executable != "pwndbg":
        return GdbController(pwndbg_executable=spec.executable)
    return GdbController()


def _default_backend_resolver(*, session_backend: Optional[str]) -> BackendSpec:
    """Resolve the backend honoring ``PAO_DEBUGGER_BACKEND`` from the env."""
    return resolve_backend(
        env_backend=os.environ.get("PAO_DEBUGGER_BACKEND"),
        session_backend=session_backend,
    )


@dataclass
class DebugSession:
    """Container for a full debugger session and associated resources."""

    session_id: str
    runtime_dir: str
    gdb: GdbController
    state: SessionState
    tools: PwndbgTools
    backend: Optional[BackendSpec] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lock: threading.RLock = field(default_factory=threading.RLock)
    driver_pid: Optional[int] = None
    closing: bool = False
    closed: bool = False
    cleanup_callback: Optional[Callable[["DebugSession"], Any]] = field(
        default=None, repr=False
    )
    cleanup_attempts: int = 0
    cleanup_result: Any = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lightweight metadata for API responses."""
        base: Dict[str, Any] = {
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
        if self.backend is not None:
            base["backend"] = self.backend.name
            base["protocol"] = self.backend.protocol
            base["backend_executable"] = self.backend.executable
        return base


class DebugSessionRegistry:
    """Tracks independent debugger sessions keyed by session id."""

    def __init__(
        self,
        runtime_paths: RuntimePaths,
        max_sessions: Optional[int] = None,
        backend_resolver: Optional[Callable[..., BackendSpec]] = None,
    ):
        self.runtime_paths = runtime_paths
        self.max_sessions = (
            DEFAULT_MAX_DEBUG_SESSIONS if max_sessions is None else max_sessions
        )
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.sessions: Dict[str, DebugSession] = {}
        self._tombstones: Dict[str, DebugSession] = {}
        self.default_session_id: Optional[str] = None
        self._lock = threading.RLock()
        self._backend_resolver = backend_resolver or _default_backend_resolver

    def _new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _runtime_dir_for(self, session_id: str) -> str:
        path = os.path.join(self.runtime_paths.sessions_dir, session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def create_session(
        self, session_id: Optional[str] = None, *, backend: Optional[str] = None
    ) -> DebugSession:
        """Create (or return existing) debugger session by id.

        ``backend`` is a ``gdb``/``lldb``/``auto`` request that is resolved
        through the registry's backend resolver; ``None`` defers to the
        resolver's default (env ``PAO_DEBUGGER_BACKEND``, then ``gdb``).
        """
        chosen_id = (
            validate_session_id(session_id) if session_id else self._new_session_id()
        )
        # Reaping may wait for a bounded GDB shutdown, so it must never run while
        # the registry-wide lock is held.
        self.reap_tombstones(lock_timeout_sec=0.0)

        while True:
            detached: Optional[DebugSession] = None
            initialization_error: Optional[Exception] = None
            new_session: Optional[DebugSession] = None
            with self._lock:
                if chosen_id in self._tombstones:
                    raise SessionLimitError(
                        f"Debug session '{chosen_id}' is pending debugger shutdown"
                    )
                existing = self.sessions.get(chosen_id)
                if existing is not None:
                    if self._session_is_destroyed(existing):
                        detached = self._detach_session_locked(existing)
                    else:
                        return existing

                if detached is None:
                    occupied = self._occupied_count_locked()
                    if occupied >= self.max_sessions:
                        raise SessionLimitError(
                            "Debug session limit reached "
                            f"({occupied}/{self.max_sessions})"
                        )

                    runtime_dir = self._runtime_dir_for(chosen_id)
                    spec = self._backend_resolver(session_backend=backend)
                    gdb = _build_controller_for_spec(spec)
                    state = SessionState(session_id=chosen_id)
                    tools = PwndbgTools(gdb, state, backend=spec)
                    new_session = DebugSession(
                        session_id=chosen_id,
                        runtime_dir=runtime_dir,
                        gdb=gdb,
                        state=state,
                        tools=tools,
                        backend=spec,
                    )
                    try:
                        gdb.initialize()
                    except Exception as exc:
                        new_session.closing = True
                        self._tombstones[chosen_id] = new_session
                        initialization_error = exc
                    else:
                        self.sessions[chosen_id] = new_session
                        if self.default_session_id is None:
                            self.default_session_id = chosen_id

            if detached is not None:
                self._close_tombstone_session(detached, lock_timeout_sec=0.0)
                # A confirmed close removes the tombstone and permits immediate
                # recreation. A still-active action leaves it capacity-consuming
                # and the next iteration returns the typed pending error.
                continue

            if initialization_error is not None:
                assert new_session is not None
                self._close_tombstone_session(
                    new_session,
                    lock_timeout_sec=DEFAULT_SESSION_CLOSE_LOCK_TIMEOUT_SEC,
                )
                raise initialization_error

            assert new_session is not None
            logger.info("Created debug session '%s'", chosen_id)
            return new_session

    def get_session(self, session_id: str) -> Optional[DebugSession]:
        lookup_id = validate_session_id(session_id)
        detached: Optional[DebugSession] = None
        with self._lock:
            session = self.sessions.get(lookup_id)
            if session is not None and self._session_is_destroyed(session):
                # The gdb process was force-killed (e.g. core-size enforcement).
                # Evict the dead session so the next resolve rebuilds a fresh
                # controller instead of dispatching MI to a corpse.
                detached = self._detach_session_locked(session)
                session = None
        if detached is not None:
            # Never race a still-unwinding MI action. A zero-timeout attempt
            # leaves a tombstone for the explicit reaper if the action owns the
            # per-session lock.
            self._close_tombstone_session(detached, lock_timeout_sec=0.0)
        return session

    @staticmethod
    def _session_is_destroyed(session: DebugSession) -> bool:
        checker = getattr(session.gdb, "is_destroyed", None)
        return bool(checker()) if callable(checker) else False

    def _detach_session_locked(
        self, session: DebugSession, *, choose_new_default: bool = False
    ) -> DebugSession:
        """Detach one session and reserve its slot as a shutdown tombstone."""
        session.closing = True
        self.sessions.pop(session.session_id, None)
        if self.default_session_id == session.session_id:
            self.default_session_id = (
                next(iter(self.sessions), None) if choose_new_default else None
            )
        self._tombstones[session.session_id] = session
        return session

    @staticmethod
    def _gdb_process_running(session: DebugSession) -> Optional[bool]:
        """Return observable debugger liveness without trusting close results."""
        checker = getattr(session.gdb, "gdb_process_running", None)
        if not callable(checker):
            return None
        try:
            running = checker()
        except Exception:
            logger.debug(
                "Could not inspect gdb liveness for session '%s'",
                session.session_id,
                exc_info=True,
            )
            return None
        return running if isinstance(running, bool) else None

    def _close_session_controller(self, session: DebugSession) -> bool:
        """Attempt bounded close and confirm the debugger no longer occupies a slot."""
        try:
            closed = session.gdb.close()
        except Exception:
            logger.exception("Failed to close gdb for session '%s'", session.session_id)
            return False

        running = self._gdb_process_running(session)
        if running is True:
            logger.error(
                "Debugger for session '%s' remains alive after close",
                session.session_id,
            )
            return False
        if running is False:
            return True
        # GdbController.close returns a bool.  The explicit result is the only
        # proof available for test doubles or legacy controllers without an
        # inspectable Popen handle.
        return closed is True

    @staticmethod
    def _cleanup_failed(result: Any) -> bool:
        """Return whether a cleanup callback explicitly reported failure."""
        if result is False:
            return True
        if not isinstance(result, dict):
            return False
        if result.get("success") is False:
            return True
        # SubprocessTools.close_session_processes reports aggregate cleanup
        # status through failure_count/failures rather than a success boolean.
        # Treat either signal as authoritative so a live helper process can
        # never be hidden behind a successful debugger-session close.
        failure_count = result.get("failure_count", 0)
        failures = result.get("failures", [])
        try:
            if int(failure_count) > 0:
                return True
        except (TypeError, ValueError):
            return True
        return bool(failures)

    @classmethod
    def _cleanup_succeeded(cls, attempts: int, result: Any) -> bool:
        """Return whether cleanup ran and did not report remaining resources."""
        return attempts > 0 and not cls._cleanup_failed(result)

    def _close_tombstone_session(
        self,
        session: DebugSession,
        *,
        lock_timeout_sec: float,
    ) -> Dict[str, Any]:
        """Close one tombstone without holding the registry-wide lock.

        Acquiring the per-session lock serializes pipe/process teardown with any
        MI action that resolved the session before it was detached. Failure to
        acquire it is a normal pending state: the tombstone remains allocated.
        """
        timeout = max(0.0, float(lock_timeout_sec))
        acquired = session.lock.acquire(timeout=timeout)
        if not acquired:
            return {
                "lock_acquired": False,
                "controller_closed": False,
                "tombstone_released": False,
                "cleanup": session.cleanup_result,
            }

        try:
            if (
                session.cleanup_callback is not None
                and not self._cleanup_succeeded(
                    session.cleanup_attempts, session.cleanup_result
                )
                and session.cleanup_attempts < MAX_CLEANUP_ATTEMPTS
            ):
                session.cleanup_attempts += 1
                try:
                    session.cleanup_result = session.cleanup_callback(session)
                except Exception as exc:
                    logger.exception(
                        "Session cleanup callback failed for '%s'", session.session_id
                    )
                    session.cleanup_result = {
                        "success": False,
                        "type": type(exc).__name__,
                        "error": str(exc),
                    }
            controller_closed = self._close_session_controller(session)
            session.closed = True
            session.closing = False
        finally:
            session.lock.release()

        cleanup_failed = self._cleanup_failed(session.cleanup_result)
        retries_exhausted = session.cleanup_attempts >= MAX_CLEANUP_ATTEMPTS
        # A cleanup failure means helper processes may still be alive: keep
        # the tombstone so a later reap_tombstones() can retry, and only
        # release the slot when cleanup succeeded or its retry budget ran out.
        released = controller_closed and (not cleanup_failed or retries_exhausted)
        if released:
            with self._lock:
                if self._tombstones.get(session.session_id) is session:
                    self._tombstones.pop(session.session_id, None)
        elif controller_closed and cleanup_failed:
            logger.warning(
                "Session '%s' cleanup failed; keeping tombstone for retry (%d/%d)",
                session.session_id,
                session.cleanup_attempts,
                MAX_CLEANUP_ATTEMPTS,
            )
        return {
            "lock_acquired": True,
            "controller_closed": controller_closed,
            "tombstone_released": released,
            "cleanup": session.cleanup_result,
        }

    def _occupied_count_locked(self) -> int:
        """Count active sessions plus shutdown tombstones."""
        return len(self.sessions) + len(self._tombstones)

    def reap_tombstones(self, *, lock_timeout_sec: float = 0.0) -> int:
        """Retry cleanup without holding the registry lock while waiting."""
        with self._lock:
            tombstones = list(self._tombstones.values())
        reaped = 0
        for session in tombstones:
            result = self._close_tombstone_session(
                session, lock_timeout_sec=lock_timeout_sec
            )
            if result["tombstone_released"]:
                reaped += 1
        return reaped

    def evict_session(self, session_id: str) -> bool:
        """Publicly drop a session (used after gdb was force-killed).

        Returns True if a session was present and detached. Safe to call with an
        unknown id. The id remains reserved until bounded controller shutdown is
        confirmed, after which a later resolve can rebuild it.
        """
        lookup_id = validate_session_id(session_id)
        with self._lock:
            session = self.sessions.get(lookup_id)
            if session is None:
                return False
            detached = self._detach_session_locked(session)
        result = self._close_tombstone_session(detached, lock_timeout_sec=0.0)
        if result["controller_closed"]:
            logger.info("Evicted destroyed debug session '%s'", lookup_id)
        else:
            logger.info(
                "Detached debug session '%s'; debugger shutdown remains pending",
                lookup_id,
            )
        return True

    def ensure_session(
        self, session_id: Optional[str] = None, *, backend: Optional[str] = None
    ) -> DebugSession:
        """Return an existing session, creating if necessary."""
        if session_id:
            validated_id = validate_session_id(session_id)
            existing = self.get_session(validated_id)
            return (
                existing
                if existing is not None
                else self.create_session(validated_id, backend=backend)
            )

        with self._lock:
            default_id = self.default_session_id
        if default_id is not None:
            default = self.get_session(default_id)
            if default is not None:
                return default
        return self.create_session("default", backend=backend)

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [session.to_dict() for session in self.sessions.values()]

    def status(self) -> Dict[str, Any]:
        """Return debugger-session capacity without exposing live objects."""
        with self._lock:
            active = len(self.sessions)
            tombstones = len(self._tombstones)
            occupied = active + tombstones
            return {
                "active": active,
                "tombstones": tombstones,
                "occupied": occupied,
                "available": max(0, self.max_sessions - occupied),
                "max_sessions": self.max_sessions,
                "default_session_id": self.default_session_id,
            }

    def close_session(
        self,
        session_id: str,
        on_closing: Optional[Callable[[DebugSession], Any]] = None,
        *,
        lock_timeout_sec: float = DEFAULT_SESSION_CLOSE_LOCK_TIMEOUT_SEC,
    ) -> Dict[str, Any]:
        """Close a session and release associated resources."""
        lookup_id = validate_session_id(session_id)
        with self._lock:
            session = self.sessions.pop(lookup_id, None)
            if not session:
                session = self._tombstones.get(lookup_id)
                if session is None:
                    return {
                        "success": False,
                        "error": f"Session '{lookup_id}' not found",
                        "type": "SessionNotFoundError",
                        "code": "NOT_FOUND",
                    }
            else:
                # Mark closing before waiting for the per-session lock. Any action
                # that already resolved this session but is queued behind an active
                # action then fails closed.
                session.closing = True
                self._tombstones[lookup_id] = session
                if self.default_session_id == lookup_id:
                    self.default_session_id = next(iter(self.sessions), None)

            if on_closing is not None and not self._cleanup_succeeded(
                session.cleanup_attempts, session.cleanup_result
            ):
                session.cleanup_callback = on_closing

        close_result = self._close_tombstone_session(
            session, lock_timeout_sec=lock_timeout_sec
        )
        cleanup_result = close_result["cleanup"]
        if not close_result["lock_acquired"]:
            logger.warning(
                "Debugger action remained active while closing session '%s'", lookup_id
            )
            result = {
                "success": False,
                "session_id": lookup_id,
                "error": (
                    "An active debugger action did not finish before the "
                    "session close timeout"
                ),
                "type": "SessionShutdownPendingError",
                "code": "SHUTDOWN_PENDING",
                "shutdown_pending": True,
                "action_lock_busy": True,
            }
        elif not close_result["controller_closed"]:
            logger.error(
                "Debugger shutdown remains pending for session '%s'", lookup_id
            )
            result = {
                "success": False,
                "session_id": lookup_id,
                "error": "Debugger process could not be confirmed stopped",
                "type": "SessionShutdownPendingError",
                "code": "SHUTDOWN_PENDING",
                "shutdown_pending": True,
            }
        elif self._cleanup_failed(cleanup_result):
            logger.error("Session cleanup failed for '%s'", lookup_id)
            result = {
                "success": False,
                "session_id": lookup_id,
                "error": "Debugger closed, but associated resource cleanup failed",
                "type": "SessionCleanupError",
                "code": "SESSION_CLEANUP_FAILED",
                "shutdown_pending": False,
            }
        else:
            logger.info("Closed debug session '%s'", lookup_id)
            result = {"success": True, "session_id": lookup_id}

        if cleanup_result is not None:
            result["cleanup"] = cleanup_result
        return result

    def close_all(self) -> None:
        """Close all tracked sessions."""
        with self._lock:
            for session in list(self.sessions.values()):
                self._detach_session_locked(session, choose_new_default=True)
            tombstones = list(self._tombstones.values())
        for session in tombstones:
            self._close_tombstone_session(
                session,
                lock_timeout_sec=DEFAULT_SESSION_CLOSE_LOCK_TIMEOUT_SEC,
            )
