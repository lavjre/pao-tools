"""Thread-safe lifecycle tracking and cancellation for MCP operations."""

from __future__ import annotations

import copy
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import os
import re
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT_OPERATIONS = int(
    os.environ.get("PAO_MAX_CONCURRENT_OPERATIONS", "32")
)
DEFAULT_OPERATION_HISTORY_LIMIT = int(
    os.environ.get("PAO_OPERATION_HISTORY_LIMIT", "1000")
)
DEFAULT_OPERATION_TTL_SECONDS = int(
    os.environ.get("PAO_OPERATION_TTL_SECONDS", str(24 * 60 * 60))
)


class OperationLimitError(RuntimeError):
    """Raised when the configured concurrent operation limit is exhausted."""


class OperationCancelledError(RuntimeError):
    """Raised when cooperative cancellation was requested for an operation."""


class OperationOwnershipError(PermissionError):
    """Raised when an operation is accessed through a different session."""


class IdempotencyConflictError(ValueError):
    """Raised when one retry key is reused with different arguments."""


_current_operation_id: ContextVar[Optional[str]] = ContextVar(
    "pao_current_operation_id", default=None
)


def current_operation_id() -> Optional[str]:
    """Return the operation associated with the current async/thread context."""
    return _current_operation_id.get()


def bind_operation(operation_id: str) -> Token[Optional[str]]:
    """Bind an operation to the current context until ``reset_operation``."""
    return _current_operation_id.set(operation_id)


def reset_operation(token: Token[Optional[str]]) -> None:
    """Restore the operation context that preceded ``bind_operation``."""
    _current_operation_id.reset(token)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class OperationRecord:
    """Mutable internal state for one public tool invocation."""

    operation_id: str
    tool: str
    session_id: Optional[str]
    state: str = "running"
    created_at: str = field(default_factory=lambda: _utc_now().isoformat())
    finished_at: Optional[str] = None
    cancellation_requested: bool = False
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    progress_message: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    _cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    _cancel_handler: Optional[Callable[[], None]] = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize public operation state without internal synchronization data."""
        return {
            "operation_id": self.operation_id,
            "tool": self.tool,
            "session_id": self.session_id,
            "state": self.state,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "cancellation_requested": self.cancellation_requested,
            "progress": {
                "current": self.progress_current,
                "total": self.progress_total,
                "message": self.progress_message,
            },
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass
class IdempotencyRecord:
    """Cached terminal response for one caller-provided retry key."""

    tool: str
    session_id: Optional[str]
    key: str
    argument_fingerprint: str
    created_at: str = field(default_factory=lambda: _utc_now().isoformat())
    response: Optional[Dict[str, Any]] = None


class OperationRegistry:
    """Track bounded operation history and dispatch cooperative cancellation."""

    TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})

    def __init__(
        self,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_OPERATIONS,
        history_limit: int = DEFAULT_OPERATION_HISTORY_LIMIT,
        ttl_seconds: int = DEFAULT_OPERATION_TTL_SECONDS,
    ) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds cannot be negative")
        self.max_concurrent = max_concurrent
        self.history_limit = history_limit
        self.ttl_seconds = ttl_seconds
        self._records: Dict[str, OperationRecord] = {}
        self._idempotency: Dict[tuple[str, Optional[str], str], IdempotencyRecord] = {}
        self._lock = threading.RLock()

    def _prune_locked(self) -> None:
        terminal = [
            record
            for record in self._records.values()
            if record.state in self.TERMINAL_STATES
        ]
        if self.ttl_seconds:
            cutoff = _utc_now() - timedelta(seconds=self.ttl_seconds)
            for record in terminal:
                if (
                    record.finished_at
                    and datetime.fromisoformat(record.finished_at) < cutoff
                ):
                    self._records.pop(record.operation_id, None)

        excess = len(self._records) - self.history_limit
        if excess > 0:
            removable = sorted(
                (
                    record
                    for record in self._records.values()
                    if record.state in self.TERMINAL_STATES
                ),
                key=lambda item: item.finished_at or item.created_at,
            )
            for record in removable[:excess]:
                self._records.pop(record.operation_id, None)

        if self.ttl_seconds:
            cutoff = _utc_now() - timedelta(seconds=self.ttl_seconds)
            for cache_key, idempotency_record in list(self._idempotency.items()):
                if (
                    idempotency_record.response is not None
                    and datetime.fromisoformat(idempotency_record.created_at) < cutoff
                ):
                    self._idempotency.pop(cache_key, None)

        completed_keys = [
            cache_key
            for cache_key, record in sorted(
                self._idempotency.items(), key=lambda item: item[1].created_at
            )
            if record.response is not None
        ]
        excess_idempotency = len(self._idempotency) - self.history_limit
        for cache_key in completed_keys[: max(0, excess_idempotency)]:
            self._idempotency.pop(cache_key, None)

    @staticmethod
    def _validate_idempotency_key(key: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", key):
            raise ValueError(
                "idempotency_key must be 1-128 safe characters "
                "(letters, digits, dot, underscore, colon, or hyphen)"
            )
        return key

    def claim_idempotency(
        self,
        tool: str,
        session_id: Optional[str],
        key: str,
        argument_fingerprint: str,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Claim a retry key or return its cached/in-progress state."""
        validated = self._validate_idempotency_key(key)
        cache_key = (tool, session_id, validated)
        with self._lock:
            self._prune_locked()
            existing = self._idempotency.get(cache_key)
            if existing is None:
                self._idempotency[cache_key] = IdempotencyRecord(
                    tool=tool,
                    session_id=session_id,
                    key=validated,
                    argument_fingerprint=argument_fingerprint,
                )
                return "claimed", None
            if existing.argument_fingerprint != argument_fingerprint:
                raise IdempotencyConflictError(
                    "idempotency_key was already used with different arguments"
                )
            if existing.response is None:
                return "in_progress", None
            return "replay", copy.deepcopy(existing.response)

    def complete_idempotency(
        self,
        tool: str,
        session_id: Optional[str],
        key: str,
        argument_fingerprint: str,
        response: Dict[str, Any],
    ) -> None:
        """Cache the normalized terminal response for deterministic retries."""
        validated = self._validate_idempotency_key(key)
        cache_key = (tool, session_id, validated)
        with self._lock:
            record = self._idempotency.get(cache_key)
            if record is None:
                record = IdempotencyRecord(
                    tool=tool,
                    session_id=session_id,
                    key=validated,
                    argument_fingerprint=argument_fingerprint,
                )
                self._idempotency[cache_key] = record
            record.response = copy.deepcopy(response)
            self._prune_locked()

    def abort_idempotency(self, tool: str, session_id: Optional[str], key: str) -> None:
        """Release a claimed key when work could not be admitted."""
        validated = self._validate_idempotency_key(key)
        with self._lock:
            self._idempotency.pop((tool, session_id, validated), None)

    def begin(
        self,
        tool: str,
        *,
        session_id: Optional[str] = None,
        bypass_limit: bool = False,
    ) -> OperationRecord:
        """Create a running operation, enforcing the global concurrency cap."""
        with self._lock:
            self._prune_locked()
            running = sum(
                record.state == "running" for record in self._records.values()
            )
            if not bypass_limit and running >= self.max_concurrent:
                raise OperationLimitError(
                    "Concurrent operation limit reached "
                    f"({running}/{self.max_concurrent})"
                )
            record = OperationRecord(
                operation_id=uuid.uuid4().hex,
                tool=tool,
                session_id=session_id,
            )
            self._records[record.operation_id] = record
            return record

    def complete(
        self,
        operation_id: str,
        *,
        success: bool,
        session_id: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> OperationRecord:
        """Mark an operation terminal and preserve its normalized error summary."""
        with self._lock:
            record = self._require_locked(operation_id)
            # Owner is fixed at creation: a backend response must never
            # reassign an operation to a foreign session. Only fill in when
            # the record was created unscoped.
            if session_id is not None and record.session_id is None:
                record.session_id = session_id
            if record.cancellation_requested or error_code == "OPERATION_CANCELLED":
                record.state = "cancelled"
                error_code = error_code or "OPERATION_CANCELLED"
                error_message = error_message or "Operation was cancelled"
            else:
                record.state = "succeeded" if success else "failed"
            record.error_code = error_code
            record.error_message = error_message
            record.finished_at = _utc_now().isoformat()
            record._cancel_handler = None
            self._prune_locked()
            return record

    def _require_locked(self, operation_id: str) -> OperationRecord:
        record = self._records.get(operation_id)
        if record is None:
            raise FileNotFoundError(f"Operation '{operation_id}' not found")
        return record

    def get(self, operation_id: str) -> Dict[str, Any]:
        """Return public state for one operation."""
        with self._lock:
            return self._require_locked(operation_id).to_dict()

    def list(
        self,
        *,
        session_id: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List newest operations with optional session/state filters."""
        if limit <= 0 or limit > self.history_limit:
            raise ValueError(f"limit must be between 1 and {self.history_limit}")
        if state is not None and state not in {
            "running",
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise ValueError("state is not a recognized operation state")
        with self._lock:
            self._prune_locked()
            records = list(self._records.values())
            records = [item for item in records if item.session_id == session_id]
            if state is not None:
                records = [item for item in records if item.state == state]
            records.sort(key=lambda item: item.created_at, reverse=True)
            return [item.to_dict() for item in records[:limit]]

    def request_cancel(self, operation_id: str) -> Dict[str, Any]:
        """Set the cancellation flag and invoke a backend-specific handler."""
        with self._lock:
            record = self._require_locked(operation_id)
            if record.state in self.TERMINAL_STATES:
                return record.to_dict()
            record.cancellation_requested = True
            record._cancel_event.set()
            handler = record._cancel_handler
        if handler is not None:
            try:
                handler()
            except Exception:
                logger.exception(
                    "Cancellation handler failed for operation '%s'", operation_id
                )
        return self.get(operation_id)

    def set_cancel_handler(
        self, operation_id: str, handler: Optional[Callable[[], None]]
    ) -> None:
        """Attach the safe backend callback used by ``request_cancel``."""
        with self._lock:
            record = self._require_locked(operation_id)
            record._cancel_handler = handler
            already_requested = record.cancellation_requested
        if already_requested and handler is not None:
            try:
                handler()
            except Exception:
                logger.exception(
                    "Late cancellation handler failed for operation '%s'",
                    operation_id,
                )

    def cancellation_event(
        self, operation_id: Optional[str] = None
    ) -> Optional[threading.Event]:
        """Return the cooperative cancellation event for an operation."""
        resolved_id = operation_id or current_operation_id()
        if resolved_id is None:
            return None
        with self._lock:
            return self._require_locked(resolved_id)._cancel_event

    def raise_if_cancelled(self, operation_id: Optional[str] = None) -> None:
        """Raise a stable exception at a cooperative cancellation checkpoint."""
        event = self.cancellation_event(operation_id)
        if event is not None and event.is_set():
            raise OperationCancelledError("Operation cancellation was requested")

    def update_progress(
        self,
        operation_id: Optional[str] = None,
        *,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish bounded progress for polling clients."""
        resolved_id = operation_id or current_operation_id()
        if resolved_id is None:
            raise RuntimeError("No operation is bound to the current context")
        if current is not None and current < 0:
            raise ValueError("progress current cannot be negative")
        if total is not None and total < 0:
            raise ValueError("progress total cannot be negative")
        if current is not None and total is not None and current > total:
            raise ValueError("progress current cannot exceed total")
        with self._lock:
            record = self._require_locked(resolved_id)
            record.progress_current = current
            record.progress_total = total
            record.progress_message = message
            return record.to_dict()

    def status(self) -> Dict[str, Any]:
        """Return operation capacity and retention policy."""
        with self._lock:
            self._prune_locked()
            running = sum(
                record.state == "running" for record in self._records.values()
            )
            return {
                "running": running,
                "available": max(0, self.max_concurrent - running),
                "tracked": len(self._records),
                "idempotency_records": len(self._idempotency),
                "policy": {
                    "max_concurrent": self.max_concurrent,
                    "history_limit": self.history_limit,
                    "ttl_seconds": self.ttl_seconds,
                },
            }
