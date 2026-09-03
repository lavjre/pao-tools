import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from fastmcp import Context

from paomcp.api import ToolError, ToolMeta, ToolResponse
from paomcp.artifacts import ArtifactStore
from paomcp.tools.backends.backend import validate_backend_name
from paomcp.operations import bind_operation, reset_operation
from paomcp.paopipe import PaoPipe
from paomcp.services import AppServices
from paomcp.state import (
    DebugSession,
    DebugSessionRegistry,
    SessionClosedError,
    SessionNotFoundError,
)
from paomcp.utils.paths import DEFAULT_WORKSPACE, resolve_workspace_path

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _error_code(error_type: Optional[str]) -> str:
    if not error_type:
        return "TOOL_ERROR"
    taxonomy = {
        "ValueError": "INVALID_ARGUMENT",
        "TypeError": "INVALID_ARGUMENT",
        "KeyError": "NOT_FOUND",
        "LookupError": "NOT_FOUND",
        "FileNotFoundError": "NOT_FOUND",
        "PermissionError": "PERMISSION_DENIED",
        "TimeoutError": "TIMEOUT",
        "TimeoutExpired": "TIMEOUT",
        "OperationCancelledError": "OPERATION_CANCELLED",
        "OperationOwnershipError": "OPERATION_ACCESS_DENIED",
        "OperationLimitError": "RESOURCE_EXHAUSTED",
        "IdempotencyConflictError": "IDEMPOTENCY_CONFLICT",
        "SessionLimitError": "RESOURCE_EXHAUSTED",
        "SessionNotFoundError": "NOT_FOUND",
        "SessionClosedError": "SESSION_CLOSED",
        "ArtifactStorageFullError": "RESOURCE_EXHAUSTED",
        "ArtifactSizeLimitError": "RESOURCE_EXHAUSTED",
        "ArtifactOwnershipError": "ARTIFACT_ACCESS_DENIED",
        "UnmanagedProcessError": "PROCESS_NOT_MANAGED",
    }
    if error_type in taxonomy:
        return taxonomy[error_type]
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", error_type).upper()
    return snake_case or "TOOL_ERROR"


def _argument_fingerprint(arguments: Dict[str, Any]) -> str:
    """Hash public tool arguments without retaining their potentially sensitive data."""
    public_arguments = {
        key: value
        for key, value in arguments.items()
        if key not in {"ctx", "idempotency_key"}
    }
    serialized = json.dumps(
        public_arguments,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: repr(value),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _missing_artifact_ids(
    value: Any, artifact_store: Optional[ArtifactStore]
) -> list[str]:
    """Return artifact references that can no longer satisfy a cached replay."""
    if artifact_store is None:
        return []
    discovered: set[str] = set()
    missing: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            artifact_id = item.get("artifact_id")
            if item.get("delivery") == "artifact" and artifact_id is not None:
                normalized = str(artifact_id)
                if normalized not in discovered:
                    discovered.add(normalized)
                    try:
                        artifact_store.get(normalized)
                    except Exception:
                        missing.add(normalized)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(missing)


TOKEN_INLINE_RAW_BUDGET_BYTES = int(os.environ.get("PAO_TOKEN_RAW_BUDGET", "8192"))
TOKEN_INLINE_LIST_KEEP = int(os.environ.get("PAO_TOKEN_LIST_KEEP", "50"))
TOKEN_RAW_PREVIEW_ITEMS = 2


def _summarize_raw_for_tokens(data: Any, warnings: list[str]) -> Any:
    """Strip verbose raw evidence from inline payloads to save LLM tokens.

    Full payloads that exceed the artifact threshold are still stored
    losslessly (see below). This only trims the inline copy that stays in
    context: ``raw_responses`` duplicates ``parsed`` and is the dominant
    token burner. Small payloads pass through untouched so unit tests and
    tiny queries keep exact shape.
    """
    if not isinstance(data, dict):
        return data
    raw = data.get("raw_responses")
    if raw is None:
        return data
    try:
        raw_size = len(
            json.dumps(raw, separators=(",", ":"), default=str).encode("utf-8")
        )
    except Exception:
        return data
    if raw_size <= TOKEN_INLINE_RAW_BUDGET_BYTES:
        return data
    omitted: int
    preview: Any
    if isinstance(raw, list):
        omitted = max(0, len(raw) - TOKEN_RAW_PREVIEW_ITEMS)
        preview = raw[:TOKEN_RAW_PREVIEW_ITEMS]
    elif isinstance(raw, dict):
        keys = list(raw.keys())
        omitted = max(0, len(keys) - TOKEN_RAW_PREVIEW_ITEMS)
        preview = {k: raw[k] for k in keys[:TOKEN_RAW_PREVIEW_ITEMS]}
    else:
        text = str(raw)
        omitted = max(0, len(text) - TOKEN_INLINE_RAW_BUDGET_BYTES)
        preview = text[:TOKEN_INLINE_RAW_BUDGET_BYTES]
    data = dict(data)
    data["raw_responses"] = {
        "truncated_for_tokens": True,
        "total_bytes": raw_size,
        "omitted_entries": omitted,
        "preview": preview,
        "note": "Full output spilled to artifact when over inline limit; parsed field is authoritative",
    }
    warnings.append(
        "raw_responses summarized for token efficiency; parsed field retained in full"
    )
    return data


def _truncate_long_lists(data: Any, warnings: list[str]) -> Any:
    """Cap very long inline lists (gadgets/strings/mappings) with explicit counts."""
    if not isinstance(data, dict):
        return data
    mutated = False
    trimmed = dict(data)
    for key, value in list(trimmed.items()):
        if isinstance(value, list) and len(value) > TOKEN_INLINE_LIST_KEEP * 4:
            # Keep head for context, report exact totals. Full list still goes
            # to artifact when the payload exceeds the inline threshold.
            trimmed[key] = value[:TOKEN_INLINE_LIST_KEEP]
            trimmed[f"{key}_truncated"] = {
                "returned": TOKEN_INLINE_LIST_KEEP,
                "total": len(value),
                "note": f"Showing first {TOKEN_INLINE_LIST_KEEP}; full list in artifact if spilled, else re-query with limit",
            }
            mutated = True
    if mutated:
        warnings.append(
            f"Long lists capped at {TOKEN_INLINE_LIST_KEEP} inline entries with totals; full data in artifact when spilled"
        )
    return trimmed


def build_tool_response(
    tool: str,
    result: Any,
    *,
    duration_ms: int = 0,
    session_id: Optional[str] = None,
    artifact_store: Optional[ArtifactStore] = None,
    auto_artifact: bool = True,
    operation_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    idempotency_replayed: bool = False,
) -> Dict[str, Any]:
    """Normalize a tool result into the versioned public response envelope.

    Large payloads spill losslessly to artifacts instead of being silently
    truncated. Inline copies are token-trimmed (raw_responses summarized,
    very long lists capped) with explicit counts so the LLM never pays
    full raw bytes just to read a summary.
    """
    if isinstance(result, ToolResponse):
        response = result
        response.meta.duration_ms = duration_ms
        response.meta.operation_id = operation_id
        response.meta.idempotency_key = idempotency_key
        response.meta.idempotency_replayed = idempotency_replayed
        return response.as_dict()

    success = True
    warnings = []
    error: Optional[ToolError] = None
    data = result

    if isinstance(result, dict):
        payload = dict(result)
        success = bool(payload.pop("success", True))
        session_id_value = payload.pop("session_id", None)
        if session_id_value is not None:
            session_id = str(session_id_value)

        raw_warnings = payload.pop("warnings", [])
        if isinstance(raw_warnings, list):
            warnings = [str(item) for item in raw_warnings]
        elif raw_warnings:
            warnings = [str(raw_warnings)]

        if success:
            data = payload
        else:
            raw_error = payload.pop("error", "Tool operation failed")
            error_type = payload.pop("type", None)
            error_code = payload.pop("code", None)
            details = None
            if isinstance(raw_error, dict):
                message = str(raw_error.get("message", "Tool operation failed"))
                error_code = raw_error.get("code", error_code)
                error_type = raw_error.get("type", error_type)
                details = raw_error.get("details")
            else:
                message = str(raw_error)
            error = ToolError(
                code=str(error_code or _error_code(error_type)),
                message=message,
                type=str(error_type) if error_type else None,
                details=details,
            )
            data = payload or None

    response = ToolResponse(
        success=success,
        tool=tool,
        session_id=session_id,
        data=data,
        error=error,
        warnings=warnings,
        meta=ToolMeta(
            duration_ms=duration_ms,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            idempotency_replayed=idempotency_replayed,
        ),
    )

    if auto_artifact and artifact_store is not None and response.data is not None:
        serialized_full = artifact_store.serialize_json(response.data)
        if artifact_store.should_store(serialized_full):
            artifact = artifact_store.save_json_bytes(
                serialized_full,
                tool=tool,
                session_id=session_id,
            )
            response.data = artifact.to_dict()
            # Attach a token-lean summary alongside the artifact descriptor
            # when the backend provided parsed evidence.
            if isinstance(result, dict) and isinstance(result.get("parsed"), dict):
                parsed = result["parsed"]
                summary_keys = [
                    k
                    for k in ("total", "count", "verdict", "state", "success")
                    if k in parsed
                ]
                if summary_keys:
                    response.warnings.append(
                        "Full payload in artifact; parsed summary keys: "
                        + ",".join(summary_keys)
                    )
        else:
            # Stays inline: trim token-heavy raw evidence, keep parsed exact.
            trimmed = _summarize_raw_for_tokens(response.data, response.warnings)
            trimmed = _truncate_long_lists(trimmed, response.warnings)
            response.data = trimmed
    else:
        # No artifact store (tests / degraded startup): still trim inline.
        if response.data is not None:
            trimmed = _summarize_raw_for_tokens(response.data, response.warnings)
            trimmed = _truncate_long_lists(trimmed, response.warnings)
            response.data = trimmed
    return response.as_dict()


def catch_errors(
    *, auto_artifact: bool = True, track_operation: bool = True
) -> Callable[[Callable[..., Any]], Any]:
    """Wrap an MCP tool with the shared response envelope and error handling."""

    def decorator(fn: Callable[..., Any]) -> Any:
        fn_signature = inspect.signature(fn)

        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = time.monotonic()
            bound_arguments = fn_signature.bind_partial(*args, **kwargs)
            bound_arguments.apply_defaults()
            requested_session_id = bound_arguments.arguments.get("session_id")
            operation_session_id = (
                str(requested_session_id) if requested_session_id is not None else None
            )
            raw_idempotency_key = bound_arguments.arguments.get("idempotency_key")
            idempotency_key = (
                str(raw_idempotency_key) if raw_idempotency_key is not None else None
            )
            argument_fingerprint = _argument_fingerprint(
                dict(bound_arguments.arguments)
            )
            idempotency_claimed = False
            artifact_store = None
            services = None
            operation = None
            operation_token = None
            context = bound_arguments.arguments.get("ctx")
            if context is not None:
                try:
                    services = context.lifespan_context.get("services")
                    if services is not None:
                        artifact_store = services.artifact_store
                except (AttributeError, RuntimeError):
                    services = None
                    artifact_store = None

            if services is not None and idempotency_key is not None:
                try:
                    claimed_state, cached_response = (
                        services.operation_registry.claim_idempotency(
                            fn.__name__,
                            operation_session_id,
                            idempotency_key,
                            argument_fingerprint,
                        )
                    )
                except Exception as exc:
                    duration_ms = int((time.monotonic() - started_at) * 1000)
                    return build_tool_response(
                        fn.__name__,
                        {
                            "success": False,
                            "error": str(exc),
                            "type": type(exc).__name__,
                        },
                        duration_ms=duration_ms,
                        session_id=operation_session_id,
                        auto_artifact=False,
                        idempotency_key=idempotency_key,
                    )
                if claimed_state == "replay" and cached_response is not None:
                    missing_artifacts = _missing_artifact_ids(
                        cached_response, artifact_store
                    )
                    if missing_artifacts:
                        duration_ms = int((time.monotonic() - started_at) * 1000)
                        return build_tool_response(
                            fn.__name__,
                            {
                                "success": False,
                                "error": {
                                    "code": "IDEMPOTENCY_RESULT_EXPIRED",
                                    "message": (
                                        "The idempotent result references artifacts "
                                        "that are no longer retained"
                                    ),
                                    "type": "IdempotencyResultExpiredError",
                                    "details": {
                                        "missing_artifact_ids": missing_artifacts
                                    },
                                },
                            },
                            duration_ms=duration_ms,
                            session_id=operation_session_id,
                            auto_artifact=False,
                            idempotency_key=idempotency_key,
                            idempotency_replayed=True,
                        )
                    cached_response.setdefault("meta", {})[
                        "idempotency_replayed"
                    ] = True
                    return cached_response
                if claimed_state == "in_progress":
                    duration_ms = int((time.monotonic() - started_at) * 1000)
                    return build_tool_response(
                        fn.__name__,
                        {
                            "success": False,
                            "code": "IDEMPOTENCY_IN_PROGRESS",
                            "error": "An operation with this idempotency key is running",
                            "type": "IdempotencyInProgressError",
                        },
                        duration_ms=duration_ms,
                        session_id=operation_session_id,
                        auto_artifact=False,
                        idempotency_key=idempotency_key,
                    )
                idempotency_claimed = True

            if services is not None and track_operation:
                try:
                    operation = services.operation_registry.begin(
                        fn.__name__,
                        session_id=operation_session_id,
                    )
                    operation_token = bind_operation(operation.operation_id)
                except Exception as exc:
                    logger.exception("could not start operation for %s", fn.__name__)
                    if idempotency_claimed and idempotency_key is not None:
                        services.operation_registry.abort_idempotency(
                            fn.__name__, operation_session_id, idempotency_key
                        )
                    duration_ms = int((time.monotonic() - started_at) * 1000)
                    return build_tool_response(
                        fn.__name__,
                        {
                            "success": False,
                            "error": str(exc),
                            "type": type(exc).__name__,
                        },
                        duration_ms=duration_ms,
                        session_id=operation_session_id,
                        artifact_store=artifact_store,
                        auto_artifact=False,
                        idempotency_key=idempotency_key,
                    )
            try:
                result = await fn(*args, **kwargs)
            except asyncio.CancelledError:
                logger.warning("tool task cancelled in %s", fn.__name__)
                if services is not None and operation is not None:
                    services.operation_registry.request_cancel(operation.operation_id)
                    services.operation_registry.complete(
                        operation.operation_id,
                        success=False,
                        session_id=operation_session_id,
                        error_code="OPERATION_CANCELLED",
                        error_message="Client cancelled the tool request",
                    )
                if (
                    services is not None
                    and idempotency_claimed
                    and idempotency_key is not None
                ):
                    services.operation_registry.abort_idempotency(
                        fn.__name__, operation_session_id, idempotency_key
                    )
                if operation_token is not None:
                    reset_operation(operation_token)
                    operation_token = None
                raise
            except Exception as e:
                logger.exception("tool error in %s", fn.__name__)
                result = {
                    "success": False,
                    "error": str(e),
                    "type": type(e).__name__,
                }
            duration_ms = int((time.monotonic() - started_at) * 1000)
            try:
                response = build_tool_response(
                    fn.__name__,
                    result,
                    duration_ms=duration_ms,
                    session_id=operation_session_id,
                    artifact_store=artifact_store,
                    auto_artifact=auto_artifact,
                    operation_id=(operation.operation_id if operation else None),
                    idempotency_key=idempotency_key,
                )
            except Exception as exc:
                logger.exception("response handling failed in %s", fn.__name__)
                response = build_tool_response(
                    fn.__name__,
                    {
                        "success": False,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    },
                    duration_ms=duration_ms,
                    session_id=operation_session_id,
                    auto_artifact=False,
                    operation_id=(operation.operation_id if operation else None),
                    idempotency_key=idempotency_key,
                )
            finally:
                if operation_token is not None:
                    reset_operation(operation_token)

            if services is not None and operation is not None:
                normalized_error = response.get("error") or {}
                completed_operation = services.operation_registry.complete(
                    operation.operation_id,
                    success=bool(response.get("success")),
                    session_id=response.get("session_id"),
                    error_code=normalized_error.get("code"),
                    error_message=normalized_error.get("message"),
                )
                if completed_operation.state == "cancelled" and response.get("success"):
                    response["success"] = False
                    response["error"] = {
                        "code": "OPERATION_CANCELLED",
                        "message": "Operation cancellation was requested",
                        "type": "OperationCancelledError",
                        "details": None,
                    }
                    response.setdefault("warnings", []).append(
                        "Backend output is preserved in data because cancellation raced completion"
                    )
            if (
                services is not None
                and idempotency_claimed
                and idempotency_key is not None
            ):
                if response.get("success"):
                    services.operation_registry.complete_idempotency(
                        fn.__name__,
                        operation_session_id,
                        idempotency_key,
                        argument_fingerprint,
                        response,
                    )
                else:
                    # Only terminal successes are cached: a transient failure
                    # must not permanently poison this tool+args combo by
                    # replaying the failure on every retry until TTL eviction.
                    services.operation_registry.abort_idempotency(
                        fn.__name__, operation_session_id, idempotency_key
                    )
            return response

        wrapper.__annotations__ = dict(wrapper.__annotations__)
        wrapper.__annotations__["return"] = ToolResponse
        setattr(
            wrapper,
            "__signature__",
            fn_signature.replace(return_annotation=ToolResponse),
        )
        return wrapper

    return decorator


def get_services(ctx: Context) -> AppServices:
    services = ctx.lifespan_context.get("services")
    if services is None:
        raise RuntimeError("Runtime services not initialized")
    return services


async def run_blocking(fn: Callable[[], T]) -> T:
    return await asyncio.to_thread(fn)


def require_session_registry(services: AppServices) -> DebugSessionRegistry:
    return services.session_registry


def resolve_debug_session(
    services: AppServices,
    session_id: Optional[str] = None,
    create_if_missing: bool = True,
    backend: Optional[str] = None,
) -> DebugSession:
    registry = require_session_registry(services)

    if session_id is not None:
        existing = registry.get_session(session_id)
        if existing:
            if backend:
                _assert_backend_matches(existing, backend)
            return existing
        if not create_if_missing:
            raise SessionNotFoundError(f"Debug session '{session_id}' not found")
        return registry.create_session(session_id, backend=backend)

    if not create_if_missing:
        raise SessionNotFoundError("session_id is required")
    return registry.ensure_session(backend=backend)


def _assert_backend_matches(session: DebugSession, backend: str) -> None:
    """Validate a requested backend against an existing session's backend."""
    validate_backend_name(backend)
    if backend == "auto":
        return
    current = getattr(session.backend, "name", "gdb") if session.backend else "gdb"
    if current != backend:
        raise ValueError(
            f"Debug session '{session.session_id}' already runs the {current} "
            f"backend; requested {backend}"
        )


def sync_session_pid(session: DebugSession) -> None:
    pid = session.gdb.get_inferior_pid() or session.state.pid
    session.state.pid = pid


def run_session_locked(
    session: DebugSession,
    action: Callable[[], T],
    *,
    sync_pid: bool = True,
) -> T:
    with session.lock:
        if session.closing or session.closed:
            raise SessionClosedError(f"Debug session '{session.session_id}' is closed")
        destroyed = getattr(session.gdb, "is_destroyed", None)
        if callable(destroyed) and destroyed():
            session.closed = True
            raise SessionClosedError(
                f"Debug session '{session.session_id}' is destroyed and must be recreated"
            )
        result = action()
        if sync_pid:
            sync_session_pid(session)
        return result


async def run_session_action(
    session: DebugSession,
    action: Callable[[], T],
    *,
    sync_pid: bool = True,
) -> T:
    return await run_blocking(
        lambda: run_session_locked(session, action, sync_pid=sync_pid)
    )


def resolve_binary_path(
    binary_path: Optional[str],
    session: DebugSession,
    require_exists: bool = True,
    workspace_root: str = DEFAULT_WORKSPACE,
) -> str:
    if binary_path:
        return resolve_workspace_path(
            binary_path,
            workspace_root=workspace_root,
            require_exists=require_exists,
            kind="binary_path",
        )

    if session.state.binary_path:
        return resolve_workspace_path(
            session.state.binary_path,
            workspace_root=workspace_root,
            require_exists=require_exists,
            kind="binary_path",
        )

    fallback = os.path.join(workspace_root, "target")
    if os.path.exists(fallback):
        return fallback
    raise RuntimeError(
        "No binary selected. Provide binary_path or call set_file first. "
        f"Binaries must be server-visible under {workspace_root}."
    )


def resolve_pipe_session_id(services: AppServices, session_id: str) -> str:
    session = resolve_debug_session(
        services, session_id=session_id, create_if_missing=False
    )
    return session.session_id


def get_paopipe(services: AppServices, session_id: str) -> Tuple[str, PaoPipe]:
    resolved_session_id = resolve_pipe_session_id(services, session_id=session_id)
    with services.paopipe_lock:
        pipe = services.paopipe_sessions.get(resolved_session_id)
        if not pipe:
            raise RuntimeError(
                f"No active paocli session for session_id='{resolved_session_id}'"
            )
        return resolved_session_id, pipe
