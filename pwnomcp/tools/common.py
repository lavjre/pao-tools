import asyncio
import inspect
import logging
import os
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from fastmcp import Context

from pwnomcp.api import ToolError, ToolMeta, ToolResponse
from pwnomcp.artifacts import ArtifactStore
from pwnomcp.pwnpipe import PwnPipe
from pwnomcp.services import AppServices
from pwnomcp.state import DebugSession, DebugSessionRegistry
from pwnomcp.utils.paths import DEFAULT_WORKSPACE, resolve_workspace_path

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _error_code(error_type: Optional[str]) -> str:
    if not error_type:
        return "TOOL_ERROR"
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", error_type).upper()
    return snake_case or "TOOL_ERROR"


def build_tool_response(
    tool: str,
    result: Any,
    *,
    duration_ms: int = 0,
    session_id: Optional[str] = None,
    artifact_store: Optional[ArtifactStore] = None,
    auto_artifact: bool = True,
) -> Dict[str, Any]:
    """Normalize a tool result into the versioned public response envelope.

    Payloads are preserved in full. This function does not truncate stdout,
    debugger responses, decompiled code, events, or any other tool data.
    """
    if isinstance(result, ToolResponse):
        response = result
        response.meta.duration_ms = duration_ms
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
        meta=ToolMeta(duration_ms=duration_ms),
    )

    if auto_artifact and artifact_store is not None and response.data is not None:
        serialized_data = artifact_store.serialize_json(response.data)
        if artifact_store.should_store(serialized_data):
            artifact = artifact_store.save_json_bytes(
                serialized_data,
                tool=tool,
                session_id=session_id,
            )
            response.data = artifact.to_dict()
    return response.as_dict()


def catch_errors(*, auto_artifact: bool = True) -> Callable[[Callable[..., Any]], Any]:
    """Wrap an MCP tool with the shared response envelope and error handling."""

    def decorator(fn: Callable[..., Any]) -> Any:
        fn_signature = inspect.signature(fn)

        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = time.monotonic()
            bound_arguments = fn_signature.bind_partial(*args, **kwargs)
            requested_session_id = bound_arguments.arguments.get("session_id")
            artifact_store = None
            context = bound_arguments.arguments.get("ctx")
            if context is not None:
                try:
                    services = context.lifespan_context.get("services")
                    if services is not None:
                        artifact_store = services.artifact_store
                except (AttributeError, RuntimeError):
                    artifact_store = None
            try:
                result = await fn(*args, **kwargs)
            except Exception as e:
                logger.exception("tool error in %s", fn.__name__)
                result = {
                    "success": False,
                    "error": str(e),
                    "type": type(e).__name__,
                }
            duration_ms = int((time.monotonic() - started_at) * 1000)
            return build_tool_response(
                fn.__name__,
                result,
                duration_ms=duration_ms,
                session_id=(
                    str(requested_session_id)
                    if requested_session_id is not None
                    else None
                ),
                artifact_store=artifact_store,
                auto_artifact=auto_artifact,
            )

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
) -> DebugSession:
    registry = require_session_registry(services)

    if session_id is not None:
        existing = registry.get_session(session_id)
        if existing:
            return existing
        if not create_if_missing:
            raise RuntimeError(f"Debug session '{session_id}' not found")
        return registry.create_session(session_id)

    if not create_if_missing:
        raise RuntimeError("session_id is required")
    return registry.ensure_session()


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


def get_pwnpipe(services: AppServices, session_id: str) -> Tuple[str, PwnPipe]:
    resolved_session_id = resolve_pipe_session_id(services, session_id=session_id)
    with services.pwnpipe_lock:
        pipe = services.pwnpipe_sessions.get(resolved_session_id)
        if not pipe:
            raise RuntimeError(
                f"No active pwncli session for session_id='{resolved_session_id}'"
            )
        return resolved_session_id, pipe
