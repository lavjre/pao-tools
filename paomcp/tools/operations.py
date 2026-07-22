"""MCP tools for operation progress, status, and cooperative cancellation."""

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.operations import OperationOwnershipError
from paomcp.tools.common import catch_errors, get_services, run_blocking


def _assert_access(operation: Dict[str, Any], session_id: Optional[str]) -> None:
    owner = operation.get("session_id")
    if owner != session_id:
        raise OperationOwnershipError(
            "The operation belongs to a different debugger session"
        )


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors(auto_artifact=False, track_operation=False)
    async def list_operations(
        session_id: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 100,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List operation state for one session, or only unscoped operations.

        Supply ``session_id`` to view work owned by that debugger session. If it
        is omitted, session-scoped operations are intentionally hidden.
        """
        services = get_services(ctx)
        operations = await run_blocking(
            lambda: services.operation_registry.list(
                session_id=session_id, state=state, limit=limit
            )
        )
        return {
            "success": True,
            "session_id": session_id,
            "count": len(operations),
            "operations": operations,
            "capacity": services.operation_registry.status(),
        }

    @mcp.tool()
    @catch_errors(auto_artifact=False, track_operation=False)
    async def get_operation(
        operation_id: str,
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get progress and terminal state for one operation."""
        services = get_services(ctx)
        operation = await run_blocking(
            lambda: services.operation_registry.get(operation_id)
        )
        _assert_access(operation, session_id)
        return {"success": True, "session_id": session_id, "operation": operation}

    @mcp.tool()
    @catch_errors(auto_artifact=False, track_operation=False)
    async def cancel_operation(
        operation_id: str,
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Request cooperative cancellation of a running operation."""
        services = get_services(ctx)
        existing = await run_blocking(
            lambda: services.operation_registry.get(operation_id)
        )
        _assert_access(existing, session_id)
        operation = await run_blocking(
            lambda: services.operation_registry.request_cancel(operation_id)
        )
        return {"success": True, "session_id": session_id, "operation": operation}
