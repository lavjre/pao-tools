from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_binary_path,
    resolve_debug_session,
    require_session_registry,
    run_session_action,
)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def create_debug_session(
        session_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Create or return a debug session by id."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=True
        )
        return {"success": True, "session": session.to_dict()}

    @mcp.tool()
    @catch_errors()
    async def list_debug_sessions(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List all active debug sessions and metadata."""
        services = get_services(ctx)
        registry = require_session_registry(services)
        sessions = registry.list_sessions()
        return {
            "success": True,
            "count": len(sessions),
            "sessions": sessions,
            "capacity": registry.status(),
        }

    @mcp.tool()
    @catch_errors()
    async def close_debug_session(
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Close an active debug session and stop any attached paocli driver."""
        services = get_services(ctx)
        registry = require_session_registry(services)
        session = registry.get_session(session_id)
        resolved_session_id = session.session_id if session else session_id

        with services.paopipe_lock:
            pipe = services.paopipe_sessions.pop(resolved_session_id, None)
        if pipe:
            pipe.kill()

        result = registry.close_session(
            session_id,
            on_closing=lambda _session: (
                services.subprocess_tools.close_session_processes(resolved_session_id)
            ),
        )
        if "cleanup" in result:
            result["managed_processes"] = result.pop("cleanup")
        return result

    @mcp.tool()
    @catch_errors()
    async def execute(
        command: str,
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Execute an arbitrary GDB/pwndbg command.

        Args:
            command: Raw GDB/pwndbg command to execute (e.g., "info registers", "vmmap").

        Returns:
            Dict containing the raw MI/console responses, a success flag, and the current GDB state.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.execute(command)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def set_file(
        binary_path: str,
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Load an executable file into GDB/pwndbg for debugging.

        Args:
            binary_path: Absolute path to the ELF to debug. Use the container-visible
                path under /workspace; relative paths resolve under /workspace. If your
                host file is ./workspace/chal, pass /workspace/chal here.

        Returns:
            Dict with MI command responses and state.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        resolved_binary = resolve_binary_path(
            binary_path,
            session,
            require_exists=True,
            workspace_root=services.runtime_paths.workspace_root,
        )
        result = await run_session_action(
            session, lambda: session.tools.set_file(resolved_binary)
        )
        result["session_id"] = session.session_id
        result["binary_path"] = resolved_binary
        return result

    @mcp.tool()
    @catch_errors()
    async def attach(
        pid: int,
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Attach to an existing process by PID using GDB/MI.

        Args:
            pid: Target process ID to attach to.

        Returns:
            Structured payload containing the MI attach result and immediate context
            snapshots (for example, backtrace and heap context).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result, context = await run_session_action(
            session, lambda: session.tools.attach(pid)
        )
        result["session_id"] = session.session_id
        response = {
            "success": bool(result.get("success")),
            "session_id": session.session_id,
            "result": result,
            "context": context,
        }
        if not response["success"]:
            response["error"] = result.get("error", f"Failed to attach to pid {pid}")
            response["type"] = result.get("type", "AttachError")
        return response

    @mcp.tool()
    @catch_errors()
    async def run(
        session_id: str,
        args: str = "",
        start: bool = False,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run the loaded program under GDB control.

        Args:
            args: Argument string passed to the inferior.
            start: If True, stop at the program entry (equivalent to --start).

        Returns:
            MI run/continue results and state.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.run(args, start)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def step_control(
        command: str,
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Execute a stepping command (c, n, s, ni, si).

        Args:
            command: One of {c, n, s, ni, si} or their long forms.

        Returns:
            Dict with MI responses and state.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.step_control(command)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def gdb_poll(
        session_id: str,
        timeout: float = 0.0,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Drain pending async GDB notifications.

        Args:
            timeout: Maximum time to wait (seconds) for the first event.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.gdb_poll(timeout)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def gdb_interrupt(
        session_id: str,
        timeout: float = 1.0,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Interrupt the inferior and drain async notifications.

        Args:
            timeout: Maximum time to wait (seconds) for stop notifications.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.gdb_interrupt(timeout)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def finish(
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run until the current function returns (MI -exec-finish)."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(session, lambda: session.tools.finish())
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def jump(
        locspec: str,
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Resume execution at a specified location (MI -exec-jump).

        Args:
            locspec: Location such as a symbol name, file:line, or address (*0x... ).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(session, lambda: session.tools.jump(locspec))
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def return_from_function(
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Force the current function to return immediately (MI -exec-return)."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.return_from_function()
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def until(
        session_id: str,
        locspec: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run until a specified location or next source line (MI -exec-until)."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(session, lambda: session.tools.until(locspec))
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def set_breakpoint(
        location: str,
        session_id: str,
        condition: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Set a breakpoint using MI (-break-insert).

        Args:
            location: Breakpoint location (symbol/address/file:line).
            condition: Optional breakpoint condition expression.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.set_breakpoint(location, condition)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def get_session_info(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return current session info (session state + GDB state) without issuing new GDB commands."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.get_session_info()
        )
        result["session_id"] = session.session_id
        result["runtime_dir"] = session.runtime_dir
        return result
