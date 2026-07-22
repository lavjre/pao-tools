from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_blocking,
    run_session_action,
)
from paomcp.utils.paths import resolve_workspace_cwd, validate_session_id


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def run_command(
        command: str,
        cwd: Optional[str] = None,
        timeout: float = 30.0,
        session_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Execute a system command and wait for completion.

        Note:
            Use this for build and helper commands.
            Do not use this to run the target binary under analysis.
            Use set_file + run (or attach) for the target.
            Use execute_python_code instead of ``python -c``.
            Use paocli for interactive exploit-driver workflows.
            Combined stdout and stderr have a hard configured byte limit. Crossing it
            terminates the process tree and returns RESOURCE_EXHAUSTED with exact
            stored and dropped byte counts.

        Args:
            command: Shell command to run.
            cwd: Working directory (defaults to /workspace). Use a container-visible path
                under /workspace; relative paths resolve under /workspace.
            timeout: Timeout in seconds.
            idempotency_key: Optional retry key. Reusing it with identical arguments
                returns the original normalized response without executing again.

        Returns:
            Structured response with stdout/stderr/exit code.
        """
        services = get_services(ctx)
        if session_id is not None:
            resolve_debug_session(services, session_id, create_if_missing=False)
        tools = services.subprocess_tools
        cwd = resolve_workspace_cwd(
            cwd, workspace_root=services.runtime_paths.workspace_root
        )
        result = await run_blocking(
            lambda: tools.run_command(
                command, cwd=cwd, timeout=timeout, session_id=session_id
            )
        )
        return result

    @mcp.tool()
    @catch_errors()
    async def spawn_process(
        command: str,
        cwd: Optional[str] = None,
        session_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Spawn a long-running background process and return its PID and log paths.

        Note:
            Use this for helper services.
            Do not use this to run the target binary under analysis.
            Use set_file + run (or attach) for debugger workflows.
            Use paocli for exploit interaction.
            Combined stdout and stderr have a hard configured byte limit. Crossing it
            terminates the process tree and is reported by process-status tools.

        Args:
            command: Shell command to spawn.
            cwd: Working directory (defaults to /workspace). Use a container-visible path
                under /workspace; relative paths resolve under /workspace.
            idempotency_key: Optional retry key that prevents duplicate process starts.
        """
        services = get_services(ctx)
        session = None
        if session_id is not None:
            session = resolve_debug_session(
                services, session_id, create_if_missing=False
            )
        tools = services.subprocess_tools
        cwd = resolve_workspace_cwd(
            cwd, workspace_root=services.runtime_paths.workspace_root
        )
        action = lambda: tools.spawn_process(command, cwd=cwd, session_id=session_id)
        if session is not None:
            result = await run_session_action(session, action, sync_pid=False)
        else:
            result = await run_blocking(action)
        return result

    @mcp.tool()
    @catch_errors()
    async def get_process(
        pid: int,
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get information about a tracked background process by PID."""
        services = get_services(ctx)
        if session_id is not None:
            session_id = validate_session_id(session_id)
        result = await run_blocking(
            lambda: services.subprocess_tools.get_process(pid, session_id)
        )
        return result

    @mcp.tool()
    @catch_errors()
    async def kill_process(
        pid: int,
        signal: int = 15,
        session_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Send a signal to a tracked background process (default SIGTERM=15)."""
        services = get_services(ctx)
        if session_id is not None:
            session_id = validate_session_id(session_id)
        result = await run_blocking(
            lambda: services.subprocess_tools.kill_process(pid, signal, session_id)
        )
        return result

    @mcp.tool()
    @catch_errors()
    async def list_processes(
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List all tracked background processes and their metadata (PID, command, log paths)."""
        services = get_services(ctx)
        if session_id is not None:
            session_id = validate_session_id(session_id)
        result = await run_blocking(
            lambda: services.subprocess_tools.list_processes(session_id)
        )
        return result
