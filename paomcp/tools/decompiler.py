"""MCP tools for local Ghidra headless decompilation."""

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import catch_errors, get_services, run_blocking
from paomcp.utils.paths import resolve_workspace_path


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def get_decompiler_status(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Report local Ghidra availability, version, limits, and script paths."""
        services = get_services(ctx)
        return await run_blocking(services.decompiler.status)

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def decompile_binary(
        binary_path: str,
        function: Optional[str] = None,
        timeout: float = 300.0,
        function_timeout: int = 30,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Decompile an ELF or one function locally with Ghidra headless.

        Args:
            binary_path: Binary under `/workspace`; relative paths resolve there.
            function: Optional exact function name or address such as `main` or
                `0x4011a0`. When omitted, all discovered functions are decompiled.
            timeout: Overall analysis timeout in seconds, greater than 0 and at most
                3600.
            function_timeout: Per-function decompiler timeout, from 1 to 600 seconds.

        Large pseudo-C, manifests, and logs are stored losslessly as artifacts.
        No external decompilation service or binary upload is used.
        """
        services = get_services(ctx)
        resolved_binary = resolve_workspace_path(
            binary_path,
            workspace_root=services.runtime_paths.workspace_root,
            require_exists=True,
            kind="binary_path",
        )
        return await run_blocking(
            lambda: services.decompiler.decompile(
                resolved_binary,
                function=function,
                timeout=timeout,
                function_timeout=function_timeout,
            )
        )
