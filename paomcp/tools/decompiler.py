"""MCP tools for local Ghidra headless decompilation."""

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.operations import current_operation_id
from paomcp.tools.common import catch_errors, get_services, run_blocking
from paomcp.utils.paths import resolve_workspace_file, stable_file_snapshot


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
        refresh: bool = False,
        idempotency_key: Optional[str] = None,
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
            refresh: Bypass the content-addressed Ghidra result cache.
            idempotency_key: Optional retry key for replaying the completed response.

        Large pseudo-C, manifests, and logs are stored losslessly as artifacts.
        No external decompilation service or binary upload is used.
        """
        services = get_services(ctx)
        resolved_binary = resolve_workspace_file(
            binary_path,
            services.runtime_paths.workspace_root,
            kind="binary_path",
        )
        operation_id = current_operation_id()
        cancellation_event = (
            services.operation_registry.cancellation_event(operation_id)
            if operation_id is not None
            else None
        )

        def decompile_snapshot() -> Dict[str, Any]:
            with stable_file_snapshot(
                resolved_binary,
                services.runtime_paths.decompiler_dir,
                workspace_root=services.runtime_paths.workspace_root,
                minimum_free_bytes=services.artifact_store.min_free_bytes,
                cancellation_requested=(
                    cancellation_event.is_set
                    if cancellation_event is not None
                    else None
                ),
            ) as snapshot_path:
                result = services.decompiler.decompile(
                    snapshot_path,
                    function=function,
                    timeout=timeout,
                    function_timeout=function_timeout,
                    refresh=refresh,
                )
                result["binary_path"] = resolved_binary
                return result

        return await run_blocking(decompile_snapshot)
