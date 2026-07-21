"""MCP tools for typed ELF and runtime inspection through pwndbg."""

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from pwnomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_binary_path,
    resolve_debug_session,
    run_session_action,
)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def analyze_elf(
        session_id: str,
        binary_path: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Analyze ELF mitigations, sections, and PLT entries with typed output.

        Args:
            session_id: Existing debugger session used for the analysis.
            binary_path: Optional ELF under the server workspace. When supplied, it is
                loaded into this debugger session before inspection.
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
            session, lambda: session.tools.analyze_elf(resolved_binary)
        )
        result["session_id"] = session.session_id
        result["binary_path"] = resolved_binary
        return result

    @mcp.tool()
    @catch_errors()
    async def checksec(
        session_id: str,
        binary_path: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return typed ELF mitigation data while preserving pwndbg evidence.

        Args:
            session_id: Existing debugger session used to execute pwndbg.
            binary_path: Optional ELF under the server workspace; otherwise use the
                binary already selected by the session or `/workspace/target`.
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
            session, lambda: session.tools.checksec(resolved_binary)
        )
        result["session_id"] = session.session_id
        result["binary_path"] = resolved_binary
        return result

    @mcp.tool()
    @catch_errors()
    async def inspect_got_plt(
        session_id: str,
        symbol_filter: Optional[str] = None,
        include_got: bool = True,
        include_plt: bool = True,
        include_readonly: bool = False,
        all_objects: bool = False,
        all_plt_symbols: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Inspect GOT and PLT entries for the selected binary/runtime.

        GOT inspection normally needs a stopped inferior. PLT inspection can work on
        a loaded ELF without running it; partial results are returned with warnings.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.inspect_got_plt(
                symbol_filter=symbol_filter,
                include_got=include_got,
                include_plt=include_plt,
                include_readonly=include_readonly,
                all_objects=all_objects,
                all_plt_symbols=all_plt_symbols,
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def memory_maps(
        session_id: str,
        address_or_module: Optional[str] = None,
        writable: bool = False,
        executable: bool = False,
        gaps: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return parsed virtual memory mappings for a stopped inferior.

        Args:
            address_or_module: Optional address expression or module path/name filter.
            writable: Return writable mappings only.
            executable: Return executable mappings only.
            gaps: Include unmapped gaps reported by pwndbg.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.memory_maps(
                address_or_module=address_or_module,
                writable=writable,
                executable=executable,
                gaps=gaps,
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def list_threads(
        session_id: str,
        limit: Optional[int] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List inferior threads as typed records.

        Args:
            limit: Optional maximum thread count between 1 and 10000.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.list_threads(limit=limit)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def get_process_info(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return typed process identity, credentials, cwd, command line, and FDs."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.get_process_info()
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def inspect_libc(
        session_id: str,
        include_link_map: bool = True,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Inspect libc type/version, mappings, symbolication, and dynamic link map."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.inspect_libc(include_link_map=include_link_map),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def inspect_address(
        address: str,
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Resolve an address to its mapping, file offsets, and ELF sections.

        Args:
            address: Safe GDB expression such as `$pc`, `$rsp+0x20`, or `0x401000`.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.inspect_address(address)
        )
        result["session_id"] = session.session_id
        return result
