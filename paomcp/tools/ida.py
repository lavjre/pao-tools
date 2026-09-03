"""MCP tools bridging mrexodia/ida-pro-mcp (GUI + idalib headless)."""

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import catch_errors, get_services, run_blocking


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors(auto_artifact=False, track_operation=False)
    async def get_ida_status(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Report IDA MCP reachability (GUI 13337 + idalib 8745)."""
        services = get_services(ctx)
        return await run_blocking(services.ida.status)

    @mcp.tool()
    @catch_errors()
    async def ida_decompile(
        addr: Any,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Decompile func at addr via IDA; needs idb session id."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.decompile(addr, database))

    @mcp.tool()
    @catch_errors()
    async def ida_disasm(
        addr: Any,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Disassemble func at addr via IDA (args, frame, details)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.disasm(addr, database))

    @mcp.tool()
    @catch_errors()
    async def ida_list_funcs(
        queries: Optional[Any] = None,
        database: Optional[str] = None,
        limit: int = 100,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List IDA funcs, paginated; default 100, max 1000."""
        services = get_services(ctx)
        return await run_blocking(
            lambda: services.ida.list_funcs(queries, database, limit)
        )

    @mcp.tool()
    @catch_errors()
    async def ida_lookup(
        queries: Any,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Lookup IDA funcs by addr or name (single or batch)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.lookup(queries, database))

    @mcp.tool()
    @catch_errors()
    async def ida_xrefs_to(
        addrs: Any,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Xrefs to addr(s) via IDA."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.xrefs_to(addrs, database))

    @mcp.tool()
    @catch_errors()
    async def ida_get_bytes(
        addrs: Any,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read raw bytes at IDA addr(s)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.get_bytes(addrs, database))

    @mcp.tool()
    @catch_errors()
    async def ida_get_string(
        addrs: Any,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read C strings at IDA addr(s)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.get_string(addrs, database))

    @mcp.tool()
    @catch_errors()
    async def ida_find_bytes(
        patterns: Any,
        database: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Find byte pattern (e.g. '48 8B ??'); max 1000 hits."""
        services = get_services(ctx)
        return await run_blocking(
            lambda: services.ida.find_bytes(patterns, database, limit, offset)
        )

    @mcp.tool()
    @catch_errors()
    async def ida_py_eval(
        code: str,
        database: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run Python in IDA (returns result/stdout/stderr)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.py_eval(code, database))

    @mcp.tool()
    @catch_errors()
    async def ida_rename(
        batch: Dict[str, Any],
        database: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Batch rename funcs/data/locals/stack via IDA."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.rename(batch, database))

    @mcp.tool()
    @catch_errors()
    async def ida_set_comments(
        items: Any,
        database: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Set IDA comments at addr(s), both views."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.set_comments(items, database))

    @mcp.tool()
    @catch_errors()
    async def ida_idb_open(
        input_path: str,
        mode: str = "prefer_headless",
        preferred_session_id: str = "",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Open binary in idalib worker; returns session id for database arg."""
        services = get_services(ctx)
        return await run_blocking(
            lambda: services.ida.idb_open(input_path, mode, preferred_session_id)
        )

    @mcp.tool()
    @catch_errors(auto_artifact=False, track_operation=False)
    async def ida_idb_list(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List open IDA sessions and GUI instances."""
        services = get_services(ctx)
        return await run_blocking(services.ida.idb_list)

    @mcp.tool()
    @catch_errors()
    async def ida_idb_close(
        database: str,
        save: bool = True,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Close IDA session; frees worker slot (GUI detached not killed)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.idb_close(database, save))

    @mcp.tool()
    @catch_errors()
    async def ida_publish_crash(
        address: Any,
        signal: str,
        database: Optional[str] = None,
        classification: Optional[str] = None,
        note: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Publish a debugger crash finding as an IDA comment (GDB->IDB)."""
        services = get_services(ctx)
        return await run_blocking(
            lambda: services.ida.publish_crash(
                address,
                signal,
                database,
                classification=classification,
                note=note,
            )
        )

    @mcp.tool()
    @catch_errors()
    async def ida_publish_note(
        address: Any,
        text: str,
        database: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Publish free text as an IDA comment at an address (GDB->IDB)."""
        services = get_services(ctx)
        return await run_blocking(
            lambda: services.ida.publish_note(address, text, database)
        )

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_start(
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Start IDA debugger (needs ?ext=dbg on IDA side)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_start(database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_continue(
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Continue IDA debugger."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_continue(database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_run_to(
        addr: Any,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run IDA debugger to addr."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_run_to(addr, database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_step(
        kind: str = "into",
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Step IDA debugger (kind into/over)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_step(kind, database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_bps(
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List IDA debugger breakpoints."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_bps(database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_add_bp(
        addrs: Any,
        database: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Add IDA breakpoint(s)."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_add_bp(addrs, database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_regs(
        names: Optional[Any] = None,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read IDA regs; names set => named subset."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_regs(names, database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_stacktrace(
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Call stack from IDA debugger."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_stacktrace(database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_read(
        regions: Any,
        database: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read memory from debugged process via IDA."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_read(regions, database))

    @mcp.tool()
    @catch_errors()
    async def ida_dbg_write(
        regions: Any,
        database: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Write memory to debugged process via IDA."""
        services = get_services(ctx)
        return await run_blocking(lambda: services.ida.dbg_write(regions, database))
