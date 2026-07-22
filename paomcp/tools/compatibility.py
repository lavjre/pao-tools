"""MCP tools for discovering the installed GDB/pwndbg capabilities."""

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_session_action,
)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def get_debugger_capabilities(
        session_id: str,
        refresh: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get installed GDB/pwndbg versions, commands, categories, and feature gaps.

        Args:
            session_id: Existing debugger session used to query the real runtime.
            refresh: Ignore the session cache and probe pwndbg again.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.get_debugger_capabilities(refresh)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def list_pwndbg_commands(
        session_id: str,
        category: Optional[str] = None,
        query: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List actual pwndbg commands with parsed fields and raw evidence.

        Args:
            session_id: Existing debugger session used for discovery.
            category: Optional category from ``pwndbg --list-categories``.
            query: Optional command/description search text. Do not combine with category.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.list_pwndbg_commands(category, query),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def get_pwndbg_command_help(
        command_name: str,
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get help for one validated GDB/pwndbg command with raw evidence."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.get_pwndbg_command_help(command_name)
        )
        result["session_id"] = session.session_id
        return result
