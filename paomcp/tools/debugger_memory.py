"""MCP tools for typed debugger state, memory, and execution control."""

from typing import Any, Dict, List, Optional

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
    async def list_breakpoints(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List breakpoints as typed records with raw GDB/MI evidence."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.list_breakpoints()
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def delete_breakpoint(
        number: int,
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Delete a GDB breakpoint by its positive numeric id."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.delete_breakpoint(number)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def toggle_breakpoint(
        number: int,
        enable: bool,
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Enable or disable a GDB breakpoint by numeric id."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.toggle_breakpoint(number, enable)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def set_watchpoint(
        expression: str,
        session_id: str,
        access: str = "write",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Set a hardware watchpoint for a validated GDB expression.

        Args:
            expression: Address or lvalue, for example `*(long*)$rsp`.
            access: One of `write`, `read`, or `access`. Hardware support depends on
                the target and runtime; unsupported cases return structured errors.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.set_watchpoint(expression, access)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def read_registers(
        session_id: str,
        names: Optional[List[str]] = None,
        refresh_names: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read selected registers, or every named target register when omitted."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.read_registers(
                names=names, refresh_names=refresh_names
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def write_register(
        name: str,
        value: str,
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Write one register and return before/after values plus raw MI evidence."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.write_register(name, value)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def evaluate_expression(
        expression: str,
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Evaluate a validated GDB expression and return its typed value."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.evaluate_expression_typed(expression)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def select_thread(
        thread_id: int,
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Select the active inferior thread by GDB's positive global thread id."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.select_thread(thread_id)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def search_memory(
        value: str,
        session_id: str,
        value_type: str = "bytes",
        mapping: Optional[str] = None,
        hex_encoded: bool = False,
        executable: bool = False,
        writable: bool = False,
        limit: Optional[int] = None,
        aligned: Optional[int] = None,
        step: Optional[int] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Search inferior memory through pwndbg and parse every returned match.

        Output is not silently truncated. Large raw/parsed results use the shared
        lossless artifact response path, with optional `limit` for caller control.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.search_memory(
                value=value,
                value_type=value_type,
                mapping=mapping,
                hex_encoded=hex_encoded,
                executable=executable,
                writable=writable,
                limit=limit,
                aligned=aligned,
                step=step,
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def write_memory(
        address: str,
        data_hex: str,
        session_id: str,
        verify: bool = True,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Write raw hex bytes to inferior memory and optionally verify read-back."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.write_memory(address, data_hex, verify=verify),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def disassemble(
        session_id: str,
        start: str = "$pc",
        byte_count: int = 64,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Disassemble a bounded byte range into typed MI instruction records."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.disassemble(start=start, byte_count=byte_count),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def step_until_event(
        event: str,
        session_id: str,
        symbol_regex: Optional[str] = None,
        mnemonic: Optional[str] = None,
        operands: Optional[str] = None,
        target: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Step until a call, jump, return, syscall, instruction, or address.

        Args:
            event: `next_call`, `next_jump`, `next_return`, `next_syscall`,
                `step_syscall`, `instruction`, or `address`.
            symbol_regex: Optional function regex for `next_call`.
            mnemonic: Required mnemonic for `instruction`.
            operands: Optional operand expression for `instruction`.
            target: Required address expression for `address`.
        """
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.step_until_event(
                event=event,
                symbol_regex=symbol_regex,
                mnemonic=mnemonic,
                operands=operands,
                target=target,
            ),
        )
        result["session_id"] = session.session_id
        return result
