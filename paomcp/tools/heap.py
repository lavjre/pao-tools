"""MCP tools for typed heap allocator discovery (Stage 6.1)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.operations import current_operation_id
from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_session_action,
)


def _debugger_command_names(debugger_capabilities: Mapping[str, Any]) -> List[str]:
    """Collect pwndbg command names from a cached compatibility report.

    The compatibility report groups commands under ``command_groups`` with an
    ``available`` list per group.  This mirrors the extraction the platform
    manifest performs, but returns a flat, de-duplicated name list for the
    allocator capability matrix.
    """
    names: set[str] = set()
    groups = debugger_capabilities.get("command_groups", {})
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            available = group.get("available", [])
            if isinstance(available, list):
                names.update(str(item) for item in available)
    return sorted(names)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def detect_allocator(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Detect the target's heap allocator family, version, and runtime profile.

        Requires a live, stopped inferior because a heap only exists at runtime.
        The result reports the allocator family (glibc/musl/jemalloc/unknown),
        the resolved libc version when available, a conservative confidence
        label, glibc safe-linking applicability, and the concrete target profile
        (pointer width, byte order, architecture) with per-field provenance.
        Every conclusion carries the raw mapping/libc evidence it was derived
        from; a heuristic is never promoted to a fact.

        Args:
            session_id: Debugger session whose live inferior should be inspected.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.detect_allocator(
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def get_allocator_capabilities(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Report typed heap-inspection capabilities for the live target.

        Detects the allocator, then reports, per allocator family, which pwndbg
        runtime primitives are available in this debugger and whether a typed
        inspection API ships yet. Availability is drawn from the cached debugger
        compatibility report; families with no known runtime primitives are
        reported as ``unknown`` rather than falsely unsupported.

        Args:
            session_id: Debugger session whose live inferior should be inspected.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        command_names = _debugger_command_names(services.debugger_compatibility or {})
        result = await run_session_action(
            session,
            lambda: session.tools.heap.get_allocator_capabilities(
                debugger_command_names=command_names,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result


__all__ = ["register"]
