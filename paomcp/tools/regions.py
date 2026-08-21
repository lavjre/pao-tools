"""MCP tools for JIT / memory-region tracking (Stage 6.12).

Two read-only tools over the live inferior's memory mappings: scan for
anonymous executable regions (V8/Chrome JIT-heap candidates) and classify
whether an address falls inside a mapped region.  Region tracking is
allocator-independent -- no family gate.  Nothing is written, so there is no
``idempotency_key`` and no mutation-set entry.
"""

from __future__ import annotations

from typing import Any, Dict

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.operations import current_operation_id
from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_session_action,
)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def scan_jit_regions(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Scan the live inferior's memory mappings for JIT-heap candidates.

        Lists anonymous executable (``jit_candidate``) regions -- the V8/Chrome
        JIT heap signature -- sorted by size descending, with each region's
        start/end/size/permissions and the executable/writable/anonymous/
        jit_candidate/rwx flags, plus a summary (count, total_bytes,
        largest_region).  Verdict is ``listed``, ``none_found``, or
        ``unavailable``.  Allocator-independent.  Read-only.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.scan_jit_regions(
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def classify_memory_region(
        session_id: str,
        address: str,
        address_kind: str = "user",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Classify whether an address falls inside a mapped memory region.

        ``address`` is a hex or decimal address.  Returns ``mapped`` with the
        containing region and its executable/writable/anonymous/
        jit_candidate/rwx flags, ``unmapped`` when the address is in a gap or
        past the end of the containing mapping, or ``unavailable`` when maps
        are unavailable.  ``address_kind`` is descriptive metadata echoed in
        the response.  Read-only.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.classify_memory_region(
                address=address,
                address_kind=address_kind,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result


__all__ = ["register"]
