"""MCP tools for the attack-helper framework (Stage 6.9).

Attack helpers execute exploitation primitives against the live target.  The
first one, ``poison_tcache_link``, computes the PROTECT_PTR-encoded ``fd``
that makes a later ``malloc`` return an arbitrary writable address -- and,
because writing into a target is a destructive act, it *never* writes by
default: ``dry_run`` is ``True`` unless the caller explicitly flips it, and
an apply path snapshots the heap first, writes through the mapping-validated
guarded writer, and verifies the result.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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
    async def poison_tcache_link(
        session_id: str,
        victim_address: str,
        target_address: str,
        address_kind: str = "chunk",
        dry_run: bool = True,
        snapshot_before: bool = True,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Re-point a freed glibc tcache chunk's fd so a later malloc returns
        target_address.

        ``victim_address`` is the chunk header address of a freed chunk
        sitting in a tcache bin (see ``inspect_tcache``:
        ``bins[i].chain[j].chunk_address``; pass ``address_kind="user"`` to
        give the user address instead).  ``target_address`` is the *user*
        pointer the following ``malloc`` of the same size class should return
        (e.g. a writable global such as ``__free_hook``-adjacent storage).

        Safe by default: ``dry_run=true`` computes the encoded bytes, the
        write plan, and the malloc follow-up and writes nothing.  Set
        ``dry_run=false`` to apply -- which captures a named pre-poison heap
        snapshot (unless ``snapshot_before=false``), writes the fd through
        the mapping-validated guarded writer, and verifies the re-read.
        Re-applying an identical poison is a no-op (``already_applied``).
        glibc only; other allocators return a typed ``unsupported`` verdict.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.poison_tcache_link(
                victim_address=victim_address,
                target_address=target_address,
                address_kind=address_kind,
                dry_run=dry_run,
                snapshot_before=snapshot_before,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result


__all__ = ["register"]
