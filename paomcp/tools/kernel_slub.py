"""MCP tools for the kernel SLUB slab analyzer (Stage 6.10).

Resolves ``struct kmem_cache`` / ``struct slab`` field offsets from BTF and
decodes the (optionally obfuscated) SLUB freelist.  Both tools are read-only:
they only ever read target memory -- nothing is written, so there is no
``idempotency_key`` and no mutation-set entry.
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
    async def inspect_slub_cache(
        session_id: str,
        cache_address: str,
        btf_path: Optional[str] = None,
        page_size: int = 4096,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Inspect a kernel ``struct kmem_cache``: name, object size,
        freepointer offset, hardening, and slab geometry.

        ``cache_address`` is the address of a ``struct kmem_cache`` (e.g. a
        ``kmalloc-*`` cache) in a live kernel target.  Field offsets are
        resolved from BTF: by default ``/sys/kernel/btf/vmlinux`` on the
        host, or pass ``btf_path`` for a BTF blob file.  Read-only.  Returns
        the cache's size class, ``random`` value, hardening state, order, and
        objects-per-slab, plus the BTF source and resolved fields.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.inspect_slub_cache(
                cache_address=cache_address,
                btf_path=btf_path,
                page_size=page_size,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def decode_slub_freelist(
        session_id: str,
        cache_address: str,
        slab_address: str,
        slab_mem_address: Optional[str] = None,
        btf_path: Optional[str] = None,
        page_size: int = 4096,
        max_objects: Optional[int] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Decode a SLUB slab's free-object freelist.

        ``cache_address`` is a ``struct kmem_cache``; ``slab_address`` is the
        address of the slab's ``struct slab`` (or ``struct page`` on older
        kernels) header.  ``slab_mem_address`` optionally names the slab page
        itself (its header lives at the page base); when omitted it is derived
        from the decoded objects.  Object slots are XOR/swap-obfuscated when
        the cache is hardened (``CONFIG_SLAB_FREELIST_HARDENED``); both the
        5.x legacy and >=5.7 ``swab`` encodings and both raw and encoded head
        interpretations are auto-detected.  Returns the decoded object
        addresses, per-link evidence, and every candidate attempt.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.decode_slub_freelist(
                cache_address=cache_address,
                slab_address=slab_address,
                slab_mem_address=slab_mem_address,
                btf_path=btf_path,
                page_size=page_size,
                max_objects=max_objects,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result


__all__ = ["register"]
