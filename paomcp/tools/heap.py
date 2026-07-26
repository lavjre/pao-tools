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

    @mcp.tool()
    @catch_errors()
    async def inspect_chunk(
        session_id: str,
        address: str,
        address_kind: str = "user",
        include_next: bool = True,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read + parse a glibc malloc_chunk at ``address`` with full evidence.

        Reads the ``2 * pointer_size`` chunk header from the live inferior,
        decodes ``prev_size``/``size``/flags (``PREV_INUSE``, ``IS_MMAPPED``,
        ``NON_MAIN_ARENA``) with per-field raw hex evidence, and returns the
        chunk vs user address, the physical next/previous chunk addresses, and
        structural checks (alignment, minimum chunk size, mapping containment).

        When ``include_next`` is true the physically next chunk's header is
        also read so the allocation status of this chunk can be inferred from
        that chunk's ``PREV_INUSE`` bit; ``IS_MMAPPED`` chunks report
        ``allocated`` by construction because they stand outside heap arenas.
        The tool refuses to parse a non-glibc target and returns
        ``verdict = "unsupported"``.

        Args:
            session_id: Debugger session whose live inferior should be inspected.
            address: Chunk or user pointer as a hex/decimal string.
            address_kind: ``"user"`` (default -- ``address`` is a pointer
                returned by ``malloc``) or ``"chunk"`` (``address`` is the
                chunk header address).
            include_next: Read the physically next chunk to infer allocation
                status.  Set to ``false`` to skip the extra read.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.inspect_chunk(
                address=address,
                address_kind=address_kind,
                include_next=include_next,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def list_arenas(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Enumerate glibc arenas by walking the ``main_arena.next`` chain.

        Requires libc built with debug symbols to resolve ``main_arena`` and
        the ``malloc_state.next`` field offset; without them the tool
        returns ``verdict = "unresolvable"`` rather than guessing.  The
        traversal is bounded (256 arenas max) with cycle detection so a
        corrupted heap cannot hang the tool.

        Each returned arena carries its address, index in the chain, a
        ``is_main`` flag, whether the address lies in any known memory
        mapping, and the next-arena pointer read from live memory.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.list_arenas(
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def inspect_arena(
        session_id: str,
        arena_address: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read and decode a full glibc ``malloc_state`` struct.

        Reads ``sizeof(struct malloc_state)`` bytes at the arena address
        and decodes each field (mutex, flags, have_fastchunks, fastbinsY,
        top, last_remainder, bins, binmap, next, next_free,
        attached_threads, system_mem, max_system_mem) using offsets
        resolved from libc debug symbols.  Fields whose offset cannot be
        resolved or whose byte range falls outside the read are reported
        under ``unreadable_fields`` rather than silently defaulted.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            arena_address: Hex/decimal address of a non-main arena
                (typically obtained from ``list_arenas``).  Omit to inspect
                main_arena.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.inspect_arena(
                arena_address=arena_address,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def inspect_fastbins(
        session_id: str,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = 128,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Walk the 10 glibc fastbin chains rooted in an arena's ``fastbinsY``.

        Reads the ten fastbin heads out of the arena's ``fastbinsY`` array
        and walks each singly-linked chain via the free-chunk ``fd``
        pointer.  When the target's glibc has safe-linking enabled
        (>= 2.32) the ``fd`` field is decoded with
        ``PROTECT_PTR = fd XOR (fd_field_addr >> 12)`` so the reported
        chain addresses are usable directly; the raw encoded value is
        preserved alongside the decoded one for evidence.

        Each chain is bounded (default 128 chunks per bin) with cycle
        detection so a corrupted allocator can never hang the tool.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            arena_address: Optional hex/decimal arena address (typically
                obtained from ``list_arenas``).  Omit to walk main_arena's
                fastbins.
            max_chunks_per_bin: Per-fastbin traversal cap (default 128).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.inspect_fastbins(
                arena_address=arena_address,
                max_chunks_per_bin=max_chunks_per_bin,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def validate_chunk(
        session_id: str,
        address: str,
        address_kind: str = "user",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return a conservative structural verdict for a glibc chunk.

        Runs the same read + parse as ``inspect_chunk`` and summarizes the
        outcome as one of ``unsupported`` (non-glibc target), ``unreadable``
        (header could not be read), ``corrupted`` (alignment / minimum-size /
        containment check failed), ``suspicious`` (header in a mapping but the
        declared size extends past its end), or ``valid`` (every check passed).
        The response includes the individual check results and the list of
        failed check names so a heuristic is never promoted to a fact.

        Args:
            session_id: Debugger session whose live inferior should be inspected.
            address: Chunk or user pointer as a hex/decimal string.
            address_kind: ``"user"`` (default) or ``"chunk"``.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.validate_chunk(
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
