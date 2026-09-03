"""MCP tools for typed heap allocator discovery (Stage 6.1) and
heap exploitation-technique analysis (Stage 6.6)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.operations import current_operation_id
from paomcp.tools.backends.heap_techniques import GlibcTechniqueAnalyzer
from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_blocking,
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
        """Inspect allocation for addr (glibc/jemalloc/libmalloc routed).

        Args:
            session_id: Session to inspect.
            address: Hex/dec chunk or user pointer.
            address_kind: "user" (malloc ptr) or "chunk" (header addr).
            include_next: Read next chunk for alloc status (glibc).
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
        """Enumerate the allocator's top-level structures, per family.

        The tool is family-routed (see ``detect_allocator``): glibc walks the
        ``main_arena.next`` chain (requiring libc debug symbols to resolve
        ``main_arena`` and the ``malloc_state.next`` offset; without them
        ``verdict = "unresolvable"`` rather than guessing; traversal bounded
        at 256 arenas with cycle detection so a corrupted heap cannot hang
        the tool), jemalloc reports its rtree extent inventory, and libmalloc
        lists its malloc zones.  Other allocators (musl / scudo / unknown)
        yield ``verdict = "unsupported"``.

        Each glibc arena carries its address, index in the chain, a
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
        """Inspect one arena-like structure deep, per allocator family.

        The tool is family-routed (see ``detect_allocator``): glibc reads and
        decodes a full ``malloc_state`` struct (mutex, flags, have_fastchunks,
        fastbinsY, top, last_remainder, bins, binmap, next, next_free,
        attached_threads, system_mem, max_system_mem) using offsets resolved
        from libc debug symbols, with unresolvable or out-of-range fields
        reported under ``unreadable_fields`` rather than silently defaulted;
        jemalloc resolves ``arena_address`` as an rtree extent; libmalloc
        inspects the given malloc zone (default zone when omitted).  Other
        allocators (musl / scudo / unknown) yield
        ``verdict = "unsupported"``.

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
    async def inspect_libmalloc_zones(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List every libmalloc zone (Apple target) with its tables.

        Reads ``malloc_zone_t`` + ``malloc_introspection_t`` structures out of
        the live inferior: zone names, function tables, and the introspection
        (debugger/statistics) table per zone.  Discovery prefers the
        ``_malloc_zones`` / ``_num_zones`` registry and falls back to the
        default zone when those symbols are unavailable.  Also reports
        best-effort Mach-O metadata for the main binary (``binary.is_macho``).

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
            lambda: session.tools.heap.inspect_libmalloc_zones(
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def inspect_malloc_zone(
        session_id: str,
        zone_address: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Inspect one libmalloc zone deep (default zone when omitted).

        Reads a single ``malloc_zone_t`` at depth: name, full function table,
        and the ``malloc_introspection_t`` table.  For the default zone also
        probes ``malloc_good_size`` over common size classes to sketch the
        size-class layout.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            zone_address: Hex/decimal address of a specific zone (from
                ``inspect_libmalloc_zones``).  Omit to inspect the default
                zone.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.inspect_malloc_zone(
                zone_address=zone_address,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def classify_malloc_pointer(
        session_id: str,
        address: str,
        address_kind: str = "user",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Classify whether an address is a live libmalloc allocation.

        Asks ``malloc_size()`` for the pointer; a nonzero size means it is a
        live allocation (``allocated``, with the size and an owning-zone hint),
        zero means ``not_allocated``.  When the inferior call fails the verdict
        is ``unresolvable``.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            address: Hex/decimal address to classify.
            address_kind: Kind of address being classified (``user``,
                ``module``, ``stack``, ``page``, ``register``, ``file``).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.classify_malloc_pointer(
                address=address,
                address_kind=address_kind,
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
        """Walk 10 glibc fastbins with safe-linking decode (glibc only).

        Args:
            session_id: Session to inspect.
            arena_address: Arena addr; omitted => main_arena.
            max_chunks_per_bin: Cap per bin (default 128).
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
    async def inspect_tcache(
        session_id: str,
        max_chunks_per_bin: int = 64,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Walk per-thread tcache bins (glibc only).

        Args:
            session_id: Session to inspect.
            max_chunks_per_bin: Cap per bin (default 64).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.inspect_tcache(
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
    async def inspect_bins(
        session_id: str,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = 1024,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Walk 127 regular bins: unsorted/small/large (glibc only).

        Args:
            session_id: Session to inspect.
            arena_address: Arena addr; omitted => main_arena.
            max_chunks_per_bin: Cap per bin (default 1024).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.inspect_bins(
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
        """Return an allocation-validity verdict, routed by allocator family.

        The tool is family-routed (see ``detect_allocator``): the verdict
        vocabulary differs by family.  On a glibc target it runs the same
        read + parse as ``inspect_chunk`` and summarizes the outcome as
        ``unsupported`` (non-glibc), ``unreadable`` (header could not be
        read), ``corrupted`` (alignment / minimum-size / containment check
        failed), ``suspicious`` (header in a mapping but the declared size
        extends past its end), or ``valid`` (every check passed).  On a
        jemalloc or libmalloc target it reports allocation ownership instead:
        ``valid`` / ``invalid`` (a live allocation via find-extent or
        ``malloc_size``), or ``unresolvable`` (the ownership check could not
        run).  The response includes the individual check results and the
        list of failed check names so a heuristic is never promoted to a
        fact.

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

    @mcp.tool()
    @catch_errors()
    async def classify_heap_corruption(
        session_id: str,
        arena_address: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Classify bin corruption; verdict valid/suspicious/corrupted.

        Args:
            session_id: Session to inspect.
            arena_address: Arena addr; omitted => main_arena.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.classify_heap_corruption(
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
    async def validate_free(
        session_id: str,
        address: str,
        address_kind: str = "user",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Safe-to-free verdict: safe/double_free/corrupted/suspicious (glibc).

        Args:
            session_id: Session to inspect.
            address: Hex/dec chunk or user pointer.
            address_kind: "user" (malloc ptr) or "chunk" (header addr).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.validate_free(
                address=address,
                address_kind=address_kind,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def capture_heap_snapshot(
        session_id: str,
        name: Optional[str] = None,
        max_chunks_per_bin: int = 512,
        max_arenas: int = 16,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Capture heap snapshot (16 entries/64MiB budget, oldest evicted).

        Args:
            session_id: Session to inspect.
            name: Name for later diff/lookup; omitted => id-only.
            max_chunks_per_bin: Cap per bin (default 512).
            max_arenas: Cap arenas walked (default 16, truncated explicitly).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.capture_heap_snapshot(
                name=name,
                max_chunks_per_bin=max_chunks_per_bin,
                max_arenas=max_arenas,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def list_heap_snapshots(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List every stored heap snapshot for a session.

        Returns a per-snapshot descriptor (``snapshot_id``, ``name``,
        ``captured_at``, ``allocator_family``, ``chunk_count``,
        ``arena_count``, ``linked_crash_ids``, ``payload_bytes``) plus
        the current budget usage (count / max_count, bytes / max_bytes).

        Args:
            session_id: Debugger session whose snapshot store should be
                enumerated.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.heap.list_heap_snapshots(),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def get_heap_snapshot(
        session_id: str,
        identifier: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Fetch a stored heap snapshot by id or name.

        The full snapshot payload is returned, including allocator
        identity, per-arena summaries, tcache summary, chunk ledger and
        the current list of ``linked_crash_ids`` (see
        ``link_snapshot_to_crash``).

        Args:
            session_id: Debugger session that owns the snapshot store.
            identifier: Either the snapshot's ``snapshot_id`` (as
                returned by ``capture_heap_snapshot``) or the human
                ``name`` it was captured under.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.heap.get_heap_snapshot(identifier=identifier),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def diff_heap_snapshots(
        session_id: str,
        before: str,
        after: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Diff two stored heap snapshots.

        Computes ``new`` / ``removed`` / ``moved`` chunk sets (by
        chunk_address, with per-chunk location before / after) plus
        per-arena presence and top-address delta.  Either identifier may
        be a ``snapshot_id`` or a name.  Diffing a snapshot against
        itself is rejected.  If either snapshot was captured on a
        non-glibc allocator (the capture is family-routed) the diff
        returns ``verdict=unsupported``.

        Args:
            session_id: Debugger session that owns both snapshots.
            before: Identifier (id or name) of the earlier snapshot.
            after: Identifier (id or name) of the later snapshot.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.heap.diff_heap_snapshots(before=before, after=after),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def link_snapshot_to_crash(
        session_id: str,
        identifier: str,
        crash_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Annotate a heap snapshot with a related crash id.

        The association is annotation-only: it does not verify that the
        crash still exists in the crash-history store, and duplicate
        ``crash_id`` values for the same snapshot are ignored.  The
        updated ``linked_crash_ids`` list is returned.

        Args:
            session_id: Debugger session that owns the snapshot.
            identifier: Either the snapshot's ``snapshot_id`` or its
                human-readable name.
            crash_id: Crash id to attach.  Non-empty string required.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session,
            lambda: session.tools.heap.link_snapshot_to_crash(
                identifier=identifier, crash_id=crash_id
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def analyze_heap_technique(
        session_id: str,
        arena_address: Optional[str] = None,
        techniques: Optional[List[str]] = None,
        scan_mmap_mappings: bool = True,
        mmap_max_mappings: int = 4,
        mmap_max_bytes: int = 1 << 20,
        max_overlap_chunks: int = 256,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Check live glibc heap for exploit primitives; verdict clean/suspicious/exploitable.

        Args:
            session_id: Session to inspect.
            arena_address: Arena addr; omitted => main_arena.
            techniques: Subset to run; omitted => all detectors.
            scan_mmap_mappings: Run bounded mmap scan.
            mmap_max_mappings: Mapping cap.
            mmap_max_bytes: Byte cap.
            max_overlap_chunks: Overlap check cap.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.analyze_heap_technique(
                arena_address=arena_address,
                techniques=techniques,
                scan_mmap_mappings=scan_mmap_mappings,
                mmap_max_mappings=mmap_max_mappings,
                mmap_max_bytes=mmap_max_bytes,
                max_overlap_chunks=max_overlap_chunks,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def detect_uaf(
        session_id: str,
        max_bytes: int = 1 << 22,
        max_mappings: int = 8,
        max_chunks: int = 512,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Scan for dangling pointers into freed chunks (UAF, bounded).

        Args:
            session_id: Session to inspect.
            max_bytes: Byte cap across mappings.
            max_mappings: Mapping cap.
            max_chunks: Freed-chunk cap.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.heap.detect_uaf(
                max_bytes=max_bytes,
                max_mappings=max_mappings,
                max_chunks=max_chunks,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors(auto_artifact=False, track_operation=False)
    async def heap_technique_catalog(
        technique: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return the static House-of-* technique catalog (no debugger needed).

        Documents 26 heap techniques across the classical Malloc Maleficarum
        era, the mid-era FSOP foundations, and the modern post-hook era, plus
        8 detector-backed primitives.  Each entry records the technique's era,
        family, primitive, the live-memory evidence to look for, the existing
        MCP tools that can confirm it, prerequisites, and whether the analyzer
        can statically detect it.

        Args:
            technique: Optional technique name to filter to a single entry.
                Omit for the full catalog.
        """
        result = await run_blocking(
            lambda: GlibcTechniqueAnalyzer.heap_technique_catalog(technique=technique)
        )
        return result


__all__ = ["register"]
