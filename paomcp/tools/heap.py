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
    async def inspect_tcache(
        session_id: str,
        max_chunks_per_bin: int = 64,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Walk the per-thread glibc tcache bins and their singly-linked chains.

        Reads the thread-local ``tcache`` pointer, resolves the layout of
        ``struct tcache_perthread_struct`` (counts/entries offsets plus
        the counts element size which changed from ``uint8_t`` to
        ``uint16_t`` between glibc 2.29 and 2.30), then reads all 64
        counts and entries in a single struct-sized gulp.  Each nonzero
        entries[i] chain is walked via ``tcache_entry.next``; when
        safe-linking is active (glibc >= 2.32) ``next`` is decoded via
        ``PROTECT_PTR``.  Cycle detection + per-bin cap (default 64)
        prevent a corrupted heap from hanging the tool.

        A NULL ``tcache`` thread-local yields
        ``verdict = "uninitialized"`` (legitimate empty state, not an
        error).  Missing debug symbols yield
        ``verdict = "unresolvable"``.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            max_chunks_per_bin: Per-tcache-bin traversal cap.
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
        """Walk all 127 regular glibc bins (unsorted / small / large).

        Reads the arena's ``bins[]`` array (254 pointer slots) in a
        single gulp for a consistent snapshot, then per bin computes the
        ``bin_at(m, i)`` synthetic mchunkptr, detects the empty case
        (fd == bin_head_addr), and walks the doubly-linked chain via
        ``fd`` until it circles back to the bin head, hits a NULL fd,
        loops, or reaches the traversal cap.

        Bin categories: index 1 is unsorted, 2..63 are small (one size
        class per bin), 64..127 are large (each covers a size range).
        Per chunk the response includes chunk_address, size_field, size,
        prev_inuse, fd, bk, and in_mapping.  Large bin secondary
        size-sorted lists (fd_nextsize / bk_nextsize) are deferred.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            arena_address: Optional hex/decimal arena address (from
                ``list_arenas``).  Omit to inspect main_arena.
            max_chunks_per_bin: Per-bin traversal cap (default 1024).
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

    @mcp.tool()
    @catch_errors()
    async def classify_heap_corruption(
        session_id: str,
        arena_address: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Classify corruption across an arena's fastbins, tcache and regular bins.

        Runs the typed bin inspectors (fastbins / tcache / regular bins) and
        synthesizes their per-chain signals into a bounded, evidence-tagged
        list of corruption findings.  Emits a conservative arena-level
        verdict: ``valid`` (no findings), ``suspicious`` (only unreadable
        findings), ``corrupted`` (any structural violation), ``unsupported``
        (non-glibc), or ``unresolvable`` (bin layout could not be resolved).

        Categories covered: ``cycle``, ``duplicate_entry`` (a chunk appearing
        in multiple bins -- a strong double-free signal), ``fd_bk_inconsistency``
        (broken doubly-linked pair in a regular bin), ``alignment``,
        ``pointer_out_of_mapping``, ``unreadable_chunk`` and
        ``unreadable_bin``.  The tcache is thread-local so its findings
        always reflect the current thread regardless of ``arena_address``.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            arena_address: Optional hex/decimal arena address (typically
                from ``list_arenas``).  Omit to classify main_arena.
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
        """Return a safe-to-free verdict for a glibc chunk.

        Runs the S6.2 chunk parse + containment checks and cross-references
        the chunk against the current thread's tcache and main_arena's
        fastbins to catch a double-free before ``free()`` fires.

        Verdict vocabulary: ``unsupported`` (non-glibc), ``unreadable``
        (header could not be read), ``double_free`` (chunk already present
        in tcache or fastbins), ``corrupted`` (alignment / mapping / min-size
        check failed), ``suspicious`` (payload extends past mapping end), or
        ``safe`` (every check passed).  Per-check results and the list of
        failed check names are returned so a heuristic is never promoted to
        a fact.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            address: Chunk or user pointer as a hex/decimal string.
            address_kind: ``"user"`` (default -- ``address`` is a pointer
                returned by ``malloc``) or ``"chunk"`` (``address`` is the
                chunk header address).
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
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Capture a typed heap snapshot for this session.

        Drives the sibling allocator / arena / bin / tcache inspectors
        into a single reproducible payload: allocator identity + target
        profile, per-arena summaries (top, last_remainder, system_mem,
        next), tcache summary, and a flat chunk ledger (address, size,
        arena_address, location kind + index) covering every fastbin,
        regular bin and tcache slot walked.

        Snapshots are stored session-locally under a bounded budget (16
        entries and 64 MiB total, oldest-first eviction).  Re-capturing
        under an existing ``name`` evicts the previous entry so the name
        always points at the latest capture.  A non-glibc allocator
        yields a stored ``unsupported`` snapshot with an empty ledger.

        Args:
            session_id: Debugger session whose live inferior should be
                inspected.
            name: Optional human-readable name for later
                ``get_heap_snapshot`` / ``diff_heap_snapshots`` lookups.
                Omit for anonymous (id-only) captures.
            max_chunks_per_bin: Per-bin traversal cap applied to fastbins,
                tcache and regular bins.  Defaults to 512.
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
        non-glibc allocator the diff returns ``verdict=unsupported``.

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


__all__ = ["register"]
