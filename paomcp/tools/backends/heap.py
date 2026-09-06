"""Typed heap allocator discovery.

Heap analysis gives an agent a typed understanding of the target's heap without
forcing it to parse pwndbg's ANSI terminal output.  Allocator discovery is the foundation of
that work: before any chunk, bin, or tcache can be parsed, the platform must
answer three questions honestly and conservatively:

* Which allocator is this process actually using (glibc ptmalloc, musl
  mallocng, jemalloc, or something unknown)?
* What is the concrete runtime profile the later parsers must obey -- pointer
  width, byte order, and architecture?
* Which inspection capabilities are genuinely available right now, and which
  are merely planned or unsupported?

Every conclusion carries the raw evidence it was derived from, and a heuristic
is never promoted to a fact.  Detection needs a *running* inferior because a
heap only exists at runtime, so the live-inferior invariant from crash triage is
reused here rather than re-implemented.

The heavy lifting lives in :meth:`AllocatorDiscovery.classify`, a pure function
over already-collected mapping/libc evidence.  That keeps the interesting logic
unit-testable without a debugger, matching the design of
:mod:`paomcp.tools.backends.foundation`.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from paomcp.tools.backends.backend import BackendSpec
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.inspection import PwndbgInspection
from paomcp.tools.backends.heap_discovery import (
    ALLOCATOR_RUNTIME_COMMANDS,
    MUSL_MALLOCNG_MIN_VERSION,
    TYPED_INSPECTION_FAMILIES,
    AllocatorDiscovery,
    encode_safe_linked,
)
from paomcp.tools.backends.heap_schemas import (
    PARSER_VERSION,
    ALLOCATOR_DETECTION_SCHEMA,
    ALLOCATOR_CAPABILITIES_SCHEMA,
    SAFE_LINKING_MIN_VERSION,
    DEFAULT_MAX_CHUNKS_PER_FASTBIN,
    DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
    DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
    HEAP_SNAPSHOT_SCHEMA,
    HEAP_SNAPSHOT_LIST_SCHEMA,
    HEAP_SNAPSHOT_DIFF_SCHEMA,
    HEAP_SNAPSHOT_LINK_SCHEMA,
    DEFAULT_MAX_HEAP_SNAPSHOTS,
    DEFAULT_MAX_HEAP_SNAPSHOT_BYTES,
)
from paomcp.tools.backends.heap_snapshots import (
    DEFAULT_MAX_SNAPSHOT_LEDGER_ENTRIES,
    HeapHistory,
    HeapSnapshotError,
)

# glibc adopted tcache/fastbin pointer "safe-linking" in 2.32.  Detection of the
# libc version therefore decides whether the corruption classifier must decode
# masked forward pointers.  Below this version the fd pointers are stored raw.


# Chunk-to-bin glibc parsers live in heap_glibc.
from paomcp.tools.backends.heap_glibc import (
    ARENA_INSPECTION_SCHEMA,
    ARENA_LIST_SCHEMA,
    ArenaLayoutResolver,
    BINMAPSIZE,
    BINS_ARRAY_LEN,
    BIN_INSPECTION_SCHEMA,
    CHUNK_INSPECTION_SCHEMA,
    CHUNK_VALIDATION_SCHEMA,
    DEFAULT_SLUB_PAGE_SIZE,
    FASTBIN_INSPECTION_SCHEMA,
    FREE_VALIDATION_SCHEMA,
    GlibcArenas,
    GlibcBins,
    GlibcHeap,
    GlibcHeapCorruption,
    HEAP_CORRUPTION_SCHEMA,
    IS_MMAPPED,
    MAX_ARENAS_TRAVERSAL,
    NBINS,
    NFASTBINS,
    NON_MAIN_ARENA,
    PREV_INUSE,
    SIZE_BITS_MASK,
    TCACHE_INSPECTION_SCHEMA,
    TCACHE_MAX_BINS,
    TcacheLayoutResolver,
)


class HeapTools:
    """Facade that groups every heap-inspection component under one object.

    Later components (arena/bin parsers, musl/jemalloc analyzers) attach here so
    the tool layer keeps a single ``session.tools.heap`` entry point regardless
    of the detected allocator family.  Delegation is intentionally boring:
    each public method routes to the appropriate sub-component untouched.

    Allocator-plugin contract
    -------------------------
    Every allocator family must be identified by :class:`AllocatorDiscovery`
    (extend ``_FAMILY_FILENAME_PATTERNS`` and the runtime-command map) and
    -- when supported -- have a sibling introspection class hanging off
    this facade (mirror :class:`GlibcHeap`, :class:`GlibcArenas`,
    :class:`GlibcBins`).

    The twelve allocator-layout-dependent methods (chunks, arenas, bins,
    corruption, free validation, techniques, UAF, snapshot capture) are
    dispatched per call through an :class:`AllocatorAnalyzer` resolved from
    the detected family (``heap_analyzers.analyzer_for_family``).  glibc,
    jemalloc and libmalloc each have a typed analyzer (the jemalloc and
    libmalloc analyzers re-verify the freshly-classified family inside
    ``_prepare_context`` so a stale dispatch hit can never run their
    family-specific reads against the wrong allocator); every other family
    resolves to ``UnsupportedAnalyzer``, which forwards to the glibc
    components that already return an honest typed ``unsupported`` verdict
    for non-glibc allocators.  A new family's parser plugs in by implementing
    the analyzer surface and adding its name to the dispatch; nothing else
    changes.
    """

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        binary_path_provider: Optional[Callable[[], Optional[str]]] = None,
        backend: Optional[BackendSpec] = None,
        patch_recorder: Optional[Any] = None,
    ) -> None:
        self.backend = backend
        self.session_id = session_id
        # ``PwndbgTools`` registers itself here when constructed, so writes
        # issued by the attack helpers land in the same session patch ledger
        # that ``list_memory_patches`` / ``revert_memory_patch`` operate on.
        self._patch_recorder = patch_recorder
        # TTL cache of the last detected allocator family (monotonic clock
        # seconds + family).  Consulted only by ``_analyzer()``; the glibc
        # components re-classify per call regardless, so a stale hit never
        # changes behavior -- it only picks which forwarder runs.
        self._family_cache: Optional[Tuple[float, Optional[str]]] = None
        self.gdb = gdb
        self.discovery = AllocatorDiscovery(
            gdb,
            inspection,
            debugger,
            session_id,
            binary_path_provider=binary_path_provider,
        )
        self.glibc = GlibcHeap(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
        )
        self.arenas = GlibcArenas(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
        )
        self.bins = GlibcBins(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            arenas=self.arenas,
        )
        self.corruption = GlibcHeapCorruption(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            arenas=self.arenas,
            bins=self.bins,
            glibc=self.glibc,
        )
        self.history = HeapHistory(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            arenas=self.arenas,
            bins=self.bins,
        )
        # Lazy imports to break module-level cycles: heap_techniques and
        # heap_analyzers both import this module at their top, so heap.py must
        # never import them at import time -- only here, when the facade is
        # actually built.
        from .attack_helpers import GlibcAttackHelper  # noqa: PLC0415
        from .heap_analyzers import analyzer_for_family  # noqa: PLC0415
        from .heap_techniques import GlibcTechniqueAnalyzer  # noqa: PLC0415
        from .slub import KernelSlab  # noqa: PLC0415

        self._analyzer_for_family = analyzer_for_family

        self.techniques = GlibcTechniqueAnalyzer(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            arenas=self.arenas,
            bins=self.bins,
            glibc=self.glibc,
            corruption=self.corruption,
        )

        # Attack helpers are the write-capable extension of the heap surface:
        # they execute primitives against the live target via the guarded
        # writer, gated behind an explicit ``dry_run=false`` opt-in.
        self._attack_for_family = GlibcAttackHelper
        self.attack = GlibcAttackHelper(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            glibc=self.glibc,
            bins=self.bins,
            history=self.history,
            patch_recorder=self._patch_recorder,
        )

        # Kernel SLUB is a dedicated self-gated surface: kernel targets are
        # not classified by the userspace allocator protocol, so this
        # component never routes through ``classify`` -- the caller names a
        # cache address and BTF resolves the layout.
        self.slub = KernelSlab(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
        )

        # musl mallocng is likewise self-gated: dedicated tools route here
        # directly while the family-routed generic tools keep returning the
        # honest ``unsupported`` verdict for musl (musl has no bins/arenas).
        from .musl import MuslAnalyzer  # noqa: PLC0415 (same cycle rule)

        self.musl = MuslAnalyzer(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
        )

        from .alloc_trace import AllocTraceManager  # noqa: PLC0415 (same cycle rule)

        self.alloc_trace = AllocTraceManager(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
        )

    # How long a detected family stays cached on this facade.  Short on
    # purpose: the components re-classify per call anyway, so this only avoids
    # a second full classification when consecutive dispatched calls land
    # within the window.
    _FAMILY_CACHE_TTL_SEC = 1.0

    def _detected_family(self) -> Optional[str]:
        """Classify the current inferior's allocator family (TTL-cached).

        Returns ``None`` when the detection cannot resolve a family (no live
        inferior, classification failure); the dispatch treats that exactly
        like ``unknown``.  Failures never propagate here -- the analyzer
        forwarders surface the same typed errors the components already raise.
        """
        now = time.monotonic()
        cached = getattr(self, "_family_cache", None)
        if cached is not None:
            ts, family = cached
            if now - ts < self._FAMILY_CACHE_TTL_SEC:
                return family
        try:
            envelope = self.discovery.detect_allocator()
        except Exception:  # noqa: BLE001 -- detection is best-effort here
            self._family_cache = (now, None)
            return None
        parsed = envelope.get("parsed") if isinstance(envelope, dict) else None
        family = (parsed or {}).get("family")
        self._family_cache = (now, family)
        return family

    def _analyzer(self) -> Any:
        """Resolve the per-family analyzer for this call's detected family.

        ``_analyzer_for_family`` is normally set in ``__init__``; fall back to
        a lazy import when the facade was assembled without ``__init__`` (the
        allocator-plugin test pattern builds ``HeapTools`` shells via
        ``__new__`` and attaches components directly).  The import is deferred
        to call time to keep the ``heap_analyzers`` <-> ``heap`` module cycle
        broken.
        """
        resolver = getattr(self, "_analyzer_for_family", None)
        if resolver is None:
            from .heap_analyzers import analyzer_for_family  # noqa: PLC0415

            resolver = analyzer_for_family
        return resolver(self._detected_family(), self)

    def _libmalloc_analyzer(self) -> Any:
        """Resolve the libmalloc analyzer, getattr-cached on the facade.

        The zone-native tools route straight to ``LibmallocAnalyzer`` regardless
        of the detected family; the analyzer itself still reports an honest
        ``unsupported`` verdict when the target is not libmalloc.  Same lazy
        import / getattr-fallback shape as ``_analyzer()``.
        """
        cached = getattr(self, "_libmalloc_analyzer_cache", None)
        if cached is not None:
            return cached
        resolver = getattr(self, "_analyzer_for_family", None)
        if resolver is None:
            from .heap_analyzers import analyzer_for_family  # noqa: PLC0415

            resolver = analyzer_for_family
        analyzer = resolver("libmalloc", self)
        self._libmalloc_analyzer_cache = analyzer
        return analyzer

    def _regions_tracker(self) -> Any:
        """Resolve the JIT/memory-region tracker, getattr-cached on the facade.

        Region tracking is allocator-independent: it never routes through
        ``_analyzer()`` or family classification.  The lazy import keeps the
        ``regions`` <-> ``heap`` module graph acyclic (``regions`` imports this
        module for ``PARSER_VERSION``).
        """
        cached = getattr(self, "_regions_tracker_cache", None)
        if cached is not None:
            return cached
        from .regions import JitRegionTracker  # noqa: PLC0415

        tracker = JitRegionTracker(self)
        self._regions_tracker_cache = tracker
        return tracker

    @property
    def _is_lldb(self) -> bool:
        backend = getattr(self, "backend", None)
        return backend is not None and getattr(backend, "protocol", None) == "lldb-cli"

    def _gdb_gate(self, capability: str) -> Optional[Dict[str, Any]]:
        """Return a typed verdict when a heap parse requires the gdb backend."""
        if not self._is_lldb:
            return None
        backend_name = getattr(self.backend, "name", "lldb")
        return {
            "success": False,
            "command": capability,
            "state": self.gdb.get_state(),
            "type": "GdbOnlyBackendError",
            "error": (
                f"'{capability}' requires the gdb/pwndbg backend; "
                f"session uses {backend_name}"
            ),
        }

    # ---- allocator discovery ------------------------------------
    def detect_allocator(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("detect_allocator")
        if gate is not None:
            return gate
        return self.discovery.detect_allocator(cancellation_check=cancellation_check)

    def get_allocator_capabilities(
        self,
        *,
        debugger_command_names: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("get_allocator_capabilities")
        if gate is not None:
            return gate
        return self.discovery.get_allocator_capabilities(
            debugger_command_names=debugger_command_names,
            cancellation_check=cancellation_check,
        )

    # ---- glibc chunks -------------------------------------------
    def inspect_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        include_next: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_chunk")
        if gate is not None:
            return gate
        return self._analyzer().inspect_chunk(
            address=address,
            address_kind=address_kind,
            include_next=include_next,
            cancellation_check=cancellation_check,
        )

    def validate_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("validate_chunk")
        if gate is not None:
            return gate
        return self._analyzer().validate_chunk(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )

    # ---- glibc arenas -------------------------------------------
    def list_arenas(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("list_arenas")
        if gate is not None:
            return gate
        return self._analyzer().list_arenas(cancellation_check=cancellation_check)

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_arena")
        if gate is not None:
            return gate
        return self._analyzer().inspect_arena(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )

    # ---- glibc fastbin traversal --------------------------------
    def inspect_fastbins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_FASTBIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_fastbins")
        if gate is not None:
            return gate
        return self._analyzer().inspect_fastbins(
            arena_address=arena_address,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    # ---- glibc tcache traversal ---------------------------------
    def inspect_tcache(
        self,
        *,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_tcache")
        if gate is not None:
            return gate
        return self._analyzer().inspect_tcache(
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    # ---- glibc regular bins (unsorted/small/large) --------------
    def inspect_bins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_bins")
        if gate is not None:
            return gate
        return self._analyzer().inspect_bins(
            arena_address=arena_address,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    # ---- attack helpers ------------------------------------------
    def poison_tcache_link(
        self,
        *,
        victim_address: str,
        target_address: str,
        address_kind: str = "chunk",
        dry_run: bool = True,
        snapshot_before: bool = True,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Re-point a freed tcache chunk's ``fd`` (dry-run by default)."""
        gate = self._gdb_gate("poison_tcache_link")
        if gate is not None:
            return gate
        attack = getattr(self, "attack", None)
        if attack is None:
            # Facade assembled without ``__init__`` (the allocator-plugin
            # test pattern); build a throwaway helper on the live components.
            from .attack_helpers import GlibcAttackHelper  # noqa: PLC0415

            attack = GlibcAttackHelper(
                self.discovery.gdb,
                self.discovery.inspection,
                self.discovery.debugger,
                self.session_id,
                discovery=self.discovery,
                glibc=self.glibc,
                bins=self.bins,
                history=self.history,
                patch_recorder=self._patch_recorder,
            )
        return attack.poison_tcache_link(
            victim_address=victim_address,
            target_address=target_address,
            address_kind=address_kind,
            dry_run=dry_run,
            snapshot_before=snapshot_before,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    def _attack_or_throwaway(self) -> Any:
        attack = getattr(self, "attack", None)
        if attack is not None:
            return attack
        from .attack_helpers import GlibcAttackHelper  # noqa: PLC0415

        return GlibcAttackHelper(
            self.discovery.gdb,
            self.discovery.inspection,
            self.discovery.debugger,
            self.session_id,
            discovery=self.discovery,
            glibc=self.glibc,
            bins=self.bins,
            history=self.history,
            patch_recorder=self._patch_recorder,
        )

    def poison_fastbin_fd(
        self,
        *,
        victim_address: str,
        target_address: str,
        address_kind: str = "chunk",
        dry_run: bool = True,
        snapshot_before: bool = True,
        max_chunks_per_bin: int = 128,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Re-point a freed fastbin chunk's ``fd`` (dry-run by default)."""
        gate = self._gdb_gate("poison_fastbin_fd")
        if gate is not None:
            return gate
        return self._attack_or_throwaway().poison_fastbin_fd(
            victim_address=victim_address,
            target_address=target_address,
            address_kind=address_kind,
            dry_run=dry_run,
            snapshot_before=snapshot_before,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    def house_of_spirit_plan(
        self,
        *,
        fake_chunk_address: str,
        fake_size: str = "0x40",
        address_kind: str = "chunk",
        dry_run: bool = True,
        snapshot_before: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Stage a House-of-Spirit fake chunk header (dry-run by default)."""
        gate = self._gdb_gate("house_of_spirit_plan")
        if gate is not None:
            return gate
        return self._attack_or_throwaway().house_of_spirit_plan(
            fake_chunk_address=fake_chunk_address,
            fake_size=fake_size,
            address_kind=address_kind,
            dry_run=dry_run,
            snapshot_before=snapshot_before,
            cancellation_check=cancellation_check,
        )

    def _alloc_tracer(self) -> Any:
        tracer = getattr(self, "alloc_trace", None)
        if tracer is not None:
            return tracer
        from .alloc_trace import AllocTraceManager  # noqa: PLC0415

        tracer = AllocTraceManager(
            self.discovery.gdb,
            self.discovery.inspection,
            self.discovery.debugger,
            self.session_id,
            discovery=self.discovery,
        )
        self.alloc_trace = tracer
        return tracer

    def start_alloc_trace(
        self,
        *,
        symbols: Optional[List[str]] = None,
        max_events: int = 10_000,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("start_alloc_trace")
        if gate is not None:
            return gate
        return self._alloc_tracer().start_alloc_trace(
            symbols=symbols,
            max_events=max_events,
            cancellation_check=cancellation_check,
        )

    def list_alloc_events(
        self,
        *,
        trace_id: str,
        limit: int = 100,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("list_alloc_events")
        if gate is not None:
            return gate
        return self._alloc_tracer().list_alloc_events(
            trace_id=trace_id,
            limit=limit,
            cancellation_check=cancellation_check,
        )

    def stop_alloc_trace(
        self,
        *,
        trace_id: str,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("stop_alloc_trace")
        if gate is not None:
            return gate
        return self._alloc_tracer().stop_alloc_trace(
            trace_id=trace_id,
            cancellation_check=cancellation_check,
        )

    # ---- kernel SLUB (dedicated self-gated surface) --------------
    def inspect_slub_cache(
        self,
        *,
        cache_address: str,
        btf_path: Optional[str] = None,
        page_size: int = DEFAULT_SLUB_PAGE_SIZE,
        byteorder: str = "little",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Decode a kernel SLUB cache's geometry + hardening state from BTF."""
        gate = self._gdb_gate("inspect_slub_cache")
        if gate is not None:
            return gate
        slub = getattr(self, "slub", None)
        if slub is None:
            # Facade assembled without ``__init__`` (the allocator-plugin
            # test pattern); build a throwaway component on the discovery.
            from .slub import KernelSlab  # noqa: PLC0415

            slub = KernelSlab(
                self.discovery.gdb,
                self.discovery.inspection,
                self.discovery.debugger,
                self.session_id,
                discovery=self.discovery,
            )
        return slub.inspect_slub_cache(
            cache_address=cache_address,
            btf_path=btf_path,
            page_size=page_size,
            byteorder=byteorder,
            cancellation_check=cancellation_check,
        )

    def decode_slub_freelist(
        self,
        *,
        cache_address: str,
        slab_address: str,
        slab_mem_address: Optional[str] = None,
        btf_path: Optional[str] = None,
        page_size: int = DEFAULT_SLUB_PAGE_SIZE,
        max_objects: Optional[int] = None,
        byteorder: str = "little",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Decode one kernel SLUB slab's obfuscated freelist chain."""
        gate = self._gdb_gate("decode_slub_freelist")
        if gate is not None:
            return gate
        slub = getattr(self, "slub", None)
        if slub is None:
            from .slub import KernelSlab  # noqa: PLC0415

            slub = KernelSlab(
                self.discovery.gdb,
                self.discovery.inspection,
                self.discovery.debugger,
                self.session_id,
                discovery=self.discovery,
            )
        return slub.decode_slub_freelist(
            cache_address=cache_address,
            slab_address=slab_address,
            slab_mem_address=slab_mem_address,
            btf_path=btf_path,
            page_size=page_size,
            max_objects=max_objects,
            byteorder=byteorder,
            cancellation_check=cancellation_check,
        )

    # ---- musl mallocng surface -------------------------------------
    def inspect_musl_chunk(
        self,
        *,
        address: str,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Resolve a user pointer to its mallocng group and validate it."""
        gate = self._gdb_gate("inspect_musl_chunk")
        if gate is not None:
            return gate
        return self.musl.inspect_musl_chunk(
            address=address,
            cancellation_check=cancellation_check,
        )

    def list_musl_groups(
        self,
        *,
        max_groups: int = 64,
        max_areas: int = 16,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Inventory musl allocation groups via ctx.meta_area_head."""
        gate = self._gdb_gate("list_musl_groups")
        if gate is not None:
            return gate
        return self.musl.list_musl_groups(
            max_groups=max_groups,
            max_areas=max_areas,
            cancellation_check=cancellation_check,
        )

    # ---- libmalloc zone surface --------------------------------
    def inspect_libmalloc_zones(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """List every libmalloc zone (name + function + introspection tables)."""
        gate = self._gdb_gate("inspect_libmalloc_zones")
        if gate is not None:
            return gate
        return self._libmalloc_analyzer().inspect_zones(
            cancellation_check=cancellation_check
        )

    def inspect_malloc_zone(
        self,
        *,
        zone_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Inspect one malloc zone deep (default zone when address omitted)."""
        gate = self._gdb_gate("inspect_malloc_zone")
        if gate is not None:
            return gate
        return self._libmalloc_analyzer().inspect_zone(
            zone_address=zone_address,
            cancellation_check=cancellation_check,
        )

    def classify_malloc_pointer(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Classify whether an address is a live libmalloc allocation."""
        gate = self._gdb_gate("classify_malloc_pointer")
        if gate is not None:
            return gate
        return self._libmalloc_analyzer().classify_pointer(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )

    # ---- JIT memory-region tracking (allocator-independent) ----
    def scan_jit_regions(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Scan anonymous executable mappings (V8 JIT-heap candidates)."""
        gate = self._gdb_gate("scan_jit_regions")
        if gate is not None:
            return gate
        return self._regions_tracker().scan_jit_regions(
            cancellation_check=cancellation_check
        )

    def classify_memory_region(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Resolve the memory mapping containing an address."""
        gate = self._gdb_gate("classify_memory_region")
        if gate is not None:
            return gate
        return self._regions_tracker().classify_memory_region(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )

    def arm_region_watch(
        self,
        *,
        address: str,
        length: int,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Snapshot a region hash for later JIT-write detection."""
        gate = self._gdb_gate("arm_region_watch")
        if gate is not None:
            return gate
        return self._regions_tracker().arm_region_watch(
            address=address,
            length=length,
            label=label,
        )

    def check_region_watches(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Re-hash armed regions; report changed byte ranges."""
        gate = self._gdb_gate("check_region_watches")
        if gate is not None:
            return gate
        return self._regions_tracker().check_region_watches(
            cancellation_check=cancellation_check
        )

    def disarm_region_watch(self, *, watch_id: str) -> Dict[str, Any]:
        """Drop one armed region watch."""
        gate = self._gdb_gate("disarm_region_watch")
        if gate is not None:
            return gate
        return self._regions_tracker().disarm_region_watch(watch_id=watch_id)

    def classify_pa_regions(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Flag PartitionAlloc-like super pages (no freelist walk)."""
        gate = self._gdb_gate("classify_pa_regions")
        if gate is not None:
            return gate
        return self._regions_tracker().classify_pa_regions(
            cancellation_check=cancellation_check
        )

    # ---- heap corruption classifier ------------------------------
    def classify_heap_corruption(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("classify_heap_corruption")
        if gate is not None:
            return gate
        return self._analyzer().classify_heap_corruption(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )

    def validate_free(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("validate_free")
        if gate is not None:
            return gate
        return self._analyzer().validate_free(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )

    # ---- heap technique analysis --------------------------------
    def analyze_heap_technique(
        self,
        *,
        arena_address: Optional[str] = None,
        techniques: Optional[Sequence[str]] = None,
        scan_mmap_mappings: bool = True,
        mmap_max_mappings: int = 4,
        mmap_max_bytes: int = 1 << 20,
        max_overlap_chunks: int = 256,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Run House-of-* / exploit-primitive detectors over the live heap.

        Delegates to :class:`GlibcTechniqueAnalyzer`; every finding is a
        heuristic tagged with raw evidence, confidence, confirmation tools and
        a playbook.  This method requires the gdb/pwndbg backend (rich heap
        parsing).
        """
        gate = self._gdb_gate("analyze_heap_technique")
        if gate is not None:
            return gate
        return self._analyzer().analyze_heap_technique(
            arena_address=arena_address,
            techniques=techniques,
            scan_mmap_mappings=scan_mmap_mappings,
            mmap_max_mappings=mmap_max_mappings,
            mmap_max_bytes=mmap_max_bytes,
            max_overlap_chunks=max_overlap_chunks,
            cancellation_check=cancellation_check,
        )

    def detect_uaf(
        self,
        *,
        max_bytes: int = 1 << 22,
        max_mappings: int = 8,
        max_chunks: int = 512,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Scan writable mappings for pointers into currently-freed chunks.

        The dangling-reference signature of a use-after-free.  Bounded
        (``max_bytes`` / ``max_mappings`` / ``max_chunks``) and excludes the
        allocator's own bookkeeping (bin link slots, arena struct, tcache
        struct).  Requires the gdb/pwndbg backend.
        """
        gate = self._gdb_gate("detect_uaf")
        if gate is not None:
            return gate
        return self._analyzer().detect_uaf(
            max_bytes=max_bytes,
            max_mappings=max_mappings,
            max_chunks=max_chunks,
            cancellation_check=cancellation_check,
        )

    def heap_technique_catalog(
        self,
        *,
        technique: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the static House-of-* technique catalog (no debugger needed)."""
        return self.techniques.heap_technique_catalog(technique=technique)

    # ---- heap snapshot history ----------------------------------
    def capture_heap_snapshot(
        self,
        *,
        name: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        max_arenas: int = 16,
        max_ledger_entries: int = DEFAULT_MAX_SNAPSHOT_LEDGER_ENTRIES,
        timeout_seconds: Optional[float] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("capture_heap_snapshot")
        if gate is not None:
            return gate
        return self._analyzer().capture_heap_snapshot(
            name=name,
            max_chunks_per_bin=max_chunks_per_bin,
            max_arenas=max_arenas,
            max_ledger_entries=max_ledger_entries,
            timeout_seconds=timeout_seconds,
            cancellation_check=cancellation_check,
        )

    def list_heap_snapshots(self) -> Dict[str, Any]:
        return self.history.list_heap_snapshots()

    def get_heap_snapshot(self, *, identifier: str) -> Dict[str, Any]:
        return self.history.get_heap_snapshot(identifier=identifier)

    def diff_heap_snapshots(self, *, before: str, after: str) -> Dict[str, Any]:
        return self.history.diff_heap_snapshots(before=before, after=after)

    def link_snapshot_to_crash(
        self, *, identifier: str, crash_id: str
    ) -> Dict[str, Any]:
        return self.history.link_snapshot_to_crash(
            identifier=identifier, crash_id=crash_id
        )


__all__ = [
    "AllocatorDiscovery",
    "GlibcHeap",
    "GlibcArenas",
    "GlibcBins",
    "GlibcHeapCorruption",
    "HeapHistory",
    "HeapSnapshotError",
    "ArenaLayoutResolver",
    "TcacheLayoutResolver",
    "HeapTools",
    "ALLOCATOR_DETECTION_SCHEMA",
    "ALLOCATOR_CAPABILITIES_SCHEMA",
    "CHUNK_INSPECTION_SCHEMA",
    "CHUNK_VALIDATION_SCHEMA",
    "ARENA_LIST_SCHEMA",
    "ARENA_INSPECTION_SCHEMA",
    "FASTBIN_INSPECTION_SCHEMA",
    "DEFAULT_MAX_CHUNKS_PER_FASTBIN",
    "TCACHE_INSPECTION_SCHEMA",
    "DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN",
    "TCACHE_MAX_BINS",
    "BIN_INSPECTION_SCHEMA",
    "DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN",
    "DEFAULT_SLUB_PAGE_SIZE",
    "HEAP_CORRUPTION_SCHEMA",
    "HEAP_TECHNIQUE_SCHEMA",
    "UAF_SCAN_SCHEMA",
    "HEAP_TECHNIQUE_CATALOG_SCHEMA",
    "FREE_VALIDATION_SCHEMA",
    "HEAP_SNAPSHOT_SCHEMA",
    "HEAP_SNAPSHOT_LIST_SCHEMA",
    "HEAP_SNAPSHOT_DIFF_SCHEMA",
    "HEAP_SNAPSHOT_LINK_SCHEMA",
    "DEFAULT_MAX_HEAP_SNAPSHOTS",
    "DEFAULT_MAX_HEAP_SNAPSHOT_BYTES",
    "SAFE_LINKING_MIN_VERSION",
    "encode_safe_linked",
    "MUSL_MALLOCNG_MIN_VERSION",
    "ALLOCATOR_RUNTIME_COMMANDS",
    "TYPED_INSPECTION_FAMILIES",
    "PREV_INUSE",
    "IS_MMAPPED",
    "NON_MAIN_ARENA",
    "SIZE_BITS_MASK",
    "NFASTBINS",
    "NBINS",
    "BINS_ARRAY_LEN",
    "BINMAPSIZE",
    "MAX_ARENAS_TRAVERSAL",
    "LIBMALLOC_ZONES_SCHEMA",
    "LIBMALLOC_ZONE_SCHEMA",
    "LIBMALLOC_POINTER_SCHEMA",
]


# PEP 562 lazy re-export.  The technique schema constants live in heap_techniques,
# which imports this module at its top; a module-level ``from
# .heap_techniques import ...`` here would create an import cycle.  Keeping the
# names in __all__ and resolving them on attribute access preserves the
# ``from paomcp.tools.backends.heap import HEAP_TECHNIQUE_SCHEMA`` spelling.
# The libmalloc schemas follow the same rule (libmalloc.py imports this
# module at its top).
def __getattr__(name: str) -> Any:
    if name in (
        "HEAP_TECHNIQUE_SCHEMA",
        "UAF_SCAN_SCHEMA",
        "HEAP_TECHNIQUE_CATALOG_SCHEMA",
    ):
        from .heap_techniques import (  # noqa: PLC0415
            HEAP_TECHNIQUE_CATALOG_SCHEMA,
            HEAP_TECHNIQUE_SCHEMA,
            UAF_SCAN_SCHEMA,
        )

        return {
            "HEAP_TECHNIQUE_SCHEMA": HEAP_TECHNIQUE_SCHEMA,
            "UAF_SCAN_SCHEMA": UAF_SCAN_SCHEMA,
            "HEAP_TECHNIQUE_CATALOG_SCHEMA": HEAP_TECHNIQUE_CATALOG_SCHEMA,
        }[name]
    if name in (
        "LIBMALLOC_ZONES_SCHEMA",
        "LIBMALLOC_ZONE_SCHEMA",
        "LIBMALLOC_POINTER_SCHEMA",
    ):
        from .libmalloc import (  # noqa: PLC0415
            LIBMALLOC_POINTER_SCHEMA,
            LIBMALLOC_ZONE_SCHEMA,
            LIBMALLOC_ZONES_SCHEMA,
        )

        return {
            "LIBMALLOC_ZONES_SCHEMA": LIBMALLOC_ZONES_SCHEMA,
            "LIBMALLOC_ZONE_SCHEMA": LIBMALLOC_ZONE_SCHEMA,
            "LIBMALLOC_POINTER_SCHEMA": LIBMALLOC_POINTER_SCHEMA,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
