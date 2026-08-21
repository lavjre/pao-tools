"""Allocator-family dispatch for the heap-analysis surface.

Every typed heap tool routes through :class:`HeapTools`.  This module is the
seam that separates *which* allocator was detected from *which* component
family does the work, so a new analyzer (``SlabAnalyzer``, ``tcmalloc``,
...) plugs in without touching the facade or the tool layer.

Contract
--------
* :class:`AllocatorAnalyzer` is the per-family surface: the twelve methods
  that depend on allocator-internal layout (chunks, arenas, bins, corruption,
  techniques, snapshot capture).  One implementer exists per supported family.
* :func:`analyzer_for_family` maps a detected family to an analyzer.  ``glibc``
  resolves to :class:`GlibcAnalyzer` (full typed structural parser); ``jemalloc``
  resolves to ``JemallocAnalyzer`` (extent/rtree core via pwndbg's plugin);
  ``libmalloc`` resolves to ``LibmallocAnalyzer`` (malloc_zone introspection);
  every other family (``musl``, ``scudo``, ``unknown``) resolves to
  :class:`UnsupportedAnalyzer`.
* :class:`GlibcAnalyzer` and :class:`UnsupportedAnalyzer` *forward* to the
  existing glibc components (``facade.glibc`` / ``facade.arenas`` /
  ``facade.bins`` / ``facade.corruption`` / ``facade.techniques`` /
  ``facade.history``).  Those components already re-classify the inferior per
  call and return the typed ``unsupported`` verdict for non-glibc families,
  so forwarding is behavior-identical and keeps every existing
  ``verdict == "unsupported"`` assertion intact.  The components are read
  from the facade **at call time**, not captured at construction -- tests
  replace ``facade.glibc`` and friends with stand-ins after ``HeapTools`` is
  built.  ``JemallocAnalyzer`` and ``LibmallocAnalyzer`` are real parsers;
  each re-verifies the freshly-classified family inside ``_prepare_context``
  and returns the typed ``unsupported`` envelope on mismatch.

  A later phase replaces :class:`UnsupportedAnalyzer`'s forwarding with a
  real per-family parser (same envelope shape, same return types); the rest
  of the codebase notices nothing.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from paomcp.tools.backends.heap import (
    DEFAULT_MAX_CHUNKS_PER_FASTBIN,
    DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
    DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
)

if TYPE_CHECKING:  # pragma: no cover -- type-only, avoids a runtime cycle
    from paomcp.tools.backends.heap import HeapTools


@runtime_checkable
class AllocatorAnalyzer(Protocol):
    """The per-family heap-analysis surface (Phase 0 seam).

    Every method mirrors the corresponding :class:`HeapTools` facade method
    and returns the standard ``build_backend_response`` envelope.  A family
    that ships a real parser implements exactly these twelve methods; nothing
    else in the server is allocator-family aware.
    """

    def inspect_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        include_next: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def validate_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def list_arenas(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def inspect_fastbins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_FASTBIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def inspect_tcache(
        self,
        *,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def inspect_bins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def classify_heap_corruption(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def validate_free(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

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
    ) -> Dict[str, Any]: ...

    def detect_uaf(
        self,
        *,
        max_bytes: int = 1 << 22,
        max_mappings: int = 8,
        max_chunks: int = 512,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...

    def capture_heap_snapshot(
        self,
        *,
        name: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]: ...


class _ForwardingAnalyzer:
    """Shared implementation: forward to the facade's live glibc components.

    The components are resolved through ``self.facade.<name>`` at call time so
    a test that swaps ``facade.glibc`` / ``facade.arenas`` / ... after
    construction keeps being honored -- the same attribute reads the facade's
    own methods performed before the dispatch seam existed.
    """

    family: Optional[str]  # the family this analyzer represents

    def __init__(self, facade: "HeapTools") -> None:
        self.facade = facade

    def inspect_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        include_next: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.facade.glibc.inspect_chunk(
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
        return self.facade.glibc.validate_chunk(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )

    def list_arenas(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.facade.arenas.list_arenas(cancellation_check=cancellation_check)

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.facade.arenas.inspect_arena(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )

    def inspect_fastbins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_FASTBIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.facade.bins.inspect_fastbins(
            arena_address=arena_address,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    def inspect_tcache(
        self,
        *,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.facade.bins.inspect_tcache(
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    def inspect_bins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.facade.bins.inspect_bins(
            arena_address=arena_address,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    def classify_heap_corruption(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.facade.corruption.classify_heap_corruption(
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
        return self.facade.corruption.validate_free(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )

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
        return self.facade.techniques.analyze_heap_technique(
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
        return self.facade.techniques.detect_uaf(
            max_bytes=max_bytes,
            max_mappings=max_mappings,
            max_chunks=max_chunks,
            cancellation_check=cancellation_check,
        )

    def capture_heap_snapshot(
        self,
        *,
        name: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.facade.history.capture_heap_snapshot(
            name=name,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )


class GlibcAnalyzer(_ForwardingAnalyzer):
    """Analyzer for the glibc family (the typed structural parser)."""

    family: Optional[str] = "glibc"


class UnsupportedAnalyzer(_ForwardingAnalyzer):
    """Analyzer for families without a typed parser yet.

    Forwards to the glibc components, which re-classify the inferior
    themselves and return the honest typed ``unsupported`` verdict.  A later
    phase replaces the forwarding on this class with a real parser.
    """

    family: Optional[str] = None


def analyzer_for_family(
    family: Optional[str], facade: "HeapTools"
) -> AllocatorAnalyzer:
    """Return the analyzer for a detected allocator family.

    ``glibc`` maps to :class:`GlibcAnalyzer`, ``jemalloc`` maps to
    :class:`~paomcp.tools.backends.jemalloc.JemallocAnalyzer` (extent/rtree
    core), ``libmalloc`` maps to
    :class:`~paomcp.tools.backends.libmalloc.LibmallocAnalyzer` (malloc_zone
    introspection); every other family -- and an unresolvable detection
    (``None``) -- maps to :class:`UnsupportedAnalyzer`.  This is the single
    extension point for a new family's parser: add the family name to the
    dispatch and the facade starts routing to it.
    """
    if family == "glibc":
        return GlibcAnalyzer(facade)
    if family == "jemalloc":
        # Lazy import: jemalloc imports heap.py, and heap.py lazily imports
        # this module -- resolve at dispatch time to keep the cycle broken.
        from .jemalloc import JemallocAnalyzer  # noqa: PLC0415

        return JemallocAnalyzer(facade)
    if family == "libmalloc":
        # Same lazy-import cycle: libmalloc.py imports heap.py; resolve the
        # analyzer only when dispatch actually needs it.
        from .libmalloc import LibmallocAnalyzer  # noqa: PLC0415

        return LibmallocAnalyzer(facade)
    return UnsupportedAnalyzer(facade)


__all__ = [
    "AllocatorAnalyzer",
    "GlibcAnalyzer",
    "UnsupportedAnalyzer",
    "analyzer_for_family",
]
