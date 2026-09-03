"""glibc ptmalloc parsers facade (implementations live in heap_glibc_*)."""

from __future__ import annotations

from paomcp.tools.backends.heap_glibc_heap import (
    CHUNK_INSPECTION_SCHEMA,
    CHUNK_VALIDATION_SCHEMA,
    GlibcHeap,
    IS_MMAPPED,
    NON_MAIN_ARENA,
    PREV_INUSE,
    SIZE_BITS_MASK,
)
from paomcp.tools.backends.heap_glibc_arenas import (
    ARENA_INSPECTION_SCHEMA,
    ARENA_LIST_SCHEMA,
    ArenaLayoutResolver,
    BINMAPSIZE,
    BINS_ARRAY_LEN,
    GlibcArenas,
    MAX_ARENAS_TRAVERSAL,
    NBINS,
    NFASTBINS,
)
from paomcp.tools.backends.heap_glibc_bins import (
    BIN_INSPECTION_SCHEMA,
    DEFAULT_SLUB_PAGE_SIZE,
    FASTBIN_INSPECTION_SCHEMA,
    GlibcBins,
    TCACHE_INSPECTION_SCHEMA,
    TCACHE_MAX_BINS,
    TcacheLayoutResolver,
)
from paomcp.tools.backends.heap_glibc_corruption import (
    FREE_VALIDATION_SCHEMA,
    GlibcHeapCorruption,
    HEAP_CORRUPTION_SCHEMA,
)

__all__ = [
    "ARENA_INSPECTION_SCHEMA",
    "ARENA_LIST_SCHEMA",
    "ArenaLayoutResolver",
    "BINMAPSIZE",
    "BINS_ARRAY_LEN",
    "BIN_INSPECTION_SCHEMA",
    "CHUNK_INSPECTION_SCHEMA",
    "CHUNK_VALIDATION_SCHEMA",
    "DEFAULT_SLUB_PAGE_SIZE",
    "FASTBIN_INSPECTION_SCHEMA",
    "FREE_VALIDATION_SCHEMA",
    "GlibcArenas",
    "GlibcBins",
    "GlibcHeap",
    "GlibcHeapCorruption",
    "HEAP_CORRUPTION_SCHEMA",
    "IS_MMAPPED",
    "MAX_ARENAS_TRAVERSAL",
    "NBINS",
    "NFASTBINS",
    "NON_MAIN_ARENA",
    "PREV_INUSE",
    "SIZE_BITS_MASK",
    "TCACHE_INSPECTION_SCHEMA",
    "TCACHE_MAX_BINS",
    "TcacheLayoutResolver",
]
