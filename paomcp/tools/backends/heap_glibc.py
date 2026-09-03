"""glibc ptmalloc parsers facade (implementations live in heap_glibc_*)."""

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
    HeapHistory,
    HeapSnapshotError,
)

from paomcp.tools.backends.heap_discovery import AllocatorDiscovery

# ----------------------------------------------------------------------
# Stage 6.2 -- glibc malloc_chunk parsing
# ----------------------------------------------------------------------

# glibc malloc_chunk flag bits (low three bits of the ``size`` field).

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
    _MALLOC_STATE_FIELDS,
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
    "_MALLOC_STATE_FIELDS",
]
