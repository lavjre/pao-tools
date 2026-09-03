"""Shared heap schemas and budgets.

Single source of truth for allocator detection, traversal caps and
snapshot budgets; discovery/snapshot/heap modules import from here.
"""

from __future__ import annotations

from typing import Tuple

PARSER_VERSION = "1"
ALLOCATOR_DETECTION_SCHEMA = "pao.allocator-detection.v1"
ALLOCATOR_CAPABILITIES_SCHEMA = "pao.allocator-capabilities.v1"
SAFE_LINKING_MIN_VERSION: Tuple[int, int] = (2, 32)
DEFAULT_MAX_CHUNKS_PER_FASTBIN = 128
DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN = 64
DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN = 1024
HEAP_SNAPSHOT_SCHEMA = "pao.heap-snapshot.v1"
HEAP_SNAPSHOT_LIST_SCHEMA = "pao.heap-snapshot-list.v1"
HEAP_SNAPSHOT_DIFF_SCHEMA = "pao.heap-snapshot-diff.v1"
HEAP_SNAPSHOT_LINK_SCHEMA = "pao.heap-snapshot-link.v1"
DEFAULT_MAX_HEAP_SNAPSHOTS = 16
DEFAULT_MAX_HEAP_SNAPSHOT_BYTES = 64 * 1024 * 1024

__all__ = [
    "PARSER_VERSION",
    "ALLOCATOR_DETECTION_SCHEMA",
    "ALLOCATOR_CAPABILITIES_SCHEMA",
    "SAFE_LINKING_MIN_VERSION",
    "DEFAULT_MAX_CHUNKS_PER_FASTBIN",
    "DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN",
    "DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN",
    "HEAP_SNAPSHOT_SCHEMA",
    "HEAP_SNAPSHOT_LIST_SCHEMA",
    "HEAP_SNAPSHOT_DIFF_SCHEMA",
    "HEAP_SNAPSHOT_LINK_SCHEMA",
    "DEFAULT_MAX_HEAP_SNAPSHOTS",
    "DEFAULT_MAX_HEAP_SNAPSHOT_BYTES",
]
