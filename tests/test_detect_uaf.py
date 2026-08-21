"""Tests for detect_uaf — dangling-reference scan across writable mappings."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import UAF_SCAN_SCHEMA, HeapTools

from tests.test_glibc_arenas import (
    MAIN_ARENA_ADDR,
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
)
from tests.test_glibc_regular_bins import (
    HEAP_BASE,
    _extend_maps_with_heap,
    _libc_version,
    _place_self_referential_bins,
)
from tests.test_glibc_tcache import (
    TCACHE_ADDR,
    TcacheAwareFakeDebugger,
    make_tcache_bytes,
    place_tcache_chain,
)

LIBC_START = 0x7FFFF7C00000

TCACHE_MAX_BINS = 64

# Freed chunks live above the tcache struct span [0x9010, 0x9290) so the
# fake's range-fallback never serves tcache bytes for the chunk header / next
# reads, and still inside the heap mapping [0x9000, 0xa000).
CHUNK_BASE = HEAP_BASE + 0x600


def _tcache(counts=None, entries=None):
    pad = lambda lst: (
        None if lst is None else list(lst) + [0] * (TCACHE_MAX_BINS - len(lst))
    )
    return make_tcache_bytes(counts=pad(counts), entries=pad(entries))


def build_uaf(
    *,
    live: bool = True,
    mappings: Optional[List[Dict[str, Any]]] = None,
    libc: Optional[Dict[str, Any]] = None,
    debugger: Optional[ArenaFakeDebugger] = None,
):
    gdb = ArenaFakeGdb(live=live)
    resolved_libc = libc if libc is not None else _libc_version("2.35")
    resolved_maps = mappings if mappings is not None else _extend_maps_with_heap()
    inspection = ArenaFakeInspection(
        gdb,
        mappings=resolved_maps,
        libc=None if libc is None and mappings is not None else resolved_libc,
    )
    dbg = debugger if debugger is not None else TcacheAwareFakeDebugger()
    heap = HeapTools(
        gdb, inspection, dbg, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    return heap, gdb


def _clean_heap(gdb: ArenaFakeGdb) -> None:
    _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
    gdb.place(TCACHE_ADDR, make_tcache_bytes())


def _place_pointer_page(gdb: ArenaFakeGdb, base: int, offset: int, value: int) -> None:
    """Place a readable 0x1000-byte page with an 8-byte pointer at `offset`."""
    page = bytearray(0x1000)
    page[offset : offset + 8] = int(value).to_bytes(8, "little")
    gdb.place(base, bytes(page))


def _place_freed_tcache_chunk(heap: HeapTools, gdb: ArenaFakeGdb, user: int) -> None:
    """Free one 0x40-size chunk to tcache bin 0; returns nothing."""
    gdb.place(TCACHE_ADDR, _tcache(counts=[1], entries=[user]))
    place_tcache_chain(gdb, [user], chunk_size_field=0x41)


class TestDetectUafGuards:
    def test_requires_live_inferior(self):
        heap, _ = build_uaf(live=False)
        with pytest.raises(LiveInferiorError):
            heap.detect_uaf()

    def test_invalid_caps_raise(self):
        heap, _ = build_uaf()
        with pytest.raises(ValueError):
            heap.detect_uaf(max_bytes=0)
        with pytest.raises(ValueError):
            heap.detect_uaf(max_mappings=0)
        with pytest.raises(ValueError):
            heap.detect_uaf(max_chunks=0)


class TestDetectUafScan:
    def test_finds_dangling_pointer_into_freed_tcache_chunk(self):
        heap, gdb = build_uaf()
        _clean_heap(gdb)
        user = CHUNK_BASE + 0x200
        chunk = user - 16
        _place_freed_tcache_chunk(heap, gdb, user)
        # user range = [user, user - 16 + 0x40) = [user, user + 0x30)
        pointer_target = user + 8
        _place_pointer_page(gdb, LIBC_START, 0x100, pointer_target)

        result = heap.detect_uaf()
        parsed = result["parsed"]
        assert parsed["schema"] == UAF_SCAN_SCHEMA
        assert parsed["verdict"] == "dangling_references_found"
        assert parsed["freed_chunks"]["count"] == 1
        candidates = parsed["candidates"]
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["address_of_pointer"] == LIBC_START + 0x100
        assert candidate["pointer_value"] == pointer_target
        assert candidate["freed_chunk_address"] == chunk
        assert candidate["freed_chunk_size"] == 0x40
        assert candidate["source_bin"] == "tcache[0]"
        assert candidate["offset_into_chunk"] == 8

    def test_no_dangling_references_when_pointer_absent(self):
        heap, gdb = build_uaf()
        _clean_heap(gdb)
        user = CHUNK_BASE + 0x200
        _place_freed_tcache_chunk(heap, gdb, user)

        result = heap.detect_uaf()
        parsed = result["parsed"]
        assert parsed["verdict"] == "no_dangling_references"
        assert parsed["candidates"] == []
        assert parsed["freed_chunks"]["count"] == 1

    def test_pointer_outside_freed_range_is_ignored(self):
        heap, gdb = build_uaf()
        _clean_heap(gdb)
        user = CHUNK_BASE + 0x200
        _place_freed_tcache_chunk(heap, gdb, user)
        # Points far past the freed range end.
        _place_pointer_page(gdb, LIBC_START, 0x100, user + 0x100)

        result = heap.detect_uaf()
        parsed = result["parsed"]
        assert parsed["verdict"] == "no_dangling_references"


class TestDetectUafCaps:
    def test_max_chunks_truncates_collection(self):
        heap, gdb = build_uaf()
        _clean_heap(gdb)
        a = CHUNK_BASE + 0x200
        b = CHUNK_BASE + 0x300
        gdb.place(TCACHE_ADDR, _tcache(counts=[1, 1], entries=[a, b]))
        place_tcache_chain(gdb, [a], chunk_size_field=0x41)
        place_tcache_chain(gdb, [b], chunk_size_field=0x41)
        _place_pointer_page(gdb, LIBC_START, 0x100, a + 8)

        result = heap.detect_uaf(max_chunks=1)
        parsed = result["parsed"]
        assert parsed["freed_chunks"]["truncated"] is True
        assert parsed["freed_chunks"]["count"] == 1

    def test_max_mappings_truncates_scan(self):
        heap, gdb = build_uaf()
        _clean_heap(gdb)
        user = CHUNK_BASE + 0x200
        _place_freed_tcache_chunk(heap, gdb, user)
        _place_pointer_page(gdb, LIBC_START, 0x100, user + 8)

        # The pointer lives in the first writable mapping, so the cap still
        # finds it; mappings_scanned counts the mapping that trips the cap.
        result = heap.detect_uaf(max_mappings=1)
        parsed = result["parsed"]
        assert parsed["verdict"] == "dangling_references_found"
        assert parsed["scan"]["mappings_scanned"] >= 1
        assert parsed["scan"]["truncated"] is True

    def test_max_bytes_truncates_scan(self):
        heap, gdb = build_uaf()
        _clean_heap(gdb)
        user = CHUNK_BASE + 0x200
        _place_freed_tcache_chunk(heap, gdb, user)
        _place_pointer_page(gdb, LIBC_START + 0x2000, 0x100, user + 8)

        # 0x800 bytes only covers part of the first page of libc; the pointer
        # sits beyond. bytes_scanned accumulates whole pages and may overshoot.
        result = heap.detect_uaf(max_bytes=0x800)
        parsed = result["parsed"]
        assert parsed["verdict"] == "no_dangling_references"
        assert parsed["scan"]["bytes_scanned"] >= 0x800
