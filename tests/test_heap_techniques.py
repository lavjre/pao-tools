"""Tests for Stage 6.6 glibc heap-technique analysis (GlibcTechniqueAnalyzer)."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    HEAP_TECHNIQUE_SCHEMA,
    NFASTBINS,
    AllocatorDiscovery,
    GlibcArenas,
    GlibcBins,
    GlibcHeap,
    GlibcHeapCorruption,
    HeapTools,
)

from tests.test_glibc_arenas import (
    GLIBC_MAPS,
    MAIN_ARENA_ADDR,
    MUSL_MAPS,
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
    make_arena_bytes,
)
from tests.test_glibc_fastbins import place_arena_with_fastbin_head, place_fastbin_chain
from tests.test_glibc_regular_bins import (
    HEAP_BASE,
    HEAP_END,
    _bin_head_address,
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

# All test chunks live above the tcache_perthread_struct span [0x9010, 0x9290)
# so separate 8-byte fd/next reads never fall back into the tcache blob (the
# fake gdb's range-fallback prefers the first covering region in insertion
# order), and still inside the heap mapping [0x9000, 0xa000).
CHUNK_BASE = HEAP_BASE + 0x600


def _tcache(counts=None, entries=None):
    """make_tcache_bytes wrapper that pads sparse lists to 64 entries."""
    pad = lambda lst: (
        None if lst is None else list(lst) + [0] * (TCACHE_MAX_BINS - len(lst))
    )
    return make_tcache_bytes(counts=pad(counts), entries=pad(entries))


# ---------------------------------------------------------------------------
# Fixture: a fully-wired HeapTools facade over the arena fakes.
# ---------------------------------------------------------------------------
def build_techniques(
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
        gdb,
        inspection,
        dbg,
        "sess",
        binary_path_provider=lambda: "/bin/chal",
    )
    return heap, gdb


def _place_clean_heap(gdb: ArenaFakeGdb) -> None:
    """Self-referential bins (all empty), zeroed tcache struct, top = 0."""
    _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
    gdb.place(TCACHE_ADDR, make_tcache_bytes())


def _place_regular_chunk(
    gdb: ArenaFakeGdb,
    chunk_address: int,
    size_field: int,
    fd: int,
    bk: int,
    *,
    prev_size: int = 0,
    byteorder: str = "little",
) -> None:
    """Write a regular-bin free chunk: header + fd + bk = 32 bytes."""
    buf = bytearray(32)
    buf[0:8] = int(prev_size).to_bytes(8, byteorder)
    buf[8:16] = int(size_field).to_bytes(8, byteorder)
    buf[16:24] = int(fd).to_bytes(8, byteorder)
    buf[24:32] = int(bk).to_bytes(8, byteorder)
    gdb.place(chunk_address, bytes(buf))


def _place_bins_containing(gdb: ArenaFakeGdb, bin_index: int, head_chunk: int) -> None:
    """Place an arena where `bin_index`'s chain holds `head_chunk` (all else empty)."""
    bins_array: List[int] = []
    for i in range(1, 127):
        other = _bin_head_address(MAIN_ARENA_ADDR, i)
        bins_array.extend(
            [other, other] if i != bin_index else [head_chunk, head_chunk]
        )
    gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))


def _place_overlapping_fastbin_pair(
    gdb: ArenaFakeGdb, a: int, *, size_field: int = 0x21
) -> int:
    """Two adjacent 0x20-size chunks in one blob (b = a + 0x10).

    a.fd aliases b.prev_size at the same physical offset, exactly like real
    glibc. Returns b.
    """
    b = a + 0x10
    buf = bytearray(40)
    buf[8:16] = int(size_field).to_bytes(8, "little")
    buf[16:24] = (b ^ ((a + 16) >> 12)).to_bytes(8, "little")  # PROTECT_PTR(b)
    buf[24:32] = int(size_field).to_bytes(8, "little")
    # b.fd stays 0
    gdb.place(a, bytes(buf))
    return b


def _finding(parsed: Dict[str, Any], technique: str) -> Dict[str, Any]:
    for entry in parsed["findings"]:
        if entry["technique"] == technique:
            return entry
    raise AssertionError(
        f"no finding for {technique!r} in {[f['technique'] for f in parsed['findings']]}"
    )


# ---------------------------------------------------------------------------
# Orchestration guards and verdicts
# ---------------------------------------------------------------------------
class TestTechniqueGuards:
    def test_requires_live_inferior(self):
        heap, _ = build_techniques(live=False)
        with pytest.raises(LiveInferiorError):
            heap.analyze_heap_technique()

    def test_musl_target_returns_unsupported(self):
        heap, _ = build_techniques(mappings=MUSL_MAPS, libc=None)
        result = heap.analyze_heap_technique()
        parsed = result["parsed"]
        assert parsed["schema"] == HEAP_TECHNIQUE_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"
        assert "findings" not in parsed

    def test_unresolvable_bins_offset(self):
        dbg = ArenaFakeDebugger(unresolved_fields=["bins"])
        heap, _ = build_techniques(debugger=dbg)
        result = heap.analyze_heap_technique()
        parsed = result["parsed"]
        assert parsed["supported"] is True
        assert parsed["verdict"] == "unresolvable"
        assert parsed["findings"] == []

    def test_invalid_technique_name_raises(self):
        heap, _ = build_techniques()
        _place_clean_heap(heap.gdb)
        with pytest.raises(ValueError):
            heap.analyze_heap_technique(techniques=["not_a_technique"])

    def test_clean_heap_is_clean(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        result = heap.analyze_heap_technique()
        parsed = result["parsed"]
        assert parsed["verdict"] == "clean"
        assert parsed["supported"] is True
        assert len(parsed["findings"]) == 13
        assert all(entry["present"] is False for entry in parsed["findings"])
        assert parsed["summary"] == {"detected": 0, "suspected": 0, "not_detected": 13}

    def test_technique_filter_restricts_findings(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        result = heap.analyze_heap_technique(techniques=["house_of_force"])
        parsed = result["parsed"]
        assert parsed["techniques_requested"] == ["house_of_force"]
        assert len(parsed["findings"]) == 1
        assert parsed["findings"][0]["technique"] == "house_of_force"


# ---------------------------------------------------------------------------
# Per-detector evidence
# ---------------------------------------------------------------------------
class TestHouseOfSpirit:
    def test_fastbin_chunk_outside_mapping(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        fake = 0x7FFF12345000  # aligned but outside every mapping
        gdb.place(fake, (0).to_bytes(8, "little") + (0x21).to_bytes(8, "little"))
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, CHUNK_BASE + 0x100)
        place_fastbin_chain(gdb, [CHUNK_BASE + 0x100, fake], safe_linking=True)
        result = heap.analyze_heap_technique(techniques=["house_of_spirit"])
        parsed = result["parsed"]
        assert parsed["verdict"] == "exploitable"
        finding = _finding(parsed, "house_of_spirit")
        assert finding["present"] is True
        assert finding["severity"] == "critical"
        assert finding["confidence"] == "high"
        assert (
            finding["evidence"]["freed_chunks_outside_mappings"][0]["chunk_address"]
            == fake
        )


class TestHouseOfEinherjar:
    def test_forged_prev_size_mismatch(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        chunk_addr = CHUNK_BASE + 0x400
        _place_bins_containing(gdb, 1, chunk_addr)
        _place_regular_chunk(
            gdb,
            chunk_addr,
            size_field=0x90,  # PREV_INUSE clear
            fd=_bin_head_address(MAIN_ARENA_ADDR, 1),
            bk=_bin_head_address(MAIN_ARENA_ADDR, 1),
            prev_size=0x400,
        )
        # Physical predecessor at chunk_addr - 0x400 really has size 0x20.
        gdb.place(
            chunk_addr - 0x400,
            (0).to_bytes(8, "little") + (0x20).to_bytes(8, "little"),
        )
        result = heap.analyze_heap_technique(techniques=["house_of_einherjar"])
        parsed = result["parsed"]
        finding = _finding(parsed, "house_of_einherjar")
        assert finding["present"] is True
        assert finding["confidence"] == "medium"
        assert (
            finding["evidence"]["prev_size_mismatches"][0]["claimed_prev_size"] == 0x400
        )


class TestHouseOfForce:
    def test_huge_top_chunk_size(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        top_addr = CHUNK_BASE + 0x800
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(top=top_addr))
        gdb.place(
            top_addr, (0).to_bytes(8, "little") + (0x7FFFFFFFFFFF).to_bytes(8, "little")
        )
        result = heap.analyze_heap_technique(techniques=["house_of_force"])
        parsed = result["parsed"]
        assert parsed["verdict"] == "exploitable"
        finding = _finding(parsed, "house_of_force")
        assert finding["present"] is True
        assert finding["severity"] == "critical"
        assert finding["evidence"]["wildly_huge"] is True

    def test_sane_top_chunk_is_not_present(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        # Top fills the heap mapping exactly: size 0x800 ends at HEAP_END.
        top_addr = HEAP_END - 0x800
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(top=top_addr))
        gdb.place(top_addr, (0).to_bytes(8, "little") + (0x800).to_bytes(8, "little"))
        result = heap.analyze_heap_technique(techniques=["house_of_force"])
        parsed = result["parsed"]
        assert parsed["verdict"] == "clean"
        finding = _finding(parsed, "house_of_force")
        assert finding["present"] is False


class TestHouseOfLore:
    def test_bk_escaping_the_heap(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        chunk_addr = CHUNK_BASE + 0x200
        _place_bins_containing(gdb, 1, chunk_addr)
        _place_regular_chunk(
            gdb,
            chunk_addr,
            size_field=0x91,
            fd=_bin_head_address(MAIN_ARENA_ADDR, 1),
            bk=0x4141414141414141,
        )
        result = heap.analyze_heap_technique(techniques=["house_of_lore"])
        parsed = result["parsed"]
        finding = _finding(parsed, "house_of_lore")
        assert finding["present"] is True
        assert finding["severity"] == "high"
        assert finding["evidence"]["pointers_outside_mappings"][0]["field"] == "bk"


class TestTcachePoisoning:
    def test_next_pointer_into_writable_nonheap(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        user = CHUNK_BASE + 0x300
        poison = LIBC_START + 0x1D0000  # inside libc rw segment
        gdb.place(TCACHE_ADDR, _tcache(counts=[1], entries=[user]))
        place_tcache_chain(gdb, [user, poison], safe_linking=True)
        result = heap.analyze_heap_technique(techniques=["tcache_poisoning"])
        parsed = result["parsed"]
        assert parsed["verdict"] == "exploitable"
        finding = _finding(parsed, "tcache_poisoning")
        assert finding["present"] is True
        assert finding["confidence"] == "high"
        assert finding["evidence"]["poisoned_links"][0]["next_decoded"] == poison


class TestTcacheDoubleFree:
    def test_same_chunk_in_two_tcache_bins(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        user = CHUNK_BASE + 0x300
        chunk = user - 16  # what the walker reports as chunk_address
        gdb.place(TCACHE_ADDR, _tcache(counts=[1, 1], entries=[user, user]))
        place_tcache_chain(gdb, [user])
        place_tcache_chain(gdb, [user])
        result = heap.analyze_heap_technique(techniques=["tcache_double_free"])
        parsed = result["parsed"]
        finding = _finding(parsed, "tcache_double_free")
        assert finding["present"] is True
        assert finding["severity"] == "critical"
        assert finding["evidence"]["duplicate_chunks"][0]["chunk_address"] == chunk


class TestHouseOfBotcake:
    def test_chunk_in_tcache_and_unsorted(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        user = CHUNK_BASE + 0x400
        chunk = user - 16
        head = _bin_head_address(MAIN_ARENA_ADDR, 1)
        gdb.place(TCACHE_ADDR, _tcache(counts=[1], entries=[user]))
        place_tcache_chain(gdb, [user])
        _place_bins_containing(gdb, 1, chunk)
        _place_regular_chunk(gdb, chunk, size_field=0x91, fd=head, bk=head)
        result = heap.analyze_heap_technique(techniques=["house_of_botcake"])
        parsed = result["parsed"]
        finding = _finding(parsed, "house_of_botcake")
        assert finding["present"] is True
        assert (
            finding["evidence"]["chunks_in_tcache_and_bins"][0]["chunk_address"]
            == chunk
        )


class TestFastbinDup:
    def test_chunk_in_two_fastbin_slots(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        a = CHUNK_BASE + 0x100
        heads = [0] * NFASTBINS
        heads[0] = a
        heads[1] = a
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(fastbinsY=heads))
        place_fastbin_chain(gdb, [a])
        result = heap.analyze_heap_technique(techniques=["fastbin_dup"])
        parsed = result["parsed"]
        finding = _finding(parsed, "fastbin_dup")
        assert finding["present"] is True
        assert finding["severity"] == "critical"
        assert finding["evidence"]["cross_bin_duplicates"][0]["chunk_address"] == a


class TestUnsafeUnlink:
    def test_fd_targets_writable_nonheap(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        chunk_addr = CHUNK_BASE + 0x200
        _place_bins_containing(gdb, 2, chunk_addr)
        _place_regular_chunk(
            gdb,
            chunk_addr,
            size_field=0x91,
            fd=MAIN_ARENA_ADDR,  # writable, non-heap
            bk=_bin_head_address(MAIN_ARENA_ADDR, 2),
        )
        result = heap.analyze_heap_technique(techniques=["unsafe_unlink"])
        parsed = result["parsed"]
        finding = _finding(parsed, "unsafe_unlink")
        assert finding["present"] is True
        assert finding["severity"] == "high"
        assert finding["evidence"]["writable_targets"][0]["pointer"] == MAIN_ARENA_ADDR


class TestOverlappingChunks:
    def test_free_ranges_intersect(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        a = CHUNK_BASE
        b = _place_overlapping_fastbin_pair(gdb, a)
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, a)
        result = heap.analyze_heap_technique(techniques=["overlapping_chunks"])
        parsed = result["parsed"]
        finding = _finding(parsed, "overlapping_chunks")
        assert finding["present"] is True
        assert finding["severity"] == "high"
        assert finding["evidence"]["overlapping_ranges"][0]["overlap_start"] == b

    def test_adjacent_free_chunks_do_not_overlap(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        a = CHUNK_BASE
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, a)
        place_fastbin_chain(gdb, [a, a + 0x20], size_field=0x21, safe_linking=True)
        result = heap.analyze_heap_technique(techniques=["overlapping_chunks"])
        parsed = result["parsed"]
        assert parsed["verdict"] == "clean"
        assert _finding(parsed, "overlapping_chunks")["present"] is False


class TestTopChunkExhaustion:
    def test_top_near_mapping_end(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        top_addr = HEAP_END - 0x800
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(top=top_addr))
        gdb.place(top_addr, (0).to_bytes(8, "little") + (0x800).to_bytes(8, "little"))
        result = heap.analyze_heap_technique(
            techniques=["top_chunk_exhaustion", "house_of_force"]
        )
        parsed = result["parsed"]
        exhaustion = _finding(parsed, "top_chunk_exhaustion")
        force = _finding(parsed, "house_of_force")
        assert exhaustion["present"] is True
        assert exhaustion["severity"] == "low"
        assert force["present"] is False  # 0x800 is a sane size


class TestLargeBinAttack:
    def test_always_not_statistically_detectable(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        result = heap.analyze_heap_technique(techniques=["large_bin_attack"])
        parsed = result["parsed"]
        finding = _finding(parsed, "large_bin_attack")
        assert finding["present"] is False
        assert "not statically detectable" in finding["reason"]


class TestMmapChunkScan:
    def test_finds_is_mmapped_header_in_writable_mapping(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        page = bytearray(0x1000)
        page[8:16] = (0x1000 | 0x2).to_bytes(8, "little")  # size | IS_MMAPPED
        gdb.place(LIBC_START + 0x1000, bytes(page))
        result = heap.analyze_heap_technique(techniques=["mmap_chunk_scan"])
        parsed = result["parsed"]
        finding = _finding(parsed, "mmap_chunk_scan")
        assert finding["present"] is True
        assert (
            finding["evidence"]["mmap_chunks"][0]["chunk_address"]
            == LIBC_START + 0x1000
        )
        assert finding["evidence"]["mappings_scanned"] >= 1

    def test_scan_can_be_disabled(self):
        heap, gdb = build_techniques()
        _place_clean_heap(gdb)
        result = heap.analyze_heap_technique(
            techniques=["mmap_chunk_scan"], scan_mmap_mappings=False
        )
        parsed = result["parsed"]
        finding = _finding(parsed, "mmap_chunk_scan")
        assert finding["present"] is False
        assert finding["reason"] == "mmap chunk scan disabled"
