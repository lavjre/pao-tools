"""Tests for Stage 6.3d typed glibc regular-bin traversal."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    BIN_INSPECTION_SCHEMA,
    NBINS,
    AllocatorDiscovery,
    GlibcArenas,
    GlibcBins,
    HeapTools,
)

from tests.test_glibc_arenas import (  # noqa: E402
    ARENA2_ADDR,
    GLIBC_MAPS,
    MAIN_ARENA_ADDR,
    MUSL_MAPS,
    STANDARD_OFFSETS_X86_64,
    STANDARD_STRUCT_SIZE_X86_64,
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
    make_arena_bytes,
)

HEAP_BASE = 0x555555559000
HEAP_END = 0x55555555A000


def _extend_maps_with_heap() -> List[Dict[str, Any]]:
    return GLIBC_MAPS + [
        {
            "file": "[heap]",
            "start": f"0x{HEAP_BASE:x}",
            "end": f"0x{HEAP_END:x}",
            "permissions": "rw-p",
            "readable": True,
            "writable": True,
            "executable": False,
        }
    ]


def _libc_version(version: Optional[str]) -> Dict[str, Any]:
    return {"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": version}


def build_bins(
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
    dbg = debugger if debugger is not None else ArenaFakeDebugger()
    discovery = AllocatorDiscovery(
        gdb, inspection, dbg, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    arenas = GlibcArenas(gdb, inspection, dbg, "sess", discovery=discovery)
    bins_ = GlibcBins(gdb, inspection, dbg, "sess", discovery=discovery, arenas=arenas)
    return bins_, gdb


def _bin_head_address(arena_address: int, bin_index: int, size_sz: int = 8) -> int:
    """Compute the synthetic bin_at(m, i) address the impl uses."""
    bins_offset = STANDARD_OFFSETS_X86_64["bins"]
    return arena_address + bins_offset + (bin_index - 1) * 2 * size_sz - 2 * size_sz


def _place_self_referential_bins(gdb: ArenaFakeGdb, arena_address: int) -> None:
    """Place an arena whose bins[] all self-reference (all bins empty).

    glibc's malloc_init_state initialises every bin to fd = bk = bin_at.
    """
    heads = list(range(1, NBINS))
    bins: List[int] = []
    for i in heads:
        head = _bin_head_address(arena_address, i)
        bins.append(head)  # fd = bin_head
        bins.append(head)  # bk = bin_head
    gdb.place(arena_address, make_arena_bytes(bins=bins))


def _place_free_chunk(
    gdb: ArenaFakeGdb,
    chunk_address: int,
    size_field: int,
    fd: int,
    bk: int,
    byteorder: str = "little",
) -> None:
    """Write a free chunk: 16-byte header + fd + bk = 32 bytes total."""
    buf = bytearray(32)
    buf[0:8] = (0).to_bytes(8, byteorder)  # prev_size
    buf[8:16] = int(size_field).to_bytes(8, byteorder)
    buf[16:24] = int(fd).to_bytes(8, byteorder)
    buf[24:32] = int(bk).to_bytes(8, byteorder)
    gdb.place(chunk_address, bytes(buf))


# ---------------------------------------------------------------------------
# Pure helper: bin_category
# ---------------------------------------------------------------------------
class TestBinCategory:
    def test_unsorted_is_bin_1(self):
        assert GlibcBins.bin_category(1) == "unsorted"

    def test_small_range(self):
        assert GlibcBins.bin_category(2) == "small"
        assert GlibcBins.bin_category(63) == "small"

    def test_large_range(self):
        assert GlibcBins.bin_category(64) == "large"
        assert GlibcBins.bin_category(NBINS - 1) == "large"

    def test_rejects_bin_0(self):
        with pytest.raises(ValueError):
            GlibcBins.bin_category(0)

    def test_rejects_bin_at_upper_bound(self):
        with pytest.raises(ValueError):
            GlibcBins.bin_category(NBINS)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            GlibcBins.bin_category(-1)


# ---------------------------------------------------------------------------
# inspect_bins orchestration
# ---------------------------------------------------------------------------
class TestInspectBins:
    def test_requires_live_inferior(self):
        bins_, _ = build_bins(live=False)
        with pytest.raises(LiveInferiorError):
            bins_.inspect_bins()

    def test_unsupported_when_target_is_musl(self):
        bins_, _ = build_bins(mappings=MUSL_MAPS, libc=None)
        result = bins_.inspect_bins()
        parsed = result["parsed"]
        assert parsed["schema"] == BIN_INSPECTION_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"

    def test_unresolvable_when_bins_offset_missing(self):
        dbg = ArenaFakeDebugger(unresolved_fields=["bins"])
        bins_, _ = build_bins(debugger=dbg)
        result = bins_.inspect_bins()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unresolvable"

    def test_invalid_max_chunks_raises(self):
        bins_, _ = build_bins()
        with pytest.raises(ValueError):
            bins_.inspect_bins(max_chunks_per_bin=0)

    def test_invalid_arena_address_raises(self):
        bins_, _ = build_bins()
        with pytest.raises(ValueError):
            bins_.inspect_bins(arena_address="not-a-number")

    def test_all_empty_bins_self_reference(self):
        # A freshly-initialised arena: every bin has fd = bk = bin_at.
        bins_, gdb = build_bins()
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        result = bins_.inspect_bins()
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["is_main_arena"] is True
        assert len(parsed["bins"]) == NBINS - 1  # bins 1..127
        for slot in parsed["bins"]:
            assert slot["empty"] is True
            assert slot["chain"] == []
            assert slot["length"] == 0
        # Categories match expectations.
        assert parsed["bins"][0]["category"] == "unsorted"  # bin 1
        assert parsed["bins"][1]["category"] == "small"  # bin 2
        assert parsed["bins"][-1]["category"] == "large"  # bin 127

    def test_unsorted_bin_single_chunk_chain(self):
        # Place one chunk in the unsorted bin (bin 1).  The chunk's fd/bk
        # both point back to the bin head, forming a length-1 circular list.
        bins_, gdb = build_bins()
        bin_index = 1
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        chunk_addr = HEAP_BASE + 0x400
        # Set bins[0] = bins[1] = chunk_addr; every other bin self-references.
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(chunk_addr)  # fd
                bins_array.append(chunk_addr)  # bk
            else:
                other_head = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(other_head)
                bins_array.append(other_head)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        _place_free_chunk(
            gdb, chunk_addr, size_field=0xA1, fd=head, bk=head
        )  # size=0xA0
        result = bins_.inspect_bins()
        unsorted = result["parsed"]["bins"][0]
        assert unsorted["index"] == 1
        assert unsorted["category"] == "unsorted"
        assert unsorted["bin_head_address"] == head
        assert unsorted["fd_head"] == chunk_addr
        assert unsorted["bk_head"] == chunk_addr
        assert unsorted["empty"] is False
        assert unsorted["length"] == 1
        entry = unsorted["chain"][0]
        assert entry["chunk_address"] == chunk_addr
        assert entry["chunk_readable"] is True
        assert entry["size"] == 0xA0
        assert entry["prev_inuse"] is True
        assert entry["fd"] == head
        assert entry["bk"] == head

    def test_small_bin_multi_chunk_chain(self):
        # Place 3 chunks in small bin 4 with a doubly-linked circular list:
        # head <-> A <-> B <-> C <-> head
        bins_, gdb = build_bins()
        bin_index = 4
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        addresses = [HEAP_BASE + 0x400 + i * 0x60 for i in range(3)]
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(addresses[0])  # fd_head -> A
                bins_array.append(addresses[-1])  # bk_head -> C
            else:
                other_head = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(other_head)
                bins_array.append(other_head)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        for pos, addr in enumerate(addresses):
            fd = addresses[pos + 1] if pos + 1 < len(addresses) else head
            bk = addresses[pos - 1] if pos > 0 else head
            _place_free_chunk(gdb, addr, size_field=0x41, fd=fd, bk=bk)
        result = bins_.inspect_bins()
        small = result["parsed"]["bins"][bin_index - 1]
        assert small["category"] == "small"
        assert small["length"] == 3
        chain_addrs = [e["chunk_address"] for e in small["chain"]]
        assert chain_addrs == addresses
        # Last entry's fd loops back to the head -- confirms termination.
        assert small["chain"][-1]["fd"] == head

    def test_large_bin_chunk_categorization(self):
        bins_, gdb = build_bins()
        bin_index = 100  # large bin
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        chunk_addr = HEAP_BASE + 0x400
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(chunk_addr)
                bins_array.append(chunk_addr)
            else:
                other_head = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(other_head)
                bins_array.append(other_head)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        _place_free_chunk(
            gdb, chunk_addr, size_field=0x501, fd=head, bk=head
        )  # size=0x500
        result = bins_.inspect_bins()
        large = result["parsed"]["bins"][bin_index - 1]
        assert large["category"] == "large"
        assert large["length"] == 1
        assert large["chain"][0]["size"] == 0x500

    def test_cycle_detection_in_non_terminating_chain(self):
        # Chain: head -> A -> B -> A (cycle, never returns to head)
        bins_, gdb = build_bins()
        bin_index = 3
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        a = HEAP_BASE + 0x400
        b = HEAP_BASE + 0x500
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(a)
                bins_array.append(a)
            else:
                other_head = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(other_head)
                bins_array.append(other_head)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        _place_free_chunk(gdb, a, size_field=0x31, fd=b, bk=head)
        _place_free_chunk(gdb, b, size_field=0x31, fd=a, bk=head)  # cycles back to a
        result = bins_.inspect_bins()
        slot = result["parsed"]["bins"][bin_index - 1]
        assert slot["cycle_detected"] is True
        assert slot["length"] == 2

    def test_truncation_respects_max_chunks_per_bin(self):
        # Long non-terminating chain (each fd -> next but never returns to
        # head).  Cap at 3.
        bins_, gdb = build_bins()
        bin_index = 5
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        addresses = [HEAP_BASE + 0x400 + i * 0x40 for i in range(5)]
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(addresses[0])
                bins_array.append(addresses[-1])
            else:
                other_head = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(other_head)
                bins_array.append(other_head)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        for pos, addr in enumerate(addresses):
            fd = (
                addresses[pos + 1]
                if pos + 1 < len(addresses)
                else HEAP_BASE
                + 0xABCD  # deliberately outside chain, prevents natural end
            )
            bk = addresses[pos - 1] if pos > 0 else head
            _place_free_chunk(gdb, addr, size_field=0x41, fd=fd, bk=bk)
        result = bins_.inspect_bins(max_chunks_per_bin=3)
        slot = result["parsed"]["bins"][bin_index - 1]
        assert slot["truncated"] is True
        assert slot["length"] == 3

    def test_chunk_read_error_records_entry_and_stops_walk(self):
        # First chunk address is nonzero but no bytes placed there -> the
        # 32-byte read fails; the entry records chunk_readable=False and
        # the walk breaks.
        bins_, gdb = build_bins()
        bin_index = 6
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        chunk_addr = HEAP_BASE + 0x400
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(chunk_addr)
                bins_array.append(chunk_addr)
            else:
                other_head = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(other_head)
                bins_array.append(other_head)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        # Deliberately do NOT place a chunk at chunk_addr.
        result = bins_.inspect_bins()
        slot = result["parsed"]["bins"][bin_index - 1]
        assert slot["length"] == 1
        entry = slot["chain"][0]
        assert entry["chunk_readable"] is False
        assert entry["chunk_read_error"]

    def test_bins_block_read_failure_yields_unreadable_verdict(self):
        # No arena bytes placed at all -> bins block read fails.
        bins_, _ = build_bins()
        result = bins_.inspect_bins()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["bins"] == []

    def test_unresolved_byteorder_emits_warning(self):
        bins_, gdb = build_bins()
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)

        def _no_endian(command, timeout_sec=60.0):
            return {"success": True, "state": gdb.state, "responses": []}

        gdb.execute_command = _no_endian  # type: ignore[method-assign]
        result = bins_.inspect_bins()
        warnings = result.get("warnings", [])
        assert any(
            "byteorder could not be resolved" in w for w in warnings
        ), f"expected byteorder warning in {warnings!r}"

    def test_explicit_arena_address_parses_non_main_arena(self):
        # Regression: verify the explicit-arena-address path.  Every other
        # passing test uses the default (main_arena), so a change that
        # ignored the arena_address kwarg (always using layout.arena_address)
        # or hardcoded is_main_arena=True would ship green.  Mirrors the
        # parallel test in test_glibc_fastbins.py.
        bins_, gdb = build_bins()
        # Self-referential bins block placed at ARENA2_ADDR, computed
        # relative to ARENA2 (bin_head_addr depends on the target arena).
        heads = list(range(1, NBINS))
        bins_array: List[int] = []
        for i in heads:
            head = _bin_head_address(ARENA2_ADDR, i)
            bins_array.append(head)  # fd
            bins_array.append(head)  # bk
        gdb.place(ARENA2_ADDR, make_arena_bytes(bins=bins_array))
        # Keep main_arena readable too so any accidental fallback would
        # yield different output.
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)

        result = bins_.inspect_bins(arena_address=f"0x{ARENA2_ADDR:x}")
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["arena_address"] == ARENA2_ADDR
        assert parsed["is_main_arena"] is False
        # All bins empty by construction.
        assert all(slot["empty"] for slot in parsed["bins"])
        # Bin heads are computed relative to ARENA2, not MAIN_ARENA_ADDR.
        assert parsed["bins"][0]["bin_head_address"] == _bin_head_address(
            ARENA2_ADDR, 1
        )

    def test_explicit_arena_address_unreadable_verdict_is_main_arena_false(self):
        # Locks down the is_main_arena=False path in the UNREADABLE-verdict
        # branch as well as the parsed one -- a regression that only
        # applied the flag when parsed=True would otherwise slip through.
        bins_, _ = build_bins()
        # Deliberately place nothing at ARENA2_ADDR -> bins block read fails.
        result = bins_.inspect_bins(arena_address=f"0x{ARENA2_ADDR:x}")
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["arena_address"] == ARENA2_ADDR
        assert parsed["is_main_arena"] is False

    def test_raises_when_pointer_bits_unresolved(self):
        maps_no_end = [
            {k: v for k, v in m.items() if k != "end"} for m in _extend_maps_with_heap()
        ]
        dbg = ArenaFakeDebugger(pointer_bytes=0)
        bins_, _ = build_bins(mappings=maps_no_end, debugger=dbg)
        with pytest.raises(RuntimeError, match="pointer width"):
            bins_.inspect_bins()

    def test_inspect_bins_32bit_end_to_end(self):
        # Regression: exercise the full inspect_bins pipeline on a 32-bit
        # target so a future change that hardcoded size_sz=8 would fail.
        # On 32-bit, size_sz=4, bin_head_addr = arena+bins_offset+(i-1)*8-8,
        # and the per-chunk read is 4*size_sz = 16 bytes.
        heap32_base = 0x08049000
        heap32_end = 0x0804A000
        arena32_addr = heap32_base + 0x10

        # Build a minimal 32-bit malloc_state -- only bins matter for this
        # test.  Layout mirrors real glibc: bins offset at 60 (immediately
        # after top/last_remainder).
        offsets_32 = {
            "mutex": 0,
            "flags": 4,
            "have_fastchunks": 8,
            "fastbinsY": 12,  # 12 + 10*4 = 52
            "top": 52,
            "last_remainder": 56,
            "bins": 60,
            "binmap": 60 + 254 * 4,  # after bins
            "next": 60 + 254 * 4 + 16,
            "next_free": 60 + 254 * 4 + 20,
            "attached_threads": 60 + 254 * 4 + 24,
            "system_mem": 60 + 254 * 4 + 28,
            "max_system_mem": 60 + 254 * 4 + 32,
        }
        struct_size_32 = 60 + 254 * 4 + 36

        maps_32 = [
            {
                "file": "/bin/chal",
                "start": "0x8048000",
                "end": "0x8049000",
                "permissions": "r-xp",
                "readable": True,
                "writable": False,
                "executable": True,
            },
            {
                "file": "/lib/libc.so.6",
                "start": "0xf7c00000",
                "end": "0xf7e00000",
                "permissions": "rw-p",
                "readable": True,
                "writable": True,
                "executable": False,
            },
            {
                "file": "[heap]",
                "start": f"0x{heap32_base:x}",
                "end": f"0x{heap32_end:x}",
                "permissions": "rw-p",
                "readable": True,
                "writable": True,
                "executable": False,
            },
        ]
        dbg = ArenaFakeDebugger(
            arena_address=arena32_addr,
            struct_size=struct_size_32,
            offsets=offsets_32,
            pointer_bytes=4,
        )
        bins_, gdb = build_bins(
            mappings=maps_32, libc=_libc_version("2.31"), debugger=dbg
        )

        # Place one chunk in the unsorted bin (bin 1) on 32-bit.
        bin_index = 1
        # bin_head_addr = arena + bins_offset + (i-1)*2*4 - 2*4 = arena+60-8
        bin_head_addr_32 = arena32_addr + 60 - 8
        chunk_addr = heap32_base + 0x400

        # Assemble a 32-bit bins[] block (254 * 4 = 1016 bytes).  Every
        # entry self-references, except bin 1 which points to our chunk.
        bins_block = bytearray(254 * 4)
        for i in range(1, NBINS):
            head_i = arena32_addr + 60 + (i - 1) * 2 * 4 - 2 * 4
            fd_off = (i - 1) * 2 * 4
            bk_off = ((i - 1) * 2 + 1) * 4
            if i == bin_index:
                bins_block[fd_off : fd_off + 4] = int(chunk_addr).to_bytes(4, "little")
                bins_block[bk_off : bk_off + 4] = int(chunk_addr).to_bytes(4, "little")
            else:
                bins_block[fd_off : fd_off + 4] = int(head_i).to_bytes(4, "little")
                bins_block[bk_off : bk_off + 4] = int(head_i).to_bytes(4, "little")
        gdb.place(arena32_addr + 60, bytes(bins_block))

        # 32-bit chunk: 8-byte header + fd + bk = 16 bytes total.
        chunk_buf = bytearray(16)
        chunk_buf[0:4] = (0).to_bytes(4, "little")  # prev_size
        chunk_buf[4:8] = (0x21).to_bytes(4, "little")  # size=0x20 | PREV_INUSE
        chunk_buf[8:12] = int(bin_head_addr_32).to_bytes(4, "little")  # fd
        chunk_buf[12:16] = int(bin_head_addr_32).to_bytes(4, "little")  # bk
        gdb.place(chunk_addr, bytes(chunk_buf))

        result = bins_.inspect_bins()
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["target_profile"]["pointer_bits"] == 32
        unsorted = parsed["bins"][0]
        assert unsorted["category"] == "unsorted"
        # Critical assertion: on 32-bit, bin_head_addr = arena+bins_offset-8,
        # not -16.  A hardcoded 16 breaks this.
        assert unsorted["bin_head_address"] == bin_head_addr_32
        assert unsorted["fd_head"] == chunk_addr
        assert unsorted["length"] == 1
        entry = unsorted["chain"][0]
        assert entry["chunk_address"] == chunk_addr
        assert entry["size"] == 0x20
        assert entry["fd"] == bin_head_addr_32
        assert entry["bk"] == bin_head_addr_32


# ---------------------------------------------------------------------------
# HeapTools facade proxy
# ---------------------------------------------------------------------------
class TestHeapToolsFacadeBins:
    def test_facade_routes_inspect_bins(self):
        gdb = ArenaFakeGdb(live=True)
        inspection = ArenaFakeInspection(
            gdb,
            mappings=_extend_maps_with_heap(),
            libc=_libc_version("2.35"),
        )
        debugger = ArenaFakeDebugger()
        facade = HeapTools.__new__(HeapTools)
        facade.discovery = AllocatorDiscovery(
            gdb,
            inspection,
            debugger,
            "sess",
            binary_path_provider=lambda: "/bin/chal",
        )
        facade.glibc = GlibcArenas(
            gdb, inspection, debugger, "sess", discovery=facade.discovery
        )
        facade.arenas = GlibcArenas(
            gdb, inspection, debugger, "sess", discovery=facade.discovery
        )
        facade.bins = GlibcBins(
            gdb,
            inspection,
            debugger,
            "sess",
            discovery=facade.discovery,
            arenas=facade.arenas,
        )
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        result = facade.inspect_bins()
        assert result["parsed"]["verdict"] == "parsed"
        # All bins should be empty.
        assert all(slot["empty"] for slot in result["parsed"]["bins"])
