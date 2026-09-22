"""Tests for typed glibc fastbin traversal."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    BINMAPSIZE,
    BINS_ARRAY_LEN,
    DEFAULT_MAX_CHUNKS_PER_FASTBIN,
    FASTBIN_INSPECTION_SCHEMA,
    NFASTBINS,
    PREV_INUSE,
    AllocatorDiscovery,
    ArenaLayoutResolver,
    GlibcArenas,
    GlibcBins,
    HeapTools,
)

# Reuse the reference layout / fakes from the arena test module so the same
# glibc 2.35 x86_64 numbers govern both suites.
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
    """GLIBC_MAPS + a [heap] mapping so fastbin chunks land in a real range."""
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


def _header(prev_size: int, size_field: int, byteorder: str = "little") -> bytes:
    return prev_size.to_bytes(8, byteorder) + size_field.to_bytes(8, byteorder)


def build_bins(
    *,
    live: bool = True,
    mappings: Optional[List[Dict[str, Any]]] = None,
    libc: Optional[Dict[str, Any]] = None,
    debugger: Optional[ArenaFakeDebugger] = None,
    arch: str = "amd64-64-little",
):
    gdb = ArenaFakeGdb(live=live)
    resolved_libc = libc if libc is not None else _libc_version("2.35")
    resolved_maps = mappings if mappings is not None else _extend_maps_with_heap()
    inspection = ArenaFakeInspection(
        gdb,
        mappings=resolved_maps,
        libc=None if libc is None and mappings is not None else resolved_libc,
        arch=arch,
    )
    dbg = debugger if debugger is not None else ArenaFakeDebugger()
    discovery = AllocatorDiscovery(
        gdb, inspection, dbg, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    arenas = GlibcArenas(gdb, inspection, dbg, "sess", discovery=discovery)
    bins_ = GlibcBins(gdb, inspection, dbg, "sess", discovery=discovery, arenas=arenas)
    return bins_, gdb, arenas


def place_arena_with_fastbin_head(
    gdb: ArenaFakeGdb,
    arena_address: int,
    slot_index: int,
    head_ptr: int,
) -> None:
    """Place a minimal arena at ``arena_address`` with one fastbinsY head set."""
    heads = [0] * NFASTBINS
    heads[slot_index] = head_ptr
    gdb.place(
        arena_address,
        make_arena_bytes(fastbinsY=heads),
    )


def place_fastbin_chain(
    gdb: ArenaFakeGdb,
    addresses: List[int],
    *,
    size_field: int = 0x21,
    safe_linking: bool = False,
) -> None:
    """Write fastbin chunks at each address with fd pointing to the next.

    The last chunk's fd is NULL (chain terminator).  When ``safe_linking``
    is true fd is stored as ``next XOR (fd_position >> 12)`` matching
    glibc's PROTECT_PTR.
    """
    for i, addr in enumerate(addresses):
        header = _header(0, size_field)
        fd_position = addr + 16
        next_addr = addresses[i + 1] if i + 1 < len(addresses) else 0
        if safe_linking:
            fd_value = next_addr ^ (fd_position >> 12)
        else:
            fd_value = next_addr
        payload = bytearray(24)  # header (16) + fd (8)
        payload[0:16] = header
        payload[16:24] = fd_value.to_bytes(8, "little")
        gdb.place(addr, bytes(payload))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_safe_linked_roundtrip(self):
        # PROTECT_PTR is its own inverse.
        for pos in (0x555555559010, 0x7FFFF7B00000, 0x40000000):
            for ptr in (0x0, 0x1, 0xDEADBEEFCAFE, 0x555555559020):
                encoded = ptr ^ (pos >> 12)
                assert GlibcBins.decode_safe_linked(encoded, pos) == ptr

    def test_fastbin_size_64bit_sequence(self):
        # 64-bit MALLOC_ALIGNMENT = 16 -> sizes 0x20, 0x30, ..., 0xb0.
        expected = [0x20 + i * 0x10 for i in range(NFASTBINS)]
        actual = [GlibcBins.fastbin_size_for_index(i, 64) for i in range(NFASTBINS)]
        assert actual == expected

    def test_fastbin_size_32bit_i386_sequence(self):
        # i386 MALLOC_ALIGNMENT = 16 since glibc 2.26 (BZ 21120) makes
        # request2size(MAX_FAST_SIZE=80) round to 96, so NFASTBINS is 11.
        # The index step itself stays SIZE_SZ-based in glibc, so the ladder
        # is 0x10/0x18/.../0x60 -- the extra slot is the 0x60 fastbin that
        # a 10-slot decoder misses.  Unresolved 32-bit defaults to i386.
        i386 = [GlibcBins.fastbin_size_for_index(i, 32) for i in range(11)]
        assert i386 == [0x10 + i * 0x08 for i in range(11)]
        assert i386[-1] == 0x60
        explicit = [
            GlibcBins.fastbin_size_for_index(i, 32, "i386-32-little") for i in range(11)
        ]
        assert explicit == i386
        with pytest.raises(ValueError):
            GlibcBins.fastbin_size_for_index(11, 32)

    def test_fastbin_size_arm32_sequence(self):
        # arm32 MALLOC_ALIGNMENT = 8 -> NFASTBINS = 10, sizes 0x10..0x58.
        expected = [0x10 + i * 0x08 for i in range(NFASTBINS)]
        actual = [
            GlibcBins.fastbin_size_for_index(i, 32, "arm-32-little")
            for i in range(NFASTBINS)
        ]
        assert actual == expected
        with pytest.raises(ValueError):
            GlibcBins.fastbin_size_for_index(NFASTBINS, 32, "arm-32-little")

    def test_fastbin_size_rejects_invalid_pointer_bits(self):
        with pytest.raises(ValueError):
            GlibcBins.fastbin_size_for_index(0, 16)

    def test_fastbin_size_rejects_out_of_range_index(self):
        with pytest.raises(ValueError):
            GlibcBins.fastbin_size_for_index(NFASTBINS, 64)
        with pytest.raises(ValueError):
            GlibcBins.fastbin_size_for_index(-1, 64)


# ---------------------------------------------------------------------------
# inspect_fastbins orchestration
# ---------------------------------------------------------------------------
class TestInspectFastbins:
    def test_requires_live_inferior(self):
        bins_, _, _ = build_bins(live=False)
        with pytest.raises(LiveInferiorError):
            bins_.inspect_fastbins()

    def test_unsupported_when_target_is_musl(self):
        bins_, _, _ = build_bins(mappings=MUSL_MAPS, libc=None)
        result = bins_.inspect_fastbins()
        parsed = result["parsed"]
        assert parsed["schema"] == FASTBIN_INSPECTION_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"

    def test_unresolvable_when_fastbins_offset_missing(self):
        debugger = ArenaFakeDebugger(unresolved_fields=["fastbinsY"])
        bins_, _, _ = build_bins(debugger=debugger)
        result = bins_.inspect_fastbins()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unresolvable"

    def test_invalid_arena_address_raises(self):
        bins_, _, _ = build_bins()
        with pytest.raises(ValueError):
            bins_.inspect_fastbins(arena_address="not-a-number")

    def test_invalid_max_chunks_raises(self):
        bins_, _, _ = build_bins()
        with pytest.raises(ValueError):
            bins_.inspect_fastbins(max_chunks_per_bin=0)

    def test_empty_fastbins_returns_all_zero_heads(self):
        bins_, gdb, _ = build_bins()
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes())  # all zeros
        result = bins_.inspect_fastbins()
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["is_main_arena"] is True
        assert len(parsed["fastbins"]) == NFASTBINS
        for slot in parsed["fastbins"]:
            assert slot["head"] == 0
            assert slot["chain"] == []
            assert slot["length"] == 0
        # Every slot carries its expected size class.
        assert [s["size_class"] for s in parsed["fastbins"]] == [
            0x20 + i * 0x10 for i in range(NFASTBINS)
        ]

    def test_single_chunk_chain_no_safe_linking(self):
        # glibc 2.31 -> safe_linking disabled -> fd_decoded == fd_raw.
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        chunk = HEAP_BASE
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, chunk)
        place_fastbin_chain(gdb, [chunk])
        result = bins_.inspect_fastbins()
        parsed = result["parsed"]
        assert parsed["safe_linking"]["state"] == "disabled"
        assert parsed["safe_linking"]["applied"] is False
        slot0 = parsed["fastbins"][0]
        assert slot0["head"] == chunk
        assert slot0["length"] == 1
        entry = slot0["chain"][0]
        assert entry["chunk_address"] == chunk
        assert entry["size"] == 0x20
        assert entry["prev_inuse"] is True
        assert entry["fd_raw"] == 0
        assert entry["fd_decoded"] == 0
        assert entry["safe_linking_applied"] is False
        assert entry["in_mapping"] is True

    def test_multi_chunk_chain_walks_to_null(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        addresses = [HEAP_BASE, HEAP_BASE + 0x40, HEAP_BASE + 0x80]
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, addresses[0])
        place_fastbin_chain(gdb, addresses)
        result = bins_.inspect_fastbins()
        slot0 = result["parsed"]["fastbins"][0]
        assert slot0["length"] == 3
        chunk_addrs = [entry["chunk_address"] for entry in slot0["chain"]]
        assert chunk_addrs == addresses
        # Last entry's fd is 0 (chain terminator).
        assert slot0["chain"][-1]["fd_decoded"] == 0

    def test_safe_linking_decode_walks_chain(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.35"))
        addresses = [HEAP_BASE, HEAP_BASE + 0x40, HEAP_BASE + 0x80]
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, addresses[0])
        place_fastbin_chain(gdb, addresses, safe_linking=True)
        result = bins_.inspect_fastbins()
        parsed = result["parsed"]
        assert parsed["safe_linking"]["state"] == "enabled"
        assert parsed["safe_linking"]["applied"] is True
        slot0 = parsed["fastbins"][0]
        assert slot0["length"] == 3
        chunk_addrs = [entry["chunk_address"] for entry in slot0["chain"]]
        assert chunk_addrs == addresses
        # fd_raw is encoded, fd_decoded is the real next-address.
        first = slot0["chain"][0]
        assert first["fd_raw"] != addresses[1]
        assert first["fd_decoded"] == addresses[1]
        assert first["safe_linking_applied"] is True

    def test_cycle_detection_marks_flag(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        a = HEAP_BASE
        b = HEAP_BASE + 0x40
        # a -> b -> a  (2-cycle)
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, a)
        # Manually set fd values because place_fastbin_chain assumes a linear chain.
        gdb.place(a, _header(0, 0x21) + b.to_bytes(8, "little"))
        gdb.place(b, _header(0, 0x21) + a.to_bytes(8, "little"))
        result = bins_.inspect_fastbins()
        slot0 = result["parsed"]["fastbins"][0]
        assert slot0["cycle_detected"] is True
        assert slot0["length"] == 2

    def test_truncation_respects_max_chunks_per_bin(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        # 5 chunks in a linear chain, cap at 3.
        addresses = [HEAP_BASE + i * 0x40 for i in range(5)]
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, addresses[0])
        place_fastbin_chain(gdb, addresses)
        result = bins_.inspect_fastbins(max_chunks_per_bin=3)
        slot0 = result["parsed"]["fastbins"][0]
        assert slot0["truncated"] is True
        assert slot0["length"] == 3

    def test_per_chunk_read_error_records_entry_and_stops_walk(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        chunk = HEAP_BASE
        # Head points to an unmapped/unplaced address -> read fails.
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, chunk)
        # Deliberately DO NOT call place_fastbin_chain for chunk
        result = bins_.inspect_fastbins()
        slot0 = result["parsed"]["fastbins"][0]
        assert slot0["length"] == 1
        entry = slot0["chain"][0]
        assert entry["readable"] is False
        assert entry["read_error"]

    def test_i386_end_to_end_walks_11_fastbins(self):
        # Regression: i386 has NFASTBINS = 11 (the extra slot is the 0x60
        # fastbin).  A hardcoded 10-slot fastbinsY read/walk silently drops
        # it.  Exercises the whole inspect_fastbins pipeline on a 32-bit
        # i386 profile.
        heap32_base = 0x0804A000
        heap32_end = 0x0804B000
        arena32 = 0x08049000 + 0x10
        offsets_32 = {
            "mutex": 0,
            "flags": 4,
            "have_fastchunks": 8,
            "fastbinsY": 12,
            "top": 56,  # 12 + 11*4
            "last_remainder": 60,
            "bins": 64,
            "binmap": 64 + BINS_ARRAY_LEN * 4,
            "next": 64 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4,
            "next_free": 64 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 4,
            "attached_threads": 64 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 8,
            "system_mem": 64 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 12,
            "max_system_mem": 64 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 16,
        }
        struct_size_32 = 64 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 20
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
            arena_address=arena32,
            struct_size=struct_size_32,
            offsets=offsets_32,
            pointer_bytes=4,
        )
        bins_, gdb, _ = build_bins(
            mappings=maps_32,
            libc=_libc_version("2.31"),
            debugger=dbg,
            arch="i386-32-little",
        )
        chunk = heap32_base + 0x100
        heads = [0] * 11
        heads[10] = chunk  # the 0x60 fastbin
        gdb.place(
            arena32,
            make_arena_bytes(
                struct_size=struct_size_32,
                pointer_bits=32,
                offsets=offsets_32,
                fastbinsY=heads,
            ),
        )
        # 32-bit chunk: 8-byte header + 4-byte fd (NULL terminator).
        gdb.place(
            chunk,
            (0).to_bytes(4, "little")
            + (0x61).to_bytes(4, "little")
            + (0).to_bytes(4, "little"),
        )
        result = bins_.inspect_fastbins()
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["target_profile"]["pointer_bits"] == 32
        assert len(parsed["fastbins"]) == 11
        assert parsed["layout"]["fastbinsY_count"] == 11
        slot10 = parsed["fastbins"][10]
        assert slot10["size_class"] == 0x60
        assert slot10["head"] == chunk
        assert slot10["length"] == 1
        assert slot10["chain"][0]["size"] == 0x60

    def test_fastbins_read_failure_yields_unreadable_verdict(self):
        # arena address resolves but no bytes placed at the arena location:
        # the fastbinsY block read fails and the top-level verdict flips.
        bins_, _, _ = build_bins(libc=_libc_version("2.31"))
        result = bins_.inspect_fastbins()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert all(slot["read_error"] for slot in parsed["fastbins"])

    def test_head_out_of_mapping_still_records_entry_before_read_failure(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        # Head points to an address in the heap mapping but no chunk placed.
        head = HEAP_BASE + 0x500  # inside [HEAP_BASE, HEAP_END)
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, head)
        result = bins_.inspect_fastbins()
        slot0 = result["parsed"]["fastbins"][0]
        assert slot0["chain"][0]["chunk_address"] == head
        assert slot0["chain"][0]["in_mapping"] is True  # heap mapping contains it
        assert slot0["chain"][0]["readable"] is False

    def test_explicit_arena_address_targets_non_main_arena(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        # Populate main_arena so its own fastbins read cleanly if inspected.
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes())
        # Populate arena2 with a nonzero fastbin head.
        chunk = HEAP_BASE
        place_arena_with_fastbin_head(gdb, ARENA2_ADDR, 3, chunk)
        place_fastbin_chain(gdb, [chunk])
        result = bins_.inspect_fastbins(arena_address=f"0x{ARENA2_ADDR:x}")
        parsed = result["parsed"]
        assert parsed["arena_address"] == ARENA2_ADDR
        assert parsed["is_main_arena"] is False
        assert parsed["fastbins"][3]["head"] == chunk
        assert parsed["fastbins"][3]["length"] == 1

    def test_unknown_safe_linking_emits_warning(self):
        # Version None -> classification.safe_linking = 'unknown'.
        bins_, gdb, _ = build_bins(libc=_libc_version(None))
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes())
        result = bins_.inspect_fastbins()
        warnings = result.get("warnings", [])
        assert any("safe-linking applicability is unknown" in w for w in warnings)

    def test_fd_read_error_records_entry_and_stops_walk(self):
        # Regression: the fd-read-failure branch in _walk_fastbin_chain
        # (header succeeds, fd read fails) must lock in fd_raw=None,
        # fd_decoded=None, a non-empty fd_read_error and
        # safe_linking_applied=False.  Reproduced by placing ONLY the
        # 16-byte header at the chunk with no fd bytes at chunk+16 --
        # the range-fallback on the fake gdb makes the header read succeed
        # while the fd read comes up short.
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        chunk = HEAP_BASE
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, chunk)
        gdb.place(chunk, _header(0, 0x21))  # header only, no fd
        result = bins_.inspect_fastbins()
        slot0 = result["parsed"]["fastbins"][0]
        assert slot0["length"] == 1
        entry = slot0["chain"][0]
        assert entry["readable"] is True
        assert entry["size"] == 0x20
        assert entry["fd_raw"] is None
        assert entry["fd_decoded"] is None
        assert entry["fd_read_error"]
        assert entry["safe_linking_applied"] is False

    def test_unresolved_byteorder_emits_warning(self):
        # Regression: silently defaulting byteorder to little would decode
        # fd pointers wrong on big-endian glibc without any diagnostic.
        # Every heap tool has its own warnings list so the parallel test
        # in tests/test_glibc_arenas.py does not cover this method.
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))

        def _no_endian(command, timeout_sec=60.0):
            return {"success": True, "state": gdb.state, "responses": []}

        gdb.execute_command = _no_endian  # type: ignore[method-assign]
        chunk = HEAP_BASE
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, chunk)
        place_fastbin_chain(gdb, [chunk])
        result = bins_.inspect_fastbins()
        warnings = result.get("warnings", [])
        assert any(
            "byteorder could not be resolved" in w for w in warnings
        ), f"expected byteorder warning in {warnings!r}"

    def test_raises_when_pointer_bits_unresolved(self):
        # Regression: the guard at inspect_fastbins that refuses to walk
        # without a resolved pointer width must fire when sizeof(void*)
        # fails AND mapping-end fallback yields no width.  A future
        # regression that silently defaulted to 64-bit would produce
        # wrong-width reads on 32-bit targets without any diagnostic.
        # Force sizeof(void*) to fail via pointer_bytes=0 and strip 'end'
        # from every mapping so _pointer_bits_from_mappings returns None.
        maps_no_end = [
            {k: v for k, v in m.items() if k != "end"} for m in _extend_maps_with_heap()
        ]
        dbg = ArenaFakeDebugger(pointer_bytes=0)
        bins_, _, _ = build_bins(mappings=maps_no_end, debugger=dbg)
        with pytest.raises(RuntimeError, match="pointer width"):
            bins_.inspect_fastbins()


# ---------------------------------------------------------------------------
# HeapTools facade proxy
# ---------------------------------------------------------------------------
class TestHeapToolsFacadeFastbins:
    def test_facade_routes_inspect_fastbins(self):
        gdb = ArenaFakeGdb(live=True)
        inspection = ArenaFakeInspection(
            gdb,
            mappings=_extend_maps_with_heap(),
            libc=_libc_version("2.31"),
        )
        debugger = ArenaFakeDebugger()
        facade = HeapTools.__new__(HeapTools)
        facade.discovery = AllocatorDiscovery(
            gdb, inspection, debugger, "sess", binary_path_provider=lambda: "/bin/chal"
        )
        facade.glibc = GlibcArenas(  # dummy stand-in; only .bins is exercised
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
        chunk = HEAP_BASE
        place_arena_with_fastbin_head(gdb, MAIN_ARENA_ADDR, 0, chunk)
        place_fastbin_chain(gdb, [chunk])
        result = facade.inspect_fastbins()
        assert result["parsed"]["verdict"] == "parsed"
        assert result["parsed"]["fastbins"][0]["length"] == 1


class TestHeaderAndLinkGulp:
    def test_contiguous_span_costs_one_roundtrip(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        chunk = HEAP_BASE + 0x800
        gdb.place(chunk, b"\x00" * 8 + b"\x21" + b"\x00" * 7 + b"\x88" * 8)
        gdb.read_calls.clear()
        header, header_err, link, link_err = bins_._read_header_and_link(chunk, 16, 8)
        assert header_err is None
        assert link_err is None
        assert len(header) == 16
        assert link == b"\x88" * 8
        assert len(gdb.read_calls) == 1

    def test_split_placement_falls_back_to_two_reads(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        chunk = HEAP_BASE + 0x900
        gdb.place(chunk, b"\x00" * 16)
        gdb.place(chunk + 16, b"\x77" * 8)
        gdb.read_calls.clear()
        header, header_err, link, link_err = bins_._read_header_and_link(chunk, 16, 8)
        assert header_err is None
        assert link_err is None
        assert link == b"\x77" * 8
        # One failed gulp plus the two legacy split reads.
        assert len(gdb.read_calls) == 3

    def test_missing_link_reports_link_error(self):
        bins_, gdb, _ = build_bins(libc=_libc_version("2.31"))
        chunk = HEAP_BASE + 0xA00
        gdb.place(chunk, b"\x00" * 16)
        header, header_err, link, link_err = bins_._read_header_and_link(chunk, 16, 8)
        assert header_err is None
        assert len(header) == 16
        assert link == b""
        assert link_err is not None
