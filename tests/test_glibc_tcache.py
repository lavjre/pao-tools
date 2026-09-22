"""Tests for typed glibc tcache traversal."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
    TCACHE_INSPECTION_SCHEMA,
    TCACHE_MAX_BINS,
    AllocatorDiscovery,
    GlibcArenas,
    GlibcBins,
    HeapTools,
    TcacheLayoutResolver,
)

# Reuse the arena test fakes so both suites share one mock stack.
from tests.test_glibc_arenas import (  # noqa: E402
    GLIBC_MAPS,
    MAIN_ARENA_ADDR,
    MUSL_MAPS,
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
)

HEAP_BASE = 0x555555559000
HEAP_END = 0x55555555A000

# Reference glibc 2.35 x86_64 tcache_perthread_struct layout:
#   uint16_t counts[64]  at offset 0   (128 bytes)
#   tcache_entry* entries[64] at offset 128 (512 bytes)
# Total = 640 bytes.
TCACHE_ADDR = HEAP_BASE + 0x10
TCACHE_STRUCT_SIZE = 640
TCACHE_COUNTS_OFFSET = 0
TCACHE_ENTRIES_OFFSET = 128
TCACHE_COUNTS_ENTRY_SIZE = 2  # uint16_t (glibc >= 2.30)


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


def _header(prev_size: int, size_field: int, byteorder: str = "little") -> bytes:
    return prev_size.to_bytes(8, byteorder) + size_field.to_bytes(8, byteorder)


class TcacheAwareFakeDebugger(ArenaFakeDebugger):
    """ArenaFakeDebugger + entries for tcache expressions.

    Populates the exact GDB expressions TcacheLayoutResolver issues so tests
    can toggle each one to unresolved by passing overrides.
    """

    def __init__(
        self,
        *,
        tcache_address: Optional[int] = TCACHE_ADDR,
        tcache_struct_size: Optional[int] = TCACHE_STRUCT_SIZE,
        counts_offset: Optional[int] = TCACHE_COUNTS_OFFSET,
        entries_offset: Optional[int] = TCACHE_ENTRIES_OFFSET,
        counts_entry_size: Optional[int] = TCACHE_COUNTS_ENTRY_SIZE,
        entries_entry_size: Optional[int] = 8,
        **arena_kwargs: Any,
    ) -> None:
        super().__init__(**arena_kwargs)
        if tcache_address is not None:
            self.expression_map["(void*)tcache"] = f"(void *) 0x{tcache_address:x}"
        if tcache_struct_size is not None:
            self.expression_map[
                "(unsigned long)sizeof(struct tcache_perthread_struct)"
            ] = str(tcache_struct_size)
        if counts_entry_size is not None:
            self.expression_map[
                "(unsigned long)sizeof(((struct tcache_perthread_struct*)0)->counts[0])"
            ] = str(counts_entry_size)
        if entries_entry_size is not None:
            self.expression_map[
                "(unsigned long)sizeof(((struct tcache_perthread_struct*)0)->entries[0])"
            ] = str(entries_entry_size)
        if counts_offset is not None:
            self.expression_map[TcacheLayoutResolver._offset_expression("counts")] = (
                str(counts_offset)
            )
        if entries_offset is not None:
            self.expression_map[TcacheLayoutResolver._offset_expression("entries")] = (
                str(entries_offset)
            )


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
    dbg = debugger if debugger is not None else TcacheAwareFakeDebugger()
    discovery = AllocatorDiscovery(
        gdb, inspection, dbg, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    arenas = GlibcArenas(gdb, inspection, dbg, "sess", discovery=discovery)
    bins_ = GlibcBins(gdb, inspection, dbg, "sess", discovery=discovery, arenas=arenas)
    return bins_, gdb


def make_tcache_bytes(
    *,
    counts: Optional[List[int]] = None,
    entries: Optional[List[int]] = None,
    byteorder: str = "little",
) -> bytes:
    """Build a raw 640-byte tcache_perthread_struct.

    ``counts`` defaults to all zeros; each entry is a uint16_t so counts[i]
    < 65536.  ``entries`` defaults to all NULL pointers.
    """
    resolved_counts = counts if counts is not None else [0] * TCACHE_MAX_BINS
    resolved_entries = entries if entries is not None else [0] * TCACHE_MAX_BINS
    assert len(resolved_counts) == TCACHE_MAX_BINS
    assert len(resolved_entries) == TCACHE_MAX_BINS
    buf = bytearray(TCACHE_STRUCT_SIZE)
    for i, count in enumerate(resolved_counts):
        offset = TCACHE_COUNTS_OFFSET + i * TCACHE_COUNTS_ENTRY_SIZE
        buf[offset : offset + TCACHE_COUNTS_ENTRY_SIZE] = int(count).to_bytes(
            TCACHE_COUNTS_ENTRY_SIZE, byteorder
        )
    for i, entry in enumerate(resolved_entries):
        offset = TCACHE_ENTRIES_OFFSET + i * 8
        buf[offset : offset + 8] = int(entry).to_bytes(8, byteorder)
    return bytes(buf)


def place_tcache_chain(
    gdb: ArenaFakeGdb,
    user_addresses: List[int],
    *,
    chunk_size_field: int = 0x21,
    safe_linking: bool = False,
) -> None:
    """Write a tcache chain: each entry's ``next`` points to the next user
    address (or NULL for the last).  Each entry's containing chunk gets a
    16-byte header at user - 16 so inspect_tcache can decode ``size``.
    """
    for i, user_addr in enumerate(user_addresses):
        chunk_addr = user_addr - 16
        next_user = user_addresses[i + 1] if i + 1 < len(user_addresses) else 0
        next_value = next_user ^ (user_addr >> 12) if safe_linking else next_user
        header = _header(0, chunk_size_field)
        # place chunk header
        gdb.place(chunk_addr, header)
        # place next pointer at user_addr (offset 0 of tcache_entry)
        gdb.place(user_addr, next_value.to_bytes(8, "little"))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestTcacheSize:
    def test_size_for_index_64bit_sequence(self):
        expected = [(2 + i) * 16 for i in range(TCACHE_MAX_BINS)]
        actual = [
            GlibcBins.tcache_size_for_index(i, 64) for i in range(TCACHE_MAX_BINS)
        ]
        assert actual == expected
        # Bin 0 == min chunk == 0x20; bin 63 == 0x410 (max tcache chunk).
        assert actual[0] == 0x20
        assert actual[-1] == 0x410

    def test_size_for_index_32bit_i386_sequence(self):
        # i386 uses MALLOC_ALIGNMENT = 16 since glibc 2.26 (BZ 21120), so
        # the tcache ladder steps by 16: bin 0 = 0x10, bin 63 = 0x400.
        # An unresolved 32-bit profile defaults to the i386 layout.
        actual = [
            GlibcBins.tcache_size_for_index(i, 32) for i in range(TCACHE_MAX_BINS)
        ]
        assert actual[0] == 0x10
        # 16 + 63*16 = 1024 = 0x400 on i386.
        assert actual[-1] == 0x400
        explicit = [
            GlibcBins.tcache_size_for_index(i, 32, "i386-32-little")
            for i in range(TCACHE_MAX_BINS)
        ]
        assert explicit == actual

    def test_size_for_index_arm32_sequence(self):
        # arm32's generic MALLOC_ALIGNMENT is 8 -> 0x10/0x18/.../0x208.
        actual = [
            GlibcBins.tcache_size_for_index(i, 32, "arm-32-little")
            for i in range(TCACHE_MAX_BINS)
        ]
        assert actual[0] == 0x10
        # 16 + 63*8 = 520 = 0x208 on arm32.
        assert actual[-1] == 0x208

    def test_size_rejects_out_of_range_index(self):
        with pytest.raises(ValueError):
            GlibcBins.tcache_size_for_index(TCACHE_MAX_BINS, 64)
        with pytest.raises(ValueError):
            GlibcBins.tcache_size_for_index(-1, 64)

    def test_size_rejects_invalid_pointer_bits(self):
        with pytest.raises(ValueError):
            GlibcBins.tcache_size_for_index(0, 16)


# ---------------------------------------------------------------------------
# TcacheLayoutResolver
# ---------------------------------------------------------------------------
class TestTcacheLayoutResolver:
    def test_full_resolution_returns_all_layout_fields(self):
        resolver = TcacheLayoutResolver(TcacheAwareFakeDebugger())
        layout = resolver.resolve()
        assert layout["tcache_address"] == TCACHE_ADDR
        assert layout["struct_size"] == TCACHE_STRUCT_SIZE
        assert layout["counts_entry_size"] == TCACHE_COUNTS_ENTRY_SIZE
        assert layout["entries_entry_size"] == 8
        assert layout["offsets"]["counts"] == TCACHE_COUNTS_OFFSET
        assert layout["offsets"]["entries"] == TCACHE_ENTRIES_OFFSET
        assert layout["unresolved_fields"] == []
        assert layout["resolution_source"] == "gdb_debug_symbols"

    def test_counts_entry_size_derived_from_offset_gap_when_sizeof_fails(self):
        # sizeof-of-counts-element fails, but the offset gap is intact
        # (entries starts at offset 128, so counts-element = 128/64 = 2).
        dbg = TcacheAwareFakeDebugger(counts_entry_size=None)
        resolver = TcacheLayoutResolver(dbg)
        layout = resolver.resolve()
        assert layout["counts_entry_size"] == 2  # derived from (128-0)/64

    def test_unresolvable_when_offsets_missing(self):
        dbg = TcacheAwareFakeDebugger(counts_offset=None, entries_offset=None)
        resolver = TcacheLayoutResolver(dbg)
        layout = resolver.resolve()
        assert set(layout["unresolved_fields"]) == {"counts", "entries"}
        assert layout["offsets"]["counts"] is None
        assert layout["offsets"]["entries"] is None

    def test_no_debug_info_yields_all_none(self):
        dbg = TcacheAwareFakeDebugger(
            tcache_address=None,
            tcache_struct_size=None,
            counts_offset=None,
            entries_offset=None,
            counts_entry_size=None,
            entries_entry_size=None,
        )
        resolver = TcacheLayoutResolver(dbg)
        layout = resolver.resolve()
        assert layout["tcache_address"] is None
        assert layout["struct_size"] is None
        assert layout["counts_entry_size"] is None
        assert layout["offsets"]["counts"] is None
        assert layout["offsets"]["entries"] is None


# ---------------------------------------------------------------------------
# inspect_tcache orchestration
# ---------------------------------------------------------------------------
class TestInspectTcache:
    def test_requires_live_inferior(self):
        bins_, _ = build_bins(live=False)
        with pytest.raises(LiveInferiorError):
            bins_.inspect_tcache()

    def test_unsupported_when_target_is_musl(self):
        bins_, _ = build_bins(mappings=MUSL_MAPS, libc=None)
        result = bins_.inspect_tcache()
        parsed = result["parsed"]
        assert parsed["schema"] == TCACHE_INSPECTION_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"

    def test_unresolvable_when_offsets_missing(self):
        dbg = TcacheAwareFakeDebugger(counts_offset=None, entries_offset=None)
        bins_, _ = build_bins(debugger=dbg)
        result = bins_.inspect_tcache()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unresolvable"
        assert parsed["bins"] == []

    def test_invalid_max_chunks_raises(self):
        bins_, _ = build_bins()
        with pytest.raises(ValueError):
            bins_.inspect_tcache(max_chunks_per_bin=0)

    def test_uninitialized_tcache_returns_empty_bins(self):
        # tcache_address == 0 -> valid empty state, verdict="uninitialized"
        dbg = TcacheAwareFakeDebugger(tcache_address=0)
        bins_, gdb = build_bins(debugger=dbg)
        result = bins_.inspect_tcache()
        parsed = result["parsed"]
        assert parsed["verdict"] == "uninitialized"
        assert parsed["tcache_address"] == 0
        assert len(parsed["bins"]) == TCACHE_MAX_BINS
        for slot in parsed["bins"]:
            assert slot["count"] == 0
            assert slot["chain"] == []
            assert slot["head"] is None
        # No memory reads performed (tcache is NULL, nothing to read).
        assert len(gdb.read_calls) == 0

    def test_empty_but_initialized_returns_all_zero_bins(self):
        bins_, gdb = build_bins()
        gdb.place(TCACHE_ADDR, make_tcache_bytes())  # all zeros
        result = bins_.inspect_tcache()
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert len(parsed["bins"]) == TCACHE_MAX_BINS
        for slot in parsed["bins"]:
            assert slot["count"] == 0
            assert slot["head"] == 0
            assert slot["chain"] == []
        assert [s["size_class"] for s in parsed["bins"]] == [
            (2 + i) * 16 for i in range(TCACHE_MAX_BINS)
        ]

    def test_single_chunk_chain_no_safe_linking(self):
        bins_, gdb = build_bins(libc=_libc_version("2.31"))
        user_addr = HEAP_BASE + 0x400  # inside heap mapping
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 1
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = user_addr
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        place_tcache_chain(gdb, [user_addr])
        result = bins_.inspect_tcache()
        parsed = result["parsed"]
        assert parsed["safe_linking"]["state"] == "disabled"
        assert parsed["safe_linking"]["applied"] is False
        slot0 = parsed["bins"][0]
        assert slot0["count"] == 1
        assert slot0["head"] == user_addr
        assert slot0["length"] == 1
        entry = slot0["chain"][0]
        assert entry["user_address"] == user_addr
        assert entry["chunk_address"] == user_addr - 16
        assert entry["chunk_readable"] is True
        assert entry["size"] == 0x20
        assert entry["next_raw"] == 0
        assert entry["next_decoded"] == 0
        assert entry["safe_linking_applied"] is False

    def test_multi_chunk_chain_walks_to_null(self):
        bins_, gdb = build_bins(libc=_libc_version("2.31"))
        addresses = [HEAP_BASE + 0x400 + i * 0x40 for i in range(3)]
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 3
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = addresses[0]
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        place_tcache_chain(gdb, addresses)
        result = bins_.inspect_tcache()
        slot0 = result["parsed"]["bins"][0]
        assert slot0["length"] == 3
        user_addrs = [entry["user_address"] for entry in slot0["chain"]]
        assert user_addrs == addresses
        assert slot0["chain"][-1]["next_decoded"] == 0

    def test_safe_linking_decode_walks_chain(self):
        bins_, gdb = build_bins(libc=_libc_version("2.35"))
        addresses = [HEAP_BASE + 0x400 + i * 0x40 for i in range(3)]
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 3
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = addresses[0]
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        place_tcache_chain(gdb, addresses, safe_linking=True)
        result = bins_.inspect_tcache()
        parsed = result["parsed"]
        assert parsed["safe_linking"]["state"] == "enabled"
        assert parsed["safe_linking"]["applied"] is True
        slot0 = parsed["bins"][0]
        assert slot0["length"] == 3
        user_addrs = [entry["user_address"] for entry in slot0["chain"]]
        assert user_addrs == addresses
        # First entry's raw != decoded; decoded matches the real next.
        first = slot0["chain"][0]
        assert first["next_raw"] != addresses[1]
        assert first["next_decoded"] == addresses[1]
        assert first["safe_linking_applied"] is True

    def test_cycle_detection(self):
        bins_, gdb = build_bins(libc=_libc_version("2.31"))
        a = HEAP_BASE + 0x400
        b = HEAP_BASE + 0x500
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 2
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = a
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        # a -> b -> a (2-cycle in the singly-linked next chain)
        gdb.place(a - 16, _header(0, 0x21))
        gdb.place(b - 16, _header(0, 0x21))
        gdb.place(a, b.to_bytes(8, "little"))
        gdb.place(b, a.to_bytes(8, "little"))
        result = bins_.inspect_tcache()
        slot0 = result["parsed"]["bins"][0]
        assert slot0["cycle_detected"] is True
        assert slot0["length"] == 2

    def test_truncation_respects_max_chunks_per_bin(self):
        bins_, gdb = build_bins(libc=_libc_version("2.31"))
        # 5 chunks in a linear chain, cap at 3.
        addresses = [HEAP_BASE + 0x400 + i * 0x40 for i in range(5)]
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 5
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = addresses[0]
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        place_tcache_chain(gdb, addresses)
        result = bins_.inspect_tcache(max_chunks_per_bin=3)
        slot0 = result["parsed"]["bins"][0]
        assert slot0["truncated"] is True
        assert slot0["length"] == 3

    def test_chunk_header_read_error_records_entry_and_continues_chain(self):
        # Header at chunk_address fails to read but the next pointer at
        # user_address succeeds -- the entry records chunk_readable=False
        # yet still follows the chain to completion.
        bins_, gdb = build_bins(libc=_libc_version("2.31"))
        a = HEAP_BASE + 0x400
        b = HEAP_BASE + 0x500
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 2
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = a
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        # No header placement for a (chunk_address = a-16 not placed) so
        # the chunk read fails, but place the next pointer at a itself.
        gdb.place(a, b.to_bytes(8, "little"))
        # b is a complete chunk with next=0.
        place_tcache_chain(gdb, [b])
        result = bins_.inspect_tcache()
        slot0 = result["parsed"]["bins"][0]
        assert slot0["length"] == 2
        first = slot0["chain"][0]
        assert first["chunk_readable"] is False
        assert first["size"] is None
        assert first["next_decoded"] == b

    def test_next_read_error_records_entry_and_stops_walk(self):
        # Chunk header succeeds via range fallback but next pointer fails.
        bins_, gdb = build_bins(libc=_libc_version("2.31"))
        user_addr = HEAP_BASE + 0x400
        chunk_addr = user_addr - 16
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 1
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = user_addr
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        # Place only the header at chunk_addr (16 bytes), no next at user_addr.
        gdb.place(chunk_addr, _header(0, 0x21))
        result = bins_.inspect_tcache()
        slot0 = result["parsed"]["bins"][0]
        assert slot0["length"] == 1
        entry = slot0["chain"][0]
        assert entry["chunk_readable"] is True
        assert entry["size"] == 0x20
        assert entry["next_raw"] is None
        assert entry["next_decoded"] is None
        assert entry["next_read_error"]
        assert entry["safe_linking_applied"] is False

    def test_tcache_struct_read_failure_yields_unreadable_verdict(self):
        # tcache_address resolves but no bytes placed at it -> struct
        # read fails, verdict flips to unreadable.
        bins_, _ = build_bins(libc=_libc_version("2.31"))
        result = bins_.inspect_tcache()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["bins"] == []

    def test_unresolved_byteorder_emits_warning(self):
        # Guards against a silent regression that would decode next pointers
        # with wrong endian on big-endian glibc.
        bins_, gdb = build_bins(libc=_libc_version("2.31"))

        def _no_endian(command, timeout_sec=60.0):
            return {"success": True, "state": gdb.state, "responses": []}

        gdb.execute_command = _no_endian  # type: ignore[method-assign]
        gdb.place(TCACHE_ADDR, make_tcache_bytes())
        result = bins_.inspect_tcache()
        warnings = result.get("warnings", [])
        assert any(
            "byteorder could not be resolved" in w for w in warnings
        ), f"expected byteorder warning in {warnings!r}"

    def test_unknown_safe_linking_emits_warning(self):
        dbg = TcacheAwareFakeDebugger()
        bins_, gdb = build_bins(libc=_libc_version(None), debugger=dbg)
        gdb.place(TCACHE_ADDR, make_tcache_bytes())
        result = bins_.inspect_tcache()
        warnings = result.get("warnings", [])
        assert any("safe-linking applicability is unknown" in w for w in warnings)

    def test_inspect_tcache_32bit_end_to_end(self):
        # Regression: exercise the full inspect_tcache pipeline on a 32-bit
        # target so a future change that hardcoded size_sz=8 (dropping the
        # pointer_bits parameter) would fail rather than silently passing
        # the 64-bit chain-walk tests.  Verifies size_sz=4, header_size=8,
        # entries_entry_size=4 and chunk_address = user_address - 8.
        heap32_base = 0x08049000
        heap32_end = 0x0804A000
        tcache32_addr = heap32_base + 0x10
        # 32-bit layout: uint16_t counts[64] (offset 0, 128 bytes) followed
        # by 4-byte entries[64] (offset 128, 256 bytes).  Total = 384.
        tcache32_struct_size = 384
        tcache32_entries_offset = 128

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
        dbg = TcacheAwareFakeDebugger(
            tcache_address=tcache32_addr,
            tcache_struct_size=tcache32_struct_size,
            counts_offset=0,
            entries_offset=tcache32_entries_offset,
            counts_entry_size=2,
            entries_entry_size=4,
            pointer_bytes=4,
        )
        bins_, gdb = build_bins(
            mappings=maps_32, libc=_libc_version("2.31"), debugger=dbg
        )

        user_addr = heap32_base + 0x400
        chunk_addr = user_addr - 8  # 32-bit: 2 * size_sz

        # Build the 32-bit tcache struct: counts[0]=1, entries[0]=user_addr.
        buf = bytearray(tcache32_struct_size)
        buf[0:2] = (1).to_bytes(2, "little")
        buf[tcache32_entries_offset : tcache32_entries_offset + 4] = (
            user_addr
        ).to_bytes(4, "little")
        gdb.place(tcache32_addr, bytes(buf))
        # 32-bit chunk header (8 bytes): prev_size=0, size_field=0x11 (0x10 |
        # PREV_INUSE).  Min chunk on 32-bit is 4 * SIZE_SZ = 16 = 0x10.
        gdb.place(chunk_addr, (0).to_bytes(4, "little") + (0x11).to_bytes(4, "little"))
        # tcache_entry.next at user_addr (4 bytes, NULL terminator).
        gdb.place(user_addr, (0).to_bytes(4, "little"))

        result = bins_.inspect_tcache()
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["target_profile"]["pointer_bits"] == 32
        # i386 ladder (default for an unresolved 32-bit profile): the last
        # tcache bin is 0x400, not the old 2*SIZE_SZ-sized 0x208.
        assert parsed["bins"][0]["size_class"] == 0x10
        assert parsed["bins"][-1]["size_class"] == 0x400
        slot0 = parsed["bins"][0]
        assert slot0["count"] == 1
        assert slot0["head"] == user_addr
        assert slot0["length"] == 1
        entry = slot0["chain"][0]
        assert entry["user_address"] == user_addr
        # The critical assertion: on 32-bit, chunk_address = user - 2*size_sz
        # = user - 8.  A hardcoded 16 would break here.
        assert entry["chunk_address"] == user_addr - 8
        assert entry["chunk_readable"] is True
        assert entry["size"] == 0x10
        assert entry["next_raw"] == 0
        assert entry["next_decoded"] == 0

    def test_raises_when_pointer_bits_unresolved(self):
        # Force sizeof(void*) to fail AND strip 'end' from mappings so
        # _pointer_bits_from_mappings returns None.  Guard must fire.
        maps_no_end = [
            {k: v for k, v in m.items() if k != "end"} for m in _extend_maps_with_heap()
        ]
        dbg = TcacheAwareFakeDebugger(pointer_bytes=0)
        bins_, _ = build_bins(mappings=maps_no_end, debugger=dbg)
        with pytest.raises(RuntimeError, match="pointer width"):
            bins_.inspect_tcache()


# ---------------------------------------------------------------------------
# HeapTools facade proxy
# ---------------------------------------------------------------------------
class TestHeapToolsFacadeTcache:
    def test_facade_routes_inspect_tcache(self):
        gdb = ArenaFakeGdb(live=True)
        inspection = ArenaFakeInspection(
            gdb,
            mappings=_extend_maps_with_heap(),
            libc=_libc_version("2.31"),
        )
        debugger = TcacheAwareFakeDebugger()
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
        user_addr = HEAP_BASE + 0x400
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 1
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = user_addr
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        place_tcache_chain(gdb, [user_addr])
        result = facade.inspect_tcache()
        assert result["parsed"]["verdict"] == "parsed"
        assert result["parsed"]["bins"][0]["length"] == 1


class TestTcacheLayoutCache:
    def _one(self, dbg):
        bins_, gdb = build_bins(debugger=dbg, libc=_libc_version("2.31"))
        user_addr = HEAP_BASE + 0x400
        counts = [0] * TCACHE_MAX_BINS
        counts[0] = 1
        entries = [0] * TCACHE_MAX_BINS
        entries[0] = user_addr
        gdb.place(TCACHE_ADDR, make_tcache_bytes(counts=counts, entries=entries))
        place_tcache_chain(gdb, [user_addr])
        return bins_

    def test_second_inspect_reuses_cached_layout(self):
        dbg = TcacheAwareFakeDebugger()
        bins_ = self._one(dbg)
        assert bins_.inspect_tcache()["parsed"]["verdict"] == "parsed"
        assert bins_._tcache_layout_cache is not None
        # Break every layout expression except the per-thread address: a
        # fresh resolve() is now impossible, so parsed output proves the hit.
        for key in [k for k in dbg.expression_map if k != "(void*)tcache"]:
            if "tcache" in key or "main_arena" in key or "sizeof" in key:
                del dbg.expression_map[key]
        result = bins_.inspect_tcache()
        assert result["parsed"]["verdict"] == "parsed"
        assert result["parsed"]["bins"][0]["length"] == 1
