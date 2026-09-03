"""Tests for typed glibc arena discovery and malloc_state parsing."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    ARENA_INSPECTION_SCHEMA,
    ARENA_LIST_SCHEMA,
    BINMAPSIZE,
    BINS_ARRAY_LEN,
    MAX_ARENAS_TRAVERSAL,
    NFASTBINS,
    AllocatorDiscovery,
    ArenaLayoutResolver,
    GlibcArenas,
    HeapTools,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
LIBC_START = 0x7FFFF7C00000
LIBC_END = 0x7FFFF7E00000
MAIN_ARENA_ADDR = 0x7FFFF7C50000  # inside libc rw segment
ARENA2_START = 0x7FFFF6800000
ARENA2_END = 0x7FFFF6900000
ARENA2_ADDR = ARENA2_START + 0x10  # inside the arena2 mapping


def mapping(
    file: str,
    start: str,
    end: str,
    writable: bool = True,
) -> Dict[str, Any]:
    return {
        "file": file,
        "start": start,
        "end": end,
        "permissions": "rw-p" if writable else "r-xp",
        "readable": True,
        "writable": writable,
        "executable": not writable,
    }


GLIBC_MAPS = [
    mapping("/bin/chal", "0x555555554000", "0x555555556000", writable=False),
    mapping(
        "/lib/x86_64-linux-gnu/libc.so.6",
        f"0x{LIBC_START:x}",
        f"0x{LIBC_END:x}",
    ),
    mapping(
        "[anon:arena2]",
        f"0x{ARENA2_START:x}",
        f"0x{ARENA2_END:x}",
    ),
]

MUSL_MAPS = [
    mapping(
        "/lib/ld-musl-x86_64.so.1",
        "0x7f0000000000",
        "0x7f0000200000",
    ),
]

# Reference glibc 2.35 x86_64 malloc_state layout.  The exact offsets differ
# between glibc versions but the resolver reads them from debug info, so tests
# only need a consistent shape.
STANDARD_OFFSETS_X86_64: Dict[str, int] = {
    "mutex": 0,
    "flags": 4,
    "have_fastchunks": 8,
    "fastbinsY": 16,
    "top": 96,
    "last_remainder": 104,
    "bins": 112,
    "binmap": 2144,
    "next": 2160,
    "next_free": 2168,
    "attached_threads": 2176,
    "system_mem": 2184,
    "max_system_mem": 2192,
}
STANDARD_STRUCT_SIZE_X86_64 = 2200

_FIELD_KIND_COUNT: Dict[str, tuple] = {
    "mutex": ("int32", 1),
    "flags": ("int32", 1),
    "have_fastchunks": ("int32", 1),
    "fastbinsY": ("ptr_array", NFASTBINS),
    "top": ("ptr", 1),
    "last_remainder": ("ptr", 1),
    "bins": ("ptr_array", BINS_ARRAY_LEN),
    "binmap": ("int32_array", BINMAPSIZE),
    "next": ("ptr", 1),
    "next_free": ("ptr", 1),
    "attached_threads": ("size_t", 1),
    "system_mem": ("size_t", 1),
    "max_system_mem": ("size_t", 1),
}


def make_arena_bytes(
    struct_size: int = STANDARD_STRUCT_SIZE_X86_64,
    byteorder: str = "little",
    pointer_bits: int = 64,
    offsets: Optional[Dict[str, int]] = None,
    **field_values: Any,
) -> bytes:
    """Build a raw malloc_state byte buffer with field values placed at offsets."""
    resolved_offsets = offsets if offsets is not None else STANDARD_OFFSETS_X86_64
    size_sz = pointer_bits // 8
    buf = bytearray(struct_size)
    for name, value in field_values.items():
        offset = resolved_offsets[name]
        kind, _ = _FIELD_KIND_COUNT[name]
        if kind == "int32":
            buf[offset : offset + 4] = int(value).to_bytes(4, byteorder)
        elif kind == "int32_array":
            for i, v in enumerate(value):
                buf[offset + i * 4 : offset + (i + 1) * 4] = int(v).to_bytes(
                    4, byteorder
                )
        elif kind == "ptr":
            buf[offset : offset + size_sz] = int(value).to_bytes(size_sz, byteorder)
        elif kind == "ptr_array":
            for i, v in enumerate(value):
                buf[offset + i * size_sz : offset + (i + 1) * size_sz] = int(
                    v
                ).to_bytes(size_sz, byteorder)
        elif kind == "size_t":
            buf[offset : offset + size_sz] = int(value).to_bytes(size_sz, byteorder)
        else:  # pragma: no cover
            raise AssertionError(f"unhandled kind {kind}")
    return bytes(buf)


class ArenaFakeGdb:
    def __init__(self, *, live: bool = True, epoch: int = 1) -> None:
        self._live = live
        self._epoch = epoch
        self.state = "stopped" if live else "exited"
        self._memory: Dict[int, bytes] = {}
        self.read_calls: List[tuple] = []

    def get_state(self) -> str:
        return self.state

    def get_inferior_pid(self) -> Optional[int]:
        return 4242 if self._live else None

    def get_inferior_epoch(self) -> int:
        return self._epoch

    def get_last_stop_record(self) -> Optional[Dict[str, Any]]:
        if not self._live:
            return None
        return {"is_live": True, "identity": "1:1", "inferior_pid": 4242}

    def execute_command(
        self, command: str, timeout_sec: float = 60.0
    ) -> Dict[str, Any]:
        if command == "show endian":
            return {
                "success": True,
                "state": self.state,
                "responses": [
                    {
                        "type": "console",
                        "payload": "The target is assumed to be little endian\n",
                    }
                ],
            }
        return {"success": True, "state": self.state, "responses": []}

    def place(self, address: int, data: bytes) -> None:
        self._memory[address] = data

    def read_memory_bytes(self, address: str, num_bytes: int) -> Dict[str, Any]:
        self.read_calls.append((address, num_bytes))
        addr_int = int(address, 16)
        # Exact-address hit (fast path used by direct chunk placements).
        data = self._memory.get(addr_int)
        take: Optional[bytes] = None
        if data is not None:
            take = data[:num_bytes]
        else:
            # Range fallback: a read at [addr, addr+num_bytes) that falls
            # entirely inside a placed region succeeds -- matches real gdb
            # semantics and lets tests place one big struct then read at
            # arbitrary field offsets.
            for base, region in self._memory.items():
                if base <= addr_int < base + len(region):
                    offset = addr_int - base
                    end = offset + num_bytes
                    if end <= len(region):
                        take = region[offset:end]
                        break
        if take is None:
            return {"success": False, "state": self.state, "responses": []}
        return {
            "success": True,
            "state": self.state,
            "responses": [
                {
                    "type": "result",
                    "message": "done",
                    "payload": {"memory": [{"begin": address, "contents": take.hex()}]},
                }
            ],
        }


class ArenaFakeInspection:
    timeout_seconds = 60.0

    def __init__(
        self,
        gdb: ArenaFakeGdb,
        *,
        mappings: Optional[List[Dict[str, Any]]] = None,
        libc: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.gdb = gdb
        self._mappings = mappings if mappings is not None else GLIBC_MAPS
        self._libc = libc

    def memory_maps(self) -> Dict[str, Any]:
        return {
            "success": True,
            "parsed": {"count": len(self._mappings), "mappings": self._mappings},
        }

    def libc_info(self) -> Dict[str, Any]:
        if self._libc is None:
            return {"success": False, "parsed": {}}
        return {"success": True, "parsed": self._libc}

    def checksec(self, binary_path: str) -> Dict[str, Any]:
        return {
            "success": True,
            "parsed": {"arch": "amd64-64-little", "file": binary_path},
        }


class ArenaFakeDebugger:
    """A programmable debugger fake for arena tests.

    ``expression_map`` maps exact expression strings to the value GDB would
    return.  Any expression not in the map returns success=False.  By default
    the map is populated with a standard glibc 2.35 x86_64 layout so tests
    don't have to spell out every field offset.
    """

    def __init__(
        self,
        *,
        arena_address: Optional[int] = MAIN_ARENA_ADDR,
        struct_size: Optional[int] = STANDARD_STRUCT_SIZE_X86_64,
        offsets: Optional[Dict[str, Optional[int]]] = None,
        pointer_bytes: int = 8,
        unresolved_fields: Optional[List[str]] = None,
    ) -> None:
        base_offsets: Dict[str, Optional[int]] = dict(STANDARD_OFFSETS_X86_64)
        if offsets is not None:
            base_offsets.update(offsets)
        for name in unresolved_fields or []:
            base_offsets[name] = None
        self.expression_map: Dict[str, Optional[str]] = {
            "sizeof(void*)": str(pointer_bytes) if pointer_bytes else None,
        }
        if arena_address is not None:
            self.expression_map["(void*)&main_arena"] = (
                f"(void *) 0x{arena_address:x} <main_arena>"
            )
        if struct_size is not None:
            self.expression_map["(unsigned long)sizeof(struct malloc_state)"] = str(
                struct_size
            )
        for name, offset in base_offsets.items():
            expr = ArenaLayoutResolver._offset_expression(name)
            if offset is not None:
                self.expression_map[expr] = str(offset)

    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        if expression not in self.expression_map:
            return {
                "success": False,
                "parsed": {"expression": expression, "value": None},
                "error": "expression not supported by fake",
            }
        value = self.expression_map[expression]
        if value is None:
            return {
                "success": False,
                "parsed": {"expression": expression, "value": None},
                "error": "field marked unresolved",
            }
        return {
            "success": True,
            "parsed": {"expression": expression, "value": value},
        }


def build_arenas(
    *,
    live: bool = True,
    mappings: Optional[List[Dict[str, Any]]] = None,
    libc: Optional[Dict[str, Any]] = None,
    debugger: Optional[ArenaFakeDebugger] = None,
):
    gdb = ArenaFakeGdb(live=live)
    resolved_libc = (
        libc
        if libc is not None
        else {"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"}
    )
    inspection = ArenaFakeInspection(
        gdb,
        mappings=mappings,
        libc=None if libc is None and mappings is not None else resolved_libc,
    )
    dbg = debugger if debugger is not None else ArenaFakeDebugger()
    discovery = AllocatorDiscovery(
        gdb, inspection, dbg, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    arenas = GlibcArenas(gdb, inspection, dbg, "sess", discovery=discovery)
    return arenas, gdb


# ---------------------------------------------------------------------------
# Pure parser: parse_arena_struct
# ---------------------------------------------------------------------------
class TestParseArenaStruct:
    def test_full_decode_from_standard_layout(self):
        top = 0x555555559010
        last = 0x555555559C00
        raw = make_arena_bytes(
            mutex=1,
            flags=2,
            have_fastchunks=1,
            fastbinsY=[0] * (NFASTBINS - 1) + [0xDEAD],
            top=top,
            last_remainder=last,
            bins=[0] * BINS_ARRAY_LEN,
            binmap=[0, 1, 0, 0],
            next=MAIN_ARENA_ADDR,
            next_free=0,
            attached_threads=1,
            system_mem=0x21000,
            max_system_mem=0x21000,
        )
        parsed = GlibcArenas.parse_arena_struct(
            raw, STANDARD_OFFSETS_X86_64, 64, "little"
        )
        fields = parsed["fields"]
        assert fields["mutex"] == 1
        assert fields["flags"] == 2
        assert fields["have_fastchunks"] == 1
        assert fields["top"] == top
        assert fields["last_remainder"] == last
        assert fields["next"] == MAIN_ARENA_ADDR
        assert fields["next_free"] == 0
        assert fields["system_mem"] == 0x21000
        assert fields["max_system_mem"] == 0x21000
        assert fields["attached_threads"] == 1
        assert len(fields["fastbinsY"]) == NFASTBINS
        assert fields["fastbinsY"][-1] == 0xDEAD
        assert len(fields["bins"]) == BINS_ARRAY_LEN
        assert fields["binmap"] == [0, 1, 0, 0]
        assert parsed["unreadable_fields"] == []

    def test_missing_offset_marks_field_unreadable(self):
        raw = make_arena_bytes(top=0x1000)
        offsets = dict(STANDARD_OFFSETS_X86_64)
        offsets["top"] = None  # type: ignore[assignment]
        parsed = GlibcArenas.parse_arena_struct(raw, offsets, 64, "little")
        assert parsed["fields"]["top"] is None
        assert "top" in parsed["unreadable_fields"]

    def test_byte_slice_past_end_marks_field_unreadable(self):
        # Truncated buffer -- max_system_mem lives at offset 2192+ which is
        # past our 100-byte buffer.
        raw = b"\x00" * 100
        parsed = GlibcArenas.parse_arena_struct(
            raw, STANDARD_OFFSETS_X86_64, 64, "little"
        )
        assert parsed["fields"]["max_system_mem"] is None
        assert "max_system_mem" in parsed["unreadable_fields"]

    def test_ptr_array_with_one_out_of_bounds_slot_is_none(self):
        # Provide a buffer that stops right in the middle of the bins array.
        raw = b"\x00" * (STANDARD_OFFSETS_X86_64["bins"] + 8)
        parsed = GlibcArenas.parse_arena_struct(
            raw, STANDARD_OFFSETS_X86_64, 64, "little"
        )
        assert parsed["fields"]["bins"] is None
        assert "bins" in parsed["unreadable_fields"]

    def test_big_endian_decode(self):
        raw = make_arena_bytes(top=0x1234, byteorder="big")
        parsed = GlibcArenas.parse_arena_struct(raw, STANDARD_OFFSETS_X86_64, 64, "big")
        assert parsed["fields"]["top"] == 0x1234

    def test_32bit_target(self):
        offsets_32 = {
            "mutex": 0,
            "flags": 4,
            "have_fastchunks": 8,
            "fastbinsY": 12,
            "top": 52,  # 12 + 10*4
            "last_remainder": 56,
            "bins": 60,
            "binmap": 60 + BINS_ARRAY_LEN * 4,
            "next": 60 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4,
            "next_free": 60 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 4,
            "attached_threads": 60 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 8,
            "system_mem": 60 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 12,
            "max_system_mem": 60 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 16,
        }
        struct_size_32 = 60 + BINS_ARRAY_LEN * 4 + BINMAPSIZE * 4 + 20
        raw = make_arena_bytes(
            struct_size=struct_size_32,
            pointer_bits=32,
            offsets=offsets_32,
            top=0x8049000,
            next=0x8049100,
            system_mem=0x21000,
        )
        parsed = GlibcArenas.parse_arena_struct(raw, offsets_32, 32, "little")
        assert parsed["fields"]["top"] == 0x8049000
        assert parsed["fields"]["next"] == 0x8049100
        assert parsed["fields"]["system_mem"] == 0x21000

    def test_invalid_pointer_bits_raises(self):
        with pytest.raises(ValueError):
            GlibcArenas.parse_arena_struct(b"", {}, 16, "little")

    def test_default_byteorder_is_little_when_none(self):
        raw = make_arena_bytes(top=0x1234)
        parsed = GlibcArenas.parse_arena_struct(raw, STANDARD_OFFSETS_X86_64, 64, None)
        assert parsed["fields"]["top"] == 0x1234


# ---------------------------------------------------------------------------
# ArenaLayoutResolver
# ---------------------------------------------------------------------------
class TestArenaLayoutResolver:
    def test_full_resolution(self):
        resolver = ArenaLayoutResolver(ArenaFakeDebugger())
        layout = resolver.resolve()
        assert layout["arena_address"] == MAIN_ARENA_ADDR
        assert layout["struct_size"] == STANDARD_STRUCT_SIZE_X86_64
        assert layout["offsets"]["top"] == 96
        assert layout["offsets"]["next"] == 2160
        assert layout["unresolved_fields"] == []
        assert layout["resolution_source"] == "gdb_debug_symbols"

    def test_partial_resolution_lists_unresolved(self):
        resolver = ArenaLayoutResolver(
            ArenaFakeDebugger(unresolved_fields=["fastbinsY", "binmap"])
        )
        layout = resolver.resolve()
        assert layout["offsets"]["fastbinsY"] is None
        assert layout["offsets"]["binmap"] is None
        assert set(layout["unresolved_fields"]) == {"fastbinsY", "binmap"}

    def test_no_debug_info_yields_all_unresolved(self):
        resolver = ArenaLayoutResolver(
            ArenaFakeDebugger(
                arena_address=None,
                struct_size=None,
                unresolved_fields=list(ArenaLayoutResolver.FIELD_NAMES),
            )
        )
        layout = resolver.resolve()
        assert layout["arena_address"] is None
        assert layout["struct_size"] is None
        assert set(layout["unresolved_fields"]) == set(ArenaLayoutResolver.FIELD_NAMES)


# ---------------------------------------------------------------------------
# list_arenas orchestration
# ---------------------------------------------------------------------------
class TestListArenas:
    def test_requires_live_inferior(self):
        arenas, _ = build_arenas(live=False)
        with pytest.raises(LiveInferiorError):
            arenas.list_arenas()

    def test_unsupported_when_target_is_musl(self):
        arenas, _ = build_arenas(mappings=MUSL_MAPS, libc=None)
        result = arenas.list_arenas()
        parsed = result["parsed"]
        assert parsed["schema"] == ARENA_LIST_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"

    def test_unresolvable_when_main_arena_symbol_missing(self):
        debugger = ArenaFakeDebugger(
            arena_address=None,
            unresolved_fields=list(ArenaLayoutResolver.FIELD_NAMES),
        )
        arenas, _ = build_arenas(debugger=debugger)
        result = arenas.list_arenas()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unresolvable"
        assert parsed["arenas"] == []
        assert parsed["count"] == 0

    def test_single_main_arena_chain(self):
        # main_arena.next points back to itself -> ring of one.
        arenas, gdb = build_arenas()
        next_offset = STANDARD_OFFSETS_X86_64["next"]
        gdb.place(
            MAIN_ARENA_ADDR + next_offset,
            MAIN_ARENA_ADDR.to_bytes(8, "little"),
        )
        result = arenas.list_arenas()
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["count"] == 1
        entry = parsed["arenas"][0]
        assert entry["address"] == MAIN_ARENA_ADDR
        assert entry["is_main"] is True
        assert entry["in_mapping"] is True
        assert entry["next"] == MAIN_ARENA_ADDR
        assert parsed["traversal"]["truncated"] is False
        assert parsed["traversal"]["cycle_detected"] is False

    def test_multi_arena_chain(self):
        arenas, gdb = build_arenas()
        next_offset = STANDARD_OFFSETS_X86_64["next"]
        # main -> arena2 -> main
        gdb.place(MAIN_ARENA_ADDR + next_offset, ARENA2_ADDR.to_bytes(8, "little"))
        gdb.place(ARENA2_ADDR + next_offset, MAIN_ARENA_ADDR.to_bytes(8, "little"))
        result = arenas.list_arenas()
        parsed = result["parsed"]
        assert parsed["count"] == 2
        addrs = [a["address"] for a in parsed["arenas"]]
        assert addrs == [MAIN_ARENA_ADDR, ARENA2_ADDR]
        assert parsed["arenas"][0]["is_main"] is True
        assert parsed["arenas"][1]["is_main"] is False

    def test_null_terminator_stops_chain(self):
        arenas, gdb = build_arenas()
        next_offset = STANDARD_OFFSETS_X86_64["next"]
        # main.next = 0
        gdb.place(MAIN_ARENA_ADDR + next_offset, (0).to_bytes(8, "little"))
        result = arenas.list_arenas()
        parsed = result["parsed"]
        assert parsed["count"] == 1
        assert parsed["arenas"][0]["next"] == 0

    def test_cycle_detection_when_non_main_repeats(self):
        arenas, gdb = build_arenas()
        next_offset = STANDARD_OFFSETS_X86_64["next"]
        arena3_addr = ARENA2_ADDR + 0x1000
        # main -> arena2 -> arena3 -> arena2 (cycle back to non-main)
        gdb.place(MAIN_ARENA_ADDR + next_offset, ARENA2_ADDR.to_bytes(8, "little"))
        gdb.place(ARENA2_ADDR + next_offset, arena3_addr.to_bytes(8, "little"))
        gdb.place(arena3_addr + next_offset, ARENA2_ADDR.to_bytes(8, "little"))
        result = arenas.list_arenas()
        parsed = result["parsed"]
        assert parsed["traversal"]["cycle_detected"] is True
        assert parsed["count"] == 3

    def test_read_error_records_read_error_field(self):
        arenas, gdb = build_arenas()
        # Don't place anything -> read fails at main_arena+next_offset
        result = arenas.list_arenas()
        parsed = result["parsed"]
        assert parsed["count"] == 1
        entry = parsed["arenas"][0]
        assert entry["next"] is None
        assert entry["read_error"]

    def test_traversal_cap_marks_truncated(self, monkeypatch):
        # Shrink the traversal cap so the test stays fast, then build an
        # arena chain that never returns to main.
        monkeypatch.setattr(
            "paomcp.tools.backends.heap_glibc_arenas.MAX_ARENAS_TRAVERSAL", 4
        )
        arenas, gdb = build_arenas()
        next_offset = STANDARD_OFFSETS_X86_64["next"]
        # main -> 0x111 -> 0x222 -> 0x333 -> 0x444 -> 0x555 (never returns)
        chain = [MAIN_ARENA_ADDR, 0x111, 0x222, 0x333, 0x444, 0x555]
        for src, dst in zip(chain, chain[1:]):
            gdb.place(src + next_offset, dst.to_bytes(8, "little"))
        result = arenas.list_arenas()
        parsed = result["parsed"]
        assert parsed["traversal"]["truncated"] is True
        assert parsed["count"] == 4

    def test_unresolved_byteorder_emits_warning(self):
        # Regression: silently defaulting byteorder to little on a target
        # whose endian could not be resolved would decode .next pointers
        # wrong on any big-endian glibc without any diagnostic to the caller.
        # inspect_arena / inspect_chunk both warn here; list_arenas must too.
        arenas, gdb = build_arenas()

        # Force `show endian` output that _endianness_from_show cannot parse.
        def _no_endian(command, timeout_sec=60.0):
            return {"success": True, "state": gdb.state, "responses": []}

        gdb.execute_command = _no_endian  # type: ignore[method-assign]
        next_offset = STANDARD_OFFSETS_X86_64["next"]
        gdb.place(
            MAIN_ARENA_ADDR + next_offset,
            MAIN_ARENA_ADDR.to_bytes(8, "little"),
        )
        result = arenas.list_arenas()
        warnings = result.get("warnings", [])
        assert any(
            "byteorder could not be resolved" in w for w in warnings
        ), f"expected byteorder warning in {warnings!r}"


# ---------------------------------------------------------------------------
# inspect_arena orchestration
# ---------------------------------------------------------------------------
class TestInspectArena:
    def test_requires_live_inferior(self):
        arenas, _ = build_arenas(live=False)
        with pytest.raises(LiveInferiorError):
            arenas.inspect_arena()

    def test_unsupported_when_target_is_musl(self):
        arenas, _ = build_arenas(mappings=MUSL_MAPS, libc=None)
        result = arenas.inspect_arena()
        parsed = result["parsed"]
        assert parsed["schema"] == ARENA_INSPECTION_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"

    def test_unresolvable_when_struct_size_missing(self):
        debugger = ArenaFakeDebugger(struct_size=None)
        arenas, _ = build_arenas(debugger=debugger)
        result = arenas.inspect_arena()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unresolvable"

    def test_invalid_arena_address_raises(self):
        arenas, _ = build_arenas()
        with pytest.raises(ValueError):
            arenas.inspect_arena(arena_address="not-a-number")

    def test_default_inspects_main_arena_with_full_decode(self):
        arenas, gdb = build_arenas()
        gdb.place(
            MAIN_ARENA_ADDR,
            make_arena_bytes(
                mutex=0,
                flags=0,
                have_fastchunks=1,
                fastbinsY=[0] * NFASTBINS,
                top=0x555555559010,
                last_remainder=0,
                bins=[0] * BINS_ARRAY_LEN,
                binmap=[0, 0, 0, 0],
                next=MAIN_ARENA_ADDR,
                next_free=0,
                attached_threads=1,
                system_mem=0x21000,
                max_system_mem=0x21000,
            ),
        )
        result = arenas.inspect_arena()
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["arena_address"] == MAIN_ARENA_ADDR
        assert parsed["is_main"] is True
        assert parsed["fields"]["top"] == 0x555555559010
        assert parsed["fields"]["system_mem"] == 0x21000
        assert parsed["fields"]["next"] == MAIN_ARENA_ADDR
        assert parsed["unreadable_fields"] == []
        # Full raw hex evidence carried in output.
        assert len(parsed["raw_hex"]) == STANDARD_STRUCT_SIZE_X86_64 * 2
        # Positive-path mapping assertion: main_arena lives inside the libc
        # rw segment, so the mapping field must be populated (not just None).
        assert parsed["mapping"] is not None
        assert parsed["mapping"]["file"] == "/lib/x86_64-linux-gnu/libc.so.6"
        assert int(parsed["mapping"]["start"], 16) <= MAIN_ARENA_ADDR
        assert int(parsed["mapping"]["end"], 16) > MAIN_ARENA_ADDR

    def test_explicit_arena_address_parses_that_arena(self):
        arenas, gdb = build_arenas()
        # main_arena needs to exist so no read error there
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(top=0))
        gdb.place(
            ARENA2_ADDR,
            make_arena_bytes(top=0x1234, next=MAIN_ARENA_ADDR, system_mem=0x40000),
        )
        result = arenas.inspect_arena(arena_address=f"0x{ARENA2_ADDR:x}")
        parsed = result["parsed"]
        assert parsed["arena_address"] == ARENA2_ADDR
        assert parsed["is_main"] is False
        assert parsed["fields"]["top"] == 0x1234
        assert parsed["fields"]["system_mem"] == 0x40000

    def test_short_read_reports_unreadable_verdict(self):
        arenas, gdb = build_arenas()
        # Place fewer bytes than struct_size
        gdb.place(MAIN_ARENA_ADDR, b"\x00" * 100)
        result = arenas.inspect_arena()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["fields"] is None

    def test_read_failure_reports_unreadable(self):
        arenas, _ = build_arenas()
        # Nothing placed -> read returns success=False
        result = arenas.inspect_arena()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"

    def test_arena_outside_mapping_still_parses_with_warning(self):
        arenas, gdb = build_arenas()
        stray_addr = 0xDEADBEEF000
        gdb.place(stray_addr, make_arena_bytes(top=0x1000))
        result = arenas.inspect_arena(arena_address=f"0x{stray_addr:x}")
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["mapping"] is None
        assert any(
            "outside every known memory mapping" in w
            for w in result.get("warnings", [])
        )


# ---------------------------------------------------------------------------
# HeapTools facade proxy
# ---------------------------------------------------------------------------
class TestHeapToolsFacade:
    def _make(self):
        gdb = ArenaFakeGdb(live=True)
        inspection = ArenaFakeInspection(
            gdb,
            libc={"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"},
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
        )  # dummy; only .arenas is exercised here
        facade.arenas = GlibcArenas(
            gdb, inspection, debugger, "sess", discovery=facade.discovery
        )
        return facade, gdb

    def test_facade_routes_list_arenas(self):
        facade, gdb = self._make()
        next_offset = STANDARD_OFFSETS_X86_64["next"]
        gdb.place(
            MAIN_ARENA_ADDR + next_offset,
            MAIN_ARENA_ADDR.to_bytes(8, "little"),
        )
        result = facade.list_arenas()
        assert result["parsed"]["count"] == 1

    def test_facade_routes_inspect_arena(self):
        facade, gdb = self._make()
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(top=0x1234))
        result = facade.inspect_arena()
        assert result["parsed"]["fields"]["top"] == 0x1234


class TestArenaLayoutCache:
    def _ring(self, gdb):
        next_offset = STANDARD_OFFSETS_X86_64["next"]
        gdb.place(
            MAIN_ARENA_ADDR + next_offset,
            MAIN_ARENA_ADDR.to_bytes(8, "little"),
        )

    def test_second_list_reuses_cached_offsets(self):
        dbg = ArenaFakeDebugger()
        arenas, gdb = build_arenas(debugger=dbg)
        self._ring(gdb)
        first = arenas.list_arenas()
        assert first["parsed"]["verdict"] == "parsed"
        assert arenas._layout_cache is not None
        # Break every field-offset expression: a fresh resolve() is now
        # impossible, so a second parsed verdict proves the cache hit.
        for key in [k for k in dbg.expression_map if "main_arena." in k]:
            del dbg.expression_map[key]
        second = arenas.list_arenas()
        assert second["parsed"]["verdict"] == "parsed"
        assert second["parsed"]["count"] == 1

    def test_libc_change_forces_reresolve(self):
        dbg = ArenaFakeDebugger()
        arenas, gdb = build_arenas(debugger=dbg)
        self._ring(gdb)
        assert arenas.list_arenas()["parsed"]["verdict"] == "parsed"
        # Same version, different libc file: the cache key differs, so the
        # layout is resolved again instead of reused.
        alt_maps = [
            (
                dict(m, file="/opt/custom/libc.so.6")
                if m.get("file") == "/lib/x86_64-linux-gnu/libc.so.6"
                else m
            )
            for m in GLIBC_MAPS
        ]
        arenas.inspection._mappings = alt_maps
        for key in [k for k in dbg.expression_map if "main_arena." in k]:
            del dbg.expression_map[key]
        result = arenas.list_arenas()
        assert result["parsed"]["verdict"] == "unresolvable"

    def test_cache_key_needs_glibc_mapping(self):
        from paomcp.tools.backends.heap_glibc_arenas import arena_layout_cache_key

        profile = {"pointer_bits": 64}
        classification = {
            "family": "glibc",
            "version": "2.35",
            "evidence": [
                {"source": "mapping", "family": "glibc", "file": "/lib/libc.so.6"},
            ],
        }
        assert arena_layout_cache_key(classification, profile) == (
            "glibc",
            "2.35",
            "/lib/libc.so.6",
            64,
        )
        assert (
            arena_layout_cache_key({**classification, "family": "musl"}, profile)
            is None
        )
        assert (
            arena_layout_cache_key({**classification, "evidence": []}, profile) is None
        )
