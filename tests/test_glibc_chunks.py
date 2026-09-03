"""Tests for typed glibc heap chunk parsing."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    CHUNK_INSPECTION_SCHEMA,
    CHUNK_VALIDATION_SCHEMA,
    IS_MMAPPED,
    NON_MAIN_ARENA,
    PREV_INUSE,
    AllocatorDiscovery,
    GlibcHeap,
    HeapTools,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def mapping(
    file: str,
    start: str = "0x555555554000",
    end: str = "0x555555556000",
    writable: bool = False,
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


HEAP_BASE = 0x555555559000
HEAP_END = 0x55555555A000

GLIBC_MAPS = [
    mapping("/bin/chal", "0x555555554000", "0x555555556000"),
    mapping("[heap]", f"0x{HEAP_BASE:x}", f"0x{HEAP_END:x}", writable=True),
    mapping(
        "/lib/x86_64-linux-gnu/libc.so.6",
        "0x7ffff7c00000",
        "0x7ffff7e00000",
    ),
]
MUSL_MAPS = [mapping("/lib/ld-musl-x86_64.so.1", "0x7f0000000000", "0x7f0000200000")]


def header_bytes(
    prev_size: int,
    size_field: int,
    pointer_bits: int = 64,
    byteorder: str = "little",
) -> bytes:
    size_sz = pointer_bits // 8
    return prev_size.to_bytes(size_sz, byteorder) + size_field.to_bytes(
        size_sz, byteorder
    )


# ---------------------------------------------------------------------------
# Pure parser: no debugger, exhaustive coverage of the malloc_chunk layout.
# ---------------------------------------------------------------------------
class TestParseChunkHeader:
    def test_allocated_chunk_layout(self):
        raw = header_bytes(prev_size=0, size_field=0x21)  # size=0x20, PREV_INUSE
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["readable"] is True
        fields = parsed["fields"]
        assert fields["size"] == 0x20
        assert fields["size_field"] == 0x21
        assert fields["flags"]["prev_inuse"] is True
        assert fields["flags"]["is_mmapped"] is False
        assert fields["flags"]["non_main_arena"] is False
        assert parsed["physical_next"]["address"] == HEAP_BASE + 0x20
        assert parsed["physical_next"]["known"] is True
        assert parsed["physical_prev"]["address"] is None
        assert parsed["physical_prev"]["reason"] == "prev_inuse_set"
        assert parsed["user_address"] == HEAP_BASE + 16
        assert parsed["size_sz"] == 8
        assert parsed["alignment"] == 16
        assert parsed["min_chunk_size"] == 32

    def test_free_chunk_exposes_physical_prev(self):
        raw = header_bytes(prev_size=0x30, size_field=0x20)  # PREV_INUSE clear
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["fields"]["flags"]["prev_inuse"] is False
        assert parsed["physical_prev"]["address"] == HEAP_BASE - 0x30
        assert parsed["physical_prev"]["known"] is True
        assert parsed["physical_prev"]["reason"] is None

    def test_prev_size_zero_leaves_prev_unknown(self):
        raw = header_bytes(prev_size=0, size_field=0x20)
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["physical_prev"]["address"] is None
        assert parsed["physical_prev"]["reason"] == "prev_size_zero"

    def test_prev_size_below_min_chunk_is_untrusted(self):
        raw = header_bytes(prev_size=8, size_field=0x20)
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["physical_prev"]["address"] is None
        assert parsed["physical_prev"]["reason"] == "prev_size_below_min_chunk"

    def test_is_mmapped_flag_detected(self):
        raw = header_bytes(prev_size=0, size_field=0x22)  # 0x20 | IS_MMAPPED
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["fields"]["flags"]["is_mmapped"] is True
        assert parsed["fields"]["flags"]["prev_inuse"] is False

    def test_non_main_arena_flag_detected(self):
        raw = header_bytes(prev_size=0, size_field=0x24)
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["fields"]["flags"]["non_main_arena"] is True

    def test_all_three_flags_can_coexist(self):
        raw = header_bytes(prev_size=0, size_field=0x20 | 0x7)
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        flags = parsed["fields"]["flags"]
        assert flags["raw"] == 0x7
        assert flags["prev_inuse"] and flags["is_mmapped"] and flags["non_main_arena"]
        assert parsed["fields"]["size"] == 0x20

    def test_size_mask_strips_flags_only(self):
        raw = header_bytes(prev_size=0, size_field=0x121)  # size=0x120
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["fields"]["size"] == 0x120
        assert parsed["fields"]["size_field"] == 0x121
        assert parsed["fields"]["flags"]["prev_inuse"] is True

    def test_32bit_target_profile(self):
        chunk_addr = 0x08049000
        raw = header_bytes(prev_size=0, size_field=0x11, pointer_bits=32)
        parsed = GlibcHeap.parse_chunk_header(raw, chunk_addr, 32, "little")
        assert parsed["size_sz"] == 4
        assert parsed["alignment"] == 8
        assert parsed["min_chunk_size"] == 16
        assert parsed["fields"]["size"] == 0x10
        assert parsed["user_address"] == chunk_addr + 8
        assert parsed["physical_next"]["address"] == chunk_addr + 0x10

    def test_big_endian_decode(self):
        raw = (16).to_bytes(8, "big") + (0x21).to_bytes(8, "big")
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "big")
        assert parsed["fields"]["size"] == 0x20
        assert parsed["fields"]["prev_size"] == 16
        assert parsed["fields"]["flags"]["prev_inuse"] is True

    def test_insufficient_bytes_reports_unreadable(self):
        parsed = GlibcHeap.parse_chunk_header(b"\x00" * 8, HEAP_BASE, 64, "little")
        assert parsed["readable"] is False
        assert "insufficient bytes" in parsed["reason"]
        assert parsed["raw_hex"] == "0" * 16

    def test_misaligned_chunk_address_fails_alignment(self):
        raw = header_bytes(prev_size=0, size_field=0x21)
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE + 1, 64, "little")
        assert parsed["structural_checks"]["alignment_ok"] is False

    def test_misaligned_size_fails_alignment(self):
        raw = header_bytes(prev_size=0, size_field=0x29)  # size=0x28
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["fields"]["size"] == 0x28
        assert parsed["structural_checks"]["alignment_ok"] is False

    def test_undersized_chunk_fails_min_size(self):
        raw = header_bytes(prev_size=0, size_field=0x11)  # size=0x10 < 32
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        assert parsed["structural_checks"]["min_size_ok"] is False

    def test_size_below_header_leaves_next_unknown(self):
        raw = header_bytes(
            prev_size=0, size_field=0x11
        )  # size=0x10 (< 2*ptr on 64bit? 0x10==16, header=16)
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        # size == header, next equals chunk_address exactly -> known but at same spot
        assert parsed["physical_next"]["address"] == HEAP_BASE + 0x10

    def test_raw_evidence_matches_slice_bytes(self):
        raw = header_bytes(prev_size=0xABCD, size_field=0x123)
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, "little")
        ev = parsed["raw_evidence"]
        assert ev["prev_size"]["hex"] == raw[0:8].hex()
        assert ev["size_field"]["hex"] == raw[8:16].hex()
        assert ev["prev_size"]["offset"] == 0
        assert ev["size_field"]["offset"] == 8

    def test_invalid_pointer_bits_raises(self):
        with pytest.raises(ValueError):
            GlibcHeap.parse_chunk_header(b"\x00" * 16, HEAP_BASE, 16, "little")

    def test_default_byteorder_is_little_when_none(self):
        raw = header_bytes(prev_size=1, size_field=0x21)
        parsed = GlibcHeap.parse_chunk_header(raw, HEAP_BASE, 64, None)
        assert parsed["fields"]["size"] == 0x20
        assert parsed["fields"]["prev_size"] == 1


# ---------------------------------------------------------------------------
# Live-target orchestration fakes.
# ---------------------------------------------------------------------------
class ChunkFakeGdb:
    def __init__(self, *, live: bool = True) -> None:
        self._live = live
        self.state = "stopped" if live else "exited"
        self._memory: Dict[int, bytes] = {}
        self.read_calls: List[tuple] = []

    def get_state(self) -> str:
        return self.state

    def get_inferior_pid(self) -> Optional[int]:
        return 4242 if self._live else None

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
        data = self._memory.get(addr_int)
        if data is None:
            return {"success": False, "state": self.state, "responses": []}
        take = data[:num_bytes]
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


class FakeInspection:
    timeout_seconds = 60.0

    def __init__(
        self,
        gdb: ChunkFakeGdb,
        *,
        mappings: Optional[List[Dict[str, Any]]] = None,
        libc: Optional[Dict[str, Any]] = None,
        mappings_ok: bool = True,
    ) -> None:
        self.gdb = gdb
        self._mappings = mappings if mappings is not None else GLIBC_MAPS
        self._libc = libc
        self._mappings_ok = mappings_ok

    def memory_maps(self) -> Dict[str, Any]:
        return {
            "success": self._mappings_ok,
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


class FakeDebugger:
    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        if expression == "sizeof(void*)":
            return {"success": True, "parsed": {"value": "8"}}
        return {"success": False, "parsed": {}}


def build_heap(
    *,
    live: bool = True,
    mappings: Optional[List[Dict[str, Any]]] = None,
    libc: Optional[Dict[str, Any]] = None,
):
    gdb = ChunkFakeGdb(live=live)
    resolved_libc = (
        libc
        if libc is not None
        else {"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"}
    )
    inspection = FakeInspection(
        gdb,
        mappings=mappings,
        libc=None if libc is None and mappings is not None else resolved_libc,
    )
    debugger = FakeDebugger()
    discovery = AllocatorDiscovery(
        gdb, inspection, debugger, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    heap = GlibcHeap(gdb, inspection, debugger, "sess", discovery=discovery)
    return heap, gdb


# ---------------------------------------------------------------------------
# inspect_chunk live orchestration.
# ---------------------------------------------------------------------------
class TestInspectChunk:
    def test_requires_live_inferior(self):
        heap, _ = build_heap(live=False)
        with pytest.raises(LiveInferiorError):
            heap.inspect_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")

    def test_invalid_address_raises(self):
        heap, _ = build_heap()
        with pytest.raises(ValueError):
            heap.inspect_chunk(address="not-a-number", address_kind="chunk")

    def test_invalid_address_kind_raises(self):
        heap, gdb = build_heap()
        gdb.place(HEAP_BASE, header_bytes(0, 0x21))
        with pytest.raises(ValueError):
            heap.inspect_chunk(address=f"0x{HEAP_BASE:x}", address_kind="pointer")

    def test_unsupported_when_target_is_musl(self):
        heap, _ = build_heap(mappings=MUSL_MAPS, libc=None)
        result = heap.inspect_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        assert result["success"] is True
        parsed = result["parsed"]
        assert parsed["schema"] == CHUNK_INSPECTION_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"
        assert "glibc only" in parsed["reason"]

    def test_allocated_chunk_full_orchestration(self):
        heap, gdb = build_heap()
        chunk_addr = HEAP_BASE
        next_addr = chunk_addr + 0x20
        gdb.place(chunk_addr, header_bytes(0, 0x21))
        gdb.place(next_addr, header_bytes(0x20, 0x21))
        result = heap.inspect_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["supported"] is True
        assert parsed["chunk"]["fields"]["size"] == 0x20
        assert parsed["allocation"]["status"] == "allocated"
        assert parsed["allocation"]["reason"] == "next_prev_inuse_set"
        assert parsed["allocation"]["next_chunk"]["address"] == next_addr

    def test_free_when_next_prev_inuse_clear(self):
        heap, gdb = build_heap()
        chunk_addr = HEAP_BASE
        next_addr = chunk_addr + 0x20
        gdb.place(chunk_addr, header_bytes(0, 0x20))
        gdb.place(next_addr, header_bytes(0x20, 0x20))  # next PREV_INUSE clear
        result = heap.inspect_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        assert result["parsed"]["allocation"]["status"] == "free"
        assert result["parsed"]["allocation"]["reason"] == "next_prev_inuse_clear"

    def test_mmapped_chunk_reports_allocated_without_next_read(self):
        heap, gdb = build_heap()
        chunk_addr = HEAP_BASE
        # size 0x1020 | IS_MMAPPED
        gdb.place(chunk_addr, header_bytes(0, 0x1022))
        result = heap.inspect_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["allocation"]["status"] == "allocated"
        assert parsed["allocation"]["reason"] == "is_mmapped"
        assert parsed["allocation"]["next_chunk"] is None
        # Exactly one read: the header itself.
        assert len(gdb.read_calls) == 1

    def test_next_out_of_mapping_gives_unknown(self):
        heap, gdb = build_heap()
        # Chunk near heap end; its next would fall outside the heap mapping.
        chunk_addr = HEAP_END - 0x20
        gdb.place(chunk_addr, header_bytes(0, 0x21))
        result = heap.inspect_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["allocation"]["status"] == "unknown"
        assert parsed["allocation"]["reason"] == "next_out_of_mapping"

    def test_undersized_chunk_skips_next_lookup(self):
        # size < MIN_CHUNK_SIZE: chunk_address + size lands inside this chunk's
        # own user payload (on 64-bit, size=16 hits user_address exactly), so
        # dereferencing it as a next-chunk header would fabricate an allocation
        # status from arbitrary user bytes.  The gate must fire before any read.
        heap, gdb = build_heap()
        chunk_addr = HEAP_BASE
        gdb.place(chunk_addr, header_bytes(0, 0x11))  # size=0x10=16, PREV_INUSE=1
        result = heap.inspect_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["allocation"]["status"] == "unknown"
        assert parsed["allocation"]["reason"] == "size_below_min_chunk"
        assert parsed["allocation"]["next_chunk"] is None
        # Only the header itself was read; no next-chunk read attempted.
        assert len(gdb.read_calls) == 1

    def test_next_unreadable_gives_unknown(self):
        heap, gdb = build_heap()
        chunk_addr = HEAP_BASE
        # Place current chunk but not the next one.  Next address is in the
        # heap mapping so it clears that gate, but the read itself fails.
        gdb.place(chunk_addr, header_bytes(0, 0x21))
        result = heap.inspect_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["allocation"]["status"] == "unknown"
        assert "read" in parsed["allocation"]["reason"]
        assert parsed["allocation"]["next_chunk"]["readable"] is False

    def test_header_unreadable_reports_verdict(self):
        heap, _ = build_heap()
        # Nothing placed at HEAP_BASE.
        result = heap.inspect_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["chunk"]["readable"] is False
        assert parsed["allocation"]["status"] == "unknown"

    def test_address_kind_user_subtracts_header(self):
        heap, gdb = build_heap()
        chunk_addr = HEAP_BASE
        user_addr = chunk_addr + 16
        next_addr = chunk_addr + 0x20
        gdb.place(chunk_addr, header_bytes(0, 0x21))
        gdb.place(next_addr, header_bytes(0x20, 0x21))
        result = heap.inspect_chunk(address=f"0x{user_addr:x}", address_kind="user")
        chunk = result["parsed"]["chunk"]
        assert chunk["chunk_address"] == chunk_addr
        assert chunk["user_address"] == user_addr

    def test_include_next_false_skips_second_read(self):
        heap, gdb = build_heap()
        gdb.place(HEAP_BASE, header_bytes(0, 0x21))
        result = heap.inspect_chunk(
            address=f"0x{HEAP_BASE:x}",
            address_kind="chunk",
            include_next=False,
        )
        assert result["parsed"]["allocation"]["reason"] == "next_lookup_disabled"
        assert len(gdb.read_calls) == 1

    def test_mapping_containment_populated_for_valid_chunk(self):
        heap, gdb = build_heap()
        gdb.place(HEAP_BASE, header_bytes(0, 0x21))
        gdb.place(HEAP_BASE + 0x20, header_bytes(0x20, 0x21))
        result = heap.inspect_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        containment = result["parsed"]["mapping_containment"]
        assert containment["chunk_in_mapping"] is True
        assert containment["header_in_mapping"] is True
        assert containment["mapping"]["file"] == "[heap]"

    def test_unreadable_byteorder_warning_absent_when_resolved(self):
        heap, gdb = build_heap()
        gdb.place(HEAP_BASE, header_bytes(0, 0x21))
        gdb.place(HEAP_BASE + 0x20, header_bytes(0x20, 0x21))
        result = heap.inspect_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        warnings = result.get("warnings", [])
        assert not any("byteorder" in w for w in warnings)


# ---------------------------------------------------------------------------
# validate_chunk verdicts.
# ---------------------------------------------------------------------------
class TestValidateChunk:
    def test_valid_verdict_on_clean_allocated_chunk(self):
        heap, gdb = build_heap()
        gdb.place(HEAP_BASE, header_bytes(0, 0x21))
        gdb.place(HEAP_BASE + 0x20, header_bytes(0x20, 0x21))
        result = heap.validate_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["schema"] == CHUNK_VALIDATION_SCHEMA
        assert parsed["verdict"] == "valid"
        assert parsed["failures"] == []
        assert parsed["chunk_summary"]["size"] == 0x20
        assert parsed["chunk_summary"]["allocation_status"] == "allocated"

    def test_corrupted_on_misaligned_chunk(self):
        heap, gdb = build_heap()
        chunk_addr = HEAP_BASE + 1  # off-by-one -> not 16-aligned
        gdb.place(chunk_addr, header_bytes(0, 0x21))
        result = heap.validate_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "corrupted"
        assert "alignment" in parsed["failures"]

    def test_corrupted_on_undersized_chunk(self):
        heap, gdb = build_heap()
        gdb.place(HEAP_BASE, header_bytes(0, 0x11))  # size=0x10 < 32
        result = heap.validate_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "corrupted"
        assert "min_size" in parsed["failures"]

    def test_corrupted_when_chunk_outside_any_mapping(self):
        heap, gdb = build_heap()
        chunk_addr = 0xDEAD0000  # not in any GLIBC_MAPS mapping
        gdb.place(chunk_addr, header_bytes(0, 0x21))
        result = heap.validate_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "corrupted"
        assert "chunk_in_mapping" in parsed["failures"]
        assert "header_in_mapping" in parsed["failures"]

    def test_suspicious_when_payload_extends_past_mapping(self):
        heap, gdb = build_heap()
        chunk_addr = HEAP_END - 0x30  # only 0x30 bytes room
        # size 0x50 -> payload_end = chunk + 0x50 > mapping end
        gdb.place(chunk_addr, header_bytes(0, 0x51))
        result = heap.validate_chunk(address=f"0x{chunk_addr:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "suspicious"
        assert parsed["failures"] == ["payload_in_mapping"]

    def test_unreadable_verdict_when_header_read_fails(self):
        heap, _ = build_heap()
        # Nothing placed => read fails.
        result = heap.validate_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert "readable" in parsed["failures"]

    def test_unsupported_verdict_for_non_glibc_target(self):
        heap, _ = build_heap(mappings=MUSL_MAPS, libc=None)
        result = heap.validate_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "unsupported"
        assert parsed["chunk_summary"] is None


# ---------------------------------------------------------------------------
# HeapTools facade wires discovery + chunk parsing under one entry point.
# ---------------------------------------------------------------------------
class TestHeapToolsFacade:
    def _make_facade(self) -> tuple:
        gdb = ChunkFakeGdb(live=True)
        inspection = FakeInspection(
            gdb,
            libc={"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"},
        )
        debugger = FakeDebugger()

        class _Wrap(HeapTools):
            pass

        facade = _Wrap.__new__(_Wrap)
        facade.discovery = AllocatorDiscovery(
            gdb,
            inspection,
            debugger,
            "sess",
            binary_path_provider=lambda: "/bin/chal",
        )
        facade.glibc = GlibcHeap(
            gdb, inspection, debugger, "sess", discovery=facade.discovery
        )
        return facade, gdb

    def test_facade_routes_detect_allocator(self):
        facade, _ = self._make_facade()
        result = facade.detect_allocator()
        assert result["parsed"]["family"] == "glibc"

    def test_facade_routes_inspect_chunk(self):
        facade, gdb = self._make_facade()
        gdb.place(HEAP_BASE, header_bytes(0, 0x21))
        gdb.place(HEAP_BASE + 0x20, header_bytes(0x20, 0x21))
        result = facade.inspect_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        assert result["parsed"]["verdict"] == "parsed"

    def test_facade_routes_validate_chunk(self):
        facade, gdb = self._make_facade()
        gdb.place(HEAP_BASE, header_bytes(0, 0x21))
        gdb.place(HEAP_BASE + 0x20, header_bytes(0x20, 0x21))
        result = facade.validate_chunk(address=f"0x{HEAP_BASE:x}", address_kind="chunk")
        assert result["parsed"]["verdict"] == "valid"
