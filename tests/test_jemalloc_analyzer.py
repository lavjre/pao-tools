"""Tests for the Phase 1 jemalloc analyzer (extent/rtree core only).

The analyzer runs pwndbg's ``jemalloc`` command surface, so these tests use a
fake gdb whose ``execute_command`` returns canned console records -- exactly
the record shape the real controller emits (``{"type": "console", "payload"}``).
The discovery stack is the shared heap fake (ArenaFakeGdb / ArenaFakeInspection
/ ArenaFakeDebugger) over jemalloc mapping evidence.
"""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    ARENA_INSPECTION_SCHEMA,
    BIN_INSPECTION_SCHEMA,
    FASTBIN_INSPECTION_SCHEMA,
    FREE_VALIDATION_SCHEMA,
    HEAP_CORRUPTION_SCHEMA,
    HEAP_SNAPSHOT_SCHEMA,
    HEAP_TECHNIQUE_SCHEMA,
    TCACHE_INSPECTION_SCHEMA,
    UAF_SCAN_SCHEMA,
)
from paomcp.tools.backends.jemalloc import (
    JEMALLOC_EXTENT_LIST_SCHEMA,
    JEMALLOC_EXTENT_SCHEMA,
)

from tests.test_glibc_arenas import (
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
    mapping,
)

# A libjemalloc mapping (the detection signal) plus a heap-extent anon region.
JEMALLOC_MAPS = [
    mapping("/bin/chal", "0x555555554000", "0x555555556000", writable=False),
    mapping("/usr/lib/libjemalloc.so.2", "0x7ffff7c00000", "0x7ffff7e00000"),
    mapping("[anon:je_extent]", "0x7ffff6800000", "0x7ffff6a00000"),
]

USER_PTR = 0x7FFFF6801000
EXTENT_ADDR = 0x7FFFF6800000

_BANNER = "This command was tested only for jemalloc 5.3.0 and does not support lower versions\n"

FIND_EXTENT_FOUND = [
    "\x1b[1mJemalloc find extent\x1b[0m\n",  # ANSI must be stripped
    _BANNER,
    "\n",
    f"Pointer Address: {hex(USER_PTR)}\n",
    f"Extent Address: {hex(EXTENT_ADDR)}\n",
    "\n",
    f"Allocated Address: {hex(USER_PTR)}\n",
    f"Extent Address: {hex(EXTENT_ADDR)}\n",
    "Size: 0x1000\n",
    "Small class: True\n",
    "State: active\n",
]

FIND_EXTENT_NOT_FOUND = [
    "Jemalloc find extent\n",
    _BANNER,
    "\n",
    "\x1b[31mERROR: Extent not found\x1b[0m\n",
]

FIND_EXTENT_ERROR = [
    "Jemalloc find extent\n",
    _BANNER,
    "\n",
    "ERROR: Could not read memory at 0x0\n",
]

EXTENT_INFO = [
    "Jemalloc extent info\n",
    _BANNER,
    "\n",
    f"Allocated Address: {hex(USER_PTR)}\n",
    f"Extent Address: {hex(EXTENT_ADDR)}\n",
    "Size: 0x1000\n",
    "Small class: True\n",
    "State: active\n",
]

HEAP_TWO_EXTENTS = [
    "Jemalloc heap\n",
    _BANNER,
    "\n",
    f"Allocated Address: {hex(USER_PTR)}\n",
    f"Extent Address: {hex(EXTENT_ADDR)}\n",
    "Size: 0x1000\n",
    "Small class: True\n",
    "State: active\n",
    "\n",
    "Allocated Address: 0x7ffff6902000\n",
    "Extent Address: 0x7ffff6900000\n",
    "Size: 0x2000\n",
    "Small class: False\n",
    "State: inactive\n",
    "\n",
]

HEAP_NO_EXTENTS = [
    "Jemalloc heap\n",
    _BANNER,
    "\n",
    "No extents found\n",
]

HEAP_ERROR = [
    "Jemalloc heap\n",
    _BANNER,
    "\n",
    "ERROR: No rtree symbols found\n",
]


class JemallocFakeGdb(ArenaFakeGdb):
    """ArenaFakeGdb plus canned jemalloc command console output."""

    def __init__(
        self, *, commands: Optional[Dict[str, List[str]]] = None, live: bool = True
    ) -> None:
        super().__init__(live=live)
        self._commands = commands or {}
        self.executed: List[str] = []

    def execute_command(
        self, command: str, timeout_sec: float = 60.0
    ) -> Dict[str, Any]:
        self.executed.append(command)
        if command == "show endian":
            return super().execute_command(command)
        for prefix, lines in self._commands.items():
            if command.startswith(prefix):
                return {
                    "success": True,
                    "state": self.state,
                    "responses": [
                        {"type": "console", "payload": line} for line in lines
                    ],
                }
        return {"success": True, "state": self.state, "responses": []}


def build_jemalloc(
    *,
    live: bool = True,
    commands: Optional[Dict[str, List[str]]] = None,
):
    gdb = JemallocFakeGdb(live=live, commands=commands)
    inspection = ArenaFakeInspection(gdb, mappings=JEMALLOC_MAPS, libc=None)
    debugger = ArenaFakeDebugger()
    from paomcp.tools.backends.heap import HeapTools

    heap = HeapTools(
        gdb, inspection, debugger, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    return heap, gdb


# ---------------------------------------------------------------------------
# Extent lookups.
# ---------------------------------------------------------------------------
class TestInspectChunk:
    def test_requires_live_inferior(self):
        heap, _ = build_jemalloc(live=False)
        with pytest.raises(LiveInferiorError):
            heap.inspect_chunk(address=hex(USER_PTR))

    def test_bad_address_raises(self):
        heap, _ = build_jemalloc(commands={"jemalloc find-extent": FIND_EXTENT_FOUND})
        with pytest.raises(ValueError):
            heap.inspect_chunk(address="not-a-number")

    def test_found_extent_parses_fields(self):
        heap, gdb = build_jemalloc(commands={"jemalloc find-extent": FIND_EXTENT_FOUND})
        result = heap.inspect_chunk(address=hex(USER_PTR))
        assert result["success"] is True
        parsed = result["parsed"]
        assert parsed["schema"] == JEMALLOC_EXTENT_SCHEMA
        assert parsed["supported"] is True
        assert parsed["verdict"] == "allocated"
        assert parsed["address"] == USER_PTR
        assert parsed["allocator"]["family"] == "jemalloc"
        extent = parsed["extent"]
        assert extent["pointer_address"] == USER_PTR
        assert extent["extent_address"] == EXTENT_ADDR
        assert extent["allocated_address"] == USER_PTR
        assert extent["size"] == 0x1000
        assert extent["has_slab"] is True
        assert extent["state"] == "active"
        # The pwndbg command must be issued with a decimal address (argparse
        # type=int rejects hex literals).
        assert gdb.executed[-1] == f"jemalloc find-extent {USER_PTR}"

    def test_not_allocated(self):
        heap, _ = build_jemalloc(
            commands={"jemalloc find-extent": FIND_EXTENT_NOT_FOUND}
        )
        parsed = heap.inspect_chunk(address=hex(USER_PTR))["parsed"]
        assert parsed["verdict"] == "not_allocated"
        assert parsed["extent"] is None

    def test_unresolvable_error(self):
        heap, _ = build_jemalloc(commands={"jemalloc find-extent": FIND_EXTENT_ERROR})
        parsed = heap.inspect_chunk(address=hex(USER_PTR))["parsed"]
        assert parsed["verdict"] == "unresolvable"
        assert "Could not read memory" in parsed["reason"]


class TestValidateChunk:
    def test_valid_when_extent_found(self):
        heap, _ = build_jemalloc(commands={"jemalloc find-extent": FIND_EXTENT_FOUND})
        parsed = heap.validate_chunk(address=hex(USER_PTR))["parsed"]
        assert parsed["verdict"] == "valid"

    def test_invalid_when_not_allocated(self):
        heap, _ = build_jemalloc(
            commands={"jemalloc find-extent": FIND_EXTENT_NOT_FOUND}
        )
        parsed = heap.validate_chunk(address=hex(USER_PTR))["parsed"]
        assert parsed["verdict"] == "invalid"


# ---------------------------------------------------------------------------
# Extent inventory + extent-info.
# ---------------------------------------------------------------------------
class TestListArenas:
    def test_lists_extents(self):
        heap, _ = build_jemalloc(commands={"jemalloc heap": HEAP_TWO_EXTENTS})
        parsed = heap.list_arenas()["parsed"]
        assert parsed["schema"] == JEMALLOC_EXTENT_LIST_SCHEMA
        assert parsed["verdict"] == "listed"
        assert parsed["extent_count"] == 2
        assert parsed["extents"][0]["size"] == 0x1000
        assert parsed["extents"][1]["state"] == "inactive"
        assert parsed["extents"][1]["has_slab"] is False

    def test_no_extents(self):
        heap, _ = build_jemalloc(commands={"jemalloc heap": HEAP_NO_EXTENTS})
        parsed = heap.list_arenas()["parsed"]
        assert parsed["verdict"] == "no_extents"
        assert parsed["extent_count"] == 0

    def test_unresolvable(self):
        heap, _ = build_jemalloc(commands={"jemalloc heap": HEAP_ERROR})
        parsed = heap.list_arenas()["parsed"]
        assert parsed["verdict"] == "unresolvable"
        assert "rtree symbols" in parsed["reason"]


class TestInspectArena:
    def test_unsupported_without_address(self):
        heap, _ = build_jemalloc()
        parsed = heap.inspect_arena()["parsed"]
        assert parsed["verdict"] == "unsupported"
        assert parsed["schema"] == ARENA_INSPECTION_SCHEMA

    def test_extent_info_with_address(self):
        heap, _ = build_jemalloc(commands={"jemalloc extent-info": EXTENT_INFO})
        parsed = heap.inspect_arena(arena_address=hex(EXTENT_ADDR))["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["schema"] == JEMALLOC_EXTENT_SCHEMA
        assert parsed["extent"]["extent_address"] == EXTENT_ADDR


# ---------------------------------------------------------------------------
# Unsupported Phase 1 surface.
# ---------------------------------------------------------------------------
class TestUnsupportedSurface:
    @pytest.mark.parametrize(
        "method, schema",
        [
            ("inspect_fastbins", FASTBIN_INSPECTION_SCHEMA),
            ("inspect_tcache", TCACHE_INSPECTION_SCHEMA),
            ("inspect_bins", BIN_INSPECTION_SCHEMA),
            ("classify_heap_corruption", HEAP_CORRUPTION_SCHEMA),
            ("validate_free", FREE_VALIDATION_SCHEMA),
            ("analyze_heap_technique", HEAP_TECHNIQUE_SCHEMA),
            ("detect_uaf", UAF_SCAN_SCHEMA),
            ("capture_heap_snapshot", HEAP_SNAPSHOT_SCHEMA),
        ],
    )
    def test_returns_typed_unsupported(self, method, schema):
        heap, _ = build_jemalloc()
        kwargs: Dict[str, Any] = {}
        if method == "validate_free":
            kwargs["address"] = "0x1000"
        parsed = getattr(heap, method)(**kwargs)["parsed"]
        assert parsed["verdict"] == "unsupported"
        assert parsed["supported"] is False
        assert parsed["schema"] == schema
        assert parsed["allocator"]["family"] == "jemalloc"
